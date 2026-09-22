# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible Images API adapter (OpenRouter, fal OpenAI shim, local servers).

Speaks the documented Images surface used by ``openai_provider`` —
``POST {base_url}/images/generations`` (JSON) and ``POST {base_url}/images/edits``
(multipart with ``image[]``) — against any host that implements it. That covers
OpenRouter image models, fal's OpenAI-compatible endpoint, and several local
servers that expose the same shape.

Unlike the built-in ``openai`` provider, the base URL, model name, key
environment variable and transparency strategy come from the providers config
(``kind = "openai_compatible"``). A provider marked ``billed = true`` is
explicit-only and announces its charge the same way the built-in openai route
does (구독 우선 불변식).

Known limits (No Silent Fallback — refused rather than ignored):

* ``--resolution`` is refused: this API sizes from ``size`` / ``--aspect-ratio``.
* Quality levels outside ``quality_levels`` (config, default ``auto`` only) are
  refused instead of downgraded.
* ``native`` transparency is refused unless the config declares
  ``transparency = "native"`` *and* the host accepts ``background: transparent``.
* More refs than ``max_refs`` (config, default 1 for generations-only hosts) fail
  before upload.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from .base import (
    GEN_TIMEOUT_SECONDS,
    QUALITIES,
    TRANSPARENCY_NATIVE,
    GenRequest,
    ProviderRun,
    announce_api_billing,
    publish_png,
)
from .registry import ProviderSpec

_REFERENCE_MIMES = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}

# Aspect ratios this adapter maps to a concrete `size` when the config does not
# override `sizes`. Same constraint family as gpt-image (sides % 16 == 0).
_DEFAULT_SIZES = {
    "auto": "auto",
    "1:1": "1024x1024",
    "3:2": "1536x1024",
    "2:3": "1024x1536",
    "4:3": "1024x768",
    "3:4": "768x1024",
    "16:9": "1536x864",
    "9:16": "864x1536",
    "2:1": "1440x720",
    "1:2": "720x1440",
}


class OpenAICompatibleProvider:
    """Config-driven Images API client. Implements ``Provider``."""

    def __init__(self, spec: ProviderSpec) -> None:
        self.spec = spec
        self.name = spec.name
        self.transparency = spec.transparency
        opts = spec.options
        base = str(opts.get("base_url") or "").rstrip("/")
        if not base:
            raise SystemExit(
                f"gen: providers.{spec.name}: openai_compatible needs base_url "
                f"(from {spec.source})"
            )
        self.base_url = base
        self.default_model = opts.get("model")
        self.api_key_env = str(opts.get("api_key_env") or "OPENAI_API_KEY")
        self.auth_header = str(opts.get("auth_header") or "Authorization")
        self.auth_scheme = str(opts.get("auth_scheme") or "Bearer")  # empty = raw key
        self.max_refs = int(opts.get("max_refs", 1))
        self.supported_qualities = tuple(opts.get("quality_levels") or ("auto",))
        self.sizes = {**_DEFAULT_SIZES, **{str(k): str(v) for k, v in (opts.get("sizes") or {}).items()}}
        self.default_size = str(opts.get("size") or self.sizes["1:1"])
        self.vision_model = opts.get("vision_model")

    # -- credentials ---------------------------------------------------------

    def resolve_credential(self) -> str:
        env = os.environ
        key = (env.get(self.api_key_env) or "").strip()
        if key:
            return key
        if self.api_key_env in env:
            raise SystemExit(
                f"gen[{self.name}]: {self.api_key_env} is set but empty; give it a value"
            )
        raise SystemExit(
            f"gen[{self.name}]: no credential — {self.api_key_env} is not set "
            f"(config: {self.spec.source})"
        )

    def _auth_value(self, token: str) -> str:
        return f"{self.auth_scheme} {token}".strip()

    # -- request shaping -----------------------------------------------------

    def _fields(self, request: GenRequest) -> dict[str, Any]:
        if not request.prompt.strip():
            raise SystemExit(f"gen[{self.name}]: empty prompt")
        if len(request.refs) > self.max_refs:
            raise SystemExit(
                f"gen[{self.name}]: at most {self.max_refs} reference image(s) "
                f"are configured (max_refs); got {len(request.refs)}"
            )
        if request.resolution is not None:
            raise SystemExit(
                f"gen[{self.name}]: resolution {request.resolution!r} is grok Imagine's "
                "vocabulary; this API sizes from --aspect-ratio"
            )
        quality = request.quality or "auto"
        if quality not in self.supported_qualities:
            raise SystemExit(
                f"gen[{self.name}]: unsupported quality {quality!r}; "
                f"expected one of {', '.join(self.supported_qualities)}"
            )
        if request.aspect_ratio is None:
            size = self.default_size
        elif request.aspect_ratio in self.sizes:
            size = self.sizes[request.aspect_ratio]
        else:
            raise SystemExit(
                f"gen[{self.name}]: aspect ratio {request.aspect_ratio!r} has no mapped size; "
                f"expected one of {', '.join(self.sizes)}"
            )
        model = request.model or self.default_model
        if not model:
            raise SystemExit(
                f"gen[{self.name}]: no model — pass --model or set providers.{self.name}.model"
            )
        fields: dict[str, Any] = {
            "model": model,
            "prompt": request.prompt,
            "n": 1,
            "size": size,
        }
        if quality != "auto" or "auto" not in self.supported_qualities:
            fields["quality"] = quality
        if request.native_alpha:
            if self.transparency != TRANSPARENCY_NATIVE:
                raise SystemExit(
                    f"gen[{self.name}]: transparency strategy is {self.transparency!r}; "
                    "native alpha is not a capability of this provider"
                )
            fields["background"] = "transparent"
            fields["output_format"] = "png"
        return fields

    def _reference(self, path: Path, index: int) -> tuple[str, str, bytes]:
        try:
            data = path.read_bytes()
            with Image.open(io.BytesIO(data)) as source:
                source.load()
                mime = Image.MIME.get(source.format)
            if mime not in _REFERENCE_MIMES:
                raise ValueError("reference must be PNG, JPEG or WebP")
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            raise SystemExit(f"gen[{self.name}]: invalid reference image {path}") from exc
        return f"image-{index}.{_REFERENCE_MIMES[mime]}", mime, data

    def _multipart(self, fields: dict[str, Any], refs: list[Path]) -> tuple[bytes, str]:
        boundary = "----sprite-gen-" + uuid.uuid4().hex
        marker = f"--{boundary}\r\n".encode("utf-8")
        chunks: list[bytes] = []
        for name, value in fields.items():
            chunks.append(marker)
            chunks.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
            chunks.append(str(value).encode("utf-8") + b"\r\n")
        for index, ref in enumerate(refs):
            filename, mime, data = self._reference(ref, index)
            chunks.append(marker)
            chunks.append(
                f'Content-Disposition: form-data; name="image[]"; filename="{filename}"\r\n'
                f"Content-Type: {mime}\r\n\r\n".encode("utf-8")
            )
            chunks.append(data + b"\r\n")
        chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
        return b"".join(chunks), f"multipart/form-data; boundary={boundary}"

    def _post(self, path: str, token: str, data: bytes, content_type: str) -> tuple[int, Any]:
        url = self.base_url + path
        request = urllib.request.Request(url, data=data, method="POST")
        request.add_header(self.auth_header, self._auth_value(token))
        request.add_header("Accept", "application/json")
        request.add_header("Content-Type", content_type)
        try:
            with urllib.request.urlopen(request, timeout=GEN_TIMEOUT_SECONDS) as response:
                raw = response.read().decode("utf-8")
                return response.status, (json.loads(raw) if raw else {})
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise SystemExit(f"gen[{self.name}]: response was not valid JSON") from exc
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", "replace")
            try:
                return exc.code, json.loads(raw)
            except json.JSONDecodeError:
                return exc.code, {"raw": raw[:400]}
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise SystemExit(
                f"gen[{self.name}]: request failed or timed out; no automatic retry "
                "(the server may have accepted it)"
            ) from exc

    def _publish(self, item: dict, path: Path) -> None:
        encoded = item.get("b64_json")
        url = item.get("url")
        if isinstance(encoded, str) and encoded:
            try:
                data = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise SystemExit(f"gen[{self.name}]: response image is invalid; nothing published") from exc
            publish_png(data, path, label=f"gen[{self.name}]")
            return
        if isinstance(url, str) and url:
            # Some OpenAI-compatible hosts return a URL instead of b64. Download
            # once; a signed URL is not logged or reported.
            try:
                with urllib.request.urlopen(url, timeout=GEN_TIMEOUT_SECONDS) as response:
                    data = response.read()
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                raise SystemExit(
                    f"gen[{self.name}]: failed to download the result image; nothing published"
                ) from exc
            publish_png(data, path, label=f"gen[{self.name}]")
            return
        raise SystemExit(
            f"gen[{self.name}]: response has neither b64_json nor url; nothing published"
        )

    # -- Provider ------------------------------------------------------------

    def inspect_facing(self, path: Path, workdir: Path) -> tuple[str, dict[str, Any]]:
        from . import facing_vision as vision

        if not self.vision_model:
            raise SystemExit(
                f"gen[{self.name}]: facing inspection needs providers.{self.name}.vision_model "
                "in the providers config"
            )
        token = self.resolve_credential()
        if self.spec.billed:
            announce_api_billing(self.name, self.api_key_env, " (facing vision check).")
        body = json.dumps(vision.request_body(path, str(self.vision_model))).encode("utf-8")
        # Vision rides the chat-completions surface, not /images/*.
        status, reply = self._post("/chat/completions", token, body, "application/json")
        text, metadata = vision.response_text(status, reply)
        return text, {
            **metadata,
            "auth_source": self.api_key_env,
            "transport": "openai-compatible",
            "endpoint": "/chat/completions",
        }

    def generate(self, request: GenRequest, workdir: Path) -> ProviderRun:
        fields = self._fields(request)
        refs = [Path(ref) for ref in request.refs]
        if refs:
            endpoint = "/images/edits"
            data, content_type = self._multipart(fields, refs)
        else:
            endpoint = "/images/generations"
            data, content_type = json.dumps(fields).encode("utf-8"), "application/json"
        token = self.resolve_credential()
        if self.spec.billed:
            announce_api_billing(
                self.name,
                self.api_key_env,
                f" (model={fields['model']}, size={fields['size']}).",
            )
        started = time.monotonic()
        status, reply = self._post(endpoint, token, data, content_type)
        if status in (401, 403):
            raise SystemExit(
                f"gen[{self.name}]: credential {self.api_key_env} rejected (HTTP {status})"
            )
        if status != 200:
            raise SystemExit(
                f"gen[{self.name}]: image request failed (HTTP {status}); no retry or provider fallback"
            )
        items = reply.get("data") if isinstance(reply, dict) else None
        if not isinstance(items, list) or len(items) != 1 or not isinstance(items[0], dict):
            raise SystemExit(
                f"gen[{self.name}]: expected exactly one image in the response; nothing published"
            )
        self._publish(items[0], request.raw)
        return ProviderRun(
            provider=self.name,
            elapsed_seconds=time.monotonic() - started,
            model=str(fields["model"]),
            extra={
                "auth_source": self.api_key_env,
                "transport": "openai-compatible",
                "endpoint": endpoint,
                "base_url": self.base_url,
                "size": fields["size"],
                "billed": self.spec.billed,
            },
        )
