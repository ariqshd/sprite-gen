# SPDX-License-Identifier: Apache-2.0
"""Google Gemini image provider (AI Studio API key; free tier / pay-per-image).

POST {GEMINI_BASE_URL:-https://generativelanguage.googleapis.com}/v1beta/models/
{model}:generateContent with ``x-goog-api-key``. Unlike the Z.ai API, Gemini
returns the image inline as base64 ``inlineData`` (no download step) and accepts
reference images as inline parts — so ``--ref`` image editing works here.
Truth is the decoded PNG bytes on disk: whatever the model claims, the base64
payload is decoded and re-encoded through Pillow into the verified PNG contract.
Transparency is the chroma strategy: Gemini's alpha support is unmeasured, so
the request prompt must carry the key background (prepared prompts already do).

Docs: https://ai.google.dev/gemini-api/docs/image-generation
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path

from PIL import Image

from .base import (
    GEN_TIMEOUT_SECONDS,
    TRANSPARENCY_CHROMA,
    GenRequest,
    GenTimeoutError,
    ProviderRun,
    verify_png,
)

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
DEFAULT_MODEL = "gemini-2.5-flash-image"

# Aspect ratios the image models accept via imageConfig (docs 2026-09). Sprite
# prompts name aspects from this same family; anything else falls back to the
# server default and is recorded in the report instead of failing the run.
_ASPECTS = frozenset({"1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"})

_PIL_TO_MIME = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}


def resolve_api_key() -> str:
    """Read GEMINI_API_KEY (or GOOGLE_API_KEY), failing loud when absent."""
    key = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "gemini-gen: GEMINI_API_KEY is not set. Create a key at https://aistudio.google.com/apikey "
            "and export GEMINI_API_KEY. Image output bills per image (a free-tier daily quota may apply)."
        )
    return key


def _ref_part(ref: Path) -> dict:
    """Encode one reference image as an inline request part, sniffing its real format."""
    path = Path(ref).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"gemini-gen: reference image not found: {path}")
    with Image.open(path) as image:
        mime = _PIL_TO_MIME.get(image.format)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        data = base64.b64encode(buffer.getvalue()).decode("ascii")
    if mime is None:
        mime = "image/png"
    return {"inlineData": {"mimeType": mime, "data": data}}


def _extract_image_part(payload: dict) -> tuple[bytes, list[str]]:
    """Pull the first inline image out of the response; return (bytes, seen part kinds)."""
    candidates = payload.get("candidates") or []
    seen: list[str] = []
    for candidate in candidates:
        parts = ((candidate or {}).get("content") or {}).get("parts") or []
        for part in parts:
            inline = part.get("inlineData") or part.get("inline_data")
            if inline:
                seen.append(str(inline.get("mimeType") or inline.get("mime_type") or "image"))
                data = inline.get("data")
                if data and str(inline.get("mimeType") or inline.get("mime_type") or "").startswith("image/"):
                    return base64.b64decode(data), seen
            elif "text" in part:
                seen.append("text")
    if not candidates:
        feedback = payload.get("promptFeedback") or {}
        raise SystemExit(
            "gemini-gen: response has no candidates"
            + (f" (blocked: {feedback.get('blockReason')})" if feedback.get("blockReason") else "")
            + f": {json.dumps(payload)[:500]}"
        )
    raise SystemExit(
        f"gemini-gen: no inline image in response (parts: {', '.join(seen) or 'none'}); "
        f"tail: {json.dumps(payload)[:500]}"
    )


class GeminiProvider:
    """Generate one image through the Gemini API generateContent endpoint."""

    name = "gemini"
    # Alpha support is unmeasured (2026-09): declare chroma so transparency runs
    # on a key background, the pipeline's most deterministic path.
    transparency = TRANSPARENCY_CHROMA

    def generate(self, request: GenRequest, workdir: Path) -> ProviderRun:
        if request.native_alpha:
            raise SystemExit(
                "gemini-gen: native alpha was requested but Gemini alpha output is unmeasured "
                f"(transparency strategy is {self.transparency!r}); generate on a chroma key instead"
            )
        # Local problems (missing ref files) surface before credential problems.
        parts: list[dict] = [_ref_part(ref) for ref in request.refs]
        parts.append({"text": request.prompt})
        api_key = resolve_api_key()
        base_url = os.environ.get("GEMINI_BASE_URL", "").strip() or DEFAULT_BASE_URL
        model = request.model or os.environ.get("GEMINI_IMAGE_MODEL", "").strip() or DEFAULT_MODEL

        generation_config: dict = {"responseModalities": ["TEXT", "IMAGE"]}
        aspect = (request.aspect_ratio or "").strip()
        if aspect in _ASPECTS:
            generation_config["imageConfig"] = {"aspectRatio": aspect}
        body = {
            "contents": [{"parts": parts}],
            "generationConfig": generation_config,
        }

        request.raw.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + GEN_TIMEOUT_SECONDS
        started = time.monotonic()
        payload = _post_generate(body, base_url, model, api_key, deadline)

        image_bytes, seen = _extract_image_part(payload)
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                image.save(request.raw, format="PNG")
        except Exception as exc:
            raise SystemExit(
                f"gemini-gen: inline image bytes are not decodable ({exc.__class__.__name__}: {exc})"
            ) from exc
        raw_bytes = verify_png(request.raw)
        elapsed = time.monotonic() - started

        finish = (payload.get("candidates") or [{}])[0].get("finishReason")
        return ProviderRun(
            provider=self.name,
            elapsed_seconds=elapsed,
            model=model,
            session_id=None,
            extra={
                "aspect_ratio": aspect or None,
                "inline_bytes": len(image_bytes),
                "png_bytes": raw_bytes,
                "part_kinds": seen,
                "finish_reason": finish,
            },
        )


def _post_generate(body: dict, base_url: str, model: str, api_key: str, deadline: float) -> dict:
    url = f"{base_url.rstrip('/')}/v1beta/models/{model}:generateContent"
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise GenTimeoutError(
            f"gemini-gen: no completion within {GEN_TIMEOUT_SECONDS}s — "
            "provider call budget exhausted before POST (gen-timeout)")
    try:
        with urllib.request.urlopen(request, timeout=remaining) as response:
            return json.loads(response.read().decode("utf-8"))
    except TimeoutError as exc:
        raise GenTimeoutError(
            f"gemini-gen: no completion within {GEN_TIMEOUT_SECONDS}s — "
            "generation POST stalled; killed so the orchestrator can retry (gen-timeout)"
        ) from exc
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()[:500]
        raise SystemExit(f"gemini-gen: HTTP {exc.code} from the Gemini API: {detail}") from exc
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"gemini-gen: Gemini API request failed: {exc}") from exc
