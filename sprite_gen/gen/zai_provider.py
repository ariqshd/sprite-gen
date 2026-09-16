# SPDX-License-Identifier: Apache-2.0
"""Z.ai GLM-Image provider (open-platform API key, pay-per-use).

POST {ZAI_BASE_URL:-https://api.z.ai}/api/paas/v4/images/generations with a
Bearer ``ZAI_API_KEY``. The response carries ``data[0].url`` — a temporary
(30-day) hosted image; the API has no base64 mode, no alpha channel and no
image-edit (reference) parameter. Truth is the decoded PNG bytes on disk: the
provider downloads the URL, re-encodes the pixels through Pillow, and hands a
verified PNG to the orchestrator — the CDN's byte format is not part of the
contract. Transparency therefore follows the chroma strategy: the request
prompt must carry the key background (the prepared sprite prompts already do).

Docs: https://docs.z.ai/api-reference/image/generate-image
"""

from __future__ import annotations

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

DEFAULT_BASE_URL = "https://api.z.ai"
DEFAULT_MODEL = "glm-image"
GENERATIONS_PATH = "/api/paas/v4/images/generations"

# glm-image enum sizes (docs 2026-09): every one is 1024..2048 and divisible by
# 32, which is also exactly the model's custom-size rule — so one validation
# covers both the enum map below and a literal "WxH" aspect passthrough.
_SIZE_RANGE = (1024, 2048)
_ASPECT_SIZES = {
    "1:1": "1280x1280",
    "3:2": "1568x1056",
    "2:3": "1056x1568",
    "4:3": "1472x1088",
    "3:4": "1088x1472",
    "16:9": "1728x960",
    "9:16": "960x1728",
}


def resolve_api_key() -> str:
    """Read ZAI_API_KEY, failing loud with setup instructions when absent."""
    key = os.environ.get("ZAI_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "zai-gen: ZAI_API_KEY is not set. Create an API key in the Z.ai open platform "
            "console (https://z.ai -> API Keys), then export ZAI_API_KEY. "
            "Image generation is pay-per-use billing on that key — it is separate from a "
            "GLM Coding Plan subscription."
        )
    return key


def _aspect_to_size(aspect: str | None) -> str | None:
    if not aspect:
        return None
    aspect = aspect.strip().lower()
    if aspect in _ASPECT_SIZES:
        return _ASPECT_SIZES[aspect]
    if "x" in aspect:
        width, _, height = aspect.partition("x")
        if width.isdigit() and height.isdigit():
            w, h = int(width), int(height)
            lo, hi = _SIZE_RANGE
            if lo <= w <= hi and lo <= h <= hi and w % 32 == 0 and h % 32 == 0:
                return f"{w}x{h}"
        raise SystemExit(
            f"zai-gen: unsupported size {aspect!r}; width/height must be 1024-2048 and "
            "divisible by 32 (glm-image rule), or an aspect like 1:1, 16:9"
        )
    return None  # "auto" and unknown aspects: let the server default decide


def _post_generations(body: dict, base_url: str, api_key: str, deadline: float) -> dict:
    url = f"{base_url.rstrip('/')}{GENERATIONS_PATH}"
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise GenTimeoutError(
            f"zai-gen: no completion within {GEN_TIMEOUT_SECONDS}s — "
            "provider call budget exhausted before POST (gen-timeout)")
    try:
        with urllib.request.urlopen(request, timeout=remaining) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except TimeoutError as exc:
        raise GenTimeoutError(
            f"zai-gen: no completion within {GEN_TIMEOUT_SECONDS}s — "
            "generation POST stalled; killed so the orchestrator can retry (gen-timeout)"
        ) from exc
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()[:500]
        raise SystemExit(f"zai-gen: HTTP {exc.code} from the image API: {detail}") from exc
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"zai-gen: image API request failed: {exc}") from exc
    if not isinstance(payload, dict) or not payload.get("data"):
        raise SystemExit(f"zai-gen: response has no data array: {json.dumps(payload)[:500]}")
    return payload


def _download_as_png(url: str, raw: Path, deadline: float) -> int:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise GenTimeoutError(
            f"zai-gen: no completion within {GEN_TIMEOUT_SECONDS}s — "
            "budget exhausted before image download (gen-timeout)")
    try:
        with urllib.request.urlopen(url, timeout=remaining) as response:
            image_bytes = response.read()
    except TimeoutError as exc:
        raise GenTimeoutError(
            f"zai-gen: no completion within {GEN_TIMEOUT_SECONDS}s — "
            "image download stalled; killed (gen-timeout)"
        ) from exc
    except (urllib.error.URLError, OSError) as exc:
        raise SystemExit(f"zai-gen: image download failed: {exc}") from exc
    if not image_bytes:
        raise SystemExit("zai-gen: image download returned zero bytes")
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.save(raw, format="PNG")
    except Exception as exc:  # Pillow decoder — anything here means non-image bytes
        raise SystemExit(
            f"zai-gen: downloaded bytes are not a decodable image ({exc.__class__.__name__}: {exc})"
        ) from exc
    return len(image_bytes)


class ZaiProvider:
    """Generate one image through the Z.ai GLM-Image generations API."""

    name = "zai"
    # GLM-Image returns RGB (JPEG/PNG without alpha) from the URL — no alpha
    # path exists, same as grok Imagine: draw on a chroma key and key it out.
    transparency = TRANSPARENCY_CHROMA

    def generate(self, request: GenRequest, workdir: Path) -> ProviderRun:
        if request.native_alpha:
            raise SystemExit(
                "zai-gen: native alpha was requested but the GLM-Image API cannot return "
                f"an alpha channel (transparency strategy is {self.transparency!r}); "
                "generate on a chroma key instead"
            )
        if request.refs:
            raise SystemExit(
                f"zai-gen: {len(request.refs)} reference image(s) attached but the "
                "GLM-Image API is text-to-image only — it has no image-edit parameter; "
                "use a provider that supports refs, or drop --ref"
            )
        api_key = resolve_api_key()
        base_url = os.environ.get("ZAI_BASE_URL", "").strip() or DEFAULT_BASE_URL
        model = request.model or os.environ.get("ZAI_IMAGE_MODEL", "").strip() or DEFAULT_MODEL
        size = _aspect_to_size(request.aspect_ratio)

        body: dict = {"model": model, "prompt": request.prompt}
        if size:
            body["size"] = size
        quality = os.environ.get("ZAI_IMAGE_QUALITY", "").strip()
        if quality:
            body["quality"] = quality

        request.raw.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + GEN_TIMEOUT_SECONDS
        started = time.monotonic()
        payload = _post_generations(body, base_url, api_key, deadline)

        url = payload["data"][0].get("url")
        if not url:
            raise SystemExit(
                f"zai-gen: data[0] carries no url: {json.dumps(payload['data'][0])[:300]}")
        downloaded = _download_as_png(url, request.raw, deadline)
        raw_bytes = verify_png(request.raw)
        elapsed = time.monotonic() - started

        filters = [
            entry.get("level")
            for entry in (payload.get("content_filter") or [])
            if isinstance(entry, dict)
        ]
        return ProviderRun(
            provider=self.name,
            elapsed_seconds=elapsed,
            model=model,
            session_id=None,
            extra={
                "size": size,
                "downloaded_bytes": downloaded,
                "png_bytes": raw_bytes,
                "content_filter_levels": filters,
                "created": payload.get("created"),
            },
        )
