# SPDX-License-Identifier: Apache-2.0
"""Custom bridge provider — run any image backend through a user command.

This is the provider-agnostic escape hatch: instead of one adapter per vendor
wire format, the bridge command IS the adapter. ``SPRITE_GEN_CUSTOM_CMD`` is a
full command line run through the OS shell (pipes and quoting available); the
request arrives as one JSON object on stdin and the bridge must write the
generated image (any format Pillow decodes — PNG/JPEG/WebP) to the ``out`` path
it names, then exit 0.

    stdin:  {"prompt": str, "model": str|null, "aspect_ratio": str|null,
             "native_alpha": bool, "refs": ["<abs path>", ...],
             "out": "<abs path>"}

Truth stays the decoded PNG bytes on disk (No Silent Fallback): whatever the
bridge wrote is re-encoded through Pillow into the verified PNG contract. A
non-zero exit fails loud with the bridge's stderr tail. Transparency is
declared once via ``SPRITE_GEN_CUSTOM_TRANSPARENCY`` (``chroma`` default;
``native`` only for a backend that genuinely returns an alpha channel).

Example — any OpenAI-images-compatible endpoint, no adapter code:

    SPRITE_GEN_CUSTOM_TRANSPARENCY=chroma
    SPRITE_GEN_CUSTOM_CMD='python ~/bridges/openai_images.py'

The bridge script reads the stdin JSON, POSTs to the endpoint, and writes the
returned image to ``out``. Local installs (A1111 ``/sdapi/v1/txt2img``,
ComfyUI), aggregators and future APIs all bridge the same way.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from PIL import Image

from .base import (
    GEN_TIMEOUT_SECONDS,
    TRANSPARENCY_CHROMA,
    TRANSPARENCY_NATIVE,
    TRANSPARENCY_STRATEGIES,
    GenRequest,
    GenTimeoutError,
    ProviderRun,
    provider_subprocess_env,
    verify_png,
)

CMD_ENV = "SPRITE_GEN_CUSTOM_CMD"
TRANSPARENCY_ENV = "SPRITE_GEN_CUSTOM_TRANSPARENCY"

_STDOUT_TAIL_CHARS = 400


def resolve_bridge_command() -> str:
    """Read SPRITE_GEN_CUSTOM_CMD, failing loud when absent or blank."""
    command = os.environ.get(CMD_ENV, "").strip()
    if not command:
        raise SystemExit(
            f"custom-gen: {CMD_ENV} is not set. Point it at the command that bridges your "
            "image backend; it receives the generation request as JSON on stdin and must "
            "write the generated image to the 'out' path it names (docs/gen.md, custom bridge)."
        )
    return command


def resolve_transparency() -> str:
    """Read the declared transparency strategy, defaulting to chroma."""
    declared = os.environ.get(TRANSPARENCY_ENV, "").strip() or TRANSPARENCY_CHROMA
    if declared not in TRANSPARENCY_STRATEGIES:
        raise SystemExit(
            f"custom-gen: {TRANSPARENCY_ENV}={declared!r} is not a transparency strategy; "
            f"expected one of {', '.join(TRANSPARENCY_STRATEGIES)}"
        )
    return declared


def _bridge_request(request: GenRequest, bridge_image: Path) -> bytes:
    return json.dumps({
        "prompt": request.prompt,
        "model": request.model,
        "aspect_ratio": request.aspect_ratio,
        "native_alpha": request.native_alpha,
        "refs": [str(ref) for ref in request.refs],
        "out": str(bridge_image),
    }).encode("utf-8")


def _tail(raw: bytes | None) -> str:
    text = (raw or b"").decode("utf-8", errors="replace").strip()
    return text[-_STDOUT_TAIL_CHARS:]


class CustomProvider:
    """Generate one image by delegating to the configured bridge command."""

    name = "custom"

    def __init__(self) -> None:
        # Read at construction: the orchestrator builds the backend per call,
        # so env changes apply to the next generation like the other providers.
        self.transparency = resolve_transparency()

    def generate(self, request: GenRequest, workdir: Path) -> ProviderRun:
        if request.native_alpha and self.transparency != TRANSPARENCY_NATIVE:
            raise SystemExit(
                "custom-gen: native alpha was requested but the bridge declares "
                f"{self.transparency!r} transparency ({TRANSPARENCY_ENV}); set it to "
                "'native' only for a backend that genuinely returns an alpha channel"
            )
        command = resolve_bridge_command()
        # '.png' lets naive `Image.save(out)` bridges infer a format; content,
        # not the extension, is what the provider trusts (Pillow sniffs bytes).
        bridge_image = workdir / "bridge-image.png"
        payload = _bridge_request(request, bridge_image)

        request.raw.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + GEN_TIMEOUT_SECONDS
        started = time.monotonic()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise GenTimeoutError(
                f"custom-gen: no completion within {GEN_TIMEOUT_SECONDS}s — "
                "bridge budget exhausted before spawn (gen-timeout)")
        try:
            completed = subprocess.run(
                command,
                shell=True,  # the command line IS the config: shell semantics are the interface
                input=payload,
                capture_output=True,
                env=provider_subprocess_env(),
                timeout=remaining,
            )
        except subprocess.TimeoutExpired as exc:
            raise GenTimeoutError(
                f"custom-gen: no completion within {GEN_TIMEOUT_SECONDS}s — "
                "bridge command killed so the orchestrator can retry (gen-timeout)"
            ) from exc
        except OSError as exc:
            raise SystemExit(f"custom-gen: bridge command could not run: {exc}") from exc
        if completed.returncode != 0:
            detail = _tail(completed.stderr) or _tail(completed.stdout)
            raise SystemExit(
                f"custom-gen: bridge exited with code {completed.returncode}"
                + (f": {detail}" if detail else "")
            )
        if not bridge_image.is_file():
            raise SystemExit(
                f"custom-gen: bridge exited 0 but wrote no image at {bridge_image} — "
                f"stdout tail: {_tail(completed.stdout) or '(empty)'}"
            )

        image_bytes = bridge_image.read_bytes()
        try:
            with Image.open(bridge_image) as image:
                image.save(request.raw, format="PNG")
        except Exception as exc:
            raise SystemExit(
                f"custom-gen: bridge image bytes are not decodable ({exc.__class__.__name__}: {exc})"
            ) from exc
        raw_bytes = verify_png(request.raw)
        elapsed = time.monotonic() - started

        return ProviderRun(
            provider=self.name,
            elapsed_seconds=elapsed,
            model=request.model,
            session_id=None,
            extra={
                "bridge_exit_code": completed.returncode,
                "bridge_image_bytes": len(image_bytes),
                "png_bytes": raw_bytes,
                "bridge_stdout_tail": _tail(completed.stdout),
            },
        )
