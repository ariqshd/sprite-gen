# SPDX-License-Identifier: Apache-2.0
"""Smoke: openai_compatible + comfy against a local mock HTTP server."""

from __future__ import annotations

import base64
import io
import json
import struct
import threading
import zlib
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from PIL import Image

from sprite_gen.gen import registry
from sprite_gen.gen.base import GenRequest


def _png_bytes(color=(40, 200, 80, 255), size=(32, 32)) -> bytes:
    """Magenta chroma background + a solid subject rectangle in the center.

    A solid single-colour frame would be keyed 100% transparent (every pixel
    reads as background), so the subject is required for a meaningful alpha check.
    """
    from PIL import ImageDraw

    im = Image.new("RGBA", size, (255, 0, 255, 255))
    draw = ImageDraw.Draw(im)
    draw.rectangle([8, 8, 23, 23], fill=color)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


class _Handler(BaseHTTPRequestHandler):
    log = []

    def log_message(self, fmt, *args):  # noqa: A003
        _Handler.log.append((self.command, self.path, args))

    def _json(self, code: int, payload: dict) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _bytes(self, code: int, data: bytes, ctype: str = "image/png") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        if self.path.endswith("/images/generations") or self.path.endswith("/images/edits"):
            self._json(200, {"data": [{"b64_json": base64.b64encode(_png_bytes()).decode("ascii")}]})
            return
        if self.path.endswith("/prompt"):
            self._json(200, {"prompt_id": "smoke-prompt-1"})
            return
        if self.path.endswith("/upload/image"):
            self._json(200, {"name": "ref.png", "subfolder": "", "type": "input"})
            return
        self._json(404, {"error": "unknown POST " + self.path})

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/history/"):
            self._json(
                200,
                {
                    "smoke-prompt-1": {
                        "status": {"status_str": "success", "completed": True},
                        "outputs": {
                            "9": {
                                "images": [
                                    {
                                        "filename": "sprite-gen_00001_.png",
                                        "subfolder": "",
                                        "type": "output",
                                    }
                                ]
                            }
                        },
                    }
                },
            )
            return
        if self.path.startswith("/view"):
            self._bytes(200, _png_bytes((200, 40, 40, 255)))
            return
        self._json(404, {"error": "unknown GET " + self.path})


def main() -> int:
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = Path(__file__).resolve().parent
    out_dir = root / "smoke-out"
    out_dir.mkdir(exist_ok=True)

    toml = root / "smoke.providers.toml"
    toml.write_text(
        f"""
[providers.mock-openai]
kind = "openai_compatible"
base_url = "http://127.0.0.1:{port}/v1"
model = "mock-image-1"
api_key_env = "SMOKE_API_KEY"
transparency = "chroma"
billed = false
max_refs = 2

[providers.mock-comfy]
kind = "comfy"
url = "http://127.0.0.1:{port}"
workflow = "workflows/comfy_txt2img.json"
transparency = "chroma"
billed = false
""",
        encoding="utf-8",
    )

    import os

    os.environ["SMOKE_API_KEY"] = "smoke-key"

    specs = registry.load_specs(toml)
    assert set(specs) == {"mock-openai", "mock-comfy"}, specs
    print("PASS load_specs:", sorted(specs))

    # --- openai_compatible end-to-end ---
    backend = registry.make_provider("mock-openai", config_path=toml)
    raw = out_dir / "openai_raw.png"
    run = backend.generate(
        GenRequest(prompt="green slime idle, magenta background", raw=raw),
        out_dir,
    )
    assert raw.is_file() and raw.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert run.provider == "mock-openai"
    assert run.extra["transport"] == "openai-compatible"
    print("PASS openai_compatible generate:", raw, f"{raw.stat().st_size} bytes")

    # --- comfy end-to-end ---
    comfy = registry.make_provider("mock-comfy", config_path=toml)
    raw2 = out_dir / "comfy_raw.png"
    run2 = comfy.generate(
        GenRequest(prompt="red slime attack, magenta background", raw=raw2),
        out_dir,
    )
    assert raw2.is_file() and raw2.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert run2.provider == "mock-comfy"
    assert run2.extra["prompt_id"] == "smoke-prompt-1"
    print("PASS comfy generate:", raw2, f"{raw2.stat().st_size} bytes")

    # --- billed custom stays explicit-only ---
    toml_billed = root / "smoke.billed.toml"
    toml_billed.write_text(
        f"""
[providers.hostile-metered]
kind = "openai_compatible"
base_url = "http://127.0.0.1:{port}/v1"
model = "mock-image-1"
api_key_env = "SMOKE_API_KEY"
transparency = "chroma"
billed = true
""",
        encoding="utf-8",
    )
    assert "hostile-metered" in registry.explicit_only_names(toml_billed)
    print("PASS billed=true is explicit-only")

    # --- CLI end-to-end through generate_image ---
    from sprite_gen.gen import generate_image

    result = generate_image(
        "mock-openai",
        "a capybara walk cycle, magenta background",
        out_dir / "cli_out.png",
        transparent=True,
        alpha_mode="chroma",
        chroma_key="magenta",
        providers_config=toml,
    )
    assert result.out.is_file()
    assert result.alpha and result.alpha["strategy"] == "chroma"
    from PIL import Image as _Image

    with _Image.open(result.out) as check:
        rgba = check.convert("RGBA")
        alphas = list(rgba.getchannel("A").get_flattened_data())
        zero = sum(1 for a in alphas if a == 0)
        opaque = sum(1 for a in alphas if a >= 8)
    assert 0 < zero < len(alphas), f"chroma should leave mixed alpha, got zero={zero}/{len(alphas)}"
    assert opaque > 0, "subject must survive chroma keying"
    print(
        f"PASS generate_image chroma pipeline: {result.out} "
        f"(alpha0={zero}, opaque={opaque}, n={len(alphas)})"
    )

    server.shutdown()
    print("\nALL SMOKE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
