# SPDX-License-Identifier: Apache-2.0
"""Full atlas pipeline smoke in a hand-painted (non-pixel) rabbit style."""

from __future__ import annotations

import base64
import io
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "rabbit-paint-out"
CELL = 96  # larger cells for a softer painted look
FUR = (242, 220, 186, 255)
FUR_DARK = (210, 175, 135, 255)
FUR_LIGHT = (255, 245, 230, 255)
PINK = (255, 175, 185, 255)
NOSE = (235, 120, 135, 255)
EYE = (70, 50, 45, 255)
BLUSH = (255, 190, 195, 120)


def _soft_ellipse(d: ImageDraw.ImageDraw, box, fill, outline=None, width=1):
    d.ellipse(box, fill=fill, outline=outline, width=width)


def draw_rabbit(im: Image.Image, cx: int, cy: int, *, pose: str, phase: float) -> None:
    """Painted chibi rabbit — soft shapes, no hard pixel grid."""
    # Work on a supersampled canvas for smoother edges when we downscale later.
    d = ImageDraw.Draw(im, "RGBA")
    bob = int(round(3 * (0.5 - abs(phase - 0.5))))
    if pose == "walk":
        leg = 5 if int(phase * 4) % 2 == 0 else -5
        body_y = cy - 4
    else:
        leg = 0
        body_y = cy - bob

    # shadowless body stack (soft watercolor feel via layered ellipses)
    # ears
    _soft_ellipse(d, [cx - 14, body_y - 48, cx - 2, body_y - 8], FUR)
    _soft_ellipse(d, [cx + 2, body_y - 48, cx + 14, body_y - 8], FUR)
    _soft_ellipse(d, [cx - 11, body_y - 42, cx - 5, body_y - 14], PINK)
    _soft_ellipse(d, [cx + 5, body_y - 42, cx + 11, body_y - 14], PINK)

    # body
    _soft_ellipse(d, [cx - 26, body_y - 8, cx + 26, body_y + 36], FUR)
    _soft_ellipse(d, [cx - 18, body_y + 2, cx + 18, body_y + 28], FUR_LIGHT)
    # head
    _soft_ellipse(d, [cx - 20, body_y - 28, cx + 20, body_y + 12], FUR)
    _soft_ellipse(d, [cx - 12, body_y - 18, cx + 12, body_y + 6], FUR_LIGHT)

    # face
    d.ellipse([cx - 8, body_y - 10, cx - 3, body_y - 5], fill=EYE)
    d.ellipse([cx + 3, body_y - 10, cx + 8, body_y - 5], fill=EYE)
    d.ellipse([cx - 6, body_y - 9, cx - 4, body_y - 7], fill=(255, 255, 255, 200))
    d.ellipse([cx + 4, body_y - 9, cx + 6, body_y - 7], fill=(255, 255, 255, 200))
    d.ellipse([cx - 3, body_y - 1, cx + 3, body_y + 4], fill=NOSE)
    # blush
    d.ellipse([cx - 16, body_y - 2, cx - 8, body_y + 4], fill=BLUSH)
    d.ellipse([cx + 8, body_y - 2, cx + 16, body_y + 4], fill=BLUSH)

    # feet
    foot = body_y + 30
    if pose == "walk":
        d.ellipse([cx - 14 + leg, foot, cx - 2 + leg, foot + 12], fill=FUR)
        d.ellipse([cx + 2 - leg, foot, cx + 14 - leg, foot + 12], fill=FUR)
    else:
        d.ellipse([cx - 14, foot, cx - 2, foot + 12], fill=FUR)
        d.ellipse([cx + 2, foot, cx + 14, foot + 12], fill=FUR)

    # fluffy tail
    d.ellipse([cx + 18, body_y + 10, cx + 34, body_y + 26], fill=(255, 252, 248, 255))


def make_strip(pose: str, frames: int) -> bytes:
    scale = 3
    w, h = CELL * frames, CELL
    big = Image.new("RGBA", (w * scale, h * scale), (255, 0, 255, 255))
    for i in range(frames):
        phase = i / max(frames - 1, 1)
        draw_rabbit(
            big,
            (CELL * i + CELL // 2) * scale,
            (CELL // 2 + 4) * scale,
            pose=pose,
            phase=phase,
        )
    # Downscale for soft painted edges (anti-alias), keep magenta key pure-ish.
    im = big.resize((w, h), Image.Resampling.LANCZOS)
    # Snap near-magenta back to pure key so chroma extraction stays clean.
    px = im.load()
    for y in range(h):
        for x in range(w):
            r, g, b, a = px[x, y]
            if a > 0 and r > 220 and b > 220 and g < 60:
                px[x, y] = (255, 0, 255, 255)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        text = body.decode("utf-8", "replace")
        action = "walk" if re.search(r"Animation action:\s*walk", text, re.I) else "idle"
        frames = 6 if action == "walk" else 4
        png = make_strip(action, frames)
        payload = json.dumps({"data": [{"b64_json": base64.b64encode(png).decode("ascii")}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    providers = ROOT / "runs" / "rabbit-paint.providers.toml"
    providers.write_text(
        f"""
[providers.mock-paint]
kind = "openai_compatible"
base_url = "http://127.0.0.1:{port}/v1"
model = "mock-paint-1"
api_key_env = "SMOKE_API_KEY"
transparency = "chroma"
billed = false
max_refs = 4
""",
        encoding="utf-8",
    )
    import os
    import shutil

    os.environ["SMOKE_API_KEY"] = "smoke"
    os.environ["SPRITE_GEN_PROVIDERS_CONFIG"] = str(providers)

    from sprite_gen import cli as sprite_cli

    base = ROOT / "smoke-out" / "rabbit_paint_base.png"
    if not base.is_file():
        print("missing painted base image:", base, file=sys.stderr)
        return 1
    run = ROOT / "runs" / "rabbit-paint-live"
    if run.exists():
        shutil.rmtree(run)

    request = ROOT / "runs" / "rabbit-paint.request.json"
    request.write_text(
        json.dumps(
            {
                "states": {
                    "idle": {"frames": 4, "fps": 6, "loop": True, "action": "idle"},
                    "walk": {"frames": 6, "fps": 10, "loop": True, "action": "walk"},
                },
                "cell": {"size": CELL},
            }
        ),
        encoding="utf-8",
    )

    print("== 1/4 prepare (painted style) ==")
    rc = sprite_cli.main(
        [
            "prepare",
            "--out-dir",
            str(run),
            "--character-id",
            "rabbit-paint",
            "--base-image",
            str(base),
            "--description",
            "cute hand-painted storybook rabbit, fluffy cream fur, pink ears",
            "--style",
            "soft watercolor gouache storybook illustration, NOT pixel art",
            "--cell-size",
            str(CELL),
            "--chroma-key",
            "#FF00FF",
            "--request",
            str(request),
        ]
    )
    if rc != 0:
        return rc

    print("== 2/4 gen-set (mock-paint) ==")
    rc = sprite_cli.main(
        ["gen-set", "--run-dir", str(run), "--provider", "mock-paint", "--concurrency", "2"]
    )
    if rc != 0:
        return rc

    print("== 3/4 extract ==")
    rc = sprite_cli.main(["extract", "--run-dir", str(run), "--allow-slot-fallback"])
    if rc != 0:
        return rc

    print("== 4/4 compose-atlas ==")
    rc = sprite_cli.main(["compose-atlas", "--run-dir", str(run)])
    if rc != 0:
        return rc

    sheet = run / "sprite-sheet-alpha.png"
    print("\nOK painted pipeline complete")
    print("  sheet:", sheet, sheet.stat().st_size if sheet.is_file() else "MISSING")

    # Previews
    if sheet.is_file():
        img = Image.open(sheet)
        img.resize((img.width * 3, img.height * 3), Image.Resampling.NEAREST).save(
            run / "sprite-sheet-preview.png"
        )
    frames_by: dict[str, list[Image.Image]] = {}
    for p in sorted((run / "frames").rglob("*.png")):
        frames_by.setdefault(p.parent.name, []).append(Image.open(p).convert("RGBA"))
    for state, ims in frames_by.items():
        scaled = [im.resize((im.width * 3, im.height * 3), Image.Resampling.LANCZOS) for im in ims]
        scaled[0].save(
            run / f"{state}-preview.gif",
            save_all=True,
            append_images=scaled[1:],
            duration=160,
            loop=0,
            disposal=2,
            transparency=0,
        )
        print("  gif", state, len(scaled), "frames")

    server.shutdown()
    return 0 if sheet.is_file() else 1


if __name__ == "__main__":
    raise SystemExit(main())
