# SPDX-License-Identifier: Apache-2.0
"""Full atlas pipeline smoke: prepare → gen-set → extract → compose-atlas on a rabbit.

The mock Images host draws real chibi-rabbit pose strips (magenta chroma bg) so
every stage of the production pipeline runs on pixels that look like game art.
"""

from __future__ import annotations

import base64
import io
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "rabbit-pipeline-out"
CELL = 64
CREAM = (245, 230, 200, 255)
PINK = (255, 180, 190, 255)
DARK = (90, 60, 50, 255)
NOSE = (255, 120, 140, 255)


def draw_rabbit(im: Image.Image, cx: int, cy: int, *, pose: str, phase: float) -> None:
    """Paint one chibi rabbit pose into `im` (magenta already filled)."""
    d = ImageDraw.Draw(im)
    bob = int(round(2 * (0.5 - abs(phase - 0.5))))  # 0..2 px vertical bob
    if pose == "walk":
        leg = 3 if int(phase * 4) % 2 == 0 else -3
        body_y = cy - 2
    else:
        leg = 0
        body_y = cy - bob

    # ears
    ear_h = 15
    d.rounded_rectangle([cx - 7, body_y - 24, cx - 3, body_y - 24 + ear_h], 2, fill=CREAM, outline=DARK)
    d.rounded_rectangle([cx + 3, body_y - 24, cx + 7, body_y - 24 + ear_h], 2, fill=CREAM, outline=DARK)
    d.rectangle([cx - 5, body_y - 20, cx - 4, body_y - 12], fill=PINK)
    d.rectangle([cx + 4, body_y - 20, cx + 5, body_y - 12], fill=PINK)

    # body — keep silhouette compact so neighbouring cells stay disconnected
    d.ellipse([cx - 11, body_y - 6, cx + 11, body_y + 12], fill=CREAM, outline=DARK)
    # head
    d.ellipse([cx - 8, body_y - 16, cx + 8, body_y + 0], fill=CREAM, outline=DARK)
    # face
    d.point((cx - 3, body_y - 9), fill=DARK)
    d.point((cx + 2, body_y - 9), fill=DARK)
    d.ellipse([cx - 1, body_y - 5, cx + 1, body_y - 3], fill=NOSE)
    # feet
    foot = body_y + 10
    if pose == "walk":
        d.ellipse([cx - 7 + leg, foot, cx - 2 + leg, foot + 4], fill=CREAM, outline=DARK)
        d.ellipse([cx + 2 - leg, foot, cx + 7 - leg, foot + 4], fill=CREAM, outline=DARK)
    else:
        d.ellipse([cx - 7, foot, cx - 2, foot + 4], fill=CREAM, outline=DARK)
        d.ellipse([cx + 2, foot, cx + 7, foot + 4], fill=CREAM, outline=DARK)
    # tail — tucked in so it cannot bridge to the next cell
    d.ellipse([cx + 8, body_y + 2, cx + 13, body_y + 7], fill=(255, 255, 255, 255), outline=DARK)


def make_strip(pose: str, frames: int) -> bytes:
    # Leave a clear gap between cells so connected-component extraction sees
    # exactly `frames` blobs (tails/ears must not bridge into the neighbour).
    im = Image.new("RGBA", (CELL * frames, CELL), (255, 0, 255, 255))
    for i in range(frames):
        phase = i / max(frames - 1, 1)
        draw_rabbit(im, CELL * i + CELL // 2, CELL // 2 + 6, pose=pose, phase=phase)
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
        # Prompt prose mentions "walk" even for idle rows — key off the action line.
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

    providers = ROOT / "runs" / "rabbit.providers.toml"
    providers.write_text(
        f"""
[providers.mock-rabbit]
kind = "openai_compatible"
base_url = "http://127.0.0.1:{port}/v1"
model = "mock-rabbit-1"
api_key_env = "SMOKE_API_KEY"
transparency = "chroma"
billed = false
max_refs = 4
""",
        encoding="utf-8",
    )
    import os

    os.environ["SMOKE_API_KEY"] = "smoke"
    os.environ["SPRITE_GEN_PROVIDERS_CONFIG"] = str(providers)

    from sprite_gen import cli as sprite_cli

    base = ROOT / "smoke-out" / "rabbit_base.png"
    run = ROOT / "runs" / "rabbit-live"
    if run.exists():
        import shutil

        shutil.rmtree(run)

    print("== 1/4 prepare ==")
    rc = sprite_cli.main(
        [
            "prepare",
            "--out-dir",
            str(run),
            "--character-id",
            "rabbit",
            "--base-image",
            str(base),
            "--description",
            "cute chibi rabbit, cream fur, long ears",
            "--style",
            "pixel art chibi",
            "--cell-size",
            str(CELL),
            "--chroma-key",
            "#FF00FF",
            "--request",
            str(ROOT / "runs" / "rabbit.request.json"),
        ]
    )
    if rc != 0:
        return rc

    print("== 2/4 gen-set (mock-rabbit provider) ==")
    rc = sprite_cli.main(
        [
            "gen-set",
            "--run-dir",
            str(run),
            "--provider",
            "mock-rabbit",
            "--concurrency",
            "2",
        ]
    )
    if rc != 0:
        print("gen-set failed", file=sys.stderr)
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
    manifest = run / "manifest.json"
    print("\nOK pipeline complete")
    print("  sheet   :", sheet, sheet.stat().st_size if sheet.is_file() else "MISSING")
    print("  manifest:", manifest)
    raw = sorted((run / "raw").glob("*.png"))
    frames = list((run / "frames").rglob("*.png"))
    print("  raw rows:", [p.name for p in raw])
    print("  frames  :", len(frames))
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        layout = data.get("frame_layout") or data
        print("  layout keys:", list(layout)[:12] if isinstance(layout, dict) else type(layout))
    server.shutdown()
    return 0 if sheet.is_file() else 1


if __name__ == "__main__":
    raise SystemExit(main())
