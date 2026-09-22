# SPDX-License-Identifier: Apache-2.0
"""Smoke: Qwen-Image-2.1 I2I via ComfyUI using the rabbit base as identity."""

from __future__ import annotations

import io
import json
import sys
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
COMFY = "http://127.0.0.1:8188"
WF = ROOT / "workflows" / "comfy_qwen21_i2i.json"
BASE = ROOT / "smoke-out" / "rabbit_paint_base.png"
OUT = ROOT / "runs" / "qwen21-smoke"


def http(path: str, payload: dict | None = None, raw: bytes | None = None, ctype: str = "application/json", timeout: float = 900):
    data = raw if raw is not None else (json.dumps(payload).encode() if payload is not None else None)
    req = urllib.request.Request(COMFY + path, data=data, method="POST" if data else "GET")
    if data:
        req.add_header("Content-Type", ctype)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
        try:
            return json.loads(body.decode())
        except Exception:
            return {"bytes": body}


def upload(path: Path) -> str:
    boundary = "----qwen" + uuid.uuid4().hex
    blob = path.read_bytes()
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{path.name}\"\r\n"
        f"Content-Type: image/png\r\n\r\n"
    ).encode() + blob + f"\r\n--{boundary}--\r\n".encode()
    reply = http("/upload/image", raw=body, ctype=f"multipart/form-data; boundary={boundary}")
    return str(reply.get("name") or path.name)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    stats = http("/system_stats", timeout=30)
    print("comfy up", stats.get("system", {}).get("comfyui_version"))

    ref_name = upload(BASE)
    print("uploaded ref", ref_name)

    text = WF.read_text(encoding="utf-8")
    prompt = (
        "same exact character as the reference: cream storybook rabbit, green vest, "
        "pink neckerchief, brown satchel and trousers. ONE rabbit only, full body, "
        "standing idle front view, solid flat magenta #FF00FF background, game sprite, "
        "no floor, no frame border"
    )
    negative = (
        "two rabbits, group, text, watermark, white frame, checkerboard, floor, ground, "
        "different costume, pixel art, blurry"
    )
    text = (
        text.replace("{{PROMPT}}", json.dumps(prompt)[1:-1])
        .replace("{{NEGATIVE}}", json.dumps(negative)[1:-1])
        .replace("{{REF}}", ref_name)
        .replace("{{SEED}}", "7")
        .replace("{{WIDTH}}", "768")
        .replace("{{HEIGHT}}", "768")
        .replace("{{CHECKPOINT}}", "")
    )
    graph = json.loads(text)
    if "prompt" not in graph:
        graph = {"prompt": graph}
    graph["client_id"] = "qwen21-smoke"

    print("submitting job…")
    reply = http("/prompt", payload=graph, timeout=60)
    pid = reply.get("prompt_id")
    print("prompt_id", pid)
    if not pid:
        print(reply)
        return 1

    deadline = time.time() + 600
    while time.time() < deadline:
        time.sleep(2)
        hist = http(f"/history/{pid}", timeout=30)
        entry = hist.get(str(pid)) or (next(iter(hist.values()), None) if hist else None)
        if not isinstance(entry, dict):
            continue
        status = entry.get("status") or {}
        if status.get("status_str") == "error":
            print("ERROR", json.dumps(status)[:500])
            return 1
        if entry.get("outputs"):
            for node_out in entry["outputs"].values():
                images = node_out.get("images") if isinstance(node_out, dict) else None
                if not images:
                    continue
                img = images[0]
                q = urllib.parse.urlencode(
                    {
                        "filename": img.get("filename", ""),
                        "subfolder": img.get("subfolder", ""),
                        "type": img.get("type", "output"),
                    }
                )
                with urllib.request.urlopen(COMFY + f"/view?{q}", timeout=120) as r:
                    blob = r.read()
                out = OUT / "qwen21-idle.png"
                out.write_bytes(blob)
                im = Image.open(io.BytesIO(blob))
                print("OK", out, im.size, im.mode, len(blob), "bytes")
                Image.open(out).resize((im.width * 2, im.height * 2), Image.Resampling.NEAREST).save(
                    OUT / "qwen21-idle-preview.png"
                )
                return 0
    print("timeout")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
