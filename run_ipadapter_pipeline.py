# SPDX-License-Identifier: Apache-2.0
"""Identity-preserving rabbit atlas via local ComfyUI IPAdapter + RealVisXL.

Uses the painted storybook base as the IPAdapter identity reference so every
pose frame shares the same costume/face/palette — unlike the mock painters.
"""

from __future__ import annotations

import io
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
COMFY = "http://127.0.0.1:8188"
BASE_NAME = "rabbit_paint_base.png"
RUN = ROOT / "runs" / "rabbit-ipadapter-live"
CELL = 512  # per-frame canvas for generation
MAGENTA = (255, 0, 255, 255)

IDLE_POSES = [
    "standing idle pose, neutral front view, arms at sides",
    "idle pose, subtle head bob up, ears lifted, front view",
    "standing idle pose, neutral front view, relaxed",
    "idle pose, subtle settle down, ears soft, front view",
]
WALK_POSES = [
    "walk cycle contact pose, left foot forward, side-front view",
    "walk cycle down pose, legs passing, side-front view",
    "walk cycle contact pose, right foot forward, side-front view",
    "walk cycle passing pose, upright, side-front view",
    "walk cycle contact pose, left foot forward again, side-front view",
    "walk cycle recover pose, ready for next step, side-front view",
]


def _http_json(path: str, payload: dict | None = None, timeout: float = 600) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(COMFY + path, data=data, method="POST" if data else "GET")
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise SystemExit(f"ComfyUI HTTP {exc.code} on {path}: {body[:500]}") from exc


def wait_history(prompt_id: str, timeout: float = 900) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        hist = _http_json(f"/history/{prompt_id}")
        entry = hist.get(prompt_id) or (next(iter(hist.values()), None) if hist else None)
        if isinstance(entry, dict) and entry.get("outputs"):
            return entry
        status = (entry or {}).get("status") or {}
        if status.get("status_str") == "error":
            raise SystemExit(f"ComfyUI job failed: {json.dumps(status)[:400]}")
        time.sleep(1.5)
    raise SystemExit(f"timeout waiting for {prompt_id}")


def download_output(entry: dict) -> bytes:
    for node_out in (entry.get("outputs") or {}).values():
        images = node_out.get("images") if isinstance(node_out, dict) else None
        if not images:
            continue
        img = images[0]
        query = urllib.parse.urlencode(
            {
                "filename": img.get("filename", ""),
                "subfolder": img.get("subfolder", ""),
                "type": img.get("type", "output"),
            }
        )
        with urllib.request.urlopen(COMFY + f"/view?{query}", timeout=120) as response:
            return response.read()
    raise SystemExit("no images in job output")


def build_graph(prompt: str, seed: int) -> dict:
    """SDXL + IPAdapter-plus identity from the uploaded rabbit base."""
    identity = (
        "same exact character as the reference image: fluffy cream storybook rabbit, "
        "long ears with pink inner fur, green vest, cream shirt, brown trousers with patch, "
        "pink neckerchief, brown leather satchel, warm painted watercolor style. "
    )
    subject = identity + prompt
    negative = (
        "pixel art, blocky pixels, different character, different clothes, new costume, "
        "text, watermark, grid lines, checkerboard, white background, black background, "
        "blurry, extra limbs, deformed"
    )
    # Wide strip: N poses left-to-right on magenta for sprite-gen extract.
    width = 384 * 4  # 4 slots per generation batch; walk done in 2 batches then stitched
    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": 22,
                "cfg": 6.5,
                "sampler_name": "euler_ancestral",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["15", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": 384, "batch_size": 1},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["4", 1], "text": subject + ", solid flat magenta #FF00FF background, game sprite strip"},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["4", 1], "text": negative},
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "RealVisXL_V5.0_fp16.safetensors"},
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"images": ["8", 0], "filename_prefix": "rabbit-ipa"},
        },
        "10": {
            "class_type": "CLIPVisionLoader",
            "inputs": {"clip_name": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"},
        },
        "11": {
            "class_type": "IPAdapterModelLoader",
            "inputs": {"ipadapter_file": "ip-adapter-plus_sdxl_vit-h.safetensors"},
        },
        "12": {
            "class_type": "LoadImage",
            "inputs": {"image": BASE_NAME},
        },
        "15": {
            "class_type": "IPAdapterAdvanced",
            "inputs": {
                "model": ["4", 0],
                "ipadapter": ["11", 0],
                "image": ["12", 0],
                "clip_vision": ["10", 0],
                "weight": 0.85,
                "weight_type": "style transfer",
                "combine_embeds": "concat",
                "start_at": 0.0,
                "end_at": 0.9,
                "embeds_scaling": "V only",
            },
        },
    }


def generate_pose_image(poses: list[str], seed: int) -> Image.Image:
    prompt = (
        "horizontal sprite strip, exactly 4 separate full-body poses left to right "
        "with even spacing and clear gaps, consistent character scale. Poses: "
        + "; ".join(poses[:4])
    )
    graph = build_graph(prompt, seed)
    reply = _http_json("/prompt", {"prompt": graph})
    prompt_id = reply.get("prompt_id")
    if not prompt_id:
        raise SystemExit(f"no prompt_id: {reply}")
    entry = wait_history(str(prompt_id))
    raw = download_output(entry)
    return Image.open(io.BytesIO(raw)).convert("RGBA")


def snap_magenta(im: Image.Image) -> Image.Image:
    """Force near-magenta pixels to pure key for clean chroma extract."""
    px = im.load()
    w, h = im.size
    for y in range(h):
        for x in range(w):
            r, g, b, a = px[x, y]
            if r > 200 and b > 200 and g < 80:
                px[x, y] = (255, 0, 255, 255)
    return im


def main() -> int:
    # Health check
    _http_json("/system_stats", timeout=30)

    out = RUN / "raw-strips"
    out.mkdir(parents=True, exist_ok=True)

    print("== IPAdapter generate idle strip ==")
    idle = generate_pose_image(IDLE_POSES, seed=11)
    idle = snap_magenta(idle.resize((384 * 4, 384), Image.Resampling.LANCZOS))
    idle.save(out / "idle.png")
    print("  idle", idle.size)

    print("== IPAdapter generate walk strips (2x4 → 6f) ==")
    walk_a = generate_pose_image(WALK_POSES[:4], seed=22)
    walk_b = generate_pose_image(WALK_POSES[4:], seed=23)
    # Keep 3 poses from A + 3 from B → 6 frames
    def crop_slots(im: Image.Image, count: int, take: slice) -> list[Image.Image]:
        slot_w = im.width // 4
        return [im.crop((i * slot_w, 0, (i + 1) * slot_w, im.height)) for i in range(4)][take][:count]

    slots = crop_slots(walk_a, 3, slice(0, 3)) + crop_slots(walk_b, 3, slice(0, 3))
    walk = Image.new("RGBA", (384 * 6, 384), MAGENTA)
    for i, slot in enumerate(slots):
        walk.paste(snap_magenta(slot.resize((384, 384), Image.Resampling.LANCZOS)), (i * 384, 0))
    walk.save(out / "walk.png")
    print("  walk", walk.size)

    # --- sprite-gen pipeline on these strips ---
    import os
    import shutil

    from sprite_gen import cli as sprite_cli

    if RUN.exists():
        # keep raw-strips as evidence
        for child in RUN.iterdir():
            if child.name != "raw-strips":
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
    request = RUN / "request.json"
    request.write_text(
        json.dumps(
            {
                "states": {
                    "idle": {"frames": 4, "fps": 6, "loop": True, "action": "idle"},
                    "walk": {"frames": 6, "fps": 10, "loop": True, "action": "walk"},
                },
                "cell": {"size": 384},
            }
        ),
        encoding="utf-8",
    )
    # Place strips where extract expects after prepare (raw/<state>.png)
    prepared = RUN / "run"
    print("== prepare ==")
    rc = sprite_cli.main(
        [
            "prepare",
            "--out-dir",
            str(prepared),
            "--character-id",
            "rabbit-ipa",
            "--base-image",
            str(ROOT / "smoke-out" / BASE_NAME),
            "--description",
            "storybook rabbit, cream fur, green vest, satchel, pink scarf",
            "--style",
            "soft painted watercolor storybook",
            "--cell-size",
            "384",
            "--chroma-key",
            "#FF00FF",
            "--request",
            str(request),
        ]
    )
    if rc != 0:
        return rc

    raw = prepared / "raw"
    raw.mkdir(exist_ok=True)
    shutil.copyfile(out / "idle.png", raw / "idle.png")
    shutil.copyfile(out / "walk.png", raw / "walk.png")

    print("== extract ==")
    rc = sprite_cli.main(["extract", "--run-dir", str(prepared), "--allow-slot-fallback"])
    if rc != 0:
        return rc

    print("== compose-atlas ==")
    rc = sprite_cli.main(["compose-atlas", "--run-dir", str(prepared)])
    if rc != 0:
        return rc

    sheet = prepared / "sprite-sheet-alpha.png"
    print("\nOK IPAdapter pipeline")
    print("  sheet:", sheet, sheet.stat().st_size if sheet.is_file() else "MISSING")
    if sheet.is_file():
        img = Image.open(sheet)
        img.resize((img.width // 2, img.height // 2), Image.Resampling.LANCZOS).save(
            prepared / "sprite-sheet-preview.png"
        )
        # scale 2x nearest for chat readability of small frames
        if img.width < 800:
            img.resize((img.width * 2, img.height * 2), Image.Resampling.NEAREST).save(
                prepared / "sprite-sheet-preview.png"
            )
    for state, n in (("idle", 4), ("walk", 6)):
        frames = sorted((prepared / "frames").rglob(f"{state}*.png"))
        print(f"  frames[{state}]:", len(frames))
    return 0 if sheet.is_file() else 1


if __name__ == "__main__":
    raise SystemExit(main())
