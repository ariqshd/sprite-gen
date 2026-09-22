# SPDX-License-Identifier: Apache-2.0
"""Engine-ready rabbit atlas: IPAdapter identity + one pose per frame + scale normalize."""

from __future__ import annotations

import io
import json
import shutil
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
RUN = ROOT / "runs" / "rabbit-clean-live"
GEN = 512  # square generation canvas per frame
MAGENTA = (255, 0, 255, 255)

# Exactly one character, one pose, full body — no groups, no props-only shots.
IDLE_POSES = [
    "ONE single rabbit only, full body, standing idle, front view, arms relaxed at sides",
    "ONE single rabbit only, full body, idle with ears slightly raised, front view",
    "ONE single rabbit only, full body, standing idle, front view, gentle breath",
    "ONE single rabbit only, full body, idle settle, ears soft, front view",
]
WALK_POSES = [
    "ONE single rabbit only, full body, walk pose left foot forward, 3/4 side view",
    "ONE single rabbit only, full body, walk pose legs passing under body, 3/4 side view",
    "ONE single rabbit only, full body, walk pose right foot forward, 3/4 side view",
    "ONE single rabbit only, full body, walk pose mid-stride upright, 3/4 side view",
    "ONE single rabbit only, full body, walk pose left foot forward, 3/4 side view",
    "ONE single rabbit only, full body, walk pose recover step, 3/4 side view",
]

IDENTITY = (
    "same exact character as the reference image: fluffy cream storybook rabbit, "
    "long ears with pink inner fur, green vest over cream shirt, brown trousers with patch, "
    "pink neckerchief, brown leather satchel, warm painted watercolor style. "
)
NEGATIVE = (
    "two rabbits, multiple characters, group, crowd, duplicate, extra animals, "
    "different costume, new clothes, armor, text, watermark, grid, checkerboard, "
    "white background, black background, pixel art, blurry, deformed, cropped head, "
    "close-up face only, sitting, lying down"
)


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
        raise SystemExit(f"ComfyUI HTTP {exc.code} on {path}: {body[:400]}") from exc


def wait_history(prompt_id: str, timeout: float = 900) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        hist = _http_json(f"/history/{prompt_id}")
        entry = hist.get(prompt_id) or (next(iter(hist.values()), None) if hist else None)
        if isinstance(entry, dict) and entry.get("outputs"):
            return entry
        status = (entry or {}).get("status") or {}
        if status.get("status_str") == "error":
            raise SystemExit(f"job failed: {json.dumps(status)[:300]}")
        time.sleep(1.2)
    raise SystemExit(f"timeout {prompt_id}")


def download_output(entry: dict) -> bytes:
    for node_out in (entry.get("outputs") or {}).values():
        images = node_out.get("images") if isinstance(node_out, dict) else None
        if images:
            img = images[0]
            q = urllib.parse.urlencode(
                {
                    "filename": img.get("filename", ""),
                    "subfolder": img.get("subfolder", ""),
                    "type": img.get("type", "output"),
                }
            )
            with urllib.request.urlopen(COMFY + f"/view?{q}", timeout=120) as r:
                return r.read()
    raise SystemExit("no image in output")


def build_graph(pose: str, seed: int) -> dict:
    text = (
        IDENTITY
        + pose
        + ", centered, feet visible, solid flat magenta #FF00FF background, "
        "game character sprite, clean silhouette, no shadow on ground"
    )
    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": 24,
                "cfg": 6.0,
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
            "inputs": {"width": GEN, "height": GEN, "batch_size": 1},
        },
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": text}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": NEGATIVE}},
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "RealVisXL_V5.0_fp16.safetensors"},
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {
            "class_type": "SaveImage",
            "inputs": {"images": ["8", 0], "filename_prefix": "rabbit-clean"},
        },
        "10": {
            "class_type": "CLIPVisionLoader",
            "inputs": {"clip_name": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"},
        },
        "11": {
            "class_type": "IPAdapterModelLoader",
            "inputs": {"ipadapter_file": "ip-adapter-plus_sdxl_vit-h.safetensors"},
        },
        "12": {"class_type": "LoadImage", "inputs": {"image": BASE_NAME}},
        "15": {
            "class_type": "IPAdapterAdvanced",
            "inputs": {
                "model": ["4", 0],
                "ipadapter": ["11", 0],
                "image": ["12", 0],
                "clip_vision": ["10", 0],
                "weight": 0.9,
                "weight_type": "style and composition",
                "combine_embeds": "concat",
                "start_at": 0.0,
                "end_at": 0.85,
                "embeds_scaling": "V only",
            },
        },
    }


def gen_frame(pose: str, seed: int) -> Image.Image:
    reply = _http_json("/prompt", {"prompt": build_graph(pose, seed)})
    pid = reply.get("prompt_id")
    if not pid:
        raise SystemExit(f"no prompt_id {reply}")
    entry = wait_history(str(pid))
    return Image.open(io.BytesIO(download_output(entry))).convert("RGBA")


def snap_magenta(im: Image.Image) -> Image.Image:
    px = im.load()
    for y in range(im.height):
        for x in range(im.width):
            r, g, b, a = px[x, y]
            if r > 190 and b > 190 and g < 90:
                px[x, y] = MAGENTA
    return im


def normalize_frame(im: Image.Image, cell: int) -> Image.Image:
    """Key magenta → alpha, bbox subject, scale to shared height, center, pad to cell."""
    im = snap_magenta(im)
    rgba = im.convert("RGBA")
    # chroma-style matte: near-magenta → transparent
    px = rgba.load()
    for y in range(rgba.height):
        for x in range(rgba.width):
            r, g, b, a = px[x, y]
            if r > 180 and b > 180 and g < 100:
                px[x, y] = (r, g, b, 0)
            elif a == 255 and r > 160 and b > 160 and g < 120:
                # fringe
                px[x, y] = (r, g, b, 0)

    bbox = rgba.getchannel("A").point(lambda v: 255 if v >= 8 else 0).getbbox()
    if bbox is None:
        raise SystemExit("frame fully transparent after keying")
    subject = rgba.crop(bbox)

    # shared height = 86% of cell so feet land near bottom with margin
    target_h = int(cell * 0.86)
    scale = target_h / subject.height
    target_w = max(1, int(round(subject.width * scale)))
    subject = subject.resize((target_w, target_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", (cell, cell), MAGENTA)
    x = (cell - target_w) // 2
    y = cell - target_h - max(2, int(cell * 0.04))  # bottom = foot line
    canvas.paste(subject, (x, y), subject)
    return snap_magenta(canvas)


def build_strip(frames: list[Image.Image], cell: int) -> Image.Image:
    strip = Image.new("RGBA", (cell * len(frames), cell), MAGENTA)
    for i, frame in enumerate(frames):
        strip.paste(frame, (i * cell, 0))
    return strip


def main() -> int:
    _http_json("/system_stats", timeout=20)
    samples = RUN / "samples"
    if RUN.exists():
        shutil.rmtree(RUN)
    samples.mkdir(parents=True)

    cell = 256  # final cell after normalize (was 384; tighter)

    print("== generate 4 idle frames (1 pose each) ==")
    idle_raw = []
    for i, pose in enumerate(IDLE_POSES):
        print(f"  idle[{i}]")
        img = gen_frame(pose, seed=100 + i)
        img.save(samples / f"idle-raw-{i}.png")
        idle_raw.append(normalize_frame(img, cell))

    print("== generate 6 walk frames (1 pose each) ==")
    walk_raw = []
    for i, pose in enumerate(WALK_POSES):
        print(f"  walk[{i}]")
        img = gen_frame(pose, seed=200 + i)
        img.save(samples / f"walk-raw-{i}.png")
        walk_raw.append(normalize_frame(img, cell))

    idle_strip = build_strip(idle_raw, cell)
    walk_strip = build_strip(walk_raw, cell)
    idle_strip.save(samples / "idle.png")
    walk_strip.save(samples / "walk.png")
    print("strips", idle_strip.size, walk_strip.size)

    from sprite_gen import cli as sprite_cli

    request = RUN / "request.json"
    request.write_text(
        json.dumps(
            {
                "states": {
                    "idle": {"frames": 4, "fps": 6, "loop": True, "action": "idle"},
                    "walk": {"frames": 6, "fps": 10, "loop": True, "action": "walk"},
                },
                "cell": {"size": cell},
            }
        ),
        encoding="utf-8",
    )
    prepared = RUN / "run"
    print("== prepare ==")
    rc = sprite_cli.main(
        [
            "prepare",
            "--out-dir",
            str(prepared),
            "--character-id",
            "rabbit-clean",
            "--base-image",
            str(ROOT / "smoke-out" / BASE_NAME),
            "--description",
            "storybook rabbit, cream fur, green vest, satchel, pink scarf",
            "--style",
            "soft painted watercolor storybook",
            "--cell-size",
            str(cell),
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
    shutil.copyfile(samples / "idle.png", raw / "idle.png")
    shutil.copyfile(samples / "walk.png", raw / "walk.png")

    print("== extract ==")
    rc = sprite_cli.main(["extract", "--run-dir", str(prepared), "--allow-slot-fallback"])
    if rc != 0:
        return rc

    print("== compose-atlas ==")
    rc = sprite_cli.main(["compose-atlas", "--run-dir", str(prepared)])
    if rc != 0:
        return rc

    sheet_path = prepared / "sprite-sheet-alpha.png"
    print("\nOK clean pipeline")
    print("  sheet:", sheet_path, sheet_path.stat().st_size if sheet_path.is_file() else "MISSING")

    if sheet_path.is_file():
        sheet = Image.open(sheet_path)
        sheet.resize((sheet.width * 2, sheet.height * 2), Image.Resampling.NEAREST).save(
            prepared / "sprite-sheet-preview.png"
        )
        # grid of frames
        frames = []
        for state in ("idle", "walk"):
            for p in sorted((prepared / "frames" / state).glob("frame-*.png")):
                frames.append(Image.open(p).convert("RGBA").resize((192, 192), Image.Resampling.NEAREST))
        if frames:
            cols = min(5, len(frames))
            rows = (len(frames) + cols - 1) // cols
            grid = Image.new("RGBA", (192 * cols, 192 * rows), (24, 24, 28, 255))
            for i, im in enumerate(frames):
                grid.paste(im, ((i % cols) * 192, (i // cols) * 192), im)
            grid.save(prepared / "frames-grid.png")

        # identity compare
        base = (
            Image.open(ROOT / "smoke-out" / BASE_NAME)
            .convert("RGBA")
            .resize((256, 256), Image.Resampling.LANCZOS)
        )
        combo = Image.new("RGBA", (max(sheet.width * 2, 280), 280 + sheet.height * 2), (18, 18, 22, 255))
        combo.paste(base, (12, 12), base)
        combo.paste(
            sheet.resize((sheet.width * 2, sheet.height * 2), Image.Resampling.NEAREST),
            (12, 280),
        )
        combo.save(prepared / "identity-compare.png")
        print("  frames:", len(frames))

    return 0 if sheet_path.is_file() else 1


if __name__ == "__main__":
    raise SystemExit(main())
