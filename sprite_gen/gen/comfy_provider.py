# SPDX-License-Identifier: Apache-2.0
"""ComfyUI provider — local SDXL / FLUX / Qwen image generation over HTTP.

StableGen (Blender) and a bare ComfyUI install share the same backend:
``POST /prompt`` with a workflow graph, poll ``/history/{id}``, download the
output image from ``/view``. This adapter is the 2D sprite path into that stack
— it does not touch Blender, TRELLIS or mesh texturing.

Workflow templates are JSON files with ``{{PLACEHOLDER}}`` tokens:

* ``{{PROMPT}}`` — the generation prompt (string)
* ``{{NEGATIVE}}`` — optional negative prompt (config ``negative``)
* ``{{SEED}}`` — integer seed (random per call unless config ``seed``)
* ``{{WIDTH}}`` / ``{{HEIGHT}}`` — from ``--aspect-ratio`` via config ``sizes``
* ``{{REF}}`` — basename of the uploaded reference (img2img / IPAdapter graphs)
* ``{{CHECKPOINT}}`` — optional config ``checkpoint``

References are uploaded to ComfyUI's ``/upload/image`` and injected as
``{{REF}}`` so LoadImage / IPAdapter nodes can read them. Without refs the
template must be a pure txt2img graph.

Transparency defaults to ``chroma`` (SDXL/FLUX paint a key background from the
prompt, exactly like the sprite-row pipeline). Declare
``transparency = "native"`` only if your workflow really returns alpha.

``billed = false`` by default: a local ComfyUI spends no API credit. Set
``billed = true`` for a shared metered GPU host so the charge is announced and
the provider stays explicit-only.
"""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .base import (
    GEN_TIMEOUT_SECONDS,
    TRANSPARENCY_NATIVE,
    GenRequest,
    ProviderRun,
    announce_api_billing,
    publish_png,
)
from .registry import ProviderSpec

_POLL_SECONDS = 1.5

_DEFAULT_SIZES = {
    "1:1": (1024, 1024),
    "3:2": (1152, 768),
    "2:3": (768, 1152),
    "4:3": (1024, 768),
    "3:4": (768, 1024),
    "16:9": (1280, 720),
    "9:16": (720, 1280),
}


class ComfyProvider:
    """Config-driven ComfyUI client. Implements ``Provider``."""

    def __init__(self, spec: ProviderSpec) -> None:
        self.spec = spec
        self.name = spec.name
        self.transparency = spec.transparency
        opts = spec.options
        url = str(opts.get("url") or "http://127.0.0.1:8188").rstrip("/")
        self.url = url
        workflow = opts.get("workflow")
        if not workflow:
            raise SystemExit(
                f"gen: providers.{spec.name}: comfy needs workflow (path to a ComfyUI "
                f"JSON template) from {spec.source}"
            )
        self.workflow_path = Path(str(workflow)).expanduser()
        if not self.workflow_path.is_file():
            # Allow a path relative to the config file's directory.
            alt = Path(spec.source).parent / self.workflow_path
            if alt.is_file():
                self.workflow_path = alt.resolve()
            else:
                raise SystemExit(
                    f"gen[{self.name}]: workflow template not found: {workflow} "
                    f"(from {spec.source})"
                )
        self.negative = str(opts.get("negative") or "")
        self.checkpoint = opts.get("checkpoint")
        self.seed = opts.get("seed")
        self.sizes = {
            str(k): (int(v[0]), int(v[1]))
            for k, v in {**_DEFAULT_SIZES, **(opts.get("sizes") or {})}.items()
        }
        self.default_size = tuple(opts.get("size") or self.sizes["1:1"])  # type: ignore[assignment]
        self.max_refs = int(opts.get("max_refs", 1))
        self.timeout = float(opts.get("timeout_seconds") or max(GEN_TIMEOUT_SECONDS, 300))

    # -- HTTP ----------------------------------------------------------------

    def _request(
        self,
        path: str,
        *,
        data: bytes | None = None,
        content_type: str = "application/json",
        method: str = "GET",
    ) -> tuple[int, Any]:
        url = self.url + path
        request = urllib.request.Request(url, data=data, method=method)
        if data is not None:
            request.add_header("Content-Type", content_type)
        request.add_header("Accept", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
                if not raw:
                    return response.status, {}
                try:
                    return response.status, json.loads(raw.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    return response.status, {"bytes": raw}
        except urllib.error.HTTPError as exc:
            body = exc.read()
            try:
                return exc.code, json.loads(body.decode("utf-8", "replace"))
            except json.JSONDecodeError:
                return exc.code, {"raw": body[:400].decode("utf-8", "replace")}
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise SystemExit(
                f"gen[{self.name}]: cannot reach ComfyUI at {self.url}: {exc}"
            ) from exc

    def _upload_ref(self, ref: Path) -> str:
        """Upload one reference via /upload/image; return the server-side name."""
        boundary = "----sprite-gen-comfy-" + ref.stem
        data = ref.read_bytes()
        mime = "image/png" if ref.suffix.lower() == ".png" else "image/jpeg"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{ref.name}"\r\n'
            f"Content-Type: {mime}\r\n\r\n"
        ).encode("utf-8") + data + f"\r\n--{boundary}--\r\n".encode("utf-8")
        status, reply = self._request(
            "/upload/image",
            data=body,
            content_type=f"multipart/form-data; boundary={boundary}",
            method="POST",
        )
        if status != 200 or not isinstance(reply, dict):
            raise SystemExit(f"gen[{self.name}]: reference upload failed (HTTP {status})")
        name = (reply.get("name") or ref.name)
        return str(name)

    def _render_workflow(self, request: GenRequest, ref_name: str | None) -> dict[str, Any]:
        template = self.workflow_path.read_text(encoding="utf-8")
        if request.aspect_ratio and request.aspect_ratio in self.sizes:
            width, height = self.sizes[request.aspect_ratio]
        else:
            width, height = (int(self.default_size[0]), int(self.default_size[1]))
        seed = int(self.seed) if self.seed is not None else random.randint(0, 2**32 - 1)
        replacements = {
            "{{PROMPT}}": request.prompt,
            "{{NEGATIVE}}": self.negative,
            "{{SEED}}": str(seed),
            "{{WIDTH}}": str(width),
            "{{HEIGHT}}": str(height),
            "{{REF}}": ref_name or "",
            "{{CHECKPOINT}}": str(self.checkpoint or ""),
        }
        for token, value in replacements.items():
            template = template.replace(token, json.dumps(value)[1:-1] if token in ("{{PROMPT}}", "{{NEGATIVE}}") else value)
        # Prompts are JSON-escaped above; seed/width/height/ref stay literal.
        # Re-do prompt/negative as proper JSON string bodies when they landed as
        # bare text inside a "text" field — templates should quote them:
        #   "text": "{{PROMPT}}"
        # After replace that becomes "text": "<escaped prompt>" which is valid JSON.
        try:
            graph = json.loads(template)
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"gen[{self.name}]: workflow template is not valid JSON after substitution: {exc}"
            ) from exc
        if not isinstance(graph, dict) or "prompt" not in graph:
            # Accept either a full /prompt body {"prompt": {...}} or a bare graph.
            graph = {"prompt": graph}
        graph["client_id"] = f"sprite-gen-{self.name}"
        return graph

    def _download_output(self, history: dict[str, Any], dest: Path) -> None:
        outputs = history.get("outputs") or {}
        for node_out in outputs.values():
            images = node_out.get("images") if isinstance(node_out, dict) else None
            if not images:
                continue
            image = images[0]
            query = urllib.parse.urlencode(
                {
                    "filename": image.get("filename", ""),
                    "subfolder": image.get("subfolder", ""),
                    "type": image.get("type", "output"),
                }
            )
            status, payload = self._request(f"/view?{query}")
            if status != 200 or not isinstance(payload, dict) or "bytes" not in payload:
                raise SystemExit(f"gen[{self.name}]: failed to download output image (HTTP {status})")
            publish_png(payload["bytes"], dest, label=f"gen[{self.name}]")
            return
        raise SystemExit(f"gen[{self.name}]: workflow finished with no images in history")

    # -- Provider ------------------------------------------------------------

    def inspect_facing(self, path: Path, workdir: Path) -> tuple[str, dict[str, Any]]:
        raise SystemExit(
            f"gen[{self.name}]: facing inspection is not implemented for ComfyUI; "
            "use --facing-fix none (record-only is handled upstream) or a cloud provider"
        )

    def generate(self, request: GenRequest, workdir: Path) -> ProviderRun:
        if not request.prompt.strip():
            raise SystemExit(f"gen[{self.name}]: empty prompt")
        if len(request.refs) > self.max_refs:
            raise SystemExit(
                f"gen[{self.name}]: at most {self.max_refs} reference image(s) supported; "
                f"got {len(request.refs)}"
            )
        if request.quality is not None:
            raise SystemExit(
                f"gen[{self.name}]: quality {request.quality!r} is not a ComfyUI knob; "
                "put steps/cfg in the workflow template"
            )
        if request.resolution is not None:
            raise SystemExit(
                f"gen[{self.name}]: resolution {request.resolution!r} is grok Imagine's "
                "vocabulary; size this graph via --aspect-ratio or providers config"
            )
        if request.native_alpha and self.transparency != TRANSPARENCY_NATIVE:
            raise SystemExit(
                f"gen[{self.name}]: transparency strategy is {self.transparency!r}; "
                "native alpha is not a capability of this workflow"
            )
        ref_name = None
        if request.refs:
            ref_name = self._upload_ref(Path(request.refs[0]))
        graph = self._render_workflow(request, ref_name)
        if self.spec.billed:
            announce_api_billing(self.name, "configured host", f" (workflow {self.workflow_path.name}).")
        started = time.monotonic()
        status, reply = self._request(
            "/prompt", data=json.dumps(graph).encode("utf-8"), method="POST"
        )
        if status not in (200, 201):
            raise SystemExit(f"gen[{self.name}]: ComfyUI rejected the workflow (HTTP {status})")
        prompt_id = reply.get("prompt_id") if isinstance(reply, dict) else None
        if not prompt_id:
            raise SystemExit(f"gen[{self.name}]: ComfyUI returned no prompt_id")
        deadline = started + self.timeout
        while time.monotonic() < deadline:
            time.sleep(_POLL_SECONDS)
            hist_status, history = self._request(f"/history/{prompt_id}")
            if hist_status != 200 or not isinstance(history, dict):
                continue
            entry = history.get(prompt_id) or (next(iter(history.values()), None) if history else None)
            if not isinstance(entry, dict):
                continue
            status_info = entry.get("status") or {}
            if status_info.get("status_str") == "error" or status_info.get("completed") is False and status_info.get("status_str"):
                raise SystemExit(f"gen[{self.name}]: ComfyUI reported an error for {prompt_id}")
            if prompt_id in history or entry.get("outputs"):
                self._download_output(entry, request.raw)
                return ProviderRun(
                    provider=self.name,
                    elapsed_seconds=time.monotonic() - started,
                    model=str(self.checkpoint or self.workflow_path.name),
                    extra={
                        "auth_source": "none" if not self.spec.billed else "configured-host",
                        "transport": "comfyui",
                        "endpoint": "/prompt",
                        "base_url": self.url,
                        "prompt_id": str(prompt_id),
                        "workflow": str(self.workflow_path),
                        "billed": self.spec.billed,
                    },
                )
        raise SystemExit(
            f"gen[{self.name}]: timed out after {self.timeout:.0f}s waiting for ComfyUI job {prompt_id}"
        )
