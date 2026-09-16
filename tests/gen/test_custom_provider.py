# SPDX-License-Identifier: Apache-2.0
"""Custom bridge provider contract tests — the bridge is a real child process.

The bridge IS the adapter, so the happy paths run a real Pillow one-liner
through the OS shell (the same `shell=True` path production uses) and lock the
stdin JSON contract and the PNG-on-disk truth. Pure error shapes (exit codes,
timeouts) fake `subprocess.run`, mirroring how the sibling providers fake HTTP.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from sprite_gen.gen import PROVIDERS, _make_provider, resolve_default_provider
from sprite_gen.gen import custom_provider
from sprite_gen.gen.base import GenRequest, GenTimeoutError
from sprite_gen.gen.custom_provider import CMD_ENV, TRANSPARENCY_ENV, CustomProvider
from sprite_gen.workflow.access import probe_access

# Single-line, single-quote-only payloads: safe inside a double-quoted -c
# argument for both cmd.exe (Windows) and /bin/sh (POSIX).
_WRITE_RGB = ("import sys,json;from PIL import Image;p=json.load(sys.stdin);"
              "Image.new('RGB',(4,4),(0,255,0)).save(p['out']);"
              "open(p['out']+'.json','w').write(json.dumps(p))")
_WRITE_RGBA = ("import sys,json;from PIL import Image;p=json.load(sys.stdin);"
               "Image.new('RGBA',(4,4),(0,255,0,128)).save(p['out'])")
_WRITE_GARBAGE = ("import sys,json;p=json.load(sys.stdin);"
                  "open(p['out'],'w').write('definitely not an image')")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv(CMD_ENV, raising=False)
    monkeypatch.delenv(TRANSPARENCY_ENV, raising=False)
    monkeypatch.delenv("SPRITE_GEN_DEFAULT_PROVIDER", raising=False)


def _request(tmp_path, **overrides) -> GenRequest:
    defaults = {"prompt": "a heart", "raw": tmp_path / "raw.png"}
    defaults.update(overrides)
    return GenRequest(**defaults)


def _bridge_env(command: str, monkeypatch, transparency: str | None = None):
    monkeypatch.setenv(CMD_ENV, command)
    if transparency:
        monkeypatch.setenv(TRANSPARENCY_ENV, transparency)


def test_missing_cmd_fails_loud(tmp_path):
    with pytest.raises(SystemExit, match=CMD_ENV):
        CustomProvider().generate(_request(tmp_path), tmp_path)


def test_bad_transparency_env_fails_loud(monkeypatch):
    monkeypatch.setenv(TRANSPARENCY_ENV, "bogus")
    with pytest.raises(SystemExit, match=TRANSPARENCY_ENV):
        CustomProvider()


def test_native_alpha_refused_by_default(tmp_path):
    with pytest.raises(SystemExit, match="native alpha"):
        CustomProvider().generate(_request(tmp_path, native_alpha=True), tmp_path)


def test_native_alpha_allowed_when_declared(tmp_path, monkeypatch):
    _bridge_env(f'"{sys.executable}" -c "{_WRITE_RGBA}"', monkeypatch, transparency="native")
    provider = CustomProvider()
    assert provider.transparency == "native"

    run = provider.generate(_request(tmp_path, native_alpha=True), tmp_path)

    assert (tmp_path / "raw.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert run.provider == "custom"


def test_happy_path_locks_stdin_json_contract(tmp_path, monkeypatch):
    ref = tmp_path / "ref.png"
    ref.write_bytes(b"\x89PNG\r\n\x1a\n rest-of-ref")
    _bridge_env(f'"{sys.executable}" -c "{_WRITE_RGB}"', monkeypatch)

    run = CustomProvider().generate(
        _request(tmp_path, model="my-model", aspect_ratio="1:1", refs=[ref]), tmp_path)

    bridge_image = tmp_path / "bridge-image.png"
    assert bridge_image.is_file()
    assert (tmp_path / "raw.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    seen = json.loads((tmp_path / "bridge-image.png.json").read_text())
    assert seen["prompt"] == "a heart"
    assert seen["model"] == "my-model"
    assert seen["aspect_ratio"] == "1:1"
    assert seen["native_alpha"] is False
    assert seen["refs"] == [str(ref)]
    assert seen["out"] == str(bridge_image)
    assert run.model == "my-model"
    assert run.extra["bridge_exit_code"] == 0
    assert run.extra["bridge_image_bytes"] > 0
    assert run.extra["png_bytes"] > 0


def test_bridge_failure_fails_loud_with_stderr_tail(tmp_path, monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0] if args else "", 1, stdout=b"", stderr=b"bridge exploded")
    monkeypatch.setattr(custom_provider.subprocess, "run", fake_run)
    _bridge_env("any-command", monkeypatch)

    with pytest.raises(SystemExit, match="bridge exploded"):
        CustomProvider().generate(_request(tmp_path), tmp_path)


def test_missing_output_fails_loud_on_zero_exit(tmp_path, monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess("", 0, stdout=b"all done", stderr=b"")
    monkeypatch.setattr(custom_provider.subprocess, "run", fake_run)
    _bridge_env("any-command", monkeypatch)

    with pytest.raises(SystemExit, match="wrote no image"):
        CustomProvider().generate(_request(tmp_path), tmp_path)


def test_undecodable_output_fails_loud(tmp_path, monkeypatch):
    _bridge_env(f'"{sys.executable}" -c "{_WRITE_GARBAGE}"', monkeypatch)

    with pytest.raises(SystemExit, match="not decodable"):
        CustomProvider().generate(_request(tmp_path), tmp_path)


def test_bridge_timeout_maps_to_gen_timeout(tmp_path, monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="any-command", timeout=180)
    monkeypatch.setattr(custom_provider.subprocess, "run", fake_run)
    _bridge_env("any-command", monkeypatch)

    with pytest.raises(GenTimeoutError, match="gen-timeout"):
        CustomProvider().generate(_request(tmp_path), tmp_path)


def test_registry_and_default_resolution(monkeypatch):
    assert "custom" in PROVIDERS
    assert isinstance(_make_provider("custom", keep_session=False), CustomProvider)
    monkeypatch.setenv("SPRITE_GEN_DEFAULT_PROVIDER", "custom")
    provider, fallback = resolve_default_provider()
    assert provider == "custom" and fallback is None


def test_access_probe_reports_bridge_configuration(monkeypatch):
    unavailable = probe_access("custom")
    assert unavailable["login"] == "unavailable"
    assert CMD_ENV in unavailable["reason"]

    monkeypatch.setenv(CMD_ENV, "python ~/bridges/bridge.py")
    ready = probe_access("custom")
    assert ready["login"] == "ready"
    assert ready["billing"] == "unknown"
