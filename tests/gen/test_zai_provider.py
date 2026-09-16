# SPDX-License-Identifier: Apache-2.0
"""Z.ai provider contract tests — the HTTP boundary is faked, never live.

Like codex/grok: network/OAuth calls are exercised as live e2e by a human with
a funded account. Here we lock the request shape, the PNG-on-disk truth, and
every fail-loud gate (no key, refs, native alpha, bad size, HTTP error body).
"""
from __future__ import annotations

import io
import json
import urllib.error

import pytest

from sprite_gen.gen import zai_provider
from sprite_gen.gen.base import GenRequest
from sprite_gen.gen.zai_provider import ZaiProvider, _aspect_to_size, resolve_api_key


@pytest.fixture(autouse=True)
def no_zai_key(monkeypatch):
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    monkeypatch.delenv("ZAI_IMAGE_MODEL", raising=False)
    monkeypatch.delenv("ZAI_IMAGE_QUALITY", raising=False)
    monkeypatch.delenv("ZAI_BASE_URL", raising=False)


def test_missing_key_fails_loud_with_setup_hint():
    with pytest.raises(SystemExit, match="ZAI_API_KEY"):
        resolve_api_key()


def test_aspect_sizes_map_to_enum_or_validate_custom():
    assert _aspect_to_size("1:1") == "1280x1280"
    assert _aspect_to_size("16:9") == "1728x960"
    assert _aspect_to_size("9:16") == "960x1728"
    assert _aspect_to_size("1536x1536") == "1536x1536"
    assert _aspect_to_size("auto") is None
    assert _aspect_to_size(None) is None
    with pytest.raises(SystemExit, match="divisible by 32"):
        _aspect_to_size("1000x1000")
    with pytest.raises(SystemExit, match="divisible by 32"):
        _aspect_to_size("100x100")


def test_refs_are_refused_text_to_image_only(tmp_path):
    with pytest.raises(SystemExit, match="text-to-image only"):
        ZaiProvider().generate(
            GenRequest(prompt="p", raw=tmp_path / "raw.png", refs=[tmp_path / "ref.png"]),
            tmp_path,
        )


def test_native_alpha_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="native alpha"):
        ZaiProvider().generate(
            GenRequest(prompt="p", raw=tmp_path / "raw.png", native_alpha=True), tmp_path)


def _fake_urlopen(monkeypatch, responses):
    """Replace urlopen with a context-manager sequence; each item is (status, body: bytes|Exception)."""
    calls = []

    class FakeResponse:
        def __init__(self, payload: bytes):
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self) -> bytes:
            return self._payload

    def urlopen(request_or_url, timeout=None):
        calls.append((str(getattr(request_or_url, "full_url", request_or_url)), timeout))
        status, body = responses.pop(0)
        if isinstance(body, Exception):
            raise body
        return FakeResponse(body)

    monkeypatch.setattr(zai_provider.urllib.request, "urlopen", urlopen)
    return calls


def _tiny_png() -> bytes:
    """A real 1x1 PNG produced by Pillow — the download contract is 'decodable image'."""
    import io as _io
    import time as _time

    from PIL import Image as _Image

    buffer = _io.BytesIO()
    _Image.new("RGB", (1, 1), (255, 0, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


PNG_1PX = _tiny_png()


def test_happy_path_writes_verified_png_and_reports_extra(tmp_path, monkeypatch):
    api_json = json.dumps({
        "created": 1760335349,
        "data": [{"url": "https://cdn.example/img/abc"}],
        "content_filter": [{"role": "assistant", "level": 1}],
    }).encode()
    calls = _fake_urlopen(monkeypatch, [
        (200, api_json),
        (200, PNG_1PX),
    ])
    monkeypatch.setenv("ZAI_API_KEY", "test-key")
    raw = tmp_path / "raw.png"

    run = ZaiProvider().generate(GenRequest(prompt="a heart", raw=raw, aspect_ratio="16:9"), tmp_path)

    assert raw.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert run.provider == "zai" and run.model == "glm-image"
    assert run.extra["size"] == "1728x960"
    assert run.extra["created"] == 1760335349
    assert run.extra["content_filter_levels"] == [1]
    post_url, post_timeout = calls[0]
    assert post_url == "https://api.z.ai/api/paas/v4/images/generations"
    assert post_timeout is not None and post_timeout > 0
    download_url, _ = calls[1]
    assert download_url == "https://cdn.example/img/abc"


def test_model_and_env_overrides_shape_request_body(tmp_path, monkeypatch):
    seen_bodies = []
    real_urlopen = zai_provider.urllib.request.urlopen

    def urlopen(request, timeout=None):
        seen_bodies.append(json.loads(request.data.decode()))
        assert request.get_header("Authorization") == "Bearer k2"
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized",
                                     {"Content-Type": "application/json"},
                                     io.BytesIO(b'{"code":401,"message":"bad key"}'))

    monkeypatch.setattr(zai_provider.urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("ZAI_API_KEY", "k2")
    monkeypatch.setenv("ZAI_IMAGE_MODEL", "cogview-4-250304")
    monkeypatch.setenv("ZAI_BASE_URL", "https://api.example.org")
    monkeypatch.setenv("ZAI_IMAGE_QUALITY", "standard")

    with pytest.raises(SystemExit, match="HTTP 401"):
        ZaiProvider().generate(GenRequest(prompt="p", raw=tmp_path / "raw.png"), tmp_path)
    assert seen_bodies == [{"model": "cogview-4-250304", "prompt": "p", "quality": "standard"}]
    assert real_urlopen is not None  # silence linters about the unused import guard
