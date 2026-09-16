# SPDX-License-Identifier: Apache-2.0
"""Gemini provider contract tests — the HTTP boundary is faked, never live.

Like codex/grok/zai: network calls are exercised as live e2e by a human with a
funded key. Here we lock the request shape (endpoint, key header,
responseModalities, aspect passthrough, inline ref parts), the PNG-on-disk
truth, and every fail-loud gate (no key, native alpha, HTTP error, blocked or
image-less response).
"""
from __future__ import annotations

import base64
import io
import json
import urllib.error

import pytest

from sprite_gen.gen import gemini_provider
from sprite_gen.gen.base import GenRequest
from sprite_gen.gen.gemini_provider import GeminiProvider, resolve_api_key
from sprite_gen.gen.zai_provider import _aspect_to_size  # noqa: F401  (sibling import sanity)


@pytest.fixture(autouse=True)
def no_gemini_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_IMAGE_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_BASE_URL", raising=False)


def test_missing_key_fails_loud_with_setup_hint():
    with pytest.raises(SystemExit, match="GEMINI_API_KEY"):
        resolve_api_key()


def test_google_api_key_alias_is_accepted(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "alias-key")
    assert resolve_api_key() == "alias-key"


def _png_b64() -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), (0, 255, 0)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _fake_urlopen(monkeypatch, responses):
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

    def urlopen(request, timeout=None):
        calls.append({
            "url": request.full_url,
            "key": request.headers.get("X-goog-api-key"),  # urllib capitalizes stored headers
            "body": json.loads(request.data.decode()),
            "timeout": timeout,
        })
        status, body = responses.pop(0)
        if isinstance(body, Exception):
            raise body
        return FakeResponse(body)

    monkeypatch.setattr(gemini_provider.urllib.request, "urlopen", urlopen)
    return calls


def test_happy_path_decodes_inline_image_to_verified_png(tmp_path, monkeypatch):
    api_json = json.dumps({
        "candidates": [{
            "finishReason": "STOP",
            "content": {"parts": [{"text": "here"}, {"inlineData": {"mimeType": "image/png", "data": _png_b64()}}]},
        }],
    }).encode()
    calls = _fake_urlopen(monkeypatch, [(200, api_json)])
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    raw = tmp_path / "raw.png"

    run = GeminiProvider().generate(GenRequest(prompt="a heart", raw=raw, aspect_ratio="16:9"), tmp_path)

    assert raw.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert run.provider == "gemini" and run.model == "gemini-2.5-flash-image"
    assert run.extra["part_kinds"] == ["text", "image/png"]
    assert run.extra["finish_reason"] == "STOP"
    assert run.extra["aspect_ratio"] == "16:9"
    call = calls[0]
    assert call["url"] == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent"
    assert call["key"] == "test-key"
    assert call["body"]["generationConfig"]["responseModalities"] == ["TEXT", "IMAGE"]
    assert call["body"]["generationConfig"]["imageConfig"] == {"aspectRatio": "16:9"}
    assert call["body"]["contents"][0]["parts"] == [{"text": "a heart"}]


def test_unknown_aspect_is_omitted_not_fatal(tmp_path, monkeypatch):
    api_json = json.dumps({
        "candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": _png_b64()}}]}}]
    }).encode()
    calls = _fake_urlopen(monkeypatch, [(200, api_json)])
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    run = GeminiProvider().generate(GenRequest(prompt="p", raw=tmp_path / "raw.png", aspect_ratio="auto"), tmp_path)

    assert "imageConfig" not in calls[0]["body"]["generationConfig"]
    assert run.extra["aspect_ratio"] == "auto"  # raw request recorded, config omitted


def test_refs_become_inline_parts(tmp_path, monkeypatch):
    from PIL import Image

    ref = tmp_path / "ref.png"
    Image.new("RGB", (2, 2), (255, 0, 255)).save(ref, format="PNG")
    api_json = json.dumps({
        "candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": _png_b64()}}]}}]
    }).encode()
    calls = _fake_urlopen(monkeypatch, [(200, api_json)])
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    GeminiProvider().generate(GenRequest(prompt="edit this", raw=tmp_path / "raw.png", refs=[ref]), tmp_path)

    parts = calls[0]["body"]["contents"][0]["parts"]
    assert len(parts) == 2
    assert parts[0]["inlineData"]["mimeType"] == "image/png"
    assert base64.b64decode(parts[0]["inlineData"]["data"])[:8] == b"\x89PNG\r\n\x1a\n"
    assert parts[1] == {"text": "edit this"}


def test_missing_ref_fails_loud(tmp_path):
    with pytest.raises(SystemExit, match="reference image not found"):
        GeminiProvider().generate(
            GenRequest(prompt="p", raw=tmp_path / "raw.png", refs=[tmp_path / "nope.png"]), tmp_path)


def test_native_alpha_is_refused(tmp_path):
    monkeypatch_free = tmp_path  # no key needed: the gate fires before auth
    with pytest.raises(SystemExit, match="native alpha"):
        GeminiProvider().generate(
            GenRequest(prompt="p", raw=monkeypatch_free / "raw.png", native_alpha=True), monkeypatch_free)


def test_http_error_surfaces_body(tmp_path, monkeypatch):
    def urlopen(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests",
                                     {"Content-Type": "application/json"},
                                     io.BytesIO(b'{"error": {"message": "quota exceeded"}}'))

    monkeypatch.setattr(gemini_provider.urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    with pytest.raises(SystemExit, match="HTTP 429"):
        GeminiProvider().generate(GenRequest(prompt="p", raw=tmp_path / "raw.png"), tmp_path)


def test_imageless_response_fails_loud(tmp_path, monkeypatch):
    api_json = json.dumps({"candidates": [{"finishReason": "SAFETY",
                                           "content": {"parts": [{"text": "cannot do"}]}}]}).encode()
    _fake_urlopen(monkeypatch, [(200, api_json)])
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    with pytest.raises(SystemExit, match="no inline image"):
        GeminiProvider().generate(GenRequest(prompt="p", raw=tmp_path / "raw.png"), tmp_path)


def test_blocked_response_names_the_reason(tmp_path, monkeypatch):
    api_json = json.dumps({"promptFeedback": {"blockReason": "SAFETY"}}).encode()
    _fake_urlopen(monkeypatch, [(200, api_json)])
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    with pytest.raises(SystemExit, match="blocked: SAFETY"):
        GeminiProvider().generate(GenRequest(prompt="p", raw=tmp_path / "raw.png"), tmp_path)
