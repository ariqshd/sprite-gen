# SPDX-License-Identifier: Apache-2.0
"""Provider registry + openai_compatible / comfy adapters (offline unit tests)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sprite_gen.gen import registry
from sprite_gen.gen.base import GenRequest, TRANSPARENCY_CHROMA
from sprite_gen.gen.comfy_provider import ComfyProvider
from sprite_gen.gen.openai_compatible import OpenAICompatibleProvider


@pytest.fixture()
def providers_toml(tmp_path: Path) -> Path:
    path = tmp_path / "sprite-gen.providers.toml"
    path.write_text(
        """
[providers.local-a]
kind = "openai_compatible"
base_url = "http://127.0.0.1:9999/v1"
model = "test-model"
api_key_env = "TEST_API_KEY"
transparency = "chroma"
billed = true
max_refs = 2
quality_levels = ["auto", "low"]

[providers.comfy-box]
kind = "comfy"
url = "http://127.0.0.1:8188"
workflow = "wf.json"
transparency = "chroma"
billed = false
""",
        encoding="utf-8",
    )
    (tmp_path / "wf.json").write_text(
        json.dumps({"prompt": {"6": {"inputs": {"text": "{{PROMPT}}"}}}}),
        encoding="utf-8",
    )
    return path


def test_load_specs_reads_custom(providers_toml: Path) -> None:
    specs = registry.load_specs(providers_toml)
    assert set(specs) == {"local-a", "comfy-box"}
    assert specs["local-a"].billed is True
    assert specs["comfy-box"].kind == "comfy"
    assert specs["comfy-box"].options["workflow"] == "wf.json"


def test_all_names_includes_builtins_and_custom(providers_toml: Path) -> None:
    names = registry.all_provider_names(providers_toml)
    assert names[:3] == ("codex", "grok", "openai")
    assert "local-a" in names and "comfy-box" in names


def test_explicit_only_covers_billed_custom(providers_toml: Path) -> None:
    explicit = registry.explicit_only_names(providers_toml)
    assert "openai" in explicit
    assert "local-a" in explicit
    assert "comfy-box" not in explicit
    assert "codex" not in explicit


def test_builtin_name_shadow_rejected(tmp_path: Path) -> None:
    path = tmp_path / "p.toml"
    path.write_text(
        '[providers.grok]\nkind = "comfy"\nurl = "http://x"\nworkflow = "w.json"\n',
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="shadows a built-in"):
        registry.load_specs(path)


def test_unknown_kind_rejected(tmp_path: Path) -> None:
    path = tmp_path / "p.toml"
    path.write_text('[providers.x]\nkind = "nope"\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="kind"):
        registry.load_specs(path)


def test_make_provider_custom(providers_toml: Path) -> None:
    backend = registry.make_provider("local-a", config_path=providers_toml)
    assert isinstance(backend, OpenAICompatibleProvider)
    assert backend.name == "local-a"
    assert backend.transparency == TRANSPARENCY_CHROMA
    comfy = registry.make_provider("comfy-box", config_path=providers_toml)
    assert isinstance(comfy, ComfyProvider)


def test_openai_compatible_refuses_resolution(providers_toml: Path) -> None:
    backend = registry.make_provider("local-a", config_path=providers_toml)
    request = GenRequest(
        prompt="p", raw=Path("raw.png"), resolution="2k"
    )
    with pytest.raises(SystemExit, match="resolution"):
        backend._fields(request)


def test_openai_compatible_refuses_unknown_quality(providers_toml: Path) -> None:
    backend = registry.make_provider("local-a", config_path=providers_toml)
    request = GenRequest(prompt="p", raw=Path("raw.png"), quality="max")
    with pytest.raises(SystemExit, match="quality"):
        backend._fields(request)


def test_openai_compatible_fields_shape(providers_toml: Path) -> None:
    backend = registry.make_provider("local-a", config_path=providers_toml)
    request = GenRequest(prompt="a slime", raw=Path("raw.png"), aspect_ratio="1:1", quality="low")
    fields = backend._fields(request)
    assert fields["model"] == "test-model"
    assert fields["size"] == "1024x1024"
    assert fields["quality"] == "low"
    assert "background" not in fields


def test_comfy_refuses_quality(providers_toml: Path) -> None:
    backend = registry.make_provider("comfy-box", config_path=providers_toml)
    request = GenRequest(prompt="p", raw=Path("raw.png"), quality="low")
    with pytest.raises(SystemExit, match="quality"):
        backend.generate(request, Path("."))


def test_comfy_workflow_substitution(providers_toml: Path) -> None:
    backend = registry.make_provider("comfy-box", config_path=providers_toml)
    request = GenRequest(prompt="red slime", raw=Path("raw.png"), aspect_ratio="1:1")
    graph = backend._render_workflow(request, None)
    assert graph["prompt"]["6"]["inputs"]["text"] == "red slime"
    assert graph["client_id"].startswith("sprite-gen-")
