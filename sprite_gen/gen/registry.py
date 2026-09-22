# SPDX-License-Identifier: Apache-2.0
"""Provider registry — name -> adapter, from built-ins plus a local config file.

sprite-gen stays subscription-first for its built-in routes (codex, grok) and
explicit-only for metered ones (openai, and any custom provider marked
``billed = true``). Custom backends are declared in a TOML file so a new host is
config, not a fork of this package.

Config discovery (first hit wins):

1. ``--providers-config PATH`` on the CLI
2. ``SPRITE_GEN_PROVIDERS_CONFIG`` environment variable
3. ``./sprite-gen.providers.toml`` in the current working directory
4. ``<repo-root>/sprite-gen.providers.toml`` next to the installed package

Example ``sprite-gen.providers.toml``::

    [providers.comfy-local]
    kind = "comfy"
    url = "http://127.0.0.1:8188"
    workflow = "workflows/sdxl_magenta.json"
    transparency = "chroma"
    billed = false

    [providers.fal-schnell]
    kind = "openai_compatible"
    base_url = "https://fal.run/openai/v1"
    model = "fal-ai/flux/schnell"
    api_key_env = "FAL_KEY"
    transparency = "chroma"
    billed = true

``billed = true`` providers are explicit-only (never a default, never a
fallback target) and announce the charge on stderr before the request leaves —
the same subscription-first invariant as the built-in ``openai`` provider.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .base import TRANSPARENCY_STRATEGIES, Provider

# Built-in adapters, always available and always first in listings.
BUILTIN_PROVIDER_NAMES = ("codex", "grok", "openai")
# A metered backend is named at the call site or it does not run (구독 우선).
BUILTIN_EXPLICIT_ONLY = ("openai",)

BUILTIN_KINDS = ("codex", "grok", "openai", "openai_compatible", "comfy")

CONFIG_ENV = "SPRITE_GEN_PROVIDERS_CONFIG"
CONFIG_FILENAME = "sprite-gen.providers.toml"

# Monkeypatchable in tests.
_load_builtin: dict[str, Callable[[], Provider]] = {}


@dataclass(frozen=True)
class ProviderSpec:
    """One configured custom provider."""

    name: str
    kind: str
    options: dict[str, Any] = field(default_factory=dict)
    transparency: str = "chroma"
    billed: bool = False
    source: str = ""  # config path, for error messages


def _repo_root() -> Path:
    # sprite_gen/gen/registry.py -> sprite_gen/ -> repo root (editable / source tree).
    return Path(__file__).resolve().parents[2]


def discover_config_path(explicit: Path | None = None) -> Path | None:
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"gen: providers config not found: {path}")
        return path
    env = (os.environ.get(CONFIG_ENV) or "").strip()
    if env:
        path = Path(env).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"gen: {CONFIG_ENV} points at a missing file: {path}")
        return path
    cwd_hit = Path.cwd() / CONFIG_FILENAME
    if cwd_hit.is_file():
        return cwd_hit.resolve()
    repo_hit = _repo_root() / CONFIG_FILENAME
    if repo_hit.is_file():
        return repo_hit.resolve()
    return None


def load_specs(config_path: Path | None = None) -> dict[str, ProviderSpec]:
    """Parse the providers TOML into ``name -> ProviderSpec``. Missing file = empty."""
    path = discover_config_path(config_path)
    if path is None:
        return {}
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise SystemExit(f"gen: cannot read providers config {path}: {exc}") from exc
    table = data.get("providers")
    if table is None:
        return {}
    if not isinstance(table, dict):
        raise SystemExit(f"gen: {path}: [providers] must be a table of named providers")
    specs: dict[str, ProviderSpec] = {}
    for name, raw in table.items():
        if not isinstance(raw, dict):
            raise SystemExit(f"gen: {path}: providers.{name} must be a table")
        kind = raw.get("kind")
        if kind not in BUILTIN_KINDS:
            raise SystemExit(
                f"gen: {path}: providers.{name}.kind={kind!r} is unknown; "
                f"expected one of {', '.join(BUILTIN_KINDS)}"
            )
        transparency = raw.get("transparency", "chroma")
        if transparency not in TRANSPARENCY_STRATEGIES:
            raise SystemExit(
                f"gen: {path}: providers.{name}.transparency={transparency!r} is unknown; "
                f"expected one of {', '.join(TRANSPARENCY_STRATEGIES)}"
            )
        reserved = {"kind", "transparency", "billed"}
        options = {k: v for k, v in raw.items() if k not in reserved}
        if name in BUILTIN_PROVIDER_NAMES:
            raise SystemExit(
                f"gen: {path}: providers.{name} shadows a built-in provider name; "
                f"rename it (for example {name}-custom)"
            )
        specs[name] = ProviderSpec(
            name=name,
            kind=kind,
            options=options,
            transparency=transparency,
            billed=bool(raw.get("billed", False)),
            source=str(path),
        )
    return specs


def all_provider_names(config_path: Path | None = None) -> tuple[str, ...]:
    return (*BUILTIN_PROVIDER_NAMES, *sorted(load_specs(config_path)))


def explicit_only_names(config_path: Path | None = None) -> frozenset[str]:
    """Metered backends that may only be reached via an explicit ``--provider``."""
    names = set(BUILTIN_EXPLICIT_ONLY)
    for spec in load_specs(config_path).values():
        if spec.billed:
            names.add(spec.name)
    return frozenset(names)


def _build_custom(spec: ProviderSpec) -> Provider:
    if spec.kind == "openai_compatible":
        from .openai_compatible import OpenAICompatibleProvider

        return OpenAICompatibleProvider(spec)
    if spec.kind == "comfy":
        from .comfy_provider import ComfyProvider

        return ComfyProvider(spec)
    raise SystemExit(f"gen: unhandled provider kind {spec.kind!r} for {spec.name!r}")


def make_provider(
    name: str,
    *,
    keep_session: bool = False,
    config_path: Path | None = None,
) -> Provider:
    """Instantiate a provider by name (built-in or configured custom)."""
    if name == "codex":
        from .codex_provider import CodexProvider

        return CodexProvider(keep_session=keep_session)
    if name == "grok":
        from .grok_provider import GrokProvider

        return GrokProvider()
    if name == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider()
    specs = load_specs(config_path)
    if name in specs:
        return _build_custom(specs[name])
    known = all_provider_names(config_path)
    raise SystemExit(f"gen: unknown provider {name!r}; expected one of {', '.join(known)}")
