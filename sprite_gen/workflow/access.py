# SPDX-License-Identifier: Apache-2.0
"""Credential readiness is not proof of a paid subscription or remaining quota."""
from __future__ import annotations

import os
import shutil
import subprocess

from sprite_gen.gen.base import provider_binary, provider_subprocess_env
from sprite_gen.gen.video import AUTH_SOURCE_API_KEY, resolve_credential


def probe_access(provider: str, *, video: bool = False) -> dict:
    result = {"provider": provider, "login": "unknown", "subscription": "unknown", "quota": "unknown",
              "billing": "subscription", "reason": "Subscription and media access need user confirmation when not independently known."}
    if provider == "codex":
        if shutil.which("codex") is None:
            return {**result, "login": "unavailable", "reason": "codex CLI is not installed"}
        try:
            check = subprocess.run([provider_binary("codex"), "login", "status"], capture_output=True,
                                   text=True, encoding="utf-8", errors="replace", timeout=15, env=provider_subprocess_env())
        except (OSError, subprocess.SubprocessError):
            return {**result, "reason": "codex login status could not be checked"}
        if check.returncode != 0:
            return {**result, "login": "unavailable", "reason": "codex login status failed; sign in and check again"}
        # Never publish the CLI's raw output: some authentication modes can name keys.
        output = (check.stdout + check.stderr).lower()
        if "chatgpt" not in output:
            return {**result, "reason": "login succeeded but ChatGPT subscription authentication was not identified"}
        return {**result, "login": "ready"}
    if provider == "gemini":
        if not (os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()):
            return {**result, "login": "unavailable",
                    "reason": "GEMINI_API_KEY is not set; create a key at https://aistudio.google.com/apikey and export GEMINI_API_KEY"}
        return {**result, "login": "ready", "billing": "api-credit",
                "reason": "gemini will use GEMINI_API_KEY; a free-tier daily quota may cover it — confirm this billing choice."}
    if provider == "zai":
        if not os.environ.get("ZAI_API_KEY", "").strip():
            return {**result, "login": "unavailable",
                    "reason": "ZAI_API_KEY is not set; create an API key in the Z.ai open platform console and export ZAI_API_KEY"}
        return {**result, "login": "ready", "billing": "api-credit",
                "reason": "zai will use ZAI_API_KEY and open-platform pay-per-use credit; confirm this billing choice."}
    if provider == "custom":
        if not os.environ.get("SPRITE_GEN_CUSTOM_CMD", "").strip():
            return {**result, "login": "unavailable",
                    "reason": "SPRITE_GEN_CUSTOM_CMD is not set; point it at the command that bridges your image backend"}
        return {**result, "login": "ready", "billing": "unknown",
                "reason": "custom runs SPRITE_GEN_CUSTOM_CMD against any backend; billing is whatever that backend charges."}
    if provider != "grok":
        raise ValueError(f"unknown provider: {provider}")
    if not video and shutil.which("grok") is None:
        return {**result, "login": "unavailable", "reason": "grok CLI is not installed"}
    try:
        # Image generation uses CLI login. Video may instead use the configured
        # API key; the existing video resolver alone owns that precedence.
        credential = resolve_credential(env=None if video else {})
    except (SystemExit, OSError, ValueError):
        return {**result, "login": "unavailable", "reason": "grok credential missing, expired or unreadable; check grok login"}
    if credential.source == AUTH_SOURCE_API_KEY:
        return {**result, "login": "ready", "billing": "api-credit",
                "reason": "Video will use XAI_API_KEY and separate API credit; confirm this billing choice."}
    return {**result, "login": "ready"}
