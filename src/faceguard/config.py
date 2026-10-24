"""Strict configuration, decision, path, and hashing helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Configuration is absent, malformed, or not approved for the operation."""


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {config_path}")
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Unable to load YAML {config_path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigurationError(f"YAML root must be a mapping: {config_path}")
    return loaded


def index_decisions(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = document.get("decisions")
    if not isinstance(entries, list):
        raise ConfigurationError("decisions must be a list")
    indexed: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise ConfigurationError("Each decision must be a mapping with a string id")
        decision_id = entry["id"]
        if decision_id in indexed:
            raise ConfigurationError(f"Duplicate decision id: {decision_id}")
        indexed[decision_id] = entry
    return indexed


def require_decision(
    decisions: dict[str, dict[str, Any]],
    decision_id: str,
    *,
    allowed_states: tuple[str, ...] = ("VERIFIED", "FROZEN"),
) -> dict[str, Any]:
    entry = decisions.get(decision_id)
    if entry is None:
        raise ConfigurationError(f"Required decision is missing: {decision_id}")
    state = entry.get("state")
    if state not in allowed_states:
        allowed = ", ".join(allowed_states)
        raise ConfigurationError(
            f"Decision {decision_id} is {state!r}; required state: {allowed}"
        )
    return entry


def require_processing_permission(
    decision_document: dict[str, Any], *, context: str
) -> None:
    if context not in {"local", "kaggle"}:
        raise ConfigurationError(f"Unsupported processing context: {context}")
    decisions = index_decisions(decision_document)
    permission = require_decision(decisions, "FRAD_PERMISSION")
    value = permission.get("value")
    if not isinstance(value, dict):
        raise ConfigurationError("FRAD_PERMISSION.value must be a mapping")
    key = "local_processing" if context == "local" else "kaggle_cloud_transfer"
    if value.get(key) is not True:
        raise ConfigurationError(
            f"FRAD_PERMISSION does not explicitly authorize {key}=true"
        )


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_under(root: str | Path, relative_path: str) -> Path:
    base = Path(root).resolve(strict=True)
    candidate = (base / relative_path).resolve(strict=True)
    if not candidate.is_relative_to(base):
        raise ConfigurationError(f"Path escapes dataset root: {relative_path}")
    return candidate
