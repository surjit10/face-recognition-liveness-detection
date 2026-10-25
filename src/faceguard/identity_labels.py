"""Private local display aliases for canonical pseudonymous gallery identities."""

from __future__ import annotations

from pathlib import Path

from .config import ConfigurationError, load_yaml_mapping


def load_identity_labels(path: str | Path | None) -> dict[str, str]:
    """Load strict UI-only labels; a missing optional file means no aliases."""
    if path is None:
        return {}
    label_path = Path(path)
    if not label_path.exists():
        return {}
    document = load_yaml_mapping(label_path)
    expected = {"schema_version", "labels"}
    keys = set(document)
    if keys != expected:
        raise ConfigurationError(
            "Identity-label config keys mismatch; "
            f"missing={sorted(expected - keys)}, unknown={sorted(keys - expected)}"
        )
    if document["schema_version"] != 1:
        raise ConfigurationError("Identity-label schema_version must be 1")
    labels = document["labels"]
    if not isinstance(labels, dict):
        raise ConfigurationError("Identity labels must be a mapping")

    normalized: dict[str, str] = {}
    for person_id, display_name in labels.items():
        if not isinstance(person_id, str) or not person_id.strip():
            raise ConfigurationError("Identity-label keys must be non-empty strings")
        if not isinstance(display_name, str) or not display_name.strip():
            raise ConfigurationError(
                f"Display label for {person_id!r} must be a non-empty string"
            )
        clean_id = person_id.strip()
        clean_name = display_name.strip()
        if len(clean_id) > 128 or len(clean_name) > 64:
            raise ConfigurationError("Identity labels exceed the permitted length")
        if any(ord(character) < 32 for character in clean_id + clean_name):
            raise ConfigurationError("Identity labels must not contain control characters")
        if clean_id in normalized:
            raise ConfigurationError(f"Duplicate identity label: {clean_id}")
        normalized[clean_id] = clean_name
    return normalized
