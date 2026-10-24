from pathlib import Path

import pytest

from faceguard.config import (
    ConfigurationError,
    index_decisions,
    require_processing_permission,
    resolve_under,
)


def test_duplicate_decisions_are_rejected() -> None:
    document = {"decisions": [{"id": "X"}, {"id": "X"}]}
    with pytest.raises(ConfigurationError, match="Duplicate decision"):
        index_decisions(document)


def test_processing_permission_requires_verified_explicit_value() -> None:
    unresolved = {
        "decisions": [
            {"id": "FRAD_PERMISSION", "state": "UNRESOLVED", "value": None}
        ]
    }
    with pytest.raises(ConfigurationError, match="required state"):
        require_processing_permission(unresolved, context="local")

    local_only = {
        "decisions": [
            {
                "id": "FRAD_PERMISSION",
                "state": "VERIFIED",
                "value": {"local_processing": True, "kaggle_cloud_transfer": False},
            }
        ]
    }
    require_processing_permission(local_only, context="local")
    with pytest.raises(ConfigurationError, match="kaggle_cloud_transfer"):
        require_processing_permission(local_only, context="kaggle")


def test_resolve_under_rejects_traversal(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    inside = root / "image.jpg"
    inside.write_bytes(b"x")
    assert resolve_under(root, "image.jpg") == inside.resolve()
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"x")
    with pytest.raises(ConfigurationError, match="escapes"):
        resolve_under(root, "../outside.jpg")
