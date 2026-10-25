from pathlib import Path

import pytest

from faceguard.config import ConfigurationError
from faceguard.identity_labels import load_identity_labels


def test_missing_optional_identity_labels_return_empty_mapping(tmp_path: Path) -> None:
    assert load_identity_labels(tmp_path / "missing.yaml") == {}


def test_identity_labels_are_trimmed_and_loaded(tmp_path: Path) -> None:
    path = tmp_path / "labels.yaml"
    path.write_text(
        'schema_version: 1\nlabels:\n  "000": "  Display Name  "\n',
        encoding="utf-8",
    )
    assert load_identity_labels(path) == {"000": "Display Name"}


@pytest.mark.parametrize(
    "contents",
    [
        'schema_version: 2\nlabels: {}\n',
        'schema_version: 1\nlabels: []\n',
        'schema_version: 1\nlabels:\n  "000": ""\n',
        'schema_version: 1\nlabels: {}\nunknown: true\n',
    ],
)
def test_invalid_identity_label_configuration_fails_closed(
    tmp_path: Path, contents: str
) -> None:
    path = tmp_path / "labels.yaml"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_identity_labels(path)
