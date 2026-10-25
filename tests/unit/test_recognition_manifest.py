import csv
from pathlib import Path

from PIL import Image
import pytest

from faceguard.config import sha256_file
from faceguard.recognition_manifest import (
    FIELDS,
    RecognitionManifestError,
    validate_recognition_manifests,
)


def _make_split(tmp_path: Path, split: str, identities: list[str]) -> Path:
    root = tmp_path / "images"
    manifest = tmp_path / f"{split}.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(FIELDS))
        writer.writeheader()
        for index, identity in enumerate(identities):
            relative = f"{split}/{index}.jpg"
            image = root / relative
            image.parent.mkdir(parents=True, exist_ok=True)
            split_color = sum(ord(value) for value in split) % 255
            Image.new("RGB", (8, 8), (split_color, index, 30)).save(
                image, format="JPEG"
            )
            writer.writerow(
                {
                    "sample_id": f"{split}-{index}",
                    "source_dataset": "synthetic-test",
                    "relative_path": relative,
                    "verified_person_id": identity,
                    "session_id": "",
                    "device_id": "",
                    "source_sha256": sha256_file(image),
                    "permission_id": "synthetic-fixture",
                }
            )
    return manifest


def _manifests(tmp_path: Path) -> dict[str, Path]:
    return {
        "enrollment": _make_split(tmp_path, "enrollment", ["a", "b"]),
        "val_known": _make_split(tmp_path, "val_known", ["a"]),
        "val_impostor": _make_split(tmp_path, "val_impostor", ["c"]),
        "test_known": _make_split(tmp_path, "test_known", ["b"]),
        "test_impostor": _make_split(tmp_path, "test_impostor", ["d"]),
    }


def test_recognition_manifests_validate_with_explicit_session_limitation(
    tmp_path: Path,
) -> None:
    report = validate_recognition_manifests(
        dataset_root=tmp_path / "images",
        manifests=_manifests(tmp_path),
        allow_unknown_session_metadata=True,
    )
    assert report["total_samples"] == 6
    assert report["enrollment_identities"] == 2


def test_recognition_manifests_reject_unknown_sessions_without_acknowledgement(
    tmp_path: Path,
) -> None:
    with pytest.raises(RecognitionManifestError, match="Session/device"):
        validate_recognition_manifests(
            dataset_root=tmp_path / "images",
            manifests=_manifests(tmp_path),
            allow_unknown_session_metadata=False,
        )
