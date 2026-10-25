import csv
from pathlib import Path

import numpy as np
import pytest

from faceguard.recognition_split import (
    RecognitionSplitError,
    create_exploratory_recognition_split,
)


def _archive(path: Path) -> None:
    ids: list[str] = []
    paths: list[str] = []
    hashes: list[str] = []
    vectors: list[np.ndarray] = []
    for identity_number in range(10):
        person_id = "000" if identity_number == 0 else f"{identity_number:04d}"
        for sample in range(6):
            ids.append(person_id)
            paths.append(f"{person_id}/{sample}.jpg")
            hashes.append(f"{identity_number * 6 + sample:064x}")
            vector = np.zeros(4, dtype=np.float32)
            vector[identity_number % 4] = 1.0
            vectors.append(vector)
    np.savez_compressed(
        path,
        embeddings=np.stack(vectors),
        person_ids=np.asarray(ids),
        relative_paths=np.asarray(paths),
        source_sha256=np.asarray(hashes),
        model_hashes_json=np.asarray(['{"w600k_r50.onnx":"' + "a" * 64 + '"}']),
    )


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_exploratory_split_is_reproducible_and_open_set(tmp_path: Path) -> None:
    archive = tmp_path / "embeddings.npz"
    _archive(archive)
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first = create_exploratory_recognition_split(
        embedding_archive=archive, output_dir=first_dir, seed=42
    )
    second = create_exploratory_recognition_split(
        embedding_archive=archive, output_dir=second_dir, seed=42
    )
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["identity_labels"]["contains_label_000"] is True
    enrollment = _read(first_dir / "recognition_enrollment.csv")
    val_unknown = _read(first_dir / "recognition_val_impostor.csv")
    test_unknown = _read(first_dir / "recognition_test_impostor.csv")
    enrolled_ids = {row["verified_person_id"] for row in enrollment}
    val_ids = {row["verified_person_id"] for row in val_unknown}
    test_ids = {row["verified_person_id"] for row in test_unknown}
    assert enrolled_ids.isdisjoint(val_ids | test_ids)
    assert val_ids.isdisjoint(test_ids)
    all_rows = sum((_read(first_dir / f"{name}.csv") for name in (
        "recognition_enrollment",
        "recognition_val_known",
        "recognition_val_impostor",
        "recognition_test_known",
        "recognition_test_impostor",
    )), [])
    assert len({row["source_sha256"] for row in all_rows}) == len(all_rows)
    assert all(row["session_id"] == row["device_id"] == "" for row in all_rows)


def test_split_refuses_overwrite(tmp_path: Path) -> None:
    archive = tmp_path / "embeddings.npz"
    _archive(archive)
    output = tmp_path / "out"
    create_exploratory_recognition_split(
        embedding_archive=archive, output_dir=output, seed=42
    )
    with pytest.raises(RecognitionSplitError, match="overwrite"):
        create_exploratory_recognition_split(
            embedding_archive=archive, output_dir=output, seed=42
        )
