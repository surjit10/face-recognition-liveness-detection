import json
from pathlib import Path

import pytest

import faceguard.pad_manifest as pad_manifest
from faceguard.pad_manifest import PadRecord, prepare_celeba_spoof_manifests


def _labels(spoof_type: int, live_spoof: int) -> list[int]:
    values = [0] * 44
    values[40] = spoof_type
    values[43] = live_spoof
    return values


def _write_sample(root: Path, relative_path: str) -> None:
    image = root / relative_path
    image.parent.mkdir(parents=True, exist_ok=True)
    image.touch()
    image.with_name(f"{image.stem}_BB.txt").write_text(
        "1 2 20 30\n", encoding="utf-8"
    )


def test_quota_validation_stops_before_scanning_all_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidates = []
    for label in (0, 1):
        for index in range(200):
            candidates.append(
                pad_manifest._PadCandidate(
                    relative_path=f"Data/train/{index:04d}/{'live' if label else 'spoof'}/{label}.jpg",
                    subject_id=f"{index:04d}",
                    source_split="train",
                    target_live=label,
                    source_live_spoof_label=1 - label,
                    spoof_type=0 if label else 1,
                )
            )

    checked: list[str] = []

    def validate(_root: Path, candidate: pad_manifest._PadCandidate) -> PadRecord:
        checked.append(candidate.relative_path)
        return PadRecord(
            relative_path=candidate.relative_path,
            subject_id=candidate.subject_id,
            source_split=candidate.source_split,
            target_live=candidate.target_live,
            source_live_spoof_label=candidate.source_live_spoof_label,
            spoof_type=candidate.spoof_type,
            bbox=(1, 2, 20, 30),
        )

    monkeypatch.setattr(pad_manifest, "_validate_candidate", validate)
    selected = pad_manifest._select_valid_records(
        tmp_path,
        candidates,
        split_name="train",
        limit=10,
        seed=42,
        validation_workers=4,
        progress_every=1_000,
    )

    assert len(selected) == 10
    assert {row.target_live for row in selected} == {0, 1}
    assert len(checked) < len(candidates)
    assert len(checked) <= 128  # one bounded batch per class


def test_prepare_reuses_verified_manifest_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_entries: dict[str, list[int]] = {}
    for subject_number in range(1, 11):
        subject = f"{subject_number:04d}"
        live = f"Data/train/{subject}/live/1.jpg"
        spoof = f"Data/train/{subject}/spoof/1.jpg"
        train_entries[live] = _labels(0, 0)
        train_entries[spoof] = _labels(1, 1)
        _write_sample(tmp_path, live)
        _write_sample(tmp_path, spoof)

    test_entries = {
        "Data/test/1001/live/1.jpg": _labels(0, 0),
        "Data/test/1001/spoof/1.jpg": _labels(1, 1),
    }
    for relative_path in test_entries:
        _write_sample(tmp_path, relative_path)

    metadata = tmp_path / "metas" / "intra_test"
    metadata.mkdir(parents=True)
    (metadata / "train_label.json").write_text(
        json.dumps(train_entries), encoding="utf-8"
    )
    (metadata / "test_label.json").write_text(
        json.dumps(test_entries), encoding="utf-8"
    )

    output = tmp_path / "manifests"
    first = prepare_celeba_spoof_manifests(
        dataset_root=tmp_path,
        output_dir=output,
        validation_subject_fraction=0.1,
        train_limit=2,
        validation_limit=2,
        test_limit=2,
        validation_workers=2,
        progress_every=100,
    )

    def should_not_rebuild(*args: object, **kwargs: object) -> list[PadRecord]:
        raise AssertionError("valid cached manifests should not be rebuilt")

    monkeypatch.setattr(pad_manifest, "_select_valid_records", should_not_rebuild)
    second = prepare_celeba_spoof_manifests(
        dataset_root=tmp_path,
        output_dir=output,
        validation_subject_fraction=0.1,
        train_limit=2,
        validation_limit=2,
        test_limit=2,
        validation_workers=2,
        progress_every=100,
    )

    assert second["build_fingerprint"] == first["build_fingerprint"]
    assert second["splits"] == first["splits"]
