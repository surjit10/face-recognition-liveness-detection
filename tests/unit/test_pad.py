from pathlib import Path
import json
import numpy as np
from PIL import Image
import pytest

from faceguard.pad_metrics import confusion, rates, roc_auc, select_eer_threshold
from faceguard.pad_manifest import (
    PadManifestError,
    PadRecord,
    _balanced_cap,
    _read_bbox,
    _safe_relative_path,
    _subject_id,
    load_official_records,
    prepare_celeba_spoof_manifests,
)


def test_pad_metrics_rates_and_eer() -> None:
    labels = np.array([1, 1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])

    counts = confusion(labels, scores, threshold=0.5)
    assert counts == {"tp": 3, "fp": 0, "tn": 3, "fn": 0}

    r = rates(counts)
    assert r["apcer"] == 0.0
    assert r["bpcer"] == 0.0
    assert r["acer"] == 0.0

    auc = roc_auc(labels, scores)
    assert auc == 1.0

    eer_result = select_eer_threshold(labels, scores)
    assert eer_result["acer"] == 0.0
    assert 0.3 < eer_result["threshold"] <= 0.7


def test_pad_manifest_subject_id_and_relative_path() -> None:
    path = "train/0012/live/0001.jpg"
    assert _subject_id(path, "train") == "0012"

    with pytest.raises(PadManifestError):
        _subject_id("test/0012/live/0001.jpg", "train")

    with pytest.raises(PadManifestError):
        _safe_relative_path("../secret/image.jpg")


def test_pad_manifest_balanced_cap() -> None:
    records = [
        PadRecord("train/01/live/1.jpg", "01", "train", 1, 0, 0, (0, 0, 10, 10)),
        PadRecord("train/01/live/2.jpg", "01", "train", 1, 0, 0, (0, 0, 10, 10)),
        PadRecord("train/01/spoof/1.jpg", "01", "train", 0, 1, 1, (0, 0, 10, 10)),
        PadRecord("train/01/spoof/2.jpg", "01", "train", 0, 1, 1, (0, 0, 10, 10)),
    ]
    capped = _balanced_cap(records, limit=2, seed=42)
    assert len(capped) == 2
    assert {r.target_live for r in capped} == {0, 1}


def test_load_official_records_filters_inconsistent_spoof_type(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def labels(spoof_type: int, live_spoof: int) -> list[int]:
        values = [0] * 44
        values[40] = spoof_type
        values[43] = live_spoof
        return values

    entries = {
        "Data/train/0001/live/valid.jpg": labels(0, 0),
        # Mirrors noisy official rows such as Data/train/4154/live/001021.jpg.
        "Data/train/4154/live/inconsistent.jpg": labels(7, 0),
        "Data/train/0002/spoof/valid.jpg": labels(7, 1),
    }
    metadata = tmp_path / "metas" / "intra_test"
    metadata.mkdir(parents=True)
    (metadata / "train_label.json").write_text(json.dumps(entries), encoding="utf-8")

    for relative_path in entries:
        image_path = tmp_path / relative_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 100), color=(128, 128, 128)).save(image_path, format="JPEG")
        image_path.with_name(f"{image_path.stem}_BB.txt").write_text(
            "50 50 40 40\n", encoding="utf-8"
        )

    records = load_official_records(tmp_path, "train")

    assert {row.relative_path for row in records} == {
        "Data/train/0001/live/valid.jpg",
        "Data/train/0002/spoof/valid.jpg",
    }
    assert "Filtered 1 noisy/missing samples; kept 2 clean records" in capsys.readouterr().out
