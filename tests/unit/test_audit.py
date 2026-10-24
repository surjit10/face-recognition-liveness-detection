import csv
import json
from pathlib import Path

from PIL import Image

from faceguard.audit import audit_dataset


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 12), color=color).save(path, format="JPEG")


def test_audit_is_non_destructive_and_marks_duplicate_and_decode_failure(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    first = dataset / "0001" / "a.jpg"
    duplicate = dataset / "0001" / "b.jpg"
    second = dataset / "0002" / "c.jpg"
    corrupt = dataset / "0002" / "bad.jpg"
    _write_image(first, (10, 20, 30))
    duplicate.write_bytes(first.read_bytes())
    _write_image(second, (40, 50, 60))
    corrupt.write_bytes(b"not-a-jpeg")
    original_hashes = {path: path.read_bytes() for path in (first, duplicate, second, corrupt)}

    inventory = tmp_path / "manifests" / "inventory.csv"
    subjects = tmp_path / "manifests" / "subjects.csv"
    report = tmp_path / "reports" / "audit.json"
    summary = audit_dataset(dataset, inventory, subjects, report)

    assert summary["candidate_subjects"] == 2
    assert summary["discovered_files"] == 4
    assert summary["decoded_files"] == 3
    assert summary["exact_duplicate_files_excluded"] == 1
    with inventory.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_path = {row["relative_path"]: row for row in rows}
    assert by_path["0001/a.jpg"]["include"] == "UNRESOLVED"
    assert by_path["0001/b.jpg"]["exclusion_reason"] == "exact_duplicate"
    assert by_path["0002/bad.jpg"]["exclusion_reason"] == "decode_failed"
    assert json.loads(report.read_text(encoding="utf-8"))["inventory_sha256"]
    with subjects.open(encoding="utf-8", newline="") as handle:
        subject_rows = list(csv.DictReader(handle))
    assert [row["candidate_folder"] for row in subject_rows] == ["0001", "0002"]
    assert all(row["include"] == "UNRESOLVED" for row in subject_rows)
    assert {path: path.read_bytes() for path in original_hashes} == original_hashes
