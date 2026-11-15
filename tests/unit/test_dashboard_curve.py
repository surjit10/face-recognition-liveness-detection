"""The dashboard's threshold simulator must use the committed validation curve verbatim."""

import json
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_dashboard_curve_matches_threshold_report() -> None:
    html = (PROJECT_ROOT / "reports/mlops_dashboard.html").read_text(encoding="utf-8")
    report = json.loads(
        (PROJECT_ROOT / "reports/evaluation/recognition_threshold.json").read_text(encoding="utf-8")
    )
    match = re.search(r"const VAL_CURVE = (\[.*?\]);\n", html)
    assert match, "dashboard no longer embeds VAL_CURVE"
    embedded = json.loads(match.group(1))
    expected = [
        [row["threshold"], row["false_matches"], row["false_non_matches"]]
        for row in report["curve"]
    ]

    assert embedded == expected
    assert f"const FROZEN_THRESHOLD = {report['threshold']!r};" in html
    assert f"const VAL_STRANGERS = {report['impostor_searches']};" in html
    assert f"const VAL_ENROLLED = {report['known_searches']};" in html
    assert f"const FROZEN_REJECTS = {report['selected']['false_non_matches']};" in html
