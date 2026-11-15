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


def test_dashboard_recorded_benchmark_matches_receipt() -> None:
    """Recorded cockpit values (shown when no live feed is connected) match the receipt."""
    html = (PROJECT_ROOT / "reports/mlops_dashboard.html").read_text(encoding="utf-8")
    session = json.loads(
        (PROJECT_ROOT / "reports/benchmarks/prototype_camera.json").read_text(encoding="utf-8")
    )["session"]

    assert f'<span id="fpsValue">{session["processed_fps"]:.2f}</span>' in html
    assert f'<span id="p95Value">{session["p95_latency_ms"]:.1f}</span>' in html
    assert f'<span id="memValue">{session["peak_memory_mb"]:.1f}</span>' in html
    assert f'{session["stable_face_verification_ms"] / 1000:.2f} <span class="metric-unit">sec' in html
