from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_DEMO = PROJECT_ROOT / "scripts" / "run_demo.py"


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RUN_DEMO), *arguments],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_demo_requires_source_unless_check_only() -> None:
    result = _run()

    assert result.returncode == 2
    assert "one of --video or --camera is required" in result.stderr


def test_check_only_rejects_video_or_camera_source() -> None:
    result = _run("--check-only", "--camera", "0")

    assert result.returncode == 2
    assert "--check-only cannot be combined" in result.stderr


def test_readiness_report_requires_check_only() -> None:
    result = _run("--camera", "0", "--readiness-report", "unused.json")

    assert result.returncode == 2
    assert "--readiness-report requires --check-only" in result.stderr
