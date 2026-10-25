import importlib.util
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "faceguard_run_demo", PROJECT_ROOT / "scripts" / "run_demo.py"
)
assert SPEC is not None and SPEC.loader is not None
RUN_DEMO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN_DEMO)


class Result:
    state = "IDENTIFIED"
    status = "verified"
    reason = "cached_continuous_track"

    @staticmethod
    def event() -> dict[str, object]:
        return {"state": "IDENTIFIED", "status": "verified"}


def test_render_reports_a_state_only_once(capsys) -> None:
    frame = SimpleNamespace()

    last_state, stop = RUN_DEMO.render_and_report(frame, Result(), 1, True, None)
    first_output = capsys.readouterr().out
    same_state, stop_again = RUN_DEMO.render_and_report(
        frame, Result(), 2, True, last_state
    )
    second_output = capsys.readouterr().out

    assert '"sequence": 1' in first_output
    assert second_output == ""
    assert same_state == last_state
    assert stop is False
    assert stop_again is False
