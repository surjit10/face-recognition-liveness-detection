import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from faceguard.ui.app import DesktopApp


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "faceguard_run_app", PROJECT_ROOT / "scripts" / "run_app.py"
)
assert SPEC is not None and SPEC.loader is not None
RUN_APP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN_APP)


def test_app_defaults_to_authorized_camera_zero() -> None:
    args = RUN_APP.parse_args([])
    assert args.camera == 0
    assert args.config == PROJECT_ROOT / "configs/runtime.yaml"
    assert args.identity_labels == PROJECT_ROOT / "configs/private/identity_labels.yaml"
    assert args.history_db == PROJECT_ROOT / "reports/private/checkins.sqlite3"
    assert args.operator_config == PROJECT_ROOT / "configs/private/operator.yaml"


def test_app_rejects_unapproved_camera_index() -> None:
    with pytest.raises(SystemExit) as exc:
        RUN_APP.parse_args(["--camera", "1"])
    assert exc.value.code == 2


def test_verified_session_stops_camera_after_starting_attendance() -> None:
    calls: list[object] = []
    app = DesktopApp.__new__(DesktopApp)
    app._verification_finishing = False
    app.record_attendance = lambda: calls.append("attendance")
    app.view = SimpleNamespace(set_controls=lambda **values: calls.append(values))
    session = SimpleNamespace(stop=lambda *, wait: calls.append(("stop", wait)))

    app._finish_verified_session(session)

    assert app._verification_finishing is True
    assert calls == [
        "attendance",
        {"check": False, "start": False, "stop": True},
        ("stop", False),
    ]


def test_stop_camera_clears_preview_and_resets_ready_status() -> None:
    calls: list[object] = []
    app = DesktopApp.__new__(DesktopApp)
    app._auto_reset_timer = "timer1"
    app.root = SimpleNamespace(after_cancel=lambda tid: calls.append(f"cancel_{tid}"))
    app._verification_finishing = True
    app._session_mode = "verification"
    app._bundle = SimpleNamespace()
    app.view = SimpleNamespace(
        clear_preview=lambda msg: calls.append(f"preview_{msg}"),
        set_status=lambda msg, detail, tone: calls.append(f"status_{msg}"),
        set_source_active=lambda active: calls.append(f"source_{active}"),
        set_controls=lambda **kw: calls.append(kw),
    )
    app._stop_active_session = lambda **kw: calls.append("stop_session")

    app.stop_camera()

    assert app._auto_reset_timer is None
    assert app._verification_finishing is False
    assert calls == [
        "cancel_timer1",
        "preview_Camera preview is off",
        "status_READY",
        "source_False",
        {"check": False, "start": True, "stop": False},
        "stop_session",
    ]



def test_app_attendance_admin_restricts_pages_and_relocks_on_leave() -> None:
    from faceguard.operator_auth import OperatorAuth

    toasts: list[str] = []
    calls: list[str] = []
    app = DesktopApp.__new__(DesktopApp)
    app.operator_auth = OperatorAuth(salt=b"0" * 16, digest=b"0" * 32, role="attendance_admin")
    app._operator_unlocked = False
    app._active_page = "verify"
    app.shell = SimpleNamespace(
        show_toast=lambda msg, **kw: toasts.append(msg),
        select=lambda name: calls.append(f"select_{name}"),
    )
    verify_page = SimpleNamespace(tkraise=lambda: calls.append("raise_verify"))
    history_page = SimpleNamespace(tkraise=lambda: calls.append("raise_history"))
    enroll_page = SimpleNamespace(tkraise=lambda: calls.append("raise_enroll"))
    app._pages = {"verify": verify_page, "history": history_page, "enroll": enroll_page}
    app._stop_active_session = lambda **kw: None
    app._clear_verification_state = lambda: None
    app._update_operator_ui = lambda: None
    app.refresh_history = lambda: calls.append("refresh_history")

    # Attempt to open enroll -> blocked by permits()
    app.show_page("enroll")
    assert app._active_page == "verify"
    assert "raise_enroll" not in calls
    assert any("manage attendance only" in t for t in toasts)

    # Open history -> requires PIN, then unlocks and shows
    app._ensure_operator = lambda: True
    app.show_page("history")
    assert app._active_page == "history"
    assert "raise_history" in calls
    assert "refresh_history" in calls

    # Now navigate back to verify -> automatically re-locks
    app._operator_unlocked = True
    app.show_page("verify")
    assert app._active_page == "verify"
    assert app._operator_unlocked is False


def test_app_delete_selected_attendance_requires_selection_and_confirmation(monkeypatch) -> None:
    calls: list[str] = []
    infos: list[str] = []
    confirms: list[str] = []

    app = DesktopApp.__new__(DesktopApp)
    app.root = None
    app.history_view = SimpleNamespace(
        selected_event_id=None,
        set_status=lambda msg, **kw: calls.append(f"status_{msg}"),
    )
    app._run_task = lambda name, op: calls.append(f"task_{name}")

    from tkinter import messagebox
    monkeypatch.setattr(messagebox, "showinfo", lambda title, msg, **kw: infos.append(msg))
    monkeypatch.setattr(messagebox, "askyesno", lambda title, msg, **kw: confirms.append(msg) or True)

    # No selection
    app.delete_selected_attendance()
    assert any("Select an attendance record" in msg for msg in infos)
    assert not any("task_history_delete" in c for c in calls)

    # Valid selection and confirmation
    app.history_view.selected_event_id = 42
    app.delete_selected_attendance()
    assert any("42" in msg for msg in confirms)
    assert "task_history_delete" in calls

