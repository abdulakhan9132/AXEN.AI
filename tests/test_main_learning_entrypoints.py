"""Root commands use the existing app worker and preserve explicit opt-in."""
from types import SimpleNamespace

import pytest

import main


def test_parser_requires_explicit_source_for_demo(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py", "autonomy", "start", "--mode", "LEARNING", "--offline"])
    args = main.parse_args()
    assert args.command == "autonomy" and args.query == "start" and args.offline
    with pytest.raises(SystemExit, match="Select --offline"):
        main.run_autonomy_demo(False, False)


def test_demo_dispatch_is_bounded_and_offline(monkeypatch):
    calls = []
    monkeypatch.setattr(main.subprocess, "run", lambda args, **kwargs: calls.append((args, kwargs)) or SimpleNamespace(returncode=0))
    assert main.run_autonomy_demo(True, False) == 0
    assert calls[0][0][-1] == "--offline"
    assert calls[0][1]["cwd"] == main.ROOT


def test_control_contacts_running_app_not_new_worker(monkeypatch):
    import requests
    calls = []
    class Response:
        def raise_for_status(self): pass
        def json(self): return {"status": "running"}
    class Session:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def post(self, url, **kwargs): calls.append((url, kwargs)); return Response()
    monkeypatch.setattr(requests, "Session", Session)
    assert main.run_autonomy_command("start", "LEARNING", True) == 0
    url, options = calls[0]
    assert url == "http://127.0.0.1:8000/api/autonomy/control"
    assert options["headers"]["X-AXEN-Control"] == "operator"
    assert options["json"]["offline_demo"] is True


def test_launcher_anchors_working_directory_and_reuses_companion(monkeypatch, tmp_path):
    import run_ui
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_ui.requests, "get", lambda *args, **kwargs: SimpleNamespace(ok=True))
    launches = []
    monkeypatch.setattr(run_ui.uvicorn, "run", lambda app, **kwargs: launches.append((app, kwargs)))
    run_ui.main()
    from pathlib import Path
    assert Path.cwd() == run_ui.ROOT
    assert launches == [("app.ui.app:app", {"host": "127.0.0.1", "port": 8000})]
