"""Route and provider contract regressions; no network or live model required."""
import importlib
from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from app.ai.chat_adapter import ProviderChatAdapter
from app.ai.aexn_gpt_provider import AEXNGPTProvider
from app.context.models import AssembledContext, ModelRequest

ui = importlib.import_module("app.ui.app")


def test_model_selection_survives_next_catalog_request(monkeypatch):
    manager = ui.ModelManager()
    monkeypatch.setattr(manager, "health_check", lambda *args: False)
    monkeypatch.setattr(ui, "model_manager", manager)
    result = ui.select_model(ui.SelectModelRequest(model_id="aexn_gpt_1.0"))
    assert result["success"]
    assert ui.get_models()["active_model"] == "aexn_gpt_1.0"
    assert isinstance(manager.get_model(), AEXNGPTProvider)


def test_unknown_model_does_not_change_selection(monkeypatch):
    manager = ui.ModelManager()
    monkeypatch.setattr(manager, "health_check", lambda *args: False)
    monkeypatch.setattr(ui, "model_manager", manager)
    with pytest.raises(HTTPException) as exc:
        ui.select_model(ui.SelectModelRequest(model_id="missing-model"))
    assert exc.value.status_code == 400
    assert manager.active_model_name == "router"


def test_controller_adapter_preserves_messages_and_generation_settings():
    provider = Mock()
    provider.generate.return_value = "result"
    adapter = ProviderChatAdapter(provider, "aexn_gpt_1.0")
    messages = [{"role": "user", "content": "Shakespeare"}]
    assert adapter.chat(messages, max_new_tokens=32) == "result"
    request = provider.generate.call_args.args[0]
    assert request.context.messages_for_model == messages
    assert request.generation_settings["max_new_tokens"] == 32


def test_native_provider_uses_model_request_settings():
    provider = AEXNGPTProvider(device="cpu")
    provider._initialized = True
    provider.model = Mock()
    provider.model.generate.return_value = "A verse"
    request = ModelRequest(AssembledContext(messages_for_model=[
        {"role": "user", "content": "Write a verse"}
    ]), {"max_tokens": 12, "temperature": 0.0}, {})
    assert provider.generate(request) == "A verse"
    assert provider.model.generate.call_args.kwargs["max_new_tokens"] == 12
    assert provider.model.generate.call_args.kwargs["temperature"] == 0.0


def test_node_features_have_one_route_each():
    for path, method in [("/api/reverse-engineering/status", "GET"),
                         ("/api/reverse-engineering", "POST"),
                         ("/api/brain", "GET")]:
        routes = [r for r in ui.app.routes if getattr(r, "path", None) == path
                  and method in getattr(r, "methods", set())]
        assert len(routes) == 1
        assert routes[0].endpoint is ui.node_feature_proxy


def test_coding_handlers_browse_read_save_and_analyze(tmp_path, monkeypatch):
    from axen.coding.coding_service import CodingService
    (tmp_path / "calculator.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    service = CodingService(root_path=tmp_path)
    monkeypatch.setattr(CodingService, "get_instance", classmethod(lambda cls: service))
    assert ui.get_coding_workspace()["workspace"]["workspace_name"] == tmp_path.name
    assert ui.get_coding_file_tree(max_depth=2)["tree"]["children"]
    assert "return a + b" in ui.read_coding_file(path="calculator.py")["file"]["content"]
    saved = ui.save_coding_file({"path": "multiply.py", "content": "def multiply(a, b):\n    return a * b\n"})
    assert saved["status"] == "ok"
    assert "return a * b" in ui.read_coding_file(path="multiply.py")["file"]["content"]
    result = ui.run_coding_chat_task({"message": "Analyze this project", "think_mode": "fast"})
    assert result["task_result"]["success"]
    assert result["timeline"]


def test_frontend_message_persistence_uses_existing_store(monkeypatch):
    import threading
    service = Mock()
    service._chat_lock = threading.Lock()
    service.conversations = {"test-session": {"messages": []}}
    monkeypatch.setattr(ui.AxenBackendService, "get_instance", classmethod(lambda cls: service))
    assert ui.append_session_message("test-session", {"role": "user", "content": "hi"}) == {"status": "ok"}
    assert service.conversations["test-session"]["messages"][0]["content"] == "hi"
    service._save_conversations_to_disk.assert_called_once()


def test_workspace_tree_and_analysis_exclude_internal_folders(tmp_path):
    from axen.coding.coding_service import CodingService
    for folder in (".agents", ".codex", ".gsd-staging", "%SystemDrive%", "app"):
        target = tmp_path / folder
        target.mkdir()
        (target / "example.py").write_text("value = 1\n", encoding="utf-8")
    service = CodingService(root_path=tmp_path)
    names = {child["name"] for child in service.get_file_tree()["children"]}
    assert names == {"app"}
    assert [p.relative_to(tmp_path).as_posix() for p in service.workspace.list_files()] == ["app/example.py"]


def test_workspace_tree_rejects_resolved_directory_escape(tmp_path, monkeypatch):
    from pathlib import Path
    from axen.coding.coding_service import CodingService
    project = tmp_path / "project"
    project.mkdir()
    link = project / "external"
    link.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    service = CodingService(root_path=project)
    original_resolve = Path.resolve

    def resolve_as_external_junction(path, *args, **kwargs):
        if path == link:
            return outside
        return original_resolve(path, *args, **kwargs)

    # Simulate Windows junction resolution without requiring symlink privileges.
    monkeypatch.setattr(Path, "resolve", resolve_as_external_junction)
    assert service.get_file_tree()["children"] == []
