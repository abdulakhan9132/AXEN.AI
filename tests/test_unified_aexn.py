"""Unified routing, proposal safety and in-process ASGI integration tests.

ASGI tests use a timer-only event loop: this Windows environment denies the
loopback sockets that asyncio normally creates internally. No network is used.
"""
import asyncio
import importlib
import json
import selectors
import subprocess
import time
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from app.ai.router import ModelRouter
from app.context.models import AssembledContext, ModelRequest
from axen.coding.coding_service import CodingService

ui = importlib.import_module('app.ui.app')


class TimerSelector(selectors.BaseSelector):
    def register(self, *args, **kwargs):
        raise RuntimeError('Network/file I/O is unavailable in in-process ASGI tests')
    def unregister(self, *args, **kwargs):
        raise RuntimeError('No file descriptors registered')
    def select(self, timeout=None):
        if timeout:
            time.sleep(min(timeout, .005))
        return []
    def get_map(self):
        return {}


class InProcessLoop(asyncio.SelectorEventLoop):
    def __init__(self):
        super().__init__(selector=TimerSelector())
    def _make_self_pipe(self):
        pass
    def _write_to_self(self):
        pass
    def _close_self_pipe(self):
        pass


def request(text, allow_cloud=False):
    return ModelRequest(AssembledContext(messages_for_model=[{'role': 'user', 'content': text}]),
                        {}, {'allow_cloud': allow_cloud})


@pytest.mark.parametrize('text,kind', [
    ('Hi AEXN', 'conversation'), ('Explain neural networks', 'conversation'),
    ('Fix bugs in my Python project', 'coding'), ('Create a beautiful website', 'coding'),
    ('Generate an image of a futuristic city', 'image'), ('Analyze this PDF', 'document'),
    ('give a simple html code', 'coding'), ('generate a landing page', 'coding'),
    ('Build a fitness app', 'coding'),
    ('give a simple Java program', 'coding'), ('write a C++ example', 'coding'),
    ('give a Go function', 'coding'), ('write a Rust script', 'coding'),
    ('Analyze this project', 'coding'),
    ('Calculate 25 * 4', 'math'), ('Continue this Shakespeare sonnet', 'literary')])
def test_automatic_task_understanding(text, kind):
    assert ModelRouter.task_kind(text) == kind


def test_local_router_selects_capabilities_without_cloud():
    resolver = Mock()
    router = ModelRouter(provider_resolver=resolver)
    for provider in [router.conversational_model, router.reasoning_model, router.coding_specialist, router.own_brain]:
        provider.health_check = Mock(return_value=True)
        provider.generate = Mock(return_value='local response')
    assert router.generate(request('Explain neural networks')) == 'local response'
    assert router.last_routed_provider == 'local_reasoning'
    router.generate(request('Write a Python function'))
    assert router.last_routed_provider == 'qwen_ollama'
    native = Mock()
    native.health_check.return_value = True
    native.generate.return_value = 'native verse'
    resolver.return_value = native
    assert router.generate(request('Write a Shakespeare sonnet')) == 'native verse'
    resolver.assert_called_once_with('aexn_gpt_1.0')


def test_cloud_key_is_not_consent():
    resolver = Mock()
    router = ModelRouter(provider_resolver=resolver)
    router.conversational_model.health_check = Mock(return_value=False)
    with pytest.raises(RuntimeError, match='No cloud request was sent'):
        router.generate(request('Explain neural networks'))
    resolver.assert_not_called()
    cloud = Mock()
    cloud.health_check.return_value = True
    cloud.generate.return_value = 'explicitly authorized cloud'
    resolver.return_value = cloud
    assert router.generate(request('Explain neural networks', allow_cloud=True)) == 'explicitly authorized cloud'


def proposed_service(tmp_path):
    (tmp_path / 'calculator.py').write_text('def add(a, b):\n    return a - b\n', encoding='utf-8')
    service = CodingService(tmp_path)
    provider = Mock()
    provider.structured_output.return_value = {'summary': 'Fix addition and add multiplication',
        'plan': ['Read calculator', 'Prepare two changes'], 'files': [
        {'path': 'calculator.py', 'content': 'def add(a, b):\n    return a + b\n'},
        {'path': 'multiply.py', 'content': 'def multiply(a, b):\n    return a * b\n'}]}
    return service, provider


def test_multifile_proposal_preview_apply_and_undo(tmp_path):
    service, provider = proposed_service(tmp_path)
    result = service.propose_task('Fix my Python function', provider)
    assert len(result['diffs']) == 2 and result['tests_run'] == 0
    assert 'return a - b' in (tmp_path / 'calculator.py').read_text()
    assert not (tmp_path / 'multiply.py').exists()
    applied = service.apply_proposal(result['proposal_id'])
    assert applied['files_modified'] == ['calculator.py', 'multiply.py']
    assert service.rollback_file('calculator.py')['success']
    assert 'return a - b' in (tmp_path / 'calculator.py').read_text()


def test_proposal_rejects_user_changes_after_preview(tmp_path):
    service, provider = proposed_service(tmp_path)
    result = service.propose_task('Fix my Python function', provider)
    (tmp_path / 'calculator.py').write_text('user_changes = True\n')
    with pytest.raises(ValueError, match='changed since preview'):
        service.apply_proposal(result['proposal_id'])
    assert not (tmp_path / 'multiply.py').exists()


@pytest.mark.parametrize('path', ['../escape.py', 'checkpoints/model.pt', '.env', 'data/train.json'])
def test_proposals_cannot_escape_or_modify_preserved_assets(tmp_path, path):
    service, provider = proposed_service(tmp_path)
    provider.structured_output.return_value['files'] = [{'path': path, 'content': 'value = 1\n'}]
    with pytest.raises((PermissionError, ValueError)):
        service.propose_task('Write code', provider)
    assert not service.pending_proposals


def test_git_checkpoint_preview_and_restore(tmp_path):
    def git(*args):
        result = subprocess.run(['git', *args], cwd=tmp_path, capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()
    git('init')
    git('config', 'user.email', 'tests@example.invalid')
    git('config', 'user.name', 'AEXN tests')
    file = tmp_path / 'calculator.py'
    file.write_text('value = 1\n')
    git('add', 'calculator.py')
    git('commit', '-m', 'Test fixture')
    service = CodingService(tmp_path)
    checkpoint = service.create_git_checkpoint()
    assert git('rev-parse', checkpoint['git_reference']) == git('rev-parse', 'HEAD')
    service.save_file('calculator.py', 'value = 2\n')
    preview = service.preview_git_rollback(checkpoint['checkpoint_id'])
    assert preview['files'] == ['calculator.py'] and preview['diffs']
    service.restore_git_checkpoint(checkpoint['checkpoint_id'])
    assert file.read_text() == 'value = 1\n'


def test_untrusted_execution_is_blocked_without_sandbox(tmp_path, monkeypatch):
    service, _ = proposed_service(tmp_path)
    monkeypatch.delenv('AEXN_SANDBOX_IMAGE', raising=False)
    with pytest.raises(PermissionError, match='No external sandbox'):
        service.cmd_runner.run_sandboxed(['python', '-m', 'pytest'])


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Isolate session and memory state from the user's running application.
    import app.ui.backend_service as backend
    from axen.assistant import AssistantController
    from axen.memory.long_term_memory import LongTermMemory
    from axen.projects.project_manager import ProjectManager
    from axen.projects.project_store import SQLiteProjectStore
    project_manager = ProjectManager(store=SQLiteProjectStore(tmp_path / 'projects.db'))
    long_memory = LongTermMemory(path=tmp_path / 'memory.json', db_path=tmp_path / 'memory.db')
    controller = AssistantController(long_term_memory=long_memory, project_manager=project_manager,
                                     knowledge_manager=Mock(), file_manager=Mock())
    monkeypatch.setattr(backend, 'AssistantController', lambda **kwargs: controller)
    monkeypatch.setattr(backend, 'CONVERSATIONS_FILE', tmp_path / 'conversations.json')
    monkeypatch.setattr(ProjectManager, '_instance', project_manager)
    service = backend.AxenBackendService()
    monkeypatch.setattr(backend.AxenBackendService, '_instance', service)
    coding, provider = proposed_service(tmp_path)
    monkeypatch.setattr(CodingService, '_instance', coding)
    manager = Mock()
    manager.get_model.return_value = provider
    provider.generate.return_value = 'Here is a complete page.\n```html\n<!DOCTYPE html><html><head><meta name="viewport" content="width=device-width, initial-scale=1"></head><body><button onclick="this.textContent=\'Hello\'">Click</button></body></html>\n```\nA responsive page with a button.\nTell me the theme you want.'
    monkeypatch.setattr(ui, 'model_manager', manager)
    with TestClient(ui.app, backend_options={'loop_factory': InProcessLoop}) as connection:
        yield connection


def frames(response):
    return [json.loads(frame[6:]) for frame in response.text.split('\n\n') if frame.startswith('data: ')]


def test_asgi_greeting_stream_and_history(client):
    session = client.post('/api/sessions/new', json={}).json()['session_id']
    response = client.post('/api/v2/chat', json={'messages': [{'role': 'user', 'content': 'Hi AEXN'}], 'session_id': session})
    assert response.status_code == 200
    assert response.headers['content-type'].startswith('text/event-stream')
    events = frames(response)
    assert any(e['type'] == 'token' for e in events)
    assert events[-1]['type'] == 'done', events
    assert events[-1]['data']
    history = client.get('/api/sessions/' + session).json()['messages']
    assert [m['role'] for m in history] == ['user', 'assistant']


def test_asgi_proposal_requires_approval(client):
    # The legacy tool API remains isolated; unified chat no longer invokes it.
    proposal = CodingService.get_instance().propose_task('Fix my Python function', ui.model_manager.get_model.return_value)
    endpoint = '/api/coding/proposals/' + proposal['proposal_id'] + '/apply'
    assert client.post(endpoint, json={}).status_code == 403
    applied = client.post(endpoint, json={'approved': True})
    assert applied.status_code == 200 and len(applied.json()['result']['files_modified']) == 2


def test_asgi_errors_and_execution_permissions(client):
    assert client.post('/api/v2/chat', json={'messages': []}).status_code == 400
    assert client.post('/api/coding/test', json={}).status_code == 403
    assert client.post('/api/coding/undo', json={'file_path': 'calculator.py'}).status_code == 403
    assert client.post('/api/coding/git/checkpoint', json={}).status_code == 403


def test_main_interface_has_one_assistant_no_selectors(client):
    html = client.get('/').text
    assert 'AEXN AI' in html
    assert 'select-active-model' not in html and 'select-think-mode' not in html
    assert 'coding-think-mode' not in html and 'coding-agent-input' not in html
    assert html.count('id="composer-input"') == 1
    assert 'btn-theme' in html
    assert 'tab-coding' not in html and 'btn-top-toggle-coding' not in html
    assert 'view-coding-workspace' not in html and 'coding-workspace.css' not in html


def test_asgi_document_without_attachment_and_history(client):
    session = client.post('/api/sessions/new', json={}).json()['session_id']
    events = frames(client.post('/api/v2/chat', json={
        'messages': [{'role': 'user', 'content': 'Analyze this PDF'}], 'session_id': session}))
    assert 'Attach the document' in events[-1]['data']
    history = client.get('/api/sessions/' + session).json()['messages']
    assert [m['role'] for m in history] == ['user', 'assistant']


def test_asgi_image_pipeline_wiring_and_history(client, monkeypatch):
    import image_brain
    pipeline = Mock()
    pipeline.generate.return_value = {'paths': ['outputs/images/example.png']}
    monkeypatch.setattr(image_brain, 'ImageBrain', lambda: pipeline)
    session = client.post('/api/sessions/new', json={}).json()['session_id']
    events = frames(client.post('/api/v2/chat', json={
        'messages': [{'role': 'user', 'content': 'Generate an image of a futuristic city'}], 'session_id': session}))
    assert events[-1]['data'] == '![Generated image](/images/example.png)'
    pipeline.generate.assert_called_once()
    assert len(client.get('/api/sessions/' + session).json()['messages']) == 2


def test_asgi_generated_image_can_be_loaded(client):
    files = sorted(ui.OUTPUTS_IMAGES_DIR.glob('*.png'))
    if not files:
        pytest.skip('No existing generated image to verify')
    # Real saved output and real StaticFiles handler; no mocked image bytes.
    from io import BytesIO
    from PIL import Image
    saved = files[-1]
    response = client.get('/images/' + saved.name)
    assert response.status_code == 200
    assert response.headers['content-type'] == 'image/png'
    assert response.content == saved.read_bytes()
    with Image.open(BytesIO(response.content)) as image:
        assert image.width > 0 and image.height > 0
        image.verify()


def test_asgi_approved_test_execution_returns_actual_output(client):
    service = CodingService.get_instance()
    test_file = service.workspace.root_path / 'test_calculator.py'
    test_file.write_text('from calculator import add\ndef test_add():\n    assert add(2, 1) == 1\n', encoding='utf-8')
    response = client.post('/api/coding/test', json={
        'approved_execution': True, 'trusted_project': True, 'target': 'test_calculator.py'})
    assert response.status_code == 200, response.text
    assert '1 passed' in response.text, response.text


def test_asgi_simple_code_uses_chat_not_project_changes(client):
    from app.ai.chat_adapter import CODING_INSTRUCTIONS
    coding = CodingService.get_instance()
    provider = ui.model_manager.get_model.return_value
    before = coding.workspace.read_file('calculator.py')
    events = frames(client.post('/api/v2/chat', json={
        'messages': [{'role': 'user', 'content': 'give a simple html code'}]}))
    assert events[-1]['type'] == 'done' and '<!DOCTYPE html>' in events[-1]['data']
    assert not any(e['type'] == 'workspace' for e in events)
    request = provider.generate.call_args.args[0]
    assert request.context.messages_for_model[0]['content'] == CODING_INSTRUCTIONS
    assert request.generation_settings['task_type'] == 'coding'
    assert not coding.pending_proposals
    assert coding.workspace.read_file('calculator.py') == before
    provider.structured_output.assert_not_called()


def test_asgi_vague_landing_page_produces_code_and_keeps_followup_context(client):
    session = client.post('/api/sessions/new', json={}).json()['session_id']
    provider = ui.model_manager.get_model.return_value
    first = frames(client.post('/api/v2/chat', json={
        'messages': [{'role': 'user', 'content': 'generate a landing page'}], 'session_id': session}))
    assert '<!DOCTYPE html>' in first[-1]['data']
    followup = frames(client.post('/api/v2/chat', json={
        'messages': [{'role': 'user', 'content': 'make the button blue'}], 'session_id': session}))
    assert followup[-1]['type'] == 'done'
    history = provider.generate.call_args.args[0].context.messages_for_model
    assert any(turn['role'] == 'assistant' and '<!DOCTYPE html>' in turn['content'] for turn in history)
    assert history[-1]['content'] == 'make the button blue'
    provider.structured_output.assert_not_called()


def test_project_edits_require_explicit_folder_selection(client):
    coding = CodingService.get_instance()
    coding.workspace_selected = False
    provider = ui.model_manager.get_model.return_value
    events = frames(client.post('/api/v2/chat', json={
        'messages': [{'role': 'user', 'content': 'Fix bugs in my Python project'}]}))
    assert '<!DOCTYPE html>' in events[-1]['data']
    assert not any(event['type'] == 'workspace' for event in events)
    assert not coding.pending_proposals
    provider.structured_output.assert_not_called()
    with pytest.raises(PermissionError, match='Choose the project folder'):
        coding.propose_task('Fix bugs', provider)
    assert client.post('/api/coding/chat', json={'message': 'Fix bugs'}).status_code == 403
    coding.set_workspace(coding.workspace.root_path)
    assert coding.workspace_selected
    assert coding.propose_task('Fix my Python function', provider)['proposal_id']


def test_code_followup_forces_coding_provider():
    router = ModelRouter()
    router.coding_specialist.health_check = Mock(return_value=True)
    router.coding_specialist.generate = Mock(return_value='updated code')
    followup = request('make the button blue')
    followup.generation_settings['task_type'] = 'coding'
    assert router.generate(followup) == 'updated code'


def test_standalone_entry_point_is_not_mistaken_for_project_edit():
    assert not ModelRouter.requires_workspace('Write the main function for a simple C program')
    assert ModelRouter.requires_workspace('Fix my Python function')


def test_fresh_code_request_does_not_inherit_previous_chat(client):
    session = client.post('/api/sessions/new', json={}).json()['session_id']
    provider = ui.model_manager.get_model.return_value
    client.post('/api/v2/chat', json={
        'messages': [{'role': 'user', 'content': 'simple html code'}], 'session_id': session})
    client.post('/api/v2/chat', json={
        'messages': [{'role': 'user', 'content': 'simple python code to add two numbers'}], 'session_id': session})
    messages = provider.generate.call_args.args[0].context.messages_for_model
    assert [turn['role'] for turn in messages] == ['system', 'user']
    assert messages[-1]['content'] == 'simple python code to add two numbers'
    assert 'Last: exactly two short lines' in messages[0]['content']
