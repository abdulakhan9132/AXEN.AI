from pathlib import Path

from axen.ui.state import ChatRepository, Message


def test_conversation_creation_and_persistence(tmp_path: Path):
    repository = ChatRepository(tmp_path / "chats.sqlite3")
    session = repository.create_session("Hello AXEN")
    session.messages.extend([Message("user", "hello"), Message("assistant", "local response")])
    repository.save_session(session)

    reopened = ChatRepository(tmp_path / "chats.sqlite3").load_session(session.id)

    assert reopened is not None
    assert reopened.title == "Hello AXEN"
    assert [(item.role, item.content) for item in reopened.messages] == [
        ("user", "hello"),
        ("assistant", "local response"),
    ]


def test_missing_session_returns_none(tmp_path: Path):
    repository = ChatRepository(tmp_path / "chats.sqlite3")

    assert repository.load_session("missing") is None
