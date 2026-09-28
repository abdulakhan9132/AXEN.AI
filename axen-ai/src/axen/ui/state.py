from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Message:
    role: str
    content: str
    timestamp: str = field(default_factory=_now)


@dataclass
class ChatSession:
    id: str
    title: str
    messages: list[Message] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)


class ChatRepository:
    """SQLite persistence for conversations and messages."""

    def __init__(self, path: str | Path = "data/axen_chats.sqlite3") -> None:
        self.path = Path(path)
        self._memory_connection = sqlite3.connect(":memory:") if str(path) == ":memory:" else None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return self._memory_connection or sqlite3.connect(self.path)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id)
                );
                """
            )

    def create_session(self, title: str = "New Chat") -> ChatSession:
        session = ChatSession(id=uuid4().hex, title=title)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO conversations VALUES (?, ?, ?, ?)",
                (session.id, session.title, session.created_at, session.updated_at),
            )
        return session

    def save_session(self, session: ChatSession) -> None:
        session.updated_at = _now()
        with self._connect() as connection:
            connection.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (session.title, session.updated_at, session.id),
            )
            connection.execute("DELETE FROM messages WHERE conversation_id = ?", (session.id,))
            connection.executemany(
                "INSERT INTO messages(conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                [(session.id, item.role, item.content, item.timestamp) for item in session.messages],
            )

    def list_sessions(self) -> list[ChatSession]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
            ).fetchall()
        return [ChatSession(id=row[0], title=row[1], created_at=row[2], updated_at=row[3]) for row in rows]

    def load_session(self, session_id: str) -> ChatSession | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?", (session_id,)
            ).fetchone()
            if row is None:
                return None
            messages = connection.execute(
                "SELECT role, content, timestamp FROM messages WHERE conversation_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        return ChatSession(
            id=row[0],
            title=row[1],
            created_at=row[2],
            updated_at=row[3],
            messages=[Message(role=item[0], content=item[1], timestamp=item[2]) for item in messages],
        )
