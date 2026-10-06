"""
axen/projects/project_store.py
==============================
Thread-safe SQLite persistent storage for AXEN Projects.

Provides:
- Projects table with WAL mode and indexing
- Project memories table for scoped factual storage
- Project files junction table
- Project knowledge collections junction table
- Soft archive and hard deletion
- Safe parameter binding and error handling
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .project_models import Project, ProjectSettings

logger = logging.getLogger("axen.projects.store")
DEFAULT_PROJECTS_DB = Path("data/axen_projects.db")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteProjectStore:
    """Persistent SQLite database storing projects and their scoped associations."""

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        self.db_path = Path(db_path) if db_path else DEFAULT_PROJECTS_DB
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=15.0,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create tables and indexes with WAL mode."""
        with self._lock, self._get_connection() as conn:
            if str(self.db_path) != ":memory:":
                try:
                    conn.execute("PRAGMA journal_mode = WAL;")
                    conn.execute("PRAGMA synchronous = NORMAL;")
                except Exception as e:
                    logger.debug("Could not enable WAL pragma: %s", e)

            # 1. Projects table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id            TEXT PRIMARY KEY,
                name          TEXT NOT NULL,
                description   TEXT DEFAULT '',
                instructions  TEXT DEFAULT '',
                icon          TEXT DEFAULT 'folder',
                archived      INTEGER DEFAULT 0,
                settings_json TEXT DEFAULT '{}',
                created_at    TEXT NOT NULL,
                updated_at    TEXT NOT NULL
            );
            """)

            # 2. Project Memory table (isolated from global memory)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS project_memories (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id  TEXT NOT NULL,
                key         TEXT NOT NULL,
                value       TEXT NOT NULL,
                memory_type TEXT DEFAULT 'fact',
                confidence  REAL DEFAULT 1.0,
                is_active   INTEGER DEFAULT 1,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            );
            """)

            # 3. Project Files junction
            conn.execute("""
            CREATE TABLE IF NOT EXISTS project_files (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id  TEXT NOT NULL,
                file_id     TEXT NOT NULL,
                added_at    TEXT NOT NULL,
                UNIQUE (project_id, file_id),
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            );
            """)

            # 4. Project Knowledge Collections junction
            conn.execute("""
            CREATE TABLE IF NOT EXISTS project_knowledge_collections (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id    TEXT NOT NULL,
                collection_id TEXT NOT NULL,
                added_at      TEXT NOT NULL,
                UNIQUE (project_id, collection_id),
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            );
            """)

            # Indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_proj_archived ON projects (archived);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_proj_name ON projects (name);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pmem_proj ON project_memories (project_id, is_active);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pmem_key ON project_memories (project_id, key);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pfiles_proj ON project_files (project_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pknow_proj ON project_knowledge_collections (project_id);")
            conn.commit()

    # -----------------------------------------------------------------------
    # Project CRUD
    # -----------------------------------------------------------------------

    def create_project(self, project: Project) -> Project:
        """Insert a new project record."""
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO projects (
                    id, name, description, instructions, icon,
                    archived, settings_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    project.id,
                    project.name,
                    project.description,
                    project.instructions,
                    project.icon,
                    1 if project.archived else 0,
                    json.dumps(project.settings.to_dict()),
                    project.created_at,
                    project.updated_at,
                ),
            )
            conn.commit()
            return project

    def get_project(self, project_id: str) -> Optional[Project]:
        """Fetch project by ID."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM projects WHERE id = ?;", (project_id,))
            row = cursor.fetchone()
            if row:
                return Project.from_row(row)
            return None

    def update_project(self, project_id: str, updates: Dict[str, Any]) -> Optional[Project]:
        """Update selected fields of a project."""
        proj = self.get_project(project_id)
        if not proj:
            return None

        allowed = {"name", "description", "instructions", "icon", "archived", "settings"}
        new_name = updates.get("name", proj.name)
        new_desc = updates.get("description", proj.description)
        new_inst = updates.get("instructions", proj.instructions)
        new_icon = updates.get("icon", proj.icon)
        new_archived = updates.get("archived", proj.archived)
        new_settings_raw = updates.get("settings", proj.settings)
        if isinstance(new_settings_raw, ProjectSettings):
            new_settings = new_settings_raw.to_dict()
        elif isinstance(new_settings_raw, dict):
            new_settings = dict(new_settings_raw)
        else:
            new_settings = ProjectSettings().to_dict()

        if "instructions" in new_settings and not updates.get("instructions"):
            new_inst = new_settings["instructions"]
        elif new_inst and not new_settings.get("instructions"):
            new_settings["instructions"] = new_inst

        settings_json = json.dumps(new_settings)

        now = _now_iso()
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                UPDATE projects
                SET name = ?, description = ?, instructions = ?, icon = ?,
                    archived = ?, settings_json = ?, updated_at = ?
                WHERE id = ?;
                """,
                (
                    new_name,
                    new_desc,
                    new_inst,
                    new_icon,
                    1 if new_archived else 0,
                    settings_json,
                    now,
                    project_id,
                ),
            )
            conn.commit()

        return self.get_project(project_id)

    def delete_project(self, project_id: str) -> bool:
        """Permanently delete a project and its scoped associations."""
        with self._lock, self._get_connection() as conn:
            conn.execute("DELETE FROM project_memories WHERE project_id = ?;", (project_id,))
            conn.execute("DELETE FROM project_files WHERE project_id = ?;", (project_id,))
            conn.execute("DELETE FROM project_knowledge_collections WHERE project_id = ?;", (project_id,))
            cursor = conn.execute("DELETE FROM projects WHERE id = ?;", (project_id,))
            conn.commit()
            return cursor.rowcount > 0

    def archive_project(self, project_id: str) -> bool:
        """Mark project as archived without deleting its data."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute(
                "UPDATE projects SET archived = 1, updated_at = ? WHERE id = ?;",
                (_now_iso(), project_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def restore_project(self, project_id: str) -> bool:
        """Restore an archived project."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute(
                "UPDATE projects SET archived = 0, updated_at = ? WHERE id = ?;",
                (_now_iso(), project_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def list_projects(
        self,
        include_archived: bool = False,
        search: Optional[str] = None,
    ) -> List[Project]:
        """List projects ordered by updated_at descending with optional search query."""
        with self._lock, self._get_connection() as conn:
            query = "SELECT * FROM projects WHERE 1=1"
            params: list[Any] = []
            if not include_archived:
                query += " AND archived = 0"
            if search and search.strip():
                query += " AND (name LIKE ? OR description LIKE ?)"
                term = f"%{search.strip()}%"
                params.extend([term, term])
            query += " ORDER BY updated_at DESC;"

            cursor = conn.execute(query, params)
            return [Project.from_row(row) for row in cursor.fetchall()]

    # -----------------------------------------------------------------------
    # Project Memory Operations (Scoped strictly to project_id)
    # -----------------------------------------------------------------------

    def add_project_memory(
        self,
        project_id: str,
        key: str,
        value: str,
        memory_type: str = "fact",
        confidence: float = 1.0,
    ) -> int:
        """Store or update a memory item strictly scoped to a project."""
        clean_key = key.strip().lower()
        clean_val = value.strip()
        now = _now_iso()
        with self._lock, self._get_connection() as conn:
            # Check if active memory already exists for this key
            cursor = conn.execute(
                "SELECT id FROM project_memories WHERE project_id = ? AND key = ? AND is_active = 1;",
                (project_id, clean_key),
            )
            existing = cursor.fetchone()
            if existing:
                # Update existing
                conn.execute(
                    """
                    UPDATE project_memories
                    SET value = ?, memory_type = ?, confidence = ?, updated_at = ?
                    WHERE id = ?;
                    """,
                    (clean_val, memory_type, confidence, now, existing["id"]),
                )
                conn.commit()
                return existing["id"]
            else:
                # Insert new
                cur = conn.execute(
                    """
                    INSERT INTO project_memories (
                        project_id, key, value, memory_type, confidence,
                        is_active, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 1, ?, ?);
                    """,
                    (project_id, clean_key, clean_val, memory_type, confidence, now, now),
                )
                conn.commit()
                return cur.lastrowid

    def list_project_memories(
        self,
        project_id: str,
        active_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """List memories scoped to a specific project."""
        with self._lock, self._get_connection() as conn:
            query = "SELECT * FROM project_memories WHERE project_id = ?"
            params: list[Any] = [project_id]
            if active_only:
                query += " AND is_active = 1"
            query += " ORDER BY updated_at DESC;"
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            return [
                {
                    "id": r["id"],
                    "project_id": r["project_id"],
                    "key": r["key"],
                    "value": r["value"],
                    "memory_type": r["memory_type"],
                    "confidence": r["confidence"],
                    "is_active": bool(r["is_active"]),
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                }
                for r in rows
            ]

    def delete_project_memory(self, project_id: str, key: str) -> bool:
        """Deactivate or remove a scoped project memory."""
        clean_key = key.strip().lower()
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute(
                "UPDATE project_memories SET is_active = 0, updated_at = ? WHERE project_id = ? AND key = ?;",
                (_now_iso(), project_id, clean_key),
            )
            conn.commit()
            return cursor.rowcount > 0

    # -----------------------------------------------------------------------
    # Project Files Association
    # -----------------------------------------------------------------------

    def associate_file(self, project_id: str, file_id: str) -> bool:
        """Associate an uploaded file with a project."""
        now = _now_iso()
        with self._lock, self._get_connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO project_files (project_id, file_id, added_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(project_id, file_id) DO NOTHING;
                    """,
                    (project_id, file_id, now),
                )
                conn.commit()
                return True
            except Exception as e:
                logger.error("Error associating file with project: %s", e)
                return False

    def remove_file_association(self, project_id: str, file_id: str) -> bool:
        """Remove file association from a project without deleting the physical file."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM project_files WHERE project_id = ? AND file_id = ?;",
                (project_id, file_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def list_project_file_ids(self, project_id: str) -> List[str]:
        """Return list of file IDs associated with a project."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT file_id FROM project_files WHERE project_id = ? ORDER BY added_at DESC;",
                (project_id,),
            )
            return [row["file_id"] for row in cursor.fetchall()]

    # -----------------------------------------------------------------------
    # Project Knowledge Collections Association
    # -----------------------------------------------------------------------

    def associate_knowledge_collection(self, project_id: str, collection_id: str) -> bool:
        """Associate a knowledge collection with a project."""
        now = _now_iso()
        with self._lock, self._get_connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO project_knowledge_collections (project_id, collection_id, added_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(project_id, collection_id) DO NOTHING;
                    """,
                    (project_id, collection_id, now),
                )
                conn.commit()
                return True
            except Exception as e:
                logger.error("Error associating collection with project: %s", e)
                return False

    def list_project_collection_ids(self, project_id: str) -> List[str]:
        """Return list of collection IDs associated with a project."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT collection_id FROM project_knowledge_collections WHERE project_id = ? ORDER BY added_at DESC;",
                (project_id,),
            )
            return [row["collection_id"] for row in cursor.fetchall()]
