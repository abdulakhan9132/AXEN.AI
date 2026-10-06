"""
axen/projects/project_models.py
===============================
Core data models for AXEN Projects / Workspaces System.

Defines:
- ProjectSettings: granular feature flags and model preferences
- Project: persistent workspace entity with instructions and metadata
- ProjectContext: assembled scoped context for agent execution
- ProjectSummary: lightweight representation for sidebar navigation
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ProjectSettings:
    """Granular project-level settings and capability flags."""
    instructions: str = ""
    web_access: bool = True
    memory_enabled: bool = True
    knowledge_enabled: bool = True
    file_access: bool = True
    model_preference: str = "AUTO"  # AUTO | FAST | REASONING | CODE | LOCAL
    response_style: str = "balanced"  # concise | balanced | detailed
    tool_permissions: Dict[str, bool] = field(default_factory=dict)

    def __init__(
        self,
        instructions: str = "",
        web_access: Optional[bool] = None,
        web_access_enabled: Optional[bool] = None,
        memory_enabled: bool = True,
        knowledge_enabled: Optional[bool] = None,
        knowledge_access_enabled: Optional[bool] = None,
        file_access: Optional[bool] = None,
        file_access_enabled: Optional[bool] = None,
        model_preference: str = "AUTO",
        response_style: str = "balanced",
        tool_permissions: Optional[Dict[str, bool]] = None,
    ) -> None:
        self.instructions = instructions
        self.web_access = web_access if web_access is not None else (web_access_enabled if web_access_enabled is not None else True)
        self.memory_enabled = memory_enabled
        self.knowledge_enabled = knowledge_enabled if knowledge_enabled is not None else (knowledge_access_enabled if knowledge_access_enabled is not None else True)
        self.file_access = file_access if file_access is not None else (file_access_enabled if file_access_enabled is not None else True)
        self.model_preference = model_preference
        self.response_style = response_style
        self.tool_permissions = tool_permissions or {}

    @property
    def web_access_enabled(self) -> bool:
        return self.web_access

    @web_access_enabled.setter
    def web_access_enabled(self, val: bool) -> None:
        self.web_access = bool(val)

    @property
    def knowledge_access_enabled(self) -> bool:
        return self.knowledge_enabled

    @knowledge_access_enabled.setter
    def knowledge_access_enabled(self, val: bool) -> None:
        self.knowledge_enabled = bool(val)

    @property
    def file_access_enabled(self) -> bool:
        return self.file_access

    @file_access_enabled.setter
    def file_access_enabled(self, val: bool) -> None:
        self.file_access = bool(val)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instructions": self.instructions,
            "web_access": self.web_access,
            "web_access_enabled": self.web_access,
            "memory_enabled": self.memory_enabled,
            "knowledge_enabled": self.knowledge_enabled,
            "knowledge_access_enabled": self.knowledge_enabled,
            "file_access": self.file_access,
            "file_access_enabled": self.file_access,
            "model_preference": self.model_preference,
            "response_style": self.response_style,
            "tool_permissions": dict(self.tool_permissions),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ProjectSettings":
        if not data:
            return cls()
        return cls(
            instructions=str(data.get("instructions", "")),
            web_access=bool(data.get("web_access", data.get("web_access_enabled", True))),
            memory_enabled=bool(data.get("memory_enabled", True)),
            knowledge_enabled=bool(data.get("knowledge_enabled", data.get("knowledge_access_enabled", True))),
            file_access=bool(data.get("file_access", data.get("file_access_enabled", True))),
            model_preference=str(data.get("model_preference", "AUTO")),
            response_style=str(data.get("response_style", "balanced")),
            tool_permissions=dict(data.get("tool_permissions", {})),
        )


@dataclass
class Project:
    """Persistent workspace grouping chats, instructions, memory, files, and knowledge."""
    id: str
    name: str
    description: str = ""
    instructions: str = ""
    icon: str = "folder"
    archived: bool = False
    settings: ProjectSettings = field(default_factory=ProjectSettings)
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "instructions": self.instructions,
            "icon": self.icon,
            "archived": self.archived,
            "settings": self.settings.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_row(cls, row: Dict[str, Any] | Any) -> "Project":
        """Construct from SQLite row dictionary or object."""
        if hasattr(row, "keys"):
            d = dict(row)
        else:
            d = row
        settings_raw = d.get("settings_json") or d.get("settings")
        if isinstance(settings_raw, str):
            try:
                settings_dict = json.loads(settings_raw)
            except Exception:
                settings_dict = {}
        elif isinstance(settings_raw, dict):
            settings_dict = settings_raw
        else:
            settings_dict = {}

        return cls(
            id=str(d.get("id")),
            name=str(d.get("name")),
            description=str(d.get("description") or ""),
            instructions=str(d.get("instructions") or ""),
            icon=str(d.get("icon") or "folder"),
            archived=bool(d.get("archived", False)),
            settings=ProjectSettings.from_dict(settings_dict),
            created_at=str(d.get("created_at") or _now_iso()),
            updated_at=str(d.get("updated_at") or _now_iso()),
        )


@dataclass
class ProjectSummary:
    """Lightweight summary of a project for sidebar and lists."""
    id: str
    name: str
    description: str
    icon: str
    archived: bool
    chat_count: int = 0
    file_count: int = 0
    updated_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProjectContext:
    """Scoped project context gathered for an agent execution."""
    project_id: str
    project_name: str
    instructions: str = ""
    memories: List[Dict[str, Any]] = field(default_factory=list)
    files: List[Dict[str, Any]] = field(default_factory=list)
    knowledge_chunks: List[Dict[str, Any]] = field(default_factory=list)
    formatted_prompt_context: str = ""
    debug_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
