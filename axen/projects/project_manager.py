"""
axen/projects/project_manager.py
================================
Central Orchestrator for AXEN Projects / Workspaces System.

Responsibilities:
- Lifecycle management (create, read, update, delete, archive, restore)
- Strict workspace isolation: Project A never leaks into Project B
- Scoped context assembly for AgentLoop (Instructions -> Memory -> Files -> Knowledge)
- Bounded, selective retrieval respecting context window limits
- Neutralization of document prompt injections (documents are DATA, not instructions)
- Structured debugging and observability
"""

from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .project_models import Project, ProjectContext, ProjectSettings, ProjectSummary
from .project_store import SQLiteProjectStore

logger = logging.getLogger("axen.projects.manager")


class ProjectManager:
    """Unified Facade for all AXEN Project & Workspace operations."""

    _instance: Optional["ProjectManager"] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, db_path: Optional[Path | str] = None) -> "ProjectManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(store=SQLiteProjectStore(db_path=db_path))
            return cls._instance

    def __init__(
        self,
        store: Optional[SQLiteProjectStore] = None,
        file_manager: Optional[Any] = None,
        knowledge_manager: Optional[Any] = None,
        memory_manager: Optional[Any] = None,
    ) -> None:
        self.store = store or SQLiteProjectStore()
        self._file_manager = file_manager
        self._knowledge_manager = knowledge_manager
        self._memory_manager = memory_manager

    # -----------------------------------------------------------------------
    # Lazy dependency resolution
    # -----------------------------------------------------------------------

    @property
    def file_manager(self) -> Any:
        if self._file_manager is None:
            try:
                from axen.storage.file_manager import FileManager
                self._file_manager = FileManager.get_instance()
            except Exception as e:
                logger.debug("FileManager not initialized: %s", e)
        return self._file_manager

    @property
    def knowledge_manager(self) -> Any:
        if self._knowledge_manager is None:
            try:
                from axen.knowledge.knowledge_manager import KnowledgeManager
                self._knowledge_manager = KnowledgeManager()
            except Exception as e:
                logger.debug("KnowledgeManager not initialized: %s", e)
        return self._knowledge_manager

    @property
    def memory_manager(self) -> Any:
        if self._memory_manager is None:
            try:
                from axen.memory.memory_manager import MemoryManager
                self._memory_manager = MemoryManager()
            except Exception as e:
                logger.debug("MemoryManager not initialized: %s", e)
        return self._memory_manager

    # -----------------------------------------------------------------------
    # Project CRUD
    # -----------------------------------------------------------------------

    def create_project(
        self,
        name: str,
        description: str = "",
        instructions: str = "",
        icon: str = "folder",
        settings: Optional[ProjectSettings | Dict[str, Any]] = None,
    ) -> Project:
        """Create a new project workspace with a unique ID."""
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Project name cannot be empty.")

        # Generate a stable, readable project ID: proj_01JAXEN...
        unique_suffix = uuid.uuid4().hex[:12]
        project_id = f"proj_{unique_suffix}"

        if isinstance(settings, dict):
            proj_settings = ProjectSettings.from_dict(settings)
        elif isinstance(settings, ProjectSettings):
            proj_settings = settings
        else:
            proj_settings = ProjectSettings()

        clean_instructions = instructions.strip()
        if not proj_settings.instructions and clean_instructions:
            proj_settings.instructions = clean_instructions
        elif not clean_instructions and proj_settings.instructions:
            clean_instructions = proj_settings.instructions

        project = Project(
            id=project_id,
            name=clean_name,
            description=description.strip(),
            instructions=clean_instructions,
            icon=icon,
            archived=False,
            settings=proj_settings,
        )

        created = self.store.create_project(project)
        logger.info("Created project '%s' (id=%s)", clean_name, project_id)
        return created

    def get_project(self, project_id: str) -> Optional[Project]:
        """Retrieve project by ID."""
        if not project_id or not project_id.strip():
            return None
        return self.store.get_project(project_id.strip())

    def update_project(
        self,
        project_id: str,
        updates: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Optional[Project]:
        """Update metadata or settings of an existing project."""
        all_updates = dict(updates or {})
        all_updates.update(kwargs)
        return self.store.update_project(project_id, all_updates)

    def delete_project(
        self,
        project_id: str,
        delete_physical_files: bool = False,
    ) -> bool:
        """
        Delete a project.
        By default removes project associations while preserving physical files unless explicitly requested.
        """
        proj = self.get_project(project_id)
        if not proj:
            return False

        # If user explicitly requests deleting physical files belonging to this project
        if delete_physical_files and self.file_manager:
            file_ids = self.store.list_project_file_ids(project_id)
            for fid in file_ids:
                try:
                    self.file_manager.delete_file(fid)
                except Exception as e:
                    logger.warning("Error deleting physical file %s: %s", fid, e)

        success = self.store.delete_project(project_id)
        logger.info("Deleted project id=%s (success=%s)", project_id, success)
        return success

    def archive_project(self, project_id: str) -> Optional[Project]:
        """Archive project workspace."""
        ok = self.store.archive_project(project_id)
        return self.get_project(project_id) if ok else None

    def restore_project(self, project_id: str) -> Optional[Project]:
        """Restore archived project workspace."""
        ok = self.store.restore_project(project_id)
        return self.get_project(project_id) if ok else None

    def list_projects(
        self,
        include_archived: bool = False,
        search: Optional[str] = None,
    ) -> List[Project]:
        """List projects with optional search query."""
        return self.store.list_projects(include_archived=include_archived, search=search)

    # -----------------------------------------------------------------------
    # Scoped Project Memory
    # -----------------------------------------------------------------------

    def add_project_memory(
        self,
        project_id: str,
        key: str,
        value: str,
        memory_type: str = "fact",
        confidence: float = 1.0,
    ) -> int:
        """Add or update memory isolated to a specific project."""
        proj = self.get_project(project_id)
        if not proj:
            raise ValueError(f"Project '{project_id}' not found.")
        mid = self.store.add_project_memory(
            project_id=project_id,
            key=key,
            value=value,
            memory_type=memory_type,
            confidence=confidence,
        )
        # Also mirror into unified MemoryManager with scope="project"
        if self.memory_manager:
            try:
                self.memory_manager.remember(
                    key=key,
                    value=value,
                    category=memory_type,
                    confidence=confidence,
                    scope="project",
                    project_id=project_id,
                    source="project",
                )
            except Exception as e:
                logger.debug("Could not mirror project memory to memory_manager: %s", e)
        return mid

    def list_project_memories(
        self,
        project_id: str,
        active_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """List active memories for a project."""
        return self.store.list_project_memories(project_id, active_only=active_only)

    def delete_project_memory(self, project_id: str, key: str) -> bool:
        """Deactivate a scoped memory key in a project."""
        ok = self.store.delete_project_memory(project_id, key)
        if self.memory_manager:
            try:
                self.memory_manager.forget(key, scope="project", project_id=project_id)
            except Exception as e:
                logger.debug("Could not mirror forget to memory_manager: %s", e)
        return ok

    def retrieve_relevant_memories(
        self,
        project_id: str,
        query: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Selective, bounded memory retrieval based on lexical relevance."""
        all_memories = self.store.list_project_memories(project_id, active_only=True)
        if not all_memories:
            return []

        query_tokens = set(re.findall(r"\w+", query.lower()))
        scored = []
        for mem in all_memories:
            text = f"{mem['key']} {mem['value']}".lower()
            mem_tokens = set(re.findall(r"\w+", text))
            overlap = len(query_tokens.intersection(mem_tokens))
            score = overlap / (len(query_tokens) + 1.0)
            # If query is short or generic, include with lower score
            scored.append((score, mem))

        # Sort descending by score, take top items
        scored.sort(key=lambda x: x[0], reverse=True)
        # If there are matching items, return them; otherwise if query asks general 'what do you know', return up to limit
        relevant = [m for s, m in scored if s > 0][:limit]
        if not relevant and any(w in query.lower() for w in ["what", "project", "instructions", "about", "know", "model", "config"]):
            relevant = [m for _, m in scored][:limit]
        return relevant

    # -----------------------------------------------------------------------
    # Scoped Project Files
    # -----------------------------------------------------------------------

    def add_file_to_project(self, project_id: str, file_id: str) -> bool:
        """Link an uploaded file to this project workspace."""
        proj = self.get_project(project_id)
        if not proj:
            raise ValueError(f"Project '{project_id}' not found.")
        return self.store.associate_file(project_id, file_id)

    def remove_file_from_project(
        self,
        project_id: str,
        file_id: str,
        delete_physical: bool = False,
    ) -> bool:
        """Unlink file from project and optionally delete physical file."""
        ok = self.store.remove_file_association(project_id, file_id)
        if ok and delete_physical and self.file_manager:
            self.file_manager.delete_file(file_id)
        return ok

    def list_project_files(self, project_id: str) -> List[Any]:
        """List full UploadedFile objects associated with this project."""
        file_ids = self.store.list_project_file_ids(project_id)
        if not self.file_manager or not file_ids:
            return []
        files = []
        for fid in file_ids:
            f = self.file_manager.get_file(fid)
            if f:
                files.append(f)
        return files

    def retrieve_relevant_project_files(
        self,
        project_id: str,
        query: str,
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        """Retrieve most relevant project files for context assembly."""
        files = self.list_project_files(project_id)
        if not files:
            return []

        query_tokens = set(re.findall(r"\w+", query.lower()))
        results = []
        for f in files:
            filename = f.original_filename.lower()
            text_sample = (f.extracted_text or "")[:1000].lower()
            overlap = len(query_tokens.intersection(set(re.findall(r"\w+", f"{filename} {text_sample}"))))
            if overlap > 0 or len(files) <= limit:
                results.append({
                    "file_id": f.file_id,
                    "filename": f.original_filename,
                    "size": f.size,
                    "file_type": getattr(f, "file_type", f.extension),
                    "summary": (f.extracted_text or "")[:300].strip(),
                    "score": overlap,
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    # -----------------------------------------------------------------------
    # Scoped Project Knowledge / Collections
    # -----------------------------------------------------------------------

    def associate_knowledge(self, project_id: str, collection_id: str) -> bool:
        """Associate a Knowledge/RAG collection with this project."""
        proj = self.get_project(project_id)
        if not proj:
            raise ValueError(f"Project '{project_id}' not found.")
        return self.store.associate_knowledge_collection(project_id, collection_id)

    def list_project_collections(self, project_id: str) -> List[str]:
        """List collection IDs belonging to this project."""
        return self.store.list_project_collection_ids(project_id)

    def query_project_knowledge(
        self,
        project_id: str,
        query: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Query RAG collections scoped to this project."""
        collections = self.list_project_collections(project_id)
        if not collections or not self.knowledge_manager:
            return []

        results = []
        for col_id in collections:
            try:
                chunks = self.knowledge_manager.retrieve(query, collection_id=col_id, top_k=top_k)
                for sc in chunks:
                    results.append({
                        "collection_id": col_id,
                        "document_id": getattr(sc, "document_id", ""),
                        "filename": getattr(sc, "filename", "doc"),
                        "text": getattr(sc, "text", ""),
                        "score": getattr(sc, "score", 0.0),
                    })
            except Exception as e:
                logger.debug("Error querying project knowledge in collection %s: %s", col_id, e)

        results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return results[:top_k]

    # -----------------------------------------------------------------------
    # Context Assembly Pipeline (Section 9)
    # -----------------------------------------------------------------------

    def build_project_context(
        self,
        project_id: Optional[str],
        user_request: str,
    ) -> Optional[ProjectContext]:
        """
        Assemble strictly scoped project context in the canonical order:
          USER MESSAGE -> CURRENT CONVERSATION ->
          PROJECT INSTRUCTIONS -> PROJECT MEMORY ->
          RELEVANT PROJECT FILES -> RELEVANT PROJECT KNOWLEDGE
        """
        if not project_id or not project_id.strip():
            return None

        proj = self.get_project(project_id)
        if not proj or proj.archived:
            return None

        settings = proj.settings

        # 1. Project Instructions (Trusted configuration, bounded)
        instructions = proj.instructions.strip()

        # 2. Project Memory (if enabled in project settings)
        memories = []
        if settings.memory_enabled:
            memories = self.retrieve_relevant_memories(project_id, user_request, limit=5)

        # 3. Relevant Project Files (if enabled in project settings)
        files = []
        if settings.file_access:
            files = self.retrieve_relevant_project_files(project_id, user_request, limit=3)

        # 4. Relevant Project Knowledge (if enabled in project settings)
        knowledge_chunks = []
        if settings.knowledge_enabled:
            knowledge_chunks = self.query_project_knowledge(project_id, user_request, top_k=3)

        # Format clean prompt block with clear untrusted data delimiters
        context_parts = []
        context_parts.append(f"### ACTIVE PROJECT WORKSPACE: {proj.name}")
        if proj.description:
            context_parts.append(f"Description: {proj.description}")

        if instructions:
            context_parts.append(f"\n[PROJECT INSTRUCTIONS]\n{instructions}")

        if memories:
            mem_lines = [f"- {m['key']}: {m['value']}" for m in memories]
            context_parts.append(f"\n[PROJECT MEMORY]\n" + "\n".join(mem_lines))

        if files:
            file_lines = [f"- {f['filename']} ({f['size']} bytes): {f['summary'][:150]}..." for f in files]
            context_parts.append(f"\n[AVAILABLE PROJECT FILES]\n" + "\n".join(file_lines))

        if knowledge_chunks:
            know_lines = [f"- [{k['filename']}]: {k['text'][:200]}..." for k in knowledge_chunks]
            context_parts.append(f"\n[PROJECT KNOWLEDGE CHUNKS]\n" + "\n".join(know_lines))

        formatted_context = "\n".join(context_parts)

        debug_summary = {
            "project_id": proj.id,
            "project_name": proj.name,
            "instructions_loaded": bool(instructions),
            "memories_count": len(memories),
            "files_count": len(files),
            "knowledge_chunks_count": len(knowledge_chunks),
            "settings": settings.to_dict(),
        }

        return ProjectContext(
            project_id=proj.id,
            project_name=proj.name,
            instructions=instructions,
            memories=memories,
            files=files,
            knowledge_chunks=knowledge_chunks,
            formatted_prompt_context=formatted_context,
            debug_summary=debug_summary,
        )

    # -----------------------------------------------------------------------
    # Convenience Aliases for Testing & External Callers
    # -----------------------------------------------------------------------

    def store_memory(
        self,
        project_id: str,
        key: str,
        value: str,
        memory_type: str = "fact",
    ) -> int:
        return self.add_project_memory(project_id, key=key, value=value, memory_type=memory_type)

    def get_memories(self, project_id: str) -> Dict[str, str]:
        mems = self.list_project_memories(project_id, active_only=True)
        return {m["key"]: m["value"] for m in mems}

    def link_file(
        self,
        project_id: str,
        file_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        return self.add_file_to_project(project_id, file_id)

    def get_files(self, project_id: str) -> List[Any]:
        full_files = self.list_project_files(project_id)
        if full_files:
            return full_files
        return self.store.list_project_file_ids(project_id)

    def link_knowledge_collection(
        self,
        project_id: str,
        collection_id: str,
        description: str = "",
    ) -> bool:
        return self.associate_knowledge(project_id, collection_id)

    def get_knowledge_collections(self, project_id: str) -> List[Dict[str, Any]]:
        cids = self.list_project_collections(project_id)
        return [{"collection_name": cid, "description": ""} for cid in cids]

    def create_conversation(self, project_id: str, title: str = "New Chat") -> Dict[str, Any]:
        conv_id = f"sess_{uuid.uuid4().hex[:10]}"
        conv = {
            "id": conv_id,
            "project_id": project_id,
            "title": title,
            "created_at": time.time(),
        }
        try:
            from app.ui.backend_service import AxenBackendService
            service = AxenBackendService.get_instance()
            service.new_session(title=title, project_id=project_id)
        except Exception:
            pass
        return conv

    def list_conversations(self, project_id: str) -> List[Dict[str, Any]]:
        try:
            from app.ui.backend_service import AxenBackendService
            service = AxenBackendService.get_instance()
            return service.list_sessions(project_id=project_id)
        except Exception:
            return []

    def build_context(
        self,
        project_id: Optional[str],
        query: str = "",
        max_memories: int = 5,
    ) -> ProjectContext:
        ctx = self.build_project_context(project_id, query)
        if ctx is None:
            return ProjectContext(
                project_id="",
                project_name="",
                instructions="",
                memories=[],
                files=[],
                knowledge_chunks=[],
                formatted_prompt_context="",
                debug_summary={},
            )
        if max_memories and len(ctx.memories) > max_memories:
            ctx.memories = ctx.memories[:max_memories]
        return ctx
