"""
axen/context/context_assembler.py
=================================
Hierarchical prompt and message assembler for AXEN Context Engine.

Enforces clear source boundaries:
- SYSTEM
- PROJECT INSTRUCTIONS
- CURRENT TASK
- CONVERSATION CONTEXT
- PROJECT MEMORY
- GLOBAL MEMORY
- FILES (UNTRUSTED DATA)
- KNOWLEDGE (UNTRUSTED DATA)
- WEB EVIDENCE (UNTRUSTED DATA)

Never concatenates everything into one unstructured string.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .context_models import AssembledContext, ContextDebugInfo
from .context_policy import ContextPolicy


class ContextAssembler:
    """Assembles clean structured context sections into a final AssembledContext."""

    def __init__(self, policy: Optional[ContextPolicy] = None) -> None:
        self.policy = policy or ContextPolicy()

    def assemble(
        self,
        user_request: str,
        system_instruction: str = "",
        project_name: Optional[str] = None,
        project_instructions: str = "",
        recent_messages: Optional[List[Dict[str, str]]] = None,
        conversation_summary: Optional[str] = None,
        project_memories: Optional[List[Dict[str, Any]]] = None,
        global_memories: Optional[List[Dict[str, Any]]] = None,
        file_contexts: Optional[List[Dict[str, Any]]] = None,
        knowledge_contexts: Optional[List[Dict[str, Any]]] = None,
        web_contexts: Optional[List[Dict[str, Any]]] = None,
        debug_info: Optional[ContextDebugInfo] = None,
    ) -> AssembledContext:
        """Construct the isolated context blocks and format messages for model execution."""
        sections = []

        # 1. System Instruction (Trusted)
        base_system = system_instruction.strip() if system_instruction else (
            "You are AXEN, a local-first, privacy-focused AI assistant. "
            "Answer the user's current request directly and truthfully. "
            "Do not fabricate facts or invent capabilities."
        )

        # 2. Project Instructions (Trusted Configuration)
        if project_name or project_instructions:
            proj_header = f"ACTIVE PROJECT: {project_name or 'Current Workspace'}"
            if project_instructions:
                proj_header += f"\n[PROJECT INSTRUCTIONS]\n{project_instructions.strip()}"
            base_system += f"\n\n{proj_header}"

        # 3. Project Memory (Isolated to this project)
        if project_memories:
            mem_lines = [f"- {m.get('key')}: {m.get('value')}" for m in project_memories]
            base_system += f"\n\n[PROJECT MEMORY]\n" + "\n".join(mem_lines)

        # 4. Global Memory (User identity, stable preferences)
        if global_memories:
            mem_lines = [f"- {m.get('key')}: {m.get('value')}" for m in global_memories]
            base_system += f"\n\n[GLOBAL USER MEMORY]\n" + "\n".join(mem_lines)

        # 5. Untrusted Data Sections (Files, Knowledge, Web)
        untrusted_sections = []
        if file_contexts:
            for fc in file_contexts:
                fname = fc.get("filename", "file")
                content = fc.get("content", fc.get("summary", ""))
                wrapped = self.policy.wrap_untrusted_file(fname, content)
                untrusted_sections.append(wrapped)

        if knowledge_contexts:
            for kc in knowledge_contexts:
                src = kc.get("filename", kc.get("source", "knowledge"))
                text = kc.get("text", "")
                wrapped = self.policy.wrap_untrusted_knowledge(src, text)
                untrusted_sections.append(wrapped)

        if web_contexts:
            for wc in web_contexts:
                url = wc.get("url", "web_source")
                snippet = wc.get("snippet", wc.get("summary", ""))
                wrapped = self.policy.wrap_untrusted_web(url, snippet)
                untrusted_sections.append(wrapped)

        # Build message history for the LLM
        messages_for_model = [{"role": "system", "content": base_system}]

        # Inject conversation summary if older messages were trimmed
        if conversation_summary:
            messages_for_model.append({
                "role": "system",
                "content": f"[EARLIER CONVERSATION SUMMARY]\n{conversation_summary.strip()}",
            })

        # Inject recent conversation turns
        if recent_messages:
            for msg in recent_messages:
                role = msg.get("role", "user")
                if role not in ("system", "user", "assistant"):
                    role = "user"
                messages_for_model.append({"role": role, "content": msg.get("content", "")})

        # Append current user request with untrusted context attached if present
        current_user_content = user_request
        if untrusted_sections:
            current_user_content += "\n\n" + "\n\n".join(untrusted_sections)

        # Ensure user request is at the end
        if not recent_messages or recent_messages[-1].get("content") != current_user_content:
            messages_for_model.append({"role": "user", "content": current_user_content})

        formatted_full_prompt = "\n\n".join([f"[{m['role'].upper()}]\n{m['content']}" for m in messages_for_model])

        return AssembledContext(
            system_instruction=base_system,
            project_instructions=project_instructions,
            current_task=user_request,
            conversation_summary=conversation_summary or "",
            recent_messages=recent_messages or [],
            project_memories=project_memories or [],
            global_memories=global_memories or [],
            file_contexts=file_contexts or [],
            knowledge_contexts=knowledge_contexts or [],
            web_contexts=web_contexts or [],
            formatted_prompt=formatted_full_prompt,
            messages_for_model=messages_for_model,
            debug_info=debug_info,
        )
