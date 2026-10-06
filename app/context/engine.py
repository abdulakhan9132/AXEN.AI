from typing import List, Dict, Any, Optional
from app.context.models import ContextRequest, ContextComponent, ModelCapability, AssembledContext, ContextPriority

class ContextEngine:
    def __init__(self):
        pass

    def assemble(self, request: ContextRequest, capability: ModelCapability) -> AssembledContext:
        """
        Assemble the final context based on priority and model capabilities.
        Implements PHASE 3D (Priority) and PHASE 3C (Budgeting).
        """
        # Step 1: Collect all components
        all_components: List[ContextComponent] = []
        
        all_components.extend(request.system_instructions)
        if request.current_message:
            all_components.append(request.current_message)
        all_components.extend(request.conversation_history)
        all_components.extend(request.long_term_memories)
        all_components.extend(request.project_context)
        
        # Step 2: Apply Trust Boundaries (PHASE 3H)
        # External data must be wrapped safely
        for comp in request.retrieved_knowledge:
            comp.content = self._wrap_untrusted("KNOWLEDGE", comp.content)
            all_components.append(comp)
            
        for comp in request.file_context:
            comp.content = self._wrap_untrusted("FILE", comp.content)
            all_components.append(comp)
            
        all_components.extend(request.tool_observations)
        all_components.extend(request.runtime_metadata)
        all_components.extend(request.constraints)

        # Step 3: Sort by priority (1 is highest priority)
        # Sort by: Priority (asc), Required (desc), Relevance (desc)
        all_components.sort(key=lambda c: (c.priority, not c.required, -c.relevance_score))

        # Step 4: Budget Allocation (PHASE 3C)
        budget = capability.context_length - capability.max_output_tokens
        # Reserve some tokens for safety padding
        budget = max(10, int(budget * 0.95))
        
        selected_components: List[ContextComponent] = []
        tokens_used = 0
        summary_injected = False
        
        for comp in all_components:
            est_tokens = comp.estimate_tokens()
            if tokens_used + est_tokens <= budget:
                selected_components.append(comp)
                tokens_used += est_tokens
            elif comp.required:
                # If a required component doesn't fit, we have to truncate it or throw an error.
                # In this design, we will just include what fits.
                # To be completely strict, we could raise an error.
                pass
            else:
                # If we hit conversation history and it doesn't fit, trigger Summarization (PHASE 3F)
                if comp.priority == ContextPriority.OLDER_CONVERSATION:
                    summary_injected = True
                
        # Step 5: Format into AssembledContext
        assembled = AssembledContext()
        assembled.tokens_used = tokens_used
        assembled.summary_injected = summary_injected
        
        # Build System Instruction
        system_parts = []
        for c in selected_components:
            if c.priority == ContextPriority.SYSTEM:
                system_parts.append(c.content)
            elif c.priority in (ContextPriority.PROJECT_CONTEXT, ContextPriority.PROJECT_MEMORY, ContextPriority.GLOBAL_MEMORY):
                system_parts.append(f"[{c.source.upper()}]\n{c.content}")
            elif c.priority in (ContextPriority.KNOWLEDGE, ContextPriority.FILE, ContextPriority.TOOL_OBSERVATIONS):
                system_parts.append(f"[{c.source.upper()}]\n{c.content}")

        assembled.system_instruction = "\n\n".join(system_parts)
        
        # Build Messages
        messages = [{"role": "system", "content": assembled.system_instruction}]
        
        # Sort conversation chronologically (assuming they were passed in chronological order)
        # We need to extract them from selected_components.
        # Since we sorted by priority, we lost chronological order. 
        # We should reconstruct the conversation.
        history_comps = [c for c in request.conversation_history if c in selected_components]
        for c in history_comps:
            role = c.metadata.get("role", "user")
            messages.append({"role": role, "content": c.content})
            
        if request.current_message and request.current_message in selected_components:
            role = request.current_message.metadata.get("role", "user")
            messages.append({"role": role, "content": request.current_message.content})
            
        assembled.messages_for_model = messages
        return assembled

    def _wrap_untrusted(self, label: str, content: str) -> str:
        """PHASE 3H - SECURITY: Wrap untrusted content to prevent injection."""
        # Sanitize known injection patterns
        sanitized = content.replace("ignore previous instructions", "[REDACTED]")
        sanitized = sanitized.replace("system prompt", "[REDACTED]")
        
        header = f"--- BEGIN UNTRUSTED DATA: {label} ---"
        footer = f"--- END UNTRUSTED DATA: {label} ---"
        return f"{header}\n{sanitized}\n{footer}"
