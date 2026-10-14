"""
axen/agent/classifier.py
========================
General Intelligence Router and Task Classifier for AXEN.

Determines:
- What is the user actually asking?
- What information is already available?
- Does AXEN need memory?
- Does AXEN need a tool?
- Does AXEN need multiple tools?
- Does AXEN need reasoning before answering?
- Does AXEN need clarification?
- Can the local model answer directly?
- Does the answer need verification?
- What is the complexity level (LOW, MEDIUM, HIGH)?

Produces structured IntentInfo objects:
{
    "intent": "math",
    "domain": "calculus",
    "complexity": "medium",
    "requires_tool": true,
    "requires_multiple_steps": false,
    "requires_memory": false,
    "requires_clarification": false,
    "confidence": 0.94
}
"""

from __future__ import annotations

import re
from typing import Any, Optional

from .complexity import Complexity, ComplexityEstimator
from .context_manager import ContextManager
from .intent import IntentInfo


class TaskClassifier:
    """General Intelligence Router and Intent Classifier for AXEN."""

    _GREETINGS = re.compile(
        r"^(?:hello|hi|hey|greetings|good\s+(?:morning|afternoon|evening|day)|howdy|sup)(?:[\s,]+(?:there|aexn|axen|friend|everyone|everybody))?[\s\.\!\?]*$|"
        r"^(?:how\s+are\s+you(?:\s+doing|\s+today)?|what'?s\s+up(?:\s+axen|\s+there)?|how's\s+it\s+going)[\s\.\!\?]*$",
        re.IGNORECASE,
    )
    _IDENTITY = re.compile(
        r"\b(?:who\s+are\s+you|what\s+is\s+your\s+name|what\s+company|your\s+hobbies|do\s+you\s+have\s+hobbies)\b",
        re.IGNORECASE,
    )
    _CALCULATION = re.compile(
        r"\b(calculate|compute|what\s+is|eval)\b.*?[\d\+\-\*\/\(\)\^\.]+|"
        r"\b[\d\.]+\s*(?:multiplied\s+by|times|divided\s+by|plus|minus|\*|\+|\/|\-|\*\*)\s*[\d\.]+|"
        r"^\s*[\d\+\-\*\/\(\)\^\.% \t]+\s*$",
        re.IGNORECASE | re.DOTALL,
    )
    _SYMBOLIC_MATH = re.compile(
        r"\b(?:derivative\s+of|differentiate|integral\s+of|integrate|limit\s+of|gcd\s+of|greatest\s+common\s+divisor|is\s+\d+\s+prime|prime\?|euler\s+phi|modular\s+inverse)\b|"
        r"^\s*solve\s+[a-z0-9\^\+\-\*\/\s=]+$|"
        r"\b(?:simplify|expand|factor)\s*[\(\[a-zA-Z0-9\^\+\-\*\/\s\.\=]+|"
        r"^\s*[fgh]\([a-zA-Z]\)\s*=.*|"
        r"\b(?:[a-zA-Z]\^[0-9]|[0-9][a-zA-Z]|[a-zA-Z]\*\*[0-9])\b|"
        r"\([0-9a-zA-Z\^\+\-\*\/\s]+\)\s*\([0-9a-zA-Z\^\+\-\*\/\s]+\)",
        re.IGNORECASE,
    )
    _CODE_EXECUTION = re.compile(
        r"\b(?:run|execute)\s*(?::\s*)?(?:python|this python|code|this code)?\b.*?\bprint\b|"
        r"\b(?:run|execute)\s+(?:this\s+)?python\b|"
        r"\b(?:run|execute)\s+(?:this\s+)?code\b|"
        r"```python|"
        r"^\s*print\s*\(.*?\)\s*$",
        re.IGNORECASE | re.DOTALL,
    )
    _FILE_OPERATION = re.compile(
        r"\b(?:read|show|view|open|cat)\s+(?:this\s+)?file[:\s]+[^\s,;]+|"
        r"\b(?:read|show|view|open)\s+[a-zA-Z0-9_\-\.\/\\]+\.(?:md|txt|py|json|csv)\b|"
        r"\b(?:read|view|inspect|summarize|explain)\s+(?:this\s+|my\s+|the\s+)?(?:file|document|pdf|code|csv)\b|"
        r"\b(?:find|search|locate)\s+.*?\b(?:in\s+(?:this\s+|the\s+)?(?:project|workspace|repo|codebase)|in\s+files)\b|"
        r"\b(?:find|search|locate)\s+(?:for\s+)?(?:all\s+)?(?:references\s+to\s+|references\s+in\s+|files\s+matching\s+|files\s+named\s+|csv|data|files?\b)[^\s,;]*|"
        r"\b(?:what\s+does\s+this\s+(?:file|document|pdf)\s+say|"
        r"tell\s+me\s+what\s+(?:it|this\s+file)\s+says|"
        r"compare\s+these\s+(?:files|documents)|"
        r"total\s+(?:of|in)\s+(?:the\s+)?(?:sales|revenue|column)|"
        r"explain\s+what\s+this\s+code\s+does)\b",
        re.IGNORECASE,
    )

    _KNOWLEDGE_DOC_QUERY = re.compile(
        r"\b(?:according\s+to\s+(?:the|my)\s+(?:document|documentation|docs|notes|pdf|file)|"
        r"what\s+does\s+(?:the|my)\s+(?:[a-zA-Z0-9_\-\.]+\s+)?(?:document|documentation|docs|guide|architecture|readme|file)\s+say|"
        r"based\s+on\s+(?:the|my)\s+(?:document|documentation|project\s+files|codebase|docs)|"
        r"in\s+(?:the|my)\s+(?:document|documentation|docs|architecture|notes|knowledge\s+base)|"
        r"search\s+(?:the|my)\s+(?:knowledge|documents|notes|files)\s+for|"
        r"summarize\s+(?:this|the|my)\s+(?:document|pdf|file)|"
        r"from\s+(?:the|my)\s+(?:indexed\s+)?(?:documents|knowledge\s+base))\b",
        re.IGNORECASE,
    )
    _CODING_WORKSPACE_TASK = re.compile(
        r"\b(?:fix\s+(?:the\s+)?bug|fix\s+(?:the\s+)?error|debug\s+(?:the\s+)?failure|"
        r"review\s+(?:this|the|my)?\s*(?:project|codebase|code)|"
        r"inspect\s+(?:this|the|my)?\s*project|"
        r"show\s+(?:this|the|my)?\s*project\s+map|"
        r"run\s+(?:the\s+)?tests?\b|"
        r"refactor\s+(?:this|the|my)?\s*(?:module|class|function|file)|"
        r"search\s+code\s+for|"
        r"where\s+is\s+.*implemented\s+in\s+(?:my|the)?\s*project|"
        r"(?:add|implement)\s+.*(?:in|to)\s+(?:this|the|my)\s*(?:project|codebase|app))\b",
        re.IGNORECASE,
    )
    _WEB_RESEARCH_TASK = re.compile(
        r"\b(?:search\s+the\s+web|look\s+up\s+online|browse\s+the\s+web|find\s+online|"
        r"what\s+is\s+the\s+latest\s+.*(?:version|release|update|price|announcement|news)|"
        r"latest\s+.*(?:release|version|announcement|specifications|specs)|"
        r"current\s+.*(?:version|release|price|status|specs|specifications)|"
        r"(?:find|search\s+for)\s+(?:the\s+)?official\s+(?:documentation|docs)\s+for|"
        r"official\s+(?:documentation|docs)\s+for|"
        r"compare\s+.*(?:from\s+(?:three|multiple|different)?\s*sources|specifications|current)|"
        r"what\s+happened\s+in\s+the\s+latest\b)\b",
        re.IGNORECASE,
    )
    _MEMORY_OP = re.compile(
        r"\b(remember\s+that|please\s+note|save\s+that|note\s+that|keep\s+in\s+mind|recall|do\s+you\s+remember|what\s+do\s+you\s+remember)\b",
        re.IGNORECASE,
    )
    _PROGRAMMING_CONCEPT = re.compile(
        r"\b(?:explain|how\s+does|what\s+is)\b.*\b(?:recursion|algorithm|function|class|decorator|generator|pointer|async|promise|variable)\b|"
        r"\b(?:explain\s+python|python\s+recursion)\b",
        re.IGNORECASE,
    )
    _PROGRAMMING_GEN = re.compile(
        r"\b(?:write|create|implement|build|generate)\s+(?:a\s+)?(?:python|javascript|rust|c\+\+|html|css)?\s*(?:function|script|program|code|portfolio|website|calculator|app|dashboard)\b|"
        r"\b(?:write\s+a\s+python\s+function|write\s+code)\b",
        re.IGNORECASE,
    )
    _DEBUGGING = re.compile(
        r"\b(?:why\s+does\s+this.*error|debug|fix\s+the\s+bug|why\s+is.*failing|find\s+the\s+error|syntaxerror|typeerror|nameerror|traceback)\b",
        re.IGNORECASE,
    )
    _WRITING = re.compile(
        r"\b(?:write|draft|compose|rewrite)\s+(?:a\s+)?(?:professional|polite|formal)?\s*(?:email|letter|essay|leave\s+email|cover\s+letter|memo|resignation)\b|"
        r"\b(?:rewrite\s+this|improve\s+the\s+tone)\b",
        re.IGNORECASE,
    )
    _TRANSLATION = re.compile(
        r"\b(?:translate\s+(?:this\s+)?(?:sentence\s+|text\s+)?into|translate\s+to|how\s+do\s+you\s+say\s+.*in)\s+([a-zA-Z]+)\b",
        re.IGNORECASE,
    )
    _PLANNING = re.compile(
        r"\b(?:create\s+a\s+\d+[\s\-]day\s+plan|plan\s+to|roadmap|itinerary|how\s+to\s+survive|budget\s+plan|survive\s+for\s+\d+\s+days)\b|"
        r"\b\d+\s+rupees.*survive\b",
        re.IGNORECASE,
    )
    _REASONING = re.compile(
        r"\b(?:why\s+does|why\s+is\s+my.*slow|explain\s+why|compare\s+and\s+contrast|pros\s+and\s+cons|what\s+causes)\b",
        re.IGNORECASE,
    )
    _MIXED_TASK = re.compile(
        r"\b(?:and\s+then|after\s+that|first\s+read.*then\s+calculate|write\s+.*and\s+(?:run|test)\s+it|"
        r"read\s+.*and\s+(?:calculate|explain|compute)|find\s+.*(?:calculate|average|sum|explain)|check\s+my.*and\s+fix|"
        r"(?:calculate|compute|differentiate|solve|convert|run|execute)\b.*\band\s+(?:explain|describe|show\s+how|walk\s+through))\b",
        re.IGNORECASE,
    )

    _COMPUTER_SYSTEMS = re.compile(
        # Conversion keywords
        r"\bconvert\s+[0-9a-fA-F]+\s+(?:from\s+\w+\s+)?to\s+(?:binary|decimal|hex(?:adecimal)?|octal)\b"
        r"|\b[0-9a-fA-F]+\s+(?:in\s+|to\s+)(?:binary|decimal|hex(?:adecimal)?|octal)\b"
        # Hex and binary prefix literals
        r"|\b0[xX][0-9a-fA-F]+\b|\b0[bB][01]+\b"
        # Bitwise operations
        r"|\b(?:0[bB])?[01]+\s+(?:and|or|xor|<<|>>)\s+(?:0[bB])?[01]+\b"
        r"|\bbitwise\b|\bleft\s+shift\b|\bright\s+shift\b"
        r"|\bset\s+bit\b|\bclear\s+bit\b|\btoggle\s+bit\b|\bcheck\s+bit\b"
        r"|\bextract\s+bits?\b|\bmask\s+bits?\b"
        # Number systems
        r"|\bbinary\s+(?:number|string|arithmetic|addition|subtraction|multiplication|division)\b"
        r"|\bhexadecimal\b|\bhex\s+(?:value|number|arithmetic|code)\b"
        # ASCII
        r"|\bascii\s+(?:of|value|code|character|decimal|binary|encoding)\b"
        r"|\bunicode\s+(?:code\s+point|encoding|character)\b"
        # Two's complement
        r"|\btwo[\u2019's]*s?\s+complement\b"
        # Integer representation
        r"|\b(?:signed|unsigned)\s+(?:\d+[-\s]?bit\s+)?integer\b"
        r"|\brepresent\s+-?\d+\s+(?:using\s+)?\d+[-\s]?bit\b"
        r"|\bshow\s+\d+\s+in\s+\d+\s+bits?\b"
        # Memory concepts
        r"|\bendian(?:ness)?\b|\bbig[-\s]endian\b|\blittle[-\s]endian\b"
        r"|\bmemory\s+(?:address|representation|layout|dump|model)\b"
        r"|\bstack\s+(?:vs|and|or|pointer|frame|overflow)\b|\bheap\s+(?:vs|memory|allocation)\b"
        r"|\bpointer\s+(?:in|to|arithmetic)\b"
        r"|\b(?:bits?\s+and\s+bytes?|bit\s+manipulation|nibble)\b"
        # Assembly & CPU
        r"|\bassembl(?:y|er|ies)\s+(?:language|instruction|code)\b"
        r"|\bopcode(?:s)?\b|\binstruction\s+(?:set|encoding|cycle)\b"
        r"|\b(?:fetch.decode.execute|fde\s+cycle)\b"
        r"|\bcpu\s+(?:architecture|execution|register|simulation|emulator|model)\b"
        r"|\b(?:x86.64|arm64|aarch64|risc.v)\s+(?:instruction|register|assembly|arch)\b"
        r"|\bhow\s+(?:does\s+a?\s*)?cpu\s+(?:work|execute|run)\b"
        r"|\bcompilation\s+pipeline\b"
        r"|\bmachine\s+(?:code|instruction|byte)\b"
        # What-is questions
        r"|\b(?:what|explain|how)\s+(?:is|does|do)\s+(?:binary|hexadecimal|octal|ascii)\b"
        # Convert text/string to binary (ASCII encoding)
        r"|\bconvert\s+[a-zA-Z][a-zA-Z\s]+\s+to\s+binary\b"
        # ASCII of X
        r"|\bascii\s+(?:of|value\s+of|code\s+of)\s+\S+\b",
        re.IGNORECASE | re.DOTALL,
    )

    def __init__(self, context_manager: Optional[ContextManager] = None) -> None:
        self.context_manager = context_manager or ContextManager()

    def analyze_intent(
        self,
        text: str,
        history: Optional[list[Any]] = None,
        long_memory: Optional[Any] = None,
    ) -> IntentInfo:
        """
        Deeply inspect request intent, complexity, domain, tool requirements, and context.

        Returns structured IntentInfo.
        """
        cleaned = text.strip()
        if not cleaned:
            return IntentInfo(
                intent="unknown",
                domain="general",
                requires_reasoning=False,
                requires_tool=False,
                complexity=ComplexityEstimator.LOW,
                confidence=0.5,
            )

        lower = cleaned.lower()
        hist = history or []

        # 0. Contextual analysis (follow-ups & ambiguity - Step 10 & 11)
        ctx = self.context_manager.analyze_context(cleaned, hist)

        if ctx.get("is_ambiguous"):
            return IntentInfo(
                intent="ambiguous",
                domain="general",
                requires_reasoning=True,
                requires_tool=False,
                requires_memory=False,
                requires_verification=False,
                complexity=ComplexityEstimator.LOW,
                needs_clarification=True,
                requires_clarification=True,
                clarification_prompt=ctx["clarification_prompt"],
                confidence=0.95,
            )

        if ctx.get("is_follow_up"):
            action = ctx.get("follow_up_action")
            resolved = ctx.get("resolved_subject")
            resolved_query = ctx.get("resolved_query")

            # Mathematical follow-up (e.g. "Now simplify it")
            if action == "simplify" and resolved:
                return IntentInfo(
                    intent="math",
                    domain="algebra",
                    requires_reasoning=True,
                    requires_tool=True,
                    requires_memory=False,
                    requires_verification=True,
                    complexity=ComplexityEstimator.MEDIUM,
                    tools_needed=["math_solver"],
                    primary_tool="math_solver",
                    confidence=0.95,
                    context_reference=resolved,
                    extracted_data={"action": action, "subject": resolved, "resolved_query": resolved_query},
                )

            # Computer systems follow-up (e.g. "Now convert that to hexadecimal")
            if action == "convert" and resolved:
                return IntentInfo(
                    intent="computer_systems",
                    domain="number_systems",
                    requires_reasoning=False,
                    requires_tool=True,
                    requires_memory=False,
                    requires_verification=True,
                    complexity=ComplexityEstimator.LOW,
                    tools_needed=["computer"],
                    primary_tool="computer",
                    confidence=0.95,
                    context_reference=resolved,
                    extracted_data={"action": action, "subject": resolved, "resolved_query": resolved_query},
                )

            # File operation follow-up
            if action == "read_file" and resolved:
                return IntentInfo(
                    intent="file_operation",
                    domain="file_system",
                    requires_reasoning=False,
                    requires_tool=True,
                    requires_memory=False,
                    requires_verification=True,
                    complexity=ComplexityEstimator.LOW,
                    tools_needed=["file"],
                    primary_tool="file",
                    confidence=0.95,
                    context_reference=resolved,
                    extracted_data={"action": action, "subject": resolved, "resolved_query": resolved_query},
                )

            return IntentInfo(
                intent="follow_up",
                domain="general",
                requires_reasoning=True,
                requires_tool=False,
                requires_memory=False,
                requires_verification=False,
                complexity=ComplexityEstimator.MEDIUM,
                context_reference=resolved,
                confidence=0.90,
                extracted_data={"action": action, "subject": resolved, "resolved_query": resolved_query},
            )

        # 0b. Greetings and identity queries -> direct conversation (NO tools, NO memory injection)
        is_greeting_match = bool(self._GREETINGS.match(cleaned))
        if is_greeting_match or self._IDENTITY.search(cleaned):
            return IntentInfo(
                intent="conversation",
                domain="social" if is_greeting_match else "identity",
                requires_reasoning=False,
                requires_tool=False,
                requires_memory=False,
                requires_verification=False,
                complexity=ComplexityEstimator.LOW,
                confidence=0.98,
                extracted_data={"is_greeting": is_greeting_match},
            )

        # 0c. Computer Systems domain (before all other classifiers)
        if self._COMPUTER_SYSTEMS.search(cleaned):
            comp_domain = "binary" if "binary" in lower else "hexadecimal" if "hex" in lower else "computer_systems"
            return IntentInfo(
                intent="computer_systems",
                domain=comp_domain,
                requires_reasoning=False,
                requires_tool=True,
                requires_memory=False,
                requires_verification=True,
                complexity=ComplexityEstimator.LOW,
                tools_needed=["computer"],
                primary_tool="computer",
                confidence=0.96,
            )

        # 1. Multi-step / mixed task detection (Step 6 & Step 12)
        if self._MIXED_TASK.search(cleaned):
            tools: list[str] = []
            if re.search(r"\b(?:file|read|open)\b", lower):
                tools.append("file")
            if re.search(r"\b(?:calculate|compute|total|sum|average|numbers)\b", lower):
                # Check if it has algebraic symbols
                if re.search(r"\b[xyz]\b|[xyz]\^|f\(", cleaned):
                    tools.append("math_solver")
                else:
                    tools.append("calculator")
            elif re.search(r"\b(?:derivative|differentiate|integral|solve|math|simplify)\b", lower):
                tools.append("math_solver")
            elif re.search(r"\b(?:convert|binary|hex|bitwise)\b", lower):
                tools.append("computer")
            elif re.search(r"\b(?:python|code|run)\b", lower):
                tools.append("python")
            if not tools:
                tools = ["multi_step"]
            return IntentInfo(
                intent="multi_step",
                domain="multi_domain",
                requires_reasoning=True,
                requires_tool=True,
                requires_multiple_steps=True,
                requires_memory=False,
                requires_verification=True,
                complexity=ComplexityEstimator.HIGH,
                tools_needed=tools,
                primary_tool=tools[0] if tools else "multi_step",
                confidence=0.92,
            )

        # 2. Memory operation
        if self._MEMORY_OP.search(cleaned):
            return IntentInfo(
                intent="memory",
                domain="memory",
                requires_reasoning=False,
                requires_tool=True,
                requires_memory=True,
                requires_verification=False,
                complexity=ComplexityEstimator.LOW,
                tools_needed=["memory"],
                primary_tool="memory",
                confidence=0.95,
            )

        # Check if query benefits from existing relevant long-term memory
        has_memory_relevance = False
        if long_memory and hasattr(long_memory, "retrieve_relevant"):
            relevant_facts = long_memory.retrieve_relevant(cleaned, top_k=1)
            if relevant_facts:
                has_memory_relevance = True

        # 3. File operation
        if self._FILE_OPERATION.search(cleaned):
            tool_name = "file_search" if re.search(r"\b(?:find|search|locate|where)\b", lower) else "file"
            return IntentInfo(
                intent="file_operation",
                domain="file_system",
                requires_reasoning=False,
                requires_tool=True,
                requires_memory=False,
                requires_verification=True,
                complexity=ComplexityEstimator.LOW,
                tools_needed=[tool_name],
                primary_tool=tool_name,
                confidence=0.95,
            )

        # 3b. Local Document Knowledge Retrieval (RAG)
        if self._KNOWLEDGE_DOC_QUERY.search(cleaned):
            return IntentInfo(
                intent="knowledge_retrieval",
                domain="knowledge_base",
                requires_reasoning=True,
                requires_tool=True,
                requires_memory=False,
                requires_verification=True,
                complexity=ComplexityEstimator.MEDIUM,
                tools_needed=["knowledge"],
                primary_tool="knowledge",
                confidence=0.96,
            )

        # 3c. Software Engineering / Workspace Coding Task
        if self._CODING_WORKSPACE_TASK.search(cleaned):
            return IntentInfo(
                intent="coding",
                domain="software_engineering",
                requires_reasoning=True,
                requires_tool=True,
                requires_memory=False,
                requires_verification=True,
                complexity=ComplexityEstimator.HIGH,
                tools_needed=["coding"],
                primary_tool="coding",
                confidence=0.96,
            )

        # 3d. Web Research / Live Web Information
        if self._WEB_RESEARCH_TASK.search(cleaned):
            return IntentInfo(
                intent="web_search",
                domain="web",
                requires_reasoning=True,
                requires_tool=True,
                requires_memory=False,
                requires_verification=True,
                complexity=ComplexityEstimator.MEDIUM,
                tools_needed=["web.research"],
                primary_tool="web.research",
                confidence=0.96,
            )

        # 4. Code execution
        if self._CODE_EXECUTION.search(cleaned):
            return IntentInfo(
                intent="code_execution",
                domain="python",
                requires_reasoning=False,
                requires_tool=True,
                requires_memory=False,
                requires_verification=True,
                complexity=ComplexityEstimator.LOW,
                tools_needed=["python"],
                primary_tool="python",
                confidence=0.96,
            )

        # 5. Symbolic mathematics (MathSolver) - Step 4 Rule:
        # A symbolic expression such as f(x)=(2x^3-5x+4)(x^2+3x-1) must NOT be sent to calculator!
        has_symbolic_indicators = bool(
            self._SYMBOLIC_MATH.search(cleaned)
            or re.search(r"^\s*f\([a-zA-Z]\)\s*=", cleaned)
            or re.search(r"\b(?:simplify|expand|factor)\b", lower)
            or (re.search(r"\b[xyz]\b", cleaned) and re.search(r"[\^\*\+\-\/]", cleaned))
            or re.search(r"\([0-9a-zA-Z\^\+\-\*\/\s]+\)\s*\([0-9a-zA-Z\^\+\-\*\/\s]+\)", cleaned)
        )
        if has_symbolic_indicators:
            math_domain = "calculus" if any(w in lower for w in ["derivative", "differentiate", "integral", "limit", "dy/dx"]) else "algebra"
            return IntentInfo(
                intent="math",
                domain=math_domain,
                requires_reasoning=True,
                requires_tool=True,
                requires_memory=False,
                requires_verification=True,
                complexity=ComplexityEstimator.MEDIUM,
                tools_needed=["math_solver"],
                primary_tool="math_solver",
                confidence=0.94,
            )

        # 6. Arithmetic calculation (pure numeric arithmetic only)
        if self._CALCULATION.search(cleaned):
            # Verify it does NOT contain algebraic variable characters
            has_variable = bool(re.search(r"\b[a-zA-Z]\b", cleaned) and not re.search(r"\b(?:plus|minus|times|divided|multiplied|by|what|is|calculate|compute|eval)\b", lower))
            if not has_variable and re.search(r"\d", cleaned) and re.search(r"[+\-*/^%]|multiplied|times|divided|plus|minus", lower):
                return IntentInfo(
                    intent="calculation",
                    domain="arithmetic",
                    requires_reasoning=False,
                    requires_tool=True,
                    requires_memory=False,
                    requires_verification=True,
                    complexity=ComplexityEstimator.LOW,
                    tools_needed=["calculator"],
                    primary_tool="calculator",
                    confidence=0.98,
                )

        # 7. Greetings / identity -> conversation (NO tool)
        if self._GREETINGS.match(cleaned) or self._IDENTITY.search(cleaned):
            return IntentInfo(
                intent="conversation",
                domain="social",
                requires_reasoning=False,
                requires_tool=False,
                requires_memory=False,
                requires_verification=False,
                complexity=ComplexityEstimator.LOW,
                confidence=0.98,
            )

        # 8. Planning & Problem-Solving
        if self._PLANNING.search(cleaned):
            return IntentInfo(
                intent="planning",
                domain="strategy",
                requires_reasoning=True,
                requires_tool=False,
                requires_memory=has_memory_relevance,
                requires_verification=False,
                complexity=ComplexityEstimator.MEDIUM,
                confidence=0.90,
            )

        # 9. Debugging
        if self._DEBUGGING.search(cleaned):
            return IntentInfo(
                intent="debugging",
                domain="troubleshooting",
                requires_reasoning=True,
                requires_tool=False,
                requires_memory=False,
                requires_verification=True,
                complexity=ComplexityEstimator.HIGH,
                confidence=0.92,
            )

        # 10. Writing / Rewriting
        if self._WRITING.search(cleaned):
            return IntentInfo(
                intent="writing",
                domain="composition",
                requires_reasoning=True,
                requires_tool=False,
                requires_memory=has_memory_relevance,
                requires_verification=False,
                complexity=ComplexityEstimator.MEDIUM,
                confidence=0.90,
            )

        # 11. Translation
        if self._TRANSLATION.search(cleaned):
            target_lang_match = self._TRANSLATION.search(cleaned)
            target_lang = target_lang_match.group(1) if target_lang_match else "unknown"
            return IntentInfo(
                intent="translation",
                domain="linguistics",
                requires_reasoning=True,
                requires_tool=False,
                requires_memory=False,
                requires_verification=False,
                complexity=ComplexityEstimator.MEDIUM,
                confidence=0.95,
                extracted_data={"target_language": target_lang},
            )

        # 12. Programming code generation
        if self._PROGRAMMING_GEN.search(cleaned):
            return IntentInfo(
                intent="programming",
                domain="software_engineering",
                requires_reasoning=True,
                requires_tool=False,
                requires_memory=has_memory_relevance,
                requires_verification=True,
                complexity=ComplexityEstimator.MEDIUM,
                confidence=0.92,
            )

        # 13. Programming concept explanation
        if self._PROGRAMMING_CONCEPT.search(cleaned):
            return IntentInfo(
                intent="programming",
                domain="computer_science",
                requires_reasoning=True,
                requires_tool=False,
                requires_memory=False,
                requires_verification=False,
                complexity=ComplexityEstimator.MEDIUM,
                confidence=0.90,
            )

        # 14. Deep reasoning & analysis
        if self._REASONING.search(cleaned):
            return IntentInfo(
                intent="reasoning",
                domain="conceptual_analysis",
                requires_reasoning=True,
                requires_tool=False,
                requires_memory=False,
                requires_verification=False,
                complexity=ComplexityEstimator.MEDIUM,
                confidence=0.88,
            )

        # 15. Knowledge questions (e.g. "What is an API?", "What is Python?")
        if re.search(r"^(what|who|when|where|why|how)\s+(?:is|are|was|were)\b", lower):
            return IntentInfo(
                intent="knowledge",
                domain="knowledge_base",
                requires_reasoning=True,
                requires_tool=False,
                requires_memory=False,
                requires_verification=False,
                complexity=ComplexityEstimator.LOW,
                confidence=0.90,
            )

        # 16. Fallback for random noise / unclassified (no alphanumeric content)
        if not re.search(r"[a-zA-Z0-9]", cleaned):
            return IntentInfo(
                intent="unknown",
                domain="general",
                requires_reasoning=False,
                requires_tool=False,
                requires_memory=False,
                requires_verification=False,
                complexity=ComplexityEstimator.LOW,
                confidence=0.50,
            )

        return IntentInfo(
            intent="conversation",
            domain="general",
            requires_reasoning=False,
            requires_tool=False,
            requires_memory=has_memory_relevance,
            requires_verification=False,
            complexity=ComplexityEstimator.LOW,
            confidence=0.80,
        )

    def classify(self, text: str, model: Optional[Any] = None) -> tuple[str, bool, Optional[str]]:
        """
        Backward-compatible classification method.

        Returns:
            (task_type, tool_required, selected_tool)
        """
        intent_info = self.analyze_intent(text)

        # Map to legacy 10 types where appropriate for existing tests
        legacy_type = intent_info.intent
        if legacy_type == "knowledge":
            legacy_type = "knowledge_retrieval"
        elif legacy_type == "multi_step":
            legacy_type = "mixed"
        elif legacy_type == "computer_systems":
            legacy_type = "computer_systems"

        selected_tool = "multi_step" if legacy_type == "mixed" else intent_info.primary_tool
        return legacy_type, intent_info.requires_tool, selected_tool
