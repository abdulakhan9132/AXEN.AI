"""
axen/coding/code_context.py
============================
Context management and relevant-file selection for the AXEN Coding Agent.
Ensures TinyLlama never receives entire repositories:
- Bounded character and token limits
- Selective line windows around relevant symbols or search matches
- Structured code context representation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from .code_search import CodeSearch
from .workspace import Workspace


@dataclass
class FileContext:
    relative_path: str
    language: str
    content_snippet: str
    total_lines: int
    start_line: int = 1
    end_line: int = 1
    symbols: List[str] = field(default_factory=list)
    is_truncated: bool = False


class CodeContextManager:
    """Selects, bounds, and structures relevant code context for LLM reasoning."""

    MAX_FILES = 4
    MAX_FILE_CHARS = 2500
    MAX_TOTAL_CHARS = 5000
    MAX_CONTEXT_TOKENS = 1200

    def __init__(self, workspace: Workspace, searcher: CodeSearch) -> None:
        self.workspace = workspace
        self.searcher = searcher

    def select_relevant_files(self, task_description: str, max_files: int = MAX_FILES) -> List[str]:
        """
        Rank and select the most relevant files for a coding task.
        Uses task keywords, symbol matches, and path matching.
        """
        # Extract potential identifier keywords (min 3 chars)
        words = re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", task_description.lower())
        stopwords = {
            "the", "and", "for", "with", "this", "that", "from", "fix", "add", "update",
            "modify", "change", "create", "make", "implement", "how", "what", "where", "why",
            "please", "code", "file", "function", "class", "project", "bug", "test",
        }
        keywords = [w for w in words if w not in stopwords and len(w) >= 3]

        all_files = self.workspace.list_files()
        scores: dict[str, float] = {}

        for f in all_files:
            rel = str(f.relative_to(self.workspace.root_path)).replace("\\", "/")
            rel_lower = rel.lower()
            fname_lower = f.name.lower()
            score = 0.0

            # 1. Path & Filename Match
            for kw in keywords:
                if kw in fname_lower:
                    score += 5.0
                elif kw in rel_lower:
                    score += 2.0

            # 2. Prefer source and test files
            if f.suffix in (".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".rs", ".go"):
                score += 1.0

            # 3. Direct mention in query
            if fname_lower in task_description.lower():
                score += 10.0

            scores[rel] = score

        # Check symbol search matches
        for kw in keywords:
            sym_matches = self.searcher.search_symbols(kw, max_results=5)
            for sm in sym_matches:
                scores[sm.file_path] = scores.get(sm.file_path, 0.0) + 4.0

            text_matches = self.searcher.search_text(kw, max_results=10)
            for tm in text_matches:
                scores[tm.file_path] = scores.get(tm.file_path, 0.0) + 1.0

        # Sort descending by score
        ranked = sorted(
            [f for f, s in scores.items() if s > 0],
            key=lambda x: scores[x],
            reverse=True,
        )

        return ranked[:max_files]

    def build_file_context(
        self,
        relative_path: str,
        focus_line: Optional[int] = None,
        window_size: int = 40,
    ) -> Optional[FileContext]:
        """Read a file and extract a bounded context window."""
        try:
            full_text = self.workspace.read_file(relative_path)
        except Exception:
            return None

        lines = full_text.splitlines()
        total_lines = len(lines)
        ext = Path(relative_path).suffix.lower()

        lang_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".json": "json",
            ".md": "markdown",
            ".txt": "text",
            ".toml": "toml",
        }
        lang = lang_map.get(ext, "text")

        # Detect symbols
        symbols: list[str] = []
        sym_pattern = re.compile(r"^\s*(?:class|def|async def|function)\s+([a-zA-Z0-9_]+)")
        for l in lines:
            m = sym_pattern.match(l)
            if m:
                symbols.append(m.group(1))

        if focus_line is not None and 1 <= focus_line <= total_lines:
            # Center window around focus line
            start = max(1, focus_line - (window_size // 2))
            end = min(total_lines, start + window_size - 1)
            selected_lines = lines[start - 1 : end]
            snippet = "\n".join(f"{i}: {line}" for i, line in enumerate(selected_lines, start=start))
            is_trunc = (start > 1 or end < total_lines)
            return FileContext(
                relative_path=relative_path,
                language=lang,
                content_snippet=snippet,
                total_lines=total_lines,
                start_line=start,
                end_line=end,
                symbols=symbols,
                is_truncated=is_trunc,
            )

        # Default: Read up to MAX_FILE_CHARS with line numbers
        snippet_lines = []
        char_count = 0
        is_trunc = False

        for idx, line in enumerate(lines, start=1):
            line_str = f"{idx}: {line}"
            if char_count + len(line_str) > self.MAX_FILE_CHARS:
                snippet_lines.append(f"... [{total_lines - idx + 1} more lines truncated]")
                is_trunc = True
                break
            snippet_lines.append(line_str)
            char_count += len(line_str) + 1

        return FileContext(
            relative_path=relative_path,
            language=lang,
            content_snippet="\n".join(snippet_lines),
            total_lines=total_lines,
            start_line=1,
            end_line=len(snippet_lines),
            symbols=symbols,
            is_truncated=is_trunc,
        )

    def assemble_context_block(self, file_contexts: List[FileContext]) -> str:
        """Format file contexts into a clean, bounded prompt block."""
        blocks = []
        total_chars = 0

        for fc in file_contexts:
            block = (
                f"### File: {fc.relative_path} ({fc.language}, {fc.total_lines} lines)\n"
                f"Symbols: {', '.join(fc.symbols) if fc.symbols else 'None'}\n"
                f"```\n{fc.content_snippet}\n```"
            )
            if total_chars + len(block) > self.MAX_TOTAL_CHARS:
                blocks.append("... [Additional files omitted to protect context limits]")
                break
            blocks.append(block)
            total_chars += len(block)

        return "\n\n".join(blocks)
