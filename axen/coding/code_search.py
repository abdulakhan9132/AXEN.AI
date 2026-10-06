"""
axen/coding/code_search.py
===========================
Fast local code search engine for the AXEN Coding Agent:
- Filename and path search
- Exact text and regex search with line numbering
- Symbol search (classes, functions, methods)
- Import dependency search
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from .workspace import Workspace


@dataclass
class SearchMatch:
    file_path: str
    line_number: int
    line_content: str
    match_type: str = "text"
    symbol_name: Optional[str] = None


class CodeSearch:
    """Fast local search engine confined to the approved workspace."""

    # Python & JS symbol definitions
    _PY_CLASS_PATTERN = re.compile(r"^\s*class\s+([a-zA-Z0-9_]+)")
    _PY_FUNC_PATTERN = re.compile(r"^\s*(?:async\s+def|def)\s+([a-zA-Z0-9_]+)")
    _JS_FUNC_PATTERN = re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([a-zA-Z0-9_]+)")
    _JS_CLASS_PATTERN = re.compile(r"^\s*(?:export\s+)?class\s+([a-zA-Z0-9_]+)")
    _JS_CONST_FUNC_PATTERN = re.compile(r"^\s*(?:export\s+)?const\s+([a-zA-Z0-9_]+)\s*=\s*(?:async\s*)?\(")

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def search_files(self, pattern: str) -> List[str]:
        """Search for filenames matching a glob or substring pattern."""
        files = self.workspace.list_files()
        results: list[str] = []
        clean_pat = pattern.strip().lower()

        for f in files:
            rel = str(f.relative_to(self.workspace.root_path)).replace("\\", "/")
            if clean_pat in rel.lower() or f.name.lower() == clean_pat:
                results.append(rel)

        return results

    def search_text(
        self,
        query: str,
        case_sensitive: bool = False,
        is_regex: bool = False,
        file_extension: Optional[str] = None,
        max_results: int = 30,
    ) -> List[SearchMatch]:
        """Search across workspace file lines for matching text or regex."""
        matches: list[SearchMatch] = []
        files = self.workspace.list_files()

        if is_regex:
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                pattern = re.compile(query, flags)
            except re.error as exc:
                raise ValueError(f"Invalid regular expression: {exc}")
        else:
            q_norm = query if case_sensitive else query.lower()

        for f in files:
            if file_extension and not f.name.endswith(file_extension):
                continue

            rel_str = str(f.relative_to(self.workspace.root_path)).replace("\\", "/")
            try:
                lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception:
                continue

            for line_idx, line in enumerate(lines, start=1):
                is_match = False
                if is_regex:
                    is_match = bool(pattern.search(line))
                else:
                    target_line = line if case_sensitive else line.lower()
                    is_match = q_norm in target_line

                if is_match:
                    matches.append(
                        SearchMatch(
                            file_path=rel_str,
                            line_number=line_idx,
                            line_content=line.strip()[:180],
                            match_type="text",
                        )
                    )
                    if len(matches) >= max_results:
                        return matches

        return matches

    def search_symbols(
        self,
        symbol_name: str,
        symbol_type: Optional[str] = None,  # "class", "function", or None
        max_results: int = 20,
    ) -> List[SearchMatch]:
        """Locate where a class or function symbol is declared."""
        matches: list[SearchMatch] = []
        target = symbol_name.strip().lower()
        files = self.workspace.list_files()

        code_extensions = {".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".rs", ".go"}

        for f in files:
            if f.suffix.lower() not in code_extensions:
                continue

            rel_str = str(f.relative_to(self.workspace.root_path)).replace("\\", "/")
            try:
                lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception:
                continue

            for line_idx, line in enumerate(lines, start=1):
                matched_sym = None
                m_type = None

                # Python Check
                cm = self._PY_CLASS_PATTERN.match(line)
                fm = self._PY_FUNC_PATTERN.match(line)
                if cm:
                    matched_sym = cm.group(1)
                    m_type = "class"
                elif fm:
                    matched_sym = fm.group(1)
                    m_type = "function"

                # JS/TS Check
                if not matched_sym:
                    jm_f = self._JS_FUNC_PATTERN.match(line)
                    jm_c = self._JS_CLASS_PATTERN.match(line)
                    jm_cf = self._JS_CONST_FUNC_PATTERN.match(line)
                    if jm_c:
                        matched_sym = jm_c.group(1)
                        m_type = "class"
                    elif jm_f:
                        matched_sym = jm_f.group(1)
                        m_type = "function"
                    elif jm_cf:
                        matched_sym = jm_cf.group(1)
                        m_type = "function"

                if matched_sym:
                    if target in matched_sym.lower():
                        if symbol_type and symbol_type.lower() != m_type:
                            continue
                        matches.append(
                            SearchMatch(
                                file_path=rel_str,
                                line_number=line_idx,
                                line_content=line.strip(),
                                match_type=m_type or "symbol",
                                symbol_name=matched_sym,
                            )
                        )
                        if len(matches) >= max_results:
                            return matches

        return matches

    def find_imports(self, module_or_symbol: str, max_results: int = 20) -> List[SearchMatch]:
        """Find where a specific module or function is imported."""
        target = module_or_symbol.strip()
        pattern = rf"\b(?:import\s+{re.escape(target)}|from\s+[\w\.]*{re.escape(target)}[\w\.]*\s+import|require\(['\"]{re.escape(target)}['\"]\))"
        return self.search_text(pattern, is_regex=True, max_results=max_results)
