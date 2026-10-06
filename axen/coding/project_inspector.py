"""
axen/coding/project_inspector.py
=================================
Analyzes project structure, configuration, manifests, and test frameworks
to generate a compact, structured Project Map without overwhelming the LLM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .workspace import Workspace


class ProjectInspector:
    """Inspects a project workspace to identify manifests, dependencies, and layout."""

    # Manifest file mappings to ecosystem/language
    MANIFEST_MAP = {
        "pyproject.toml": ("python", "Python (PEP 517/518)"),
        "requirements.txt": ("python", "Python (pip)"),
        "setup.py": ("python", "Python (setuptools)"),
        "Pipfile": ("python", "Python (Pipenv)"),
        "pytest.ini": ("python", "pytest configuration"),
        "package.json": ("javascript", "Node.js / JavaScript"),
        "tsconfig.json": ("typescript", "TypeScript"),
        "pom.xml": ("java", "Java (Maven)"),
        "build.gradle": ("java", "Java (Gradle)"),
        "Cargo.toml": ("rust", "Rust (Cargo)"),
        "go.mod": ("go", "Go (Modules)"),
        "Makefile": ("polyglot", "Make"),
    }

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def detect_manifests(self) -> Dict[str, str]:
        """Find all manifest/configuration files present in the root of the workspace."""
        detected = {}
        for filename, (_, desc) in self.MANIFEST_MAP.items():
            candidate = self.workspace.root_path / filename
            if candidate.exists() and candidate.is_file():
                detected[filename] = desc
        return detected

    def detect_ecosystem(self) -> tuple[str, Optional[str]]:
        """
        Infer primary language and recommended test runner command.
        Returns: (language, default_test_command)
        """
        manifests = self.detect_manifests()

        if any(m in manifests for m in ("pyproject.toml", "requirements.txt", "setup.py", "pytest.ini")):
            # Check if pytest is configured or present
            return "python", "pytest"

        if "package.json" in manifests:
            pkg_path = self.workspace.root_path / "package.json"
            try:
                pkg_data = json.loads(pkg_path.read_text(encoding="utf-8"))
                has_test = "test" in pkg_data.get("scripts", {})
                lang = "typescript" if "tsconfig.json" in manifests else "javascript"
                return lang, "npm test" if has_test else None
            except Exception:
                return "javascript", "npm test"

        if "Cargo.toml" in manifests:
            return "rust", "cargo test"

        if "go.mod" in manifests:
            return "go", "go test ./..."

        if "pom.xml" in manifests:
            return "java", "mvn test"

        # Check workspace files if no manifest found
        all_files = self.workspace.list_files(max_files=100)
        if any(f.suffix == ".py" for f in all_files):
            return "python", "pytest"
        if any(f.suffix in (".js", ".jsx") for f in all_files):
            return "javascript", "npm test"
        if any(f.suffix in (".ts", ".tsx") for f in all_files):
            return "typescript", "npm test"

        return "unknown", None

    def generate_project_map(self, max_depth: int = 3, max_entries: int = 40) -> str:
        """
        Produce a compact, human-readable directory tree for LLM context planning.
        Omits noisy build, cache, and dependency folders.
        """
        root = self.workspace.root_path
        lines = [f"{root.name}/"]
        entries_counted = 0

        ignore_dirs = {
            ".git", ".venv", "venv", "__pycache__", "node_modules",
            ".pytest_cache", ".mypy_cache", "dist", "build", ".egg-info",
        }

        def _walk(directory: Path, prefix: str, current_depth: int):
            nonlocal entries_counted
            if current_depth > max_depth or entries_counted >= max_entries:
                return

            try:
                items = sorted(
                    directory.iterdir(),
                    key=lambda x: (not x.is_dir(), x.name.lower()),
                )
            except PermissionError:
                return

            # Filter items
            visible_items = [
                item for item in items
                if item.name not in ignore_dirs
                and not item.name.endswith((".pyc", ".axen_backup", ".egg-info"))
            ]

            for i, item in enumerate(visible_items):
                if entries_counted >= max_entries:
                    lines.append(f"{prefix}└── ... [truncated]")
                    break

                is_last = (i == len(visible_items) - 1)
                connector = "└── " if is_last else "├── "
                sub_prefix = "    " if is_last else "│   "

                if item.is_dir():
                    lines.append(f"{prefix}{connector}{item.name}/")
                    entries_counted += 1
                    _walk(item, prefix + sub_prefix, current_depth + 1)
                else:
                    lines.append(f"{prefix}{connector}{item.name}")
                    entries_counted += 1

        _walk(root, "", 1)
        return "\n".join(lines)

    def get_summary(self) -> Dict[str, Any]:
        """Produce a complete structured project overview."""
        lang, test_cmd = self.detect_ecosystem()
        manifests = self.detect_manifests()
        files = self.workspace.list_files()

        # Identify tests folder
        test_dirs = [
            str(p.relative_to(self.workspace.root_path))
            for p in self.workspace.root_path.glob("test*")
            if p.is_dir()
        ]

        return {
            "workspace_name": self.workspace.root_path.name,
            "workspace_path": str(self.workspace.root_path),
            "primary_language": lang,
            "test_runner": test_cmd,
            "manifests": manifests,
            "test_directories": test_dirs,
            "total_files": len(files),
            "project_map": self.generate_project_map(),
        }
