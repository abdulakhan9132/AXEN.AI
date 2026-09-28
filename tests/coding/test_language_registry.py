"""Tests for AXEN Language Registry (37 Languages)."""

from __future__ import annotations

import pytest
from reasoning.coding.registry import LanguageRegistry, get_language_registry


EXPECTED_37_LANGUAGES = [
    "Python", "SQL", "JavaScript", "TypeScript", "C++", "C", "Rust", "R",
    "Julia", "Java", "Go", "Scala", "MATLAB", "Kotlin", "Swift", "C#",
    "Lisp", "Prolog", "Haskell", "Mojo", "CUDA", "Fortran", "Lua", "Ruby",
    "PHP", "Dart", "Perl", "OCaml", "Clojure", "Octave", "SAS",
    "Wolfram Language", "Scheme", "Erlang", "Elixir", "F#", "Bash",
]


def test_registry_contains_exactly_37_languages():
    registry = get_language_registry()
    assert registry.count() == 37, f"Expected 37 languages, found {registry.count()}"
    all_names = [l.name for l in registry.list_all()]
    for expected in EXPECTED_37_LANGUAGES:
        assert expected in all_names, f"Missing language from registry: {expected}"


def test_all_languages_have_complete_specifications():
    registry = get_language_registry()
    for lang in registry.list_all():
        assert lang.name, "Language must have a non-empty name"
        assert lang.slug, f"Language {lang.name} must have a slug"
        assert lang.extensions, f"Language {lang.name} must have extensions"
        assert lang.primary_uses, f"Language {lang.name} must have primary uses"
        assert lang.ecosystem, f"Language {lang.name} must have ecosystem entries"
        assert lang.compilers_or_interpreters, f"Language {lang.name} must list compilers or interpreters"
        assert lang.syntax_family, f"Language {lang.name} must specify syntax family"


def test_registry_lookup_by_alias():
    registry = get_language_registry()
    # Test case insensitivity and common aliases
    assert registry.get("py").name == "Python"
    assert registry.get("PYTHON").name == "Python"
    assert registry.get("rs").name == "Rust"
    assert registry.get("cpp").name == "C++"
    assert registry.get("c++").name == "C++"
    assert registry.get("c#").name == "C#"
    assert registry.get("csharp").name == "C#"
    assert registry.get("golang").name == "Go"
    assert registry.get("js").name == "JavaScript"
    assert registry.get("ts").name == "TypeScript"
    assert registry.get("cu").name == "CUDA"
    assert registry.get("sh").name == "Bash"
    assert registry.get("mathematica").name == "Wolfram Language"


def test_registry_lookup_by_extension():
    registry = get_language_registry()
    assert any(l.name == "Rust" for l in registry.get_by_extension(".rs"))
    assert any(l.name == "Python" for l in registry.get_by_extension(".py"))
    assert any(l.name == "SQL" for l in registry.get_by_extension(".sql"))
    assert any(l.name == "CUDA" for l in registry.get_by_extension(".cu"))
    assert any(l.name == "TypeScript" for l in registry.get_by_extension(".tsx"))


def test_distinguishes_language_supported_from_runtime_available():
    registry = get_language_registry()
    python_spec = registry.get("python")
    assert python_spec is not None
    # Python is definitely running this test, so runtime should be detected as available
    installed, path = python_spec.check_runtime()
    assert installed is True
    assert path is not None

    # SAS is likely not installed on this developer machine
    sas_spec = registry.get("sas")
    assert sas_spec is not None
    installed_sas, path_sas = sas_spec.check_runtime()
    # The language is supported in AXEN, but runtime availability is honest
    assert isinstance(installed_sas, bool)
    if not installed_sas:
        assert path_sas is None
