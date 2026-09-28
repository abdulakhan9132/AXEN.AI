"""Tests for AXEN Language Detector."""

from __future__ import annotations

from pathlib import Path
from reasoning.coding.detector import detect_language


def test_detect_explicit_flag():
    res = detect_language(prompt="Build a CLI tool --language rust")
    assert res.language is not None
    assert res.language.name == "Rust"
    assert res.confidence >= 0.95


def test_detect_prompt_keywords():
    res_py = detect_language(prompt="Write a Python script to scrape a site")
    assert res_py.language is not None
    assert res_py.language.name == "Python"

    res_sql = detect_language(prompt="Explain this SQL query with a CTE")
    assert res_sql.language is not None
    assert res_sql.language.name == "SQL"

    res_cpp = detect_language(prompt="Implement Dijkstra in C++")
    assert res_cpp.language is not None
    assert res_cpp.language.name == "C++"


def test_detect_from_file_path():
    assert detect_language(file_path="src/main.rs").language.name == "Rust"
    assert detect_language(file_path="scripts/deploy.sh").language.name == "Bash"
    assert detect_language(file_path="kernel.cu").language.name == "CUDA"
    assert detect_language(file_path="app/components/Header.tsx").language.name == "TypeScript"
    assert detect_language(file_path="analysis.R").language.name == "R"


def test_detect_from_manifests():
    assert detect_language(file_path="Cargo.toml").language.name == "Rust"
    assert detect_language(file_path="go.mod").language.name == "Go"
    assert detect_language(file_path="package.json").language.name == "JavaScript"
    assert detect_language(file_path="pubspec.yaml").language.name == "Dart"


def test_detect_from_syntax_patterns():
    cuda_code = """
    __global__ void add_vectors(float *a, float *b, float *c, int n) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < n) c[i] = a[i] + b[i];
    }
    """
    res = detect_language(code=cuda_code)
    assert res.language is not None
    assert res.language.name == "CUDA"

    rust_code = """
    fn main() {
        let mut x = 5;
        println!("Value: {}", x);
    }
    """
    assert detect_language(code=rust_code).language.name == "Rust"

    python_code = """
    def solve(nums: list[int]) -> int:
        return sum(nums)

    if __name__ == '__main__':
        print(solve([1, 2, 3]))
    """
    assert detect_language(code=python_code).language.name == "Python"


def test_uncertainty_on_ambiguous_text():
    # Plain English without code or language hints
    res = detect_language(prompt="What is the weather like in Tokyo?")
    assert res.language is None or res.confidence < 0.40
