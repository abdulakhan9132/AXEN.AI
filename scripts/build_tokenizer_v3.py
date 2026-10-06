"""
scripts/build_tokenizer_v3.py
==============================
Builds the canonical AXEN Subword Tokenizer V3 vocabulary (~2,048 tokens)
from the AXEN training corpus and core syntax/domain tokens.

Outputs:
- checkpoints/tokenizer_v3.json
- evaluation/phase10/tokenizer_training_corpus_report.json
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

TARGET_VOCAB_SIZE = 2048

# 1. Mandatory Special Tokens
SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

# 2. Mandatory Base Whitespace and Delimiters
BASE_WHITESPACE = [
    "\n", "\t", "\r", " ", "  ", "    ", "        "
]

# 3. All Printable ASCII Characters (ord 32 to 126)
ASCII_CHARS = [chr(code) for code in range(32, 127)]

# 4. Common Unicode Characters
COMMON_UNICODE = [
    "•", "—", "–", "’", "“", "”", "…", "°", "×", "÷", "±", "²", "³",
    "µ", "€", "£", "¥", "→", "←", "↔", "✓", "✗", "★", "\u200b"
]

# 5. Core Multi-character Programming and Syntax Delimiters
SYNTAX_TOKENS = [
    "==", "!=", "<=", ">=", "->", "=>", "//", "/*", "*/", ":=", "**",
    "++", "--", "+=", "-=", "*=", "/=", "&&", "||", "===", "!==",
    ": ", ", ", ": [", ": {", "{\n", "}\n", "{\n  ", '": "', '": ',
    ": true", ": false", ": null", "```", "```python", "```javascript",
    "```json", "```html", "```css", "```bash", "<div>", "</div>",
    "<span>", "</span>", "<p>", "</p>", "<a>", "</a>", "<h1>", "</h1>",
    "<h2>", "</h2>", "<h3>", "</h3>", "<html>", "</html>", "<head>",
    "</head>", "<body>", "</body>", "<button>", "</button>", "<input>",
    "<form>", "</form>", "<ul>", "</ul>", "<li>", "</li>", "<table>",
    "</table>", "<tr>", "</tr>", "<td>", "</td>", "<script>", "</script>",
    "<style>", "</style>", "<br>", "<hr>", "</br>", "/>"
]

# 6. Explicitly Requested Evaluation Examples
REQUESTED_TOKENS = [
    "hello", "conversation", "artificial intelligence", "machine learning",
    "information", "Python", "JavaScript", "function", "def", "return",
    "const", "{", "}", ":", ",", "=", "==", "->", "<div>", "</div>",
    "JSON", "12345"
]

# 7. Common Programming Keywords and Syntactic Entities
PROGRAMMING_TOKENS = [
    # Python / General
    "def", "return", "const", "let", "var", "function", "class", "import",
    "from", "export", "async", "await", "try", "catch", "finally", "if",
    "else", "elif", "while", "for", "in", "of", "break", "continue", "yield",
    "lambda", "pass", "raise", "except", "assert", "print", "self", "this",
    "true", "false", "null", "None", "True", "False", "undefined", "NaN",
    "json", "JSON", "python", "Python", "javascript", "JavaScript", "html",
    "HTML", "css", "CSS", "sql", "SQL", "select", "insert", "update",
    "delete", "where", "join", "table", "database", "api", "API", "http",
    "HTTP", "url", "URL", "query", "response", "request", "status", "data",
    "value", "key", "name", "type", "id", "message", "role", "content",
    "intent", "confidence", "tool", "arguments", "result", "error", "success",
    "failure", "verification", "model", "output", "prompt", "system", "user",
    "assistant", "console", "log", "string", "number", "boolean", "array",
    "object", "dict", "list", "int", "str", "float", "bool", "len", "range",
    "enumerate", "zip", "map", "filter", "open", "read", "write", "close",
    "append", "extend", "pop", "get", "items", "keys", "values", "update",
    # Tools / Actions
    "file_read", "file_write", "file_search", "calculator", "bash", "execute",
    "terminal", "directory", "filename", "filepath", "project", "workspace",
    "context", "memory", "reasoning", "decision", "action", "planner",
    "fallback", "specialist", "router", "brain", "agent", "session",
    # Contractions
    "don't", "can't", "won't", "it's", "I'm", "you're", "they're", "we're",
    "that's", "there's", "what's", "doesn't", "didn't", "isn't", "aren't",
    "wasn't", "weren't", "haven't", "hasn't", "hadn't", "wouldn't",
    "shouldn't", "couldn't"
]

# 8. Frequent Subwords & Morphemes
SUBWORDS = [
    "ing", "tion", "tions", "ment", "ments", "ness", "able", "ible",
    "ally", "ful", "less", "est", "ed", "er", "ers", "ly", "es", "al",
    "ize", "ized", "izing", "ous", "ic", "ive", "ity", "ities", "pre",
    "re", "un", "dis", "non", "multi", "inter", "trans", "super", "sub",
    "auto", "tele", "ance", "ence", "ant", "ent", "ism", "ist", "ists"
]


def collect_corpus_data() -> Tuple[List[str], Dict[str, Any]]:
    """Reads all existing AXEN datasets, sanitizes, and returns text records."""
    corpus_paths = [
        Path("data/training/curated/all_curated.jsonl"),
        Path("data/training/train.jsonl"),
        Path("data/training/events/success.jsonl"),
        Path("training/intents.jsonl"),
        Path("data/axen_conversations.json"),
    ]

    records: List[str] = []
    unique_texts: Set[str] = set()
    duplicates_count = 0
    total_examples = 0
    source_stats: Dict[str, int] = {}

    for path in corpus_paths:
        if not path.exists():
            continue
        count_in_file = 0
        if path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    # Extract text content
                    text = ""
                    if "input" in obj and "target" in obj:
                        text = f"{obj['input']} {obj['target']}"
                    elif "text" in obj:
                        text = obj["text"]
                    elif "user_request" in obj:
                        text = f"{obj.get('user_request', '')} {obj.get('requested_action', '')}"
                    
                    if text.strip():
                        total_examples += 1
                        count_in_file += 1
                        if text in unique_texts:
                            duplicates_count += 1
                        else:
                            unique_texts.add(text)
                            records.append(text)
        elif path.suffix == ".json":
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                try:
                    data = json.load(f)
                    if isinstance(data, dict):
                        for _, conv in data.items():
                            if isinstance(conv, dict) and "messages" in conv:
                                for msg in conv["messages"]:
                                    c = msg.get("content", "")
                                    if c.strip():
                                        total_examples += 1
                                        count_in_file += 1
                                        if c in unique_texts:
                                            duplicates_count += 1
                                        else:
                                            unique_texts.add(c)
                                            records.append(c)
                except Exception as e:
                    print(f"Error loading {path}: {e}")

        source_stats[str(path)] = count_in_file

    total_chars = sum(len(t) for t in records)
    total_words = sum(len(t.split()) for t in records)

    meta = {
        "source_files": [str(p) for p in corpus_paths if p.exists()],
        "source_stats": source_stats,
        "total_examples": total_examples,
        "unique_text_count": len(records),
        "duplicate_count": duplicates_count,
        "duplicate_percentage": round((duplicates_count / max(1, total_examples)) * 100.0, 2),
        "total_characters": total_chars,
        "total_words": total_words,
    }
    return records, meta


def build_vocabulary(corpus_records: List[str], target_size: int = TARGET_VOCAB_SIZE) -> Dict[str, int]:
    """Builds deterministic vocabulary of exactly target_size tokens."""
    vocab: Dict[str, int] = {}

    def add_token(t: str) -> bool:
        if t not in vocab and len(vocab) < target_size:
            vocab[t] = len(vocab)
            return True
        return False

    # 1. Special tokens (0..3)
    for tok in SPECIAL_TOKENS:
        add_token(tok)

    # 2. Whitespace tokens
    for tok in BASE_WHITESPACE:
        add_token(tok)

    # 3. All printable ASCII
    for tok in ASCII_CHARS:
        add_token(tok)

    # 4. Common Unicode
    for tok in COMMON_UNICODE:
        add_token(tok)

    # 5. Core Syntax & Delimiters
    for tok in SYNTAX_TOKENS:
        add_token(tok)

    # 6. Explicit Requested Tokens
    for tok in REQUESTED_TOKENS:
        add_token(tok)

    # 7. Programming tokens
    for tok in PROGRAMMING_TOKENS:
        add_token(tok)

    # 8. Common subwords
    for tok in SUBWORDS:
        add_token(tok)

    # 9. Mine Corpus: Word & Subword frequencies
    # Count words and subwords with regex token extraction
    freq_counter = Counter()
    for text in corpus_records:
        # Extract word tokens (both case sensitive and lowercase)
        tokens = re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]+|\s+", text)
        for t in tokens:
            if len(t) > 1 and t not in vocab:
                freq_counter[t] += 1
            # Also count lowercased word
            if t.isalpha() and not t.islower() and t.lower() not in vocab:
                freq_counter[t.lower()] += 1

        # Also mine frequent 2-6 character substrings within words
        for w in text.split():
            clean_w = re.sub(r"[^\w]", "", w)
            if len(clean_w) >= 3:
                for sublen in (3, 4, 5):
                    for i in range(len(clean_w) - sublen + 1):
                        sub = clean_w[i : i + sublen]
                        if sub not in vocab:
                            freq_counter[sub] += 1

    # Sort candidates deterministically by (-frequency, len(token), token)
    sorted_candidates = sorted(
        freq_counter.items(),
        key=lambda item: (-item[1], -len(item[0]), item[0])
    )

    for tok, _ in sorted_candidates:
        if len(vocab) >= target_size:
            break
        add_token(tok)

    # If still below target_size, add numeric ranges and common short patterns
    curr_num = 0
    while len(vocab) < target_size:
        num_str = str(curr_num)
        if num_str not in vocab:
            add_token(num_str)
        curr_num += 1

    return vocab


def main() -> None:
    print(f"Building AXEN Tokenizer V3 vocabulary (target size: {TARGET_VOCAB_SIZE})...")
    records, corpus_meta = collect_corpus_data()
    print(f"Corpus processed: {corpus_meta['total_examples']} records ({corpus_meta['unique_text_count']} unique).")

    vocab = build_vocabulary(records, target_size=TARGET_VOCAB_SIZE)
    print(f"Constructed vocabulary size: {len(vocab)} tokens.")

    # Save to checkpoints/tokenizer_v3.json
    out_path = Path("checkpoints/tokenizer_v3.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": "v3",
        "algorithm": "longest_prefix_match",
        "target_vocab_size": TARGET_VOCAB_SIZE,
        "vocab_size": len(vocab),
        "special_tokens": {
            "<PAD>": vocab["<PAD>"],
            "<UNK>": vocab["<UNK>"],
            "<BOS>": vocab["<BOS>"],
            "<EOS>": vocab["<EOS>"],
        },
        "vocab": vocab,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved tokenizer to {out_path}")

    # Also save report metadata
    rep_path = Path("evaluation/phase10/tokenizer_training_corpus_report.json")
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(corpus_meta, f, indent=2, ensure_ascii=False)
    print(f"Saved corpus statistics to {rep_path}")


if __name__ == "__main__":
    main()
