import json
import random
import re
from pathlib import Path
from collections import defaultdict

# Optional: suppress huggingface hub warnings
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

TARGET_EXAMPLES = 10000
MAX_SHARE = 0.15
SEED = 42
random.seed(SEED)

def detect_language(text: str) -> str:
    # Basic code fence heuristic
    match = re.search(r"```([a-zA-Z0-9+#_]+)", text)
    if match:
        lang = match.group(1).lower()
        if lang in ["python", "py"]: return "python"
        if lang in ["javascript", "js", "node"]: return "javascript"
        if lang in ["typescript", "ts"]: return "typescript"
        if lang in ["java"]: return "java"
        if lang in ["c++", "cpp", "cxx"]: return "cpp"
        if lang in ["c"]: return "c"
        if lang in ["go", "golang"]: return "go"
        if lang in ["rust", "rs"]: return "rust"
        if lang in ["html"]: return "html"
        if lang in ["css"]: return "css"
        if lang in ["sql"]: return "sql"
        if lang in ["bash", "sh", "shell"]: return "bash"
        if lang in ["json"]: return "json"
        return lang
        
    # Content heuristic
    if "def " in text and ":" in text and "self" in text: return "python"
    if "function(" in text or "=>" in text or "const " in text: return "javascript"
    if "fn " in text and "->" in text and "let mut" in text: return "rust"
    if "func " in text and "fmt.Println" in text: return "go"
    if "#include" in text and "std::" in text: return "cpp"
    if "public static void main" in text or "public class " in text: return "java"
    
    return "unknown"

def has_code(text: str) -> bool:
    return "```" in text

def estimate_tokens(text: str) -> int:
    return len(text) // 4  # Rough heuristic

def format_magicoder(row):
    return {
        "messages": [
            {"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": row["response"]}
        ]
    }

def format_evol(row):
    return {
        "messages": [
            {"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": row["output"]}
        ]
    }

def format_alpaca(row):
    prompt = row["instruction"]
    if row.get("input"):
        prompt += "\n" + row["input"]
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": row["output"]}
        ]
    }

def process_dataset(ds_name, split, formatter):
    print(f"Loading {ds_name}...")
    try:
        ds = load_dataset(ds_name, split=split)
    except Exception as e:
        print(f"Failed to load {ds_name}: {e}")
        return []
        
    results = []
    seen = set()
    for row in ds:
        formatted = formatter(row)
        user_msg = formatted["messages"][0]["content"]
        asst_msg = formatted["messages"][1]["content"]
        
        # Deduplicate
        if user_msg in seen:
            continue
        seen.add(user_msg)
        
        # Filter by size and code presence
        total_len = estimate_tokens(user_msg) + estimate_tokens(asst_msg)
        if total_len < 20 or total_len > 2048:
            continue
        if not has_code(asst_msg):
            continue
            
        lang = detect_language(asst_msg)
        results.append({
            "messages": formatted["messages"],
            "lang": lang
        })
    return results

def main():
    if not HAS_DATASETS:
        print("Please install datasets: pip install datasets")
        return
        
    print("Starting dataset preparation...")
    all_examples = []
    
    # 1. Load from Hugging Face
    all_examples.extend(process_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", "train", format_magicoder))
    all_examples.extend(process_dataset("nickrosh/Evol-Instruct-Code-80k-v1", "train", format_evol))
    all_examples.extend(process_dataset("sahil2801/CodeAlpaca-20k", "train", format_alpaca))
    
    print(f"Total raw examples after initial filters: {len(all_examples)}")
    
    # 2. Balance by language
    by_lang = defaultdict(list)
    for ex in all_examples:
        by_lang[ex["lang"]].append(ex)
        
    print("\nInitial Language Counts:")
    for lang, items in sorted(by_lang.items(), key=lambda x: len(x[1]), reverse=True)[:15]:
        print(f"{lang:15s}: {len(items)}")
        
    max_per_lang = int(TARGET_EXAMPLES * MAX_SHARE)
    balanced_examples = []
    
    for lang, items in by_lang.items():
        random.shuffle(items)
        # Cap at max_per_lang
        balanced_examples.extend(items[:max_per_lang])
        
    # We might have more than TARGET_EXAMPLES now. Shuffle and trim.
    random.shuffle(balanced_examples)
    if len(balanced_examples) > TARGET_EXAMPLES:
        balanced_examples = balanced_examples[:TARGET_EXAMPLES]
        
    print(f"\nExamples after balancing: {len(balanced_examples)}")
    
    # 3. Merge custom examples
    my_examples_path = Path("training/my_examples.jsonl")
    if my_examples_path.exists():
        with open(my_examples_path, "r", encoding="utf-8") as f:
            custom = [json.loads(line) for line in f if line.strip()]
        print(f"Found {len(custom)} custom examples. Upsampling 3x...")
        balanced_examples.extend(custom * 3)
    else:
        print("No custom examples found at training/my_examples.jsonl")
        
    # 4. Final shuffle and split
    random.shuffle(balanced_examples)
    # Remove the 'lang' key for the final JSONL to keep it clean
    for ex in balanced_examples:
        if "lang" in ex:
            del ex["lang"]
            
    split_idx = int(len(balanced_examples) * 0.95)
    train_data = balanced_examples[:split_idx]
    eval_data = balanced_examples[split_idx:]
    
    Path("training").mkdir(exist_ok=True)
    with open("training/train.jsonl", "w", encoding="utf-8") as f:
        for ex in train_data:
            f.write(json.dumps(ex) + "\n")
            
    with open("training/eval.jsonl", "w", encoding="utf-8") as f:
        for ex in eval_data:
            f.write(json.dumps(ex) + "\n")
            
    print(f"\nDone! Created training/train.jsonl ({len(train_data)}) and training/eval.jsonl ({len(eval_data)})")

if __name__ == "__main__":
    main()
