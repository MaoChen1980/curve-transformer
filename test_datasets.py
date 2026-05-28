"""Try to load Chinese datasets from HuggingFace."""
from datasets import load_dataset

# Test a few Chinese datasets
candidates = [
    ("wikitext", "zh"),
    ("oscar", "unshuffled_deduplicated_zh"),
    (" CMC", None),
    ("liamchu/chinese-webtext", None),
    ("yeastlin/Chinese-Corpus-100M", None),
]

print("Testing available Chinese datasets...")
for name, subset in candidates:
    try:
        if subset:
            ds = load_dataset(name, subset, streaming=True, trust_remote_code=True)
        else:
            ds = load_dataset(name, streaming=True, trust_remote_code=True)
        print(f"  ✅ {name}/{subset} — loaded, rows={next(iter(ds['train']))}")
    except Exception as e:
        print(f"  ❌ {name}/{subset} — {type(e).__name__}: {e}")

# Also check what's locally available
import os
print("\nChecking E: drive space...")
import shutil
total, used, free = shutil.disk_usage("E:/")
print(f"  Total: {total/1024**3:.1f} GB | Used: {used/1024**3:.1f} GB | Free: {free/1024**3:.1f} GB")