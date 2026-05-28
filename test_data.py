"""Try loading Chinese datasets."""
from datasets import load_dataset

# Try a few common ones
targets = [
    ("chinese Wikitext", "wikitext", "zh"),
    ("CMRC (QA)", "cmrc2018", None),
    ("DRCD (QA)", "DRCD", None),
    ("ChineseBookReview", "csv", None),
    ("SimChg/LCCCC", "csv", None),
]

for label, ds_name, subset in targets:
    try:
        kw = {"streaming": True, "trust_remote_code": True}
        if subset:
            ds = load_dataset(ds_name, subset, **kw)
        else:
            ds = load_dataset(ds_name, **kw)
        print(f"✅ {label} ({ds_name}/{subset}) — keys={list(ds.keys())}")
        ex = next(iter(ds['train']))
        print(f"   Sample keys: {list(ex.keys()) if hasattr(ex,'keys') else type(ex)}")
    except Exception as e:
        print(f"❌ {label} — {type(e).__name__}: {e}")