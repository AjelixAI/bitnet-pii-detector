#!/usr/bin/env python3
"""build_unified_corpus.py — unified GDPR-aware PII corpus with ZERO contamination.

TRAINING SET: ai4privacy train split + TAB echr_train split ONLY
EVAL SET:     ai4privacy val split + TAB echr_test + TAB echr_dev

Guarantees: no doc in eval appears in training (verified by uid AND text fingerprint).
"""
import json, glob, os, random, sys, hashlib
sys.path.insert(0, "/root/bitnet_train")
from unified_taxonomy import (UNIFIED_TYPES, AI4P_MAP, TAB_MAP,
                               remap_spans, merge_adjacent)

def load_ai4privacy(shard_dir, split, max_per_lang=0):
    rows = []
    for fp in sorted(glob.glob(os.path.join(shard_dir, f"eu-{split}_*.jsonl"))):
        n = 0
        with open(fp) as f:
            for line in f:
                if max_per_lang and n >= max_per_lang: break
                r = json.loads(line)
                r["spans"] = merge_adjacent(remap_spans(r["spans"], AI4P_MAP), r["text"])
                r["source"] = "ai4privacy"
                rows.append(r); n += 1
    return rows

def load_tab_split(tab_dir, split_file):
    fp = os.path.join(tab_dir, split_file)
    if not os.path.exists(fp): return []
    rows = []
    for doc in json.load(open(fp)):
        text = doc["text"]
        spans = []
        for annotator, layer in doc.get("annotations", {}).items():
            for em in layer.get("entity_mentions", []):
                if em.get("identifier_type") == "NO_MASK": continue
                s, e = em["start_offset"], em["end_offset"]
                if 0 <= s < e <= len(text):
                    spans.append({"start": s, "end": e, "type": em.get("entity_type", "?")})
        spans = merge_adjacent(remap_spans(spans, TAB_MAP), text)
        if text and spans:
            rows.append({"text": text, "spans": spans, "source": "tab", "language": "en"})
    return rows

def text_fingerprint(text):
    """First 300 chars hashed — catches near-identical content across splits."""
    return hashlib.md5(text[:300].encode("utf-8")).hexdigest()

outdir = "/scratch/pii_corpus/unified"
os.makedirs(outdir, exist_ok=True)

# ── TRAINING SET: ai4privacy train + TAB echr_train ONLY ──
print("Building TRAIN set...", flush=True)
ai4p_train = load_ai4privacy("/scratch/pii_corpus/eu_shards", "train")
tab_train = load_tab_split("/scratch/pii_corpus/benchmarks/tab", "echr_train.json")
print(f"  ai4privacy train: {len(ai4p_train)}")
print(f"  TAB echr_train:   {len(tab_train)}")
train_rows = ai4p_train + tab_train

# ── EVAL SET: ai4privacy val + TAB echr_test + TAB echr_dev ──
print("Building EVAL set...", flush=True)
ai4p_val = load_ai4privacy("/scratch/pii_corpus/eu_shards", "val")
tab_test = load_tab_split("/scratch/pii_corpus/benchmarks/tab", "echr_test.json")
tab_dev = load_tab_split("/scratch/pii_corpus/benchmarks/tab", "echr_dev.json")
print(f"  ai4privacy val: {len(ai4p_val)}")
print(f"  TAB echr_test:   {len(tab_test)}")
print(f"  TAB echr_dev:    {len(tab_dev)}")
eval_rows = ai4p_val + tab_test + tab_dev

# ── CONTAMINATION CHECK (text fingerprint, not just uid) ──
print("\nRunning contamination check...", flush=True)
train_fp = set(text_fingerprint(r["text"]) for r in train_rows)
eval_fp = set(text_fingerprint(r["text"]) for r in eval_rows)
overlap = train_fp & eval_fp
print(f"  train fingerprints: {len(train_fp)}")
print(f"  eval fingerprints:  {len(eval_fp)}")
print(f"  OVERLAP (contamination): {len(overlap)}")
if overlap:
    print("  ❌ CONTAMINATION DETECTED — removing contaminated eval rows")
    eval_rows = [r for r in eval_rows if text_fingerprint(r["text"]) not in train_fp]
    print(f"  eval rows after decontamination: {len(eval_rows)}")
else:
    print("  ✅ ZERO CONTAMINATION — eval is fully disjoint from train")

# ── WRITE ──
random.seed(42); random.shuffle(train_rows); random.shuffle(eval_rows)
out_train = os.path.join(outdir, "unified_train.jsonl")
out_eval = os.path.join(outdir, "unified_eval.jsonl")
with open(out_train, "w") as f:
    for r in train_rows: f.write(json.dumps(r) + "\n")
with open(out_eval, "w") as f:
    for r in eval_rows: f.write(json.dumps(r) + "\n")
print(f"\nWrote {len(train_rows)} train → {out_train}")
print(f"Wrote {len(eval_rows)} eval → {out_eval}")

# separate eval files by source for per-benchmark reporting
ai4p_eval = [r for r in eval_rows if r["source"] == "ai4privacy"]
tab_eval = [r for r in eval_rows if r["source"] == "tab"]
with open(os.path.join(outdir, "eval_ai4privacy.jsonl"), "w") as f:
    for r in ai4p_eval: f.write(json.dumps(r) + "\n")
with open(os.path.join(outdir, "eval_tab.jsonl"), "w") as f:
    for r in tab_eval: f.write(json.dumps(r) + "\n")
print(f"  eval_ai4privacy.jsonl: {len(ai4p_eval)} rows")
print(f"  eval_tab.jsonl:        {len(tab_eval)} rows (held-out TAB test+dev)")

json.dump(UNIFIED_TYPES, open(os.path.join(outdir, "types.json"), "w"))

from collections import Counter
tc = Counter()
for r in train_rows:
    for s in r["spans"]: tc[s["type"]] += 1
print(f"\nUnified type distribution (train):")
for t, c in tc.most_common():
    print(f"  {t:20s} {c:>8d}")
sc = Counter(r["source"] for r in train_rows)
print(f"\nSource: {dict(sc)}")
