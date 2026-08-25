#!/usr/bin/env python3
"""build_piibench_corpus.py — rebuild our ai4privacy + TAB data using PIIBench's
published 26-type canonical taxonomy (87 source labels mapped, peer-reviewed).
"""
import json, glob, os, random, hashlib

LABEL_MAP = json.load(open("/scratch/pii_corpus/benchmarks/piibench/label_map.json"))
TYPES = LABEL_MAP["types"]
LMAP = LABEL_MAP["label_map"]

def remap(spans):
    out = []
    for s in spans:
        utype = LMAP.get(s["type"].upper()) or LMAP.get(s["type"])
        if utype:
            out.append({"start": s["start"], "end": s["end"], "type": utype})
    return out

def merge(spans, gap=2):
    if not spans: return []
    spans = sorted(spans, key=lambda s: s["start"])
    merged = [dict(spans[0])]
    for s in spans[1:]:
        last = merged[-1]
        if s["type"] == last["type"] and s["start"] <= last["end"] + gap:
            last["end"] = max(last["end"], s["end"])
        else:
            merged.append(dict(s))
    return merged

def load_ai4p(shard_dir, split):
    rows = []
    for fp in sorted(glob.glob(os.path.join(shard_dir, f"eu-{split}_*.jsonl"))):
        with open(fp) as f:
            for line in f:
                r = json.loads(line)
                r["spans"] = merge(remap(r["spans"]))
                r["source"] = "ai4privacy"
                rows.append(r)
    return rows

def load_tab(tab_dir, files):
    rows = []
    for fname in files:
        fp = os.path.join(tab_dir, fname)
        if not os.path.exists(fp): continue
        for doc in json.load(open(fp)):
            text = doc["text"]
            spans = []
            for ann, layer in doc.get("annotations", {}).items():
                for em in layer.get("entity_mentions", []):
                    if em.get("identifier_type") == "NO_MASK": continue
                    s, e = em["start_offset"], em["end_offset"]
                    if 0 <= s < e <= len(text):
                        spans.append({"start": s, "end": e, "type": em.get("entity_type", "?")})
            spans = merge(remap(spans))
            if text and spans:
                rows.append({"text": text, "spans": spans, "source": "tab", "language": "en"})
    return rows

outdir = "/scratch/pii_corpus/piibench_unified"
os.makedirs(outdir, exist_ok=True)

train = load_ai4p("/scratch/pii_corpus/eu_shards", "train") + load_tab("/scratch/pii_corpus/benchmarks/tab", ["echr_train.json"])
val = load_ai4p("/scratch/pii_corpus/eu_shards", "val") + load_tab("/scratch/pii_corpus/benchmarks/tab", ["echr_test.json", "echr_dev.json"])

# contamination check
def fp(text): return hashlib.md5(text[:300].encode()).hexdigest()
train_fps = set(fp(r["text"]) for r in train)
overlap = sum(1 for r in val if fp(r["text"]) in train_fps)
if overlap:
    print(f"Contamination: {overlap} val rows in train — removing")
    val = [r for r in val if fp(r["text"]) not in train_fps]
else:
    print("✅ ZERO contamination")

random.seed(42); random.shuffle(train); random.shuffle(val)
with open(os.path.join(outdir, "train.jsonl"), "w") as f:
    for r in train: f.write(json.dumps(r) + "\n")
with open(os.path.join(outdir, "eval.jsonl"), "w") as f:
    for r in val: f.write(json.dumps(r) + "\n")
json.dump(TYPES, open(os.path.join(outdir, "types.json"), "w"))

from collections import Counter
tc = Counter()
for r in train:
    for s in r["spans"]: tc[s["type"]] += 1
print(f"Train: {len(train)} | Eval: {len(val)}")
print(f"Types: {len(TYPES)} (PIIBench canonical)")
print("Distribution:")
for t, c in tc.most_common(): print(f"  {t:20s} {c:>8d}")
