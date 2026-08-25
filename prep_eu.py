#!/usr/bin/env python3
# prep_eu.py — stream-shard ai4privacy/openpii-1m (3.7GB jsonl) into per-language
# token-classification JSONLs compatible with bioes_pii.py ({text, spans:[{start,end,type}]}).
#
# Why per-language: fairness (per-language caps) + per-language eval + lets us exclude or
# upweight languages at train time without re-sharding.
import json, os, collections, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/scratch/pii_corpus/openpii_1m/data/train.jsonl")
ap.add_argument("--val-src", default="/scratch/pii_corpus/openpii_1m/data/validation.jsonl")
ap.add_argument("--outdir", default="/scratch/pii_corpus/eu_shards")
ap.add_argument("--cap-per-lang", type=int, default=0, help="0 = no cap")
ap.add_argument("--min-span-chars", type=int, default=1)
args = ap.parse_args()

os.makedirs(args.outdir, exist_ok=True)
LABELS = collections.Counter()
LANG = collections.Counter()
BAD = collections.Counter()

def normalize(row):
    """openpii row -> {text, spans, language, region}. Verify offsets."""
    text = row["source_text"]
    spans = []
    for s in row.get("privacy_mask", []):
        st, en = s["start"], s["end"]
        # integrity: must slice back to the value exactly (verified 5220/5220 on sample)
        if text[st:en] != s["value"]:
            BAD[row.get("language", "?")] += 1
            continue  # drop misaligned span entirely rather than train on wrong bounds
        if en - st < args.min_span_chars:
            continue
        spans.append({"start": st, "end": en, "type": s["label"]})
    return {
        "text": text,
        "spans": spans,
        "language": row.get("language", "?"),
        "region": row.get("region", "?"),
        "uid": row.get("uid", ""),
    }

def shard(src, suffix):
    writers = {}
    n = 0
    with open(src) as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec = normalize(row)
            lg = rec["language"]
            LANG[(lg, suffix)] += 1
            for s in rec["spans"]:
                LABELS[s["type"]] += 1
            if lg not in writers:
                writers[lg] = open(os.path.join(args.outdir, f"{suffix}_{lg}.jsonl"), "w")
            writers[lg].write(json.dumps(rec) + "\n")
            n += 1
    for w in writers.values():
        w.close()
    return n

n_tr = shard(args.src, "eu-train")
n_va = shard(args.val_src, "eu-val")
print("total written:", n_tr, n_va)
print("languages:")
for (lg, suf), c in sorted(LANG.items()):
    print("  ", suf, lg, c)
print("top labels:", LABELS.most_common())
print("dropped misaligned/short spans per lang:", dict(BAD))
with open(os.path.join(args.outdir, "stats.json"), "w") as f:
    json.dump({
        "languages": {f"{a}|{b}": c for (a, b), c in LANG.items()},
        "labels": dict(LABELS), "bad_offsets": dict(BAD),
    }, f, indent=2)
