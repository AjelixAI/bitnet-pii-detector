#!/usr/bin/env python3
"""Presidio vs ours — char-level DETECTION F1 on the same stratified multilingual gold set.
Presidio via default AnalyzerEngine (spacy en_core_web_sm + regex patterns + context).
Ours via saved best model, char spans recovered from token runs.
Type-agnostic: a match = exact (start, end) char span of ANY PII.
Usage: python bench_presidio.py [n_per_lang]
"""
import collections
import sys
import time

import pyarrow.parquet as pq
import torch
from datasets import Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer
from presidio_analyzer import AnalyzerEngine

PER_LANG = int(sys.argv[1]) if len(sys.argv) > 1 else 300
SEQ = 256
BATCH = 128
OURS = "/root/piiranha_hf/best"

PII_TYPES = ["ACCOUNTNUM", "BUILDINGNUM", "CITY", "CREDITCARDNUMBER", "DATEOFBIRTH",
             "DRIVERLICENSENUM", "EMAIL", "GIVENNAME", "IDCARDNUM", "PASSWORD",
             "SOCIALNUM", "STREET", "SURNAME", "TAXNUM", "TELEPHONENUM",
             "USERNAME", "ZIPCODE"]


def f1_stats(c):
    p = c["tp"] / max(1, c["p"])
    r = c["tp"] / max(1, c["g"])
    f = 2 * p * r / max(1e-9, p + r)
    return f, p, r


def overlap_tp(golds, preds):
    """gold spans hit by any pred (recall side) and preds hitting any gold (prec side)."""
    hit_g = sum(1 for g in golds if any(g[0] < p[1] and p[0] < g[1] for p in preds))
    hit_p = sum(1 for p in preds if any(g[0] < p[1] and p[0] < g[1] for g in golds))
    return hit_g, hit_p


def io_spans_to_chars(tags, offs):
    """I-only runs -> char spans [(start, end)]."""
    out, cur = [], None
    for ti, (a, b) in enumerate(offs):
        if a == b or b == 0:
            if cur:
                out.append((cur[0], cur[1]))
            cur = None
            continue
        tag = tags[ti]
        if tag == "O" or tag is None:
            if cur:
                out.append((cur[0], cur[1]))
            cur = None
            continue
        if cur and tag.startswith("I-") and ti == cur[2] + 1:
            cur = (cur[0], b, ti)
        else:
            if cur:
                out.append((cur[0], cur[1]))
            cur = (a, b, ti)
    if cur:
        out.append((cur[0], cur[1]))
    return out


def main():
    va = Dataset(pq.read_table("/root/data/pii400k_val.parquet").replace_schema_metadata({}))
    by_lang = collections.defaultdict(list)
    for i, lg in enumerate(va["language"]):
        by_lang[lg].append(i)
    sel = []
    for lg in sorted(by_lang):
        sel += by_lang[lg][:PER_LANG]
    print(f"rows: {len(sel)} ({PER_LANG} x {len(by_lang)} langs)", flush=True)

    golds = []
    for i in sel:
        r = va[i]
        spans = [(int(s["start"]), int(s["end"])) for s in r["privacy_mask"]]
        golds.append(spans)

    results = {}

    # --- Presidio ---
    engine = AnalyzerEngine()
    stat = {"g": 0, "p": 0, "tp": 0, "oi": 0}
    plang = collections.defaultdict(lambda: {"g": 0, "p": 0, "tp": 0, "oi": 0})
    t0 = time.time()
    for k, i in enumerate(sel):
        text = va[i]["source_text"]
        gd = set(golds[k])
        found = engine.analyze(text=text, language="en")
        pd = {(r.start, r.end) for r in found}
        hg, hp = overlap_tp(golds[k], pd)
        lg = va[i]["language"]
        for c in (stat, plang[lg]):
            c["g"] += len(gd)
            c["p"] += len(pd)
            c["tp"] += len(gd & pd)
            c["oi"] += max(hg, hp)
    results["presidio"] = (stat, plang)
    print(f"presidio done in {(time.time()-t0)/60:.1f} min", flush=True)

    # --- ours ---
    tok = AutoTokenizer.from_pretrained(OURS)
    model = AutoModelForTokenClassification.from_pretrained(OURS).cuda().eval()
    id2tag = {int(k): v for k, v in model.config.id2label.items()}
    stat = {"g": 0, "p": 0, "tp": 0, "oi": 0}
    plang = collections.defaultdict(lambda: {"g": 0, "p": 0, "tp": 0, "oi": 0})
    with torch.no_grad():
        for bi in range(0, len(sel), BATCH):
            chunk = sel[bi:bi + BATCH]
            texts = [va[i]["source_text"] for i in chunk]
            enc = tok(texts, padding=True, truncation=True, max_length=SEQ)
            ids = torch.tensor(enc["input_ids"]).cuda()
            attn = torch.tensor(enc["attention_mask"]).cuda()
            logits = model(input_ids=ids, attention_mask=attn).logits.cpu()
            preds = logits.argmax(-1)
            for k, j in enumerate(chunk):
                text = texts[k]
                enc1 = tok(text, truncation=True, max_length=SEQ, return_offsets_mapping=True)
                offs = enc1["offset_mapping"]
                tags = [id2tag.get(x, "O") for x in preds[k][:len(offs)].tolist()]
                pd = set()
                for a, b in io_spans_to_chars(tags, offs):
                    while a < b and text[a].isspace():
                        a += 1
                    while b > a and text[b - 1].isspace():
                        b -= 1
                    if b > a:
                        pd.add((a, b))
                gd = set(golds[bi + k])
                hg, hp = overlap_tp(golds[bi + k], list(pd))
                lg = va[j]["language"]
                for c in (stat, plang[lg]):
                    c["g"] += len(gd)
                    c["p"] += len(pd)
                    c["tp"] += len(gd & pd)
                    c["oi"] += max(hg, hp)
    results["ours"] = (stat, plang)

    print("\n=== DETECTION F1 (any PII, exact char span) ===")
    for name, (stat, plang) in results.items():
        f, p, r = f1_stats(stat)
        print(f"  {name:<10s} detF1={f:.4f}  P={p:.4f}  R={r:.4f}  support={stat['g']}")
    print("\n=== OVERLAP-BASED DETECTION F1 (gold hit / pred hit) ===")
    for name, (stat, plang) in results.items():
        h = stat["oi"] / max(1, stat["g"])
        print(f"  {name:<10s} goldRec={h:.4f}")
    print("\n  per-language:")
    for name, (stat, plang) in results.items():
        parts = []
        for lg in sorted(plang):
            f, _, _ = f1_stats(plang[lg])
            parts.append(f"{lg}={f:.3f}")
        print(f"  {name:<10s} " + " ".join(parts))


if __name__ == "__main__":
    main()
