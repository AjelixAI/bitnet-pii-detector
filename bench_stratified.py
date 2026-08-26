#!/usr/bin/env python3
"""Independent stratified benchmark: per-language + per-type span F1.
Usage: python bench_stratified.py /root/piiranha_hf/best [n_per_lang]
"""
import collections
import sys

import pyarrow.parquet as pq
import torch
from datasets import Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "/root/piiranha_hf/best"
PER_LANG = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
DATA = "/root/data/pii-masking-400k"
SEQ = 256
BATCH = 128

PII_TYPES = ["ACCOUNTNUM", "BUILDINGNUM", "CITY", "CREDITCARDNUMBER", "DATEOFBIRTH",
             "DRIVERLICENSENUM", "EMAIL", "GIVENNAME", "IDCARDNUM", "PASSWORD",
             "SOCIALNUM", "STREET", "SURNAME", "TAXNUM", "TELEPHONENUM",
             "USERNAME", "ZIPCODE"]
LABEL_NAMES = [f"I-{t}" for t in PII_TYPES] + ["O"]
O_IDX = LABEL_NAMES.index("O")
TYPE_IDX = {t: i for i, t in enumerate(PII_TYPES)}


def runs(labels, offs):
    out, cur = [], None
    for ti, (a, b) in enumerate(offs):
        if a == b or b == 0:
            if cur:
                out.append(cur)
            cur = None
            continue
        p = labels[ti]
        if p == O_IDX or p >= len(TYPE_IDX):
            if cur:
                out.append(cur)
            cur = None
            continue
        t = PII_TYPES[p]
        if cur and cur[2] == t and ti == cur[1] + 1:
            cur = (cur[0], ti, t)
        else:
            if cur:
                out.append(cur)
            cur = (ti, ti, t)
    if cur:
        out.append(cur)
    return out


def f1(c):
    p = c["tp"] / max(1, c["p"])
    r = c["tp"] / max(1, c["g"])
    return 2 * p * r / max(1e-9, p + r), p, r


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR).cuda().eval()
    print(f"model: {MODEL_DIR}", flush=True)

    va = Dataset(pq.read_table("/root/data/pii400k_val.parquet").replace_schema_metadata({}))
    by_lang = collections.defaultdict(list)
    for i, lg in enumerate(va["language"]):
        by_lang[lg].append(i)
    sel = []
    for lg in sorted(by_lang):
        sel += by_lang[lg][:PER_LANG]
    print(f"eval rows: {len(sel)} ( {PER_LANG} x {len(by_lang)} langs )", flush=True)

    # pre-tokenize
    items = []
    for i in sel:
        r = va[i]
        spans = [(int(s["start"]), int(s["end"]), s["label"])
                 for s in r["privacy_mask"] if s["label"] in TYPE_IDX]
        enc = tok(r["source_text"], truncation=True, max_length=SEQ,
                  return_offsets_mapping=True)
        items.append({"ids": enc["input_ids"], "offs": enc["offset_mapping"],
                      "offs_lbl": enc["offset_mapping"],
                      "lang": r.get("language", "?")})
        # gold labels from char spans
        lab = []
        for a, b in enc["offset_mapping"]:
            if a == b:
                lab.append(-100)
                continue
            tag = O_IDX
            for s_, e_, t_ in spans:
                if not (b <= s_ or a >= e_):
                    tag = TYPE_IDX[t_]
                    break
            lab.append(tag)
        items[-1]["labs"] = lab

    typed = collections.defaultdict(lambda: {"g": 0, "p": 0, "tp": 0})
    det = {"g": 0, "p": 0, "tp": 0}
    per_lang = collections.defaultdict(lambda: collections.defaultdict(lambda: {"g": 0, "p": 0, "tp": 0}))
    with torch.no_grad():
        for bi in range(0, len(items), BATCH):
            chunk = items[bi:bi + BATCH]
            mx = max(len(x["ids"]) for x in chunk)
            ids = torch.zeros(len(chunk), mx, dtype=torch.long)
            attn = torch.zeros(len(chunk), mx, dtype=torch.long)
            for k, x in enumerate(chunk):
                ids[k, :len(x["ids"])] = torch.tensor(x["ids"])
                attn[k, :len(x["ids"])] = 1
            logits = model(input_ids=ids.cuda(), attention_mask=attn.cuda()).logits.cpu()
            preds = logits.argmax(-1)
            for k, x in enumerate(chunk):
                L = len(x["ids"])
                g_runs = [(a, b, t) for a, b, t in runs(x["labs"][:L], x["offs_lbl"])]
                p_runs = runs(preds[k][:L].tolist(), x["offs"])
                gs, ps = set(g_runs), set(p_runs)
                for rg, rp, c in ((g_runs, p_runs, typed),):
                    pass
                # typed: per type
                for (ga, gb, gt) in g_runs:
                    typed_t = typed[gt]
                    typed_t["g"] += 1
                for (pa, pb, pt) in p_runs:
                    typed[pt]["p"] += 1
                for (ga, gb, gt) in g_runs:
                    for (pa, pb, pt) in p_runs:
                        if ga == pa and gb == pb and gt == pt:
                            typed[gt]["tp"] += 1
                            break
                # detection: any type
                gd = {(a, b) for a, b, _ in g_runs}
                pd = {(a, b) for a, b, _ in p_runs}
                det["g"] += len(gd)
                det["p"] += len(pd)
                det["tp"] += len(gd & pd)
                for (ga, gb, gt) in g_runs:
                    per_lang[x["lang"]][gt]["g"] += 1
                for (pa, pb, pt) in p_runs:
                    per_lang[x["lang"]][pt]["p"] += 1
                for (ga, gb, gt) in g_runs:
                    for (pa, pb, pt) in p_runs:
                        if ga == pa and gb == pb and gt == pt:
                            per_lang[x["lang"]][gt]["tp"] += 1
                            break

    print("\n=== DETECTION F1 (any PII, exact span) ===")
    df1, dp_, dr_ = f1(det)
    print(f"  detF1={df1:.4f}  P={dp_:.4f}  R={dr_:.4f}  (support={det['g']})")

    print("\n=== TYPE-AWARE SPAN F1 per type ===")
    for t in PII_TYPES:
        c = typed[t]
        if c["g"] == 0 and c["p"] == 0:
            continue
        f, p, r = f1(c)
        print(f"  {t:<18s} F1={f:.4f}  P={p:.3f}  R={r:.3f}  n={c['g']}")

    print("\n=== TYPE-AWARE F1 per language ===")
    for lg in sorted(per_lang):
        acc = {"g": 0, "p": 0, "tp": 0}
        for t in PII_TYPES:
            c = per_lang[lg][t]
            acc["g"] += c["g"]
            acc["p"] += c["p"]
            acc["tp"] += c["tp"]
        f, p, r = f1(acc)
        print(f"  {lg:<3s} F1={f:.4f}  P={p:.3f}  R={r:.3f}  n={acc['g']}")


if __name__ == "__main__":
    main()
