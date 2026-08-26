#!/usr/bin/env python3
"""Head-to-head: our mdeberta fine-tune vs LiquidAI lfm_pii_detector.
Same 12K stratified multilingual gold rows; each model tokenizes with its own
tokenizer; detection F1 = exact token-run match on ANY-PII spans (type-agnostic),
so taxonomy differences don't distort the comparison.
Usage: python compare_lfm.py
"""
import collections
import sys

import pyarrow.parquet as pq
import torch
from datasets import Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer

SEQ = 256
BATCH = 128
PER_LANG = 2000
THIS = {"name": "ours-mdeberta", "path": "/root/piiranha_hf/best", "scheme": "io"}
LFM = {"name": "lfm_pii_detector", "path": "/root/models/lfm_pii_detector", "scheme": "bioes"}

PII_TYPES = ["ACCOUNTNUM", "BUILDINGNUM", "CITY", "CREDITCARDNUMBER", "DATEOFBIRTH",
             "DRIVERLICENSENUM", "EMAIL", "GIVENNAME", "IDCARDNUM", "PASSWORD",
             "SOCIALNUM", "STREET", "SURNAME", "TAXNUM", "TELEPHONENUM",
             "USERNAME", "ZIPCODE"]
O_IDX = len(PII_TYPES)


def io_runs(labels, offs):
    out, cur = [], None
    for ti, (a, b) in enumerate(offs):
        if a == b or b == 0:
            if cur:
                out.append(cur)
            cur = None
            continue
        p = labels[ti]
        if p == O_IDX or p == "O":
            if cur:
                out.append(cur)
            cur = None
            continue
        t = p
        if cur and cur[2] == t and ti == cur[1] + 1:
            cur = (cur[0], ti, t)
        else:
            if cur:
                out.append(cur)
            cur = (ti, ti, t)
    if cur:
        out.append(cur)
    return out


def bioes_runs(labels, offs):
    """B-X I-X* E-X | S-X ; O otherwise. labels: dict id->(prefix,type) or 'O'."""
    out, cur = [], None
    for ti, (a, b) in enumerate(offs):
        if a == b or b == 0:
            if cur:
                out.append(cur)
            cur = None
            continue
        tag = labels[ti]
        if tag == "O":
            if cur:
                out.append(cur)
            cur = None
            continue
        pre, t = tag.split("-", 1)
        if pre == "S":
            if cur:
                out.append(cur)
            cur = (ti, ti, -1)
            out.append(cur)
            cur = None
        elif pre == "B":
            if cur:
                out.append(cur)
            cur = (ti, ti, -1)
        elif pre == "I":
            if cur and ti == cur[1] + 1:
                cur = (cur[0], ti, -1)
            else:
                if cur:
                    out.append(cur)
                cur = (ti, ti, -1)
        elif pre == "E":
            if cur and ti == cur[1] + 1:
                cur = (cur[0], ti, -1)
            if cur:
                out.append(cur)
            cur = None
        else:
            if cur:
                out.append(cur)
            cur = None
    if cur:
        out.append(cur)
    return out


def decode_scheme(scheme):
    return io_runs if scheme == "io" else bioes_runs


def main():
    va = Dataset(pq.read_table("/root/data/pii400k_val.parquet").replace_schema_metadata({}))
    by_lang = collections.defaultdict(list)
    for i, lg in enumerate(va["language"]):
        by_lang[lg].append(i)
    sel = []
    for lg in sorted(by_lang):
        sel += by_lang[lg][:PER_LANG]
    print(f"gold rows: {len(sel)}", flush=True)

    results = {}
    for cfg in (THIS, LFM):
        tok = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True)
        model = AutoModelForTokenClassification.from_pretrained(
            cfg["path"], trust_remote_code=True).cuda().eval()
        cfg["id2tag"] = {int(k): v for k, v in model.config.id2label.items()}
        det = {"g": 0, "p": 0, "tp": 0}
        per_lang = collections.defaultdict(lambda: {"g": 0, "p": 0, "tp": 0})
        # gold spans per row, per this tokenizer
        gold_runs = []
        for i in sel:
            r = va[i]
            spans = [(int(s["start"]), int(s["end"])) for s in r["privacy_mask"]]
            enc = tok(r["source_text"], truncation=True, max_length=SEQ,
                      return_offsets_mapping=True)
            offs = enc["offset_mapping"]
            runs = []
            for ti, (a, b) in enumerate(offs):
                if a == b:
                    continue
                for s_, e_ in spans:
                    if not (b <= s_ or a >= e_):
                        runs.append(ti)
                        break
            # contiguous token runs of PII → gold spans
            grs = []
            for ti in runs:
                if grs and ti == grs[-1][1] + 1:
                    grs[-1] = (grs[-1][0], ti)
                else:
                    grs.append((ti, ti))
            gold_runs.append(grs)
        with torch.no_grad():
            for bi in range(0, len(sel), BATCH):
                chunk = sel[bi:bi + BATCH]
                enc = tok([va[i]["source_text"] for i in chunk], padding=True,
                          truncation=True, max_length=SEQ)
                ids = torch.tensor(enc["input_ids"]).cuda()
                attn = torch.tensor(enc["attention_mask"]).cuda()
                logits = model(input_ids=ids, attention_mask=attn).logits.cpu()
                preds = logits.argmax(-1)
                for k, i in enumerate(chunk):
                    text = va[i]["source_text"]
                    enc1 = tok(text, truncation=True, max_length=SEQ,
                               return_offsets_mapping=True)
                    offs = enc1["offset_mapping"]
                    pk = preds[k][:len(offs)]
                    tags = [cfg["id2tag"].get(x, "O") for x in pk.tolist()]
                    pruns = decode_scheme(cfg["scheme"])(tags, offs)
                    pd = {(a, b) for a, b, _ in pruns}
                    gd = {(a, b) for a, b in gold_runs[bi + k]}
                    lg = va[i]["language"]
                    for c in (det, per_lang[lg]):
                        c["g"] += len(gd)
                        c["p"] += len(pd)
                        c["tp"] += len(gd & pd)
        results[cfg["name"]] = (det, per_lang)

    print("\n=== DETECTION F1 (any PII, exact token-run) — 12000 rows, 6 langs ===")
    for name, (det, per_lang) in results.items():
        p = det["tp"] / max(1, det["p"])
        r = det["tp"] / max(1, det["g"])
        f = 2 * p * r / max(1e-9, p + r)
        print(f"  {name:<20s} detF1={f:.4f}  P={p:.4f}  R={r:.4f}  support={det['g']}")
    print("\n  per-language:")
    for name, (det, per_lang) in results.items():
        parts = []
        for lg in sorted(per_lang):
            c = per_lang[lg]
            p = c["tp"] / max(1, c["p"])
            r = c["tp"] / max(1, c["g"])
            f = 2 * p * r / max(1e-9, p + r)
            parts.append(f"{lg}={f:.3f}")
        print(f"  {name:<20s} " + " ".join(parts))


if __name__ == "__main__":
    main()
