#!/usr/bin/env python3
"""PIIBench cross-domain evaluation — v2.

Their grid: mBERT subword tokens + BIO labels (per test.jsonl).
Our grid: DeBERTa spans over the original text.
Bridge: tokenize text with bert-base-multilingual-cased + offset_mapping to get
char ranges per grid token; gold runs = their labels on that grid (verify same
tokenization); pred runs = our model's char spans projected onto grid indices.
Compare exact token-run match (seqeval-style), type-agnostic + type-aware.

Usage: python eval_piibench.py [max_per_source]
"""
import collections
import json
import sys

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

MODEL = "/root/piiranha_hf/best"
TEST = "/root/pii-bench/data/test.jsonl"
SEQ = 256
BATCH = 128
MAX_PER_SOURCE = int(sys.argv[1]) if len(sys.argv) > 1 else 10**9

PII_TYPES = ["ACCOUNTNUM", "BUILDINGNUM", "CITY", "CREDITCARDNUMBER", "DATEOFBIRTH",
             "DRIVERLICENSENUM", "EMAIL", "GIVENNAME", "IDCARDNUM", "PASSWORD",
             "SOCIALNUM", "STREET", "SURNAME", "TAXNUM", "TELEPHONENUM",
             "USERNAME", "ZIPCODE"]

MAP = {
    "PERSON": {"GIVENNAME", "SURNAME"}, "PER": {"GIVENNAME", "SURNAME"},
    "NAME": {"GIVENNAME", "SURNAME"},
    "GPE": {"CITY"}, "LOC": {"CITY", "STREET"},
    "LOCATION": {"CITY", "STREET", "BUILDINGNUM", "ZIPCODE"},
    "ADDRESS": {"STREET", "CITY", "BUILDINGNUM", "ZIPCODE"},
    "STREET_ADDRESS": {"STREET", "CITY", "BUILDINGNUM", "ZIPCODE"},
    "STREET": {"STREET", "CITY"}, "CITY": {"CITY", "STREET"},
    "ZIP": {"ZIPCODE"}, "ZIPCODE": {"ZIPCODE"},
    "EMAIL": {"EMAIL"}, "EMAIL_ADDRESS": {"EMAIL"},
    "PHONE_NUMBER": {"TELEPHONENUM"}, "PHONE": {"TELEPHONENUM"},
    "TELEPHONE": {"TELEPHONENUM"},
    "CREDIT_CARD": {"CREDITCARDNUMBER"}, "CREDITCARD": {"CREDITCARDNUMBER"},
    "CREDIT_CARD_NUMBER": {"CREDITCARDNUMBER"},
    "USERNAME": {"USERNAME"}, "PASSWORD": {"PASSWORD"},
    "DATE_OF_BIRTH": {"DATEOFBIRTH"}, "DOB": {"DATEOFBIRTH"}, "BIRTHDATE": {"DATEOFBIRTH"},
    "ID": {"IDCARDNUM", "SOCIALNUM", "TAXNUM", "DRIVERLICENSENUM", "ACCOUNTNUM"},
    "ID_NUMBER": {"IDCARDNUM", "SOCIALNUM", "TAXNUM", "DRIVERLICENSENUM", "ACCOUNTNUM"},
    "SSN": {"SOCIALNUM"}, "SOCIAL_SECURITY_NUMBER": {"SOCIALNUM"},
    "TAX_ID": {"TAXNUM"}, "DRIVER_LICENSE": {"DRIVERLICENSENUM"},
    "ACCOUNT_NUMBER": {"ACCOUNTNUM"}, "IBAN": {"ACCOUNTNUM"},
    "FINANCIAL_ENTITY": {"ACCOUNTNUM"},
    "COMPANY_NAME": set(), "ORG": set(), "ORGANIZATION": set(),
    "MISC": set(), "DATE": set(), "TIME": set(), "URL": set(),
    "IP_ADDRESS": set(), "CRYPTO_ADDRESS": set(),
}


def gold_runs(labels):
    """token-index runs (start, end, type) from BIO labels."""
    out, cur = [], None
    for i, lab in enumerate(labels):
        if lab == "O":
            if cur:
                out.append(cur)
            cur = None
        elif lab.startswith("B-"):
            if cur:
                out.append(cur)
            cur = (i, i, lab[2:])
        elif lab.startswith("I-") and cur:
            cur = (cur[0], i, cur[2])
        else:
            if cur:
                out.append(cur)
            cur = None
    if cur:
        out.append(cur)
    return out


def pred_runs(tags, offs):
    """DeBERTa I-only runs -> char spans then grid mapping? return char spans."""
    out, cur = [], None
    for ti, (a, b) in enumerate(offs):
        if a == b or b == 0:
            if cur:
                out.append(cur)
            cur = None
            continue
        tag = tags[ti]
        if tag == "O":
            if cur:
                out.append(cur)
            cur = None
            continue
        t = tag[2:]
        if cur and cur[2] == t and ti == cur[1] + 1:
            cur = (cur[0], b, t)
        else:
            if cur:
                out.append(cur)
            cur = (a, b, t)
    if cur:
        out.append(cur)
    return out


def main():
    recs = []
    seen = collections.Counter()
    for line in open(TEST):
        r = json.loads(line)
        s = r.get("source", "?")
        if seen[s] >= MAX_PER_SOURCE:
            continue
        seen[s] += 1
        if r.get("tokens") and r.get("labels"):
            recs.append(r)
    print(f"records: {len(recs)}", flush=True)

    bert = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForTokenClassification.from_pretrained(MODEL).cuda().eval()
    id2tag = {int(k): v for k, v in model.config.id2label.items()}

    stat = {"g": 0, "p": 0, "tp": 0, "tm": 0}
    by_src = collections.defaultdict(lambda: {"g": 0, "p": 0, "tp": 0, "tm": 0})
    n_short = n_grid_ok = 0

    with torch.no_grad():
        for bi in range(0, len(recs), BATCH):
            chunk = recs[bi:bi + BATCH]
            texts = [r["text"] if r.get("text") else " ".join(r["tokens"]) for r in chunk]
            enc = tok(texts, padding=True, truncation=True, max_length=SEQ)
            ids = torch.tensor(enc["input_ids"]).cuda()
            attn = torch.tensor(enc["attention_mask"]).cuda()
            logits = model(input_ids=ids, attention_mask=attn).logits.cpu()
            preds = logits.argmax(-1)
            for k, r in enumerate(chunk):
                text = texts[k]
                gold_toks = r["tokens"]
                # their grid via BERT tokenizer
                b_enc = bert(text, truncation=True, max_length=SEQ, return_offsets_mapping=True, add_special_tokens=False)
                if len(b_enc["input_ids"]) < len(gold_toks):
                    # truncation: their grid longer -> skip (row truncated)
                    n_short += 1
                    continue
                if len(b_enc["input_ids"]) != len(gold_toks):
                    n_grid_ok += 1
                    continue
                gruns = gold_runs(r["labels"][:len(b_enc["input_ids"])])
                if not gruns:
                    continue
                # our char spans -> their grid indices
                offs = b_enc["offset_mapping"]
                enc1 = tok(text, truncation=True, max_length=SEQ, return_offsets_mapping=True)
                tags = [id2tag.get(x, "O") for x in preds[k][:len(enc1["offset_mapping"])].tolist()]
                cspans = pred_runs(tags, enc1["offset_mapping"])
                p_gold_runs = []
                for a, b, t in cspans:
                    idxs = [i for i, (s_, e_) in enumerate(offs) if s_ < b and e_ > a and s_ != e_]
                    if idxs:
                        p_gold_runs.append((idxs[0], idxs[-1], t))
                gd = {(a, b) for a, b, _ in gruns}
                pd = {(a, b) for a, b, _ in p_gold_runs}
                tp = len(gd & pd)
                tm = 0
                for ga, gb, gt in gruns:
                    ok = MAP.get(gt.upper(), set())
                    for pa, pb, pt in p_gold_runs:
                        if (ga, gb) == (pa, pb) and pt in ok:
                            tm += 1
                            break
                src = r.get("source", "?")
                for c in (stat, by_src[src]):
                    c["g"] += len(gd)
                    c["p"] += len(pd)
                    c["tp"] += tp
                    c["tm"] += tm

    def rep(c):
        p_ = c["tp"] / max(1, c["p"])
        r_ = c["tp"] / max(1, c["g"])
        f = 2 * p_ * r_ / max(1e-9, p_ + r_)
        return f, p_, r_, c["tm"] / max(1, c["g"])

    print(f"skipped: {n_short} truncated, {n_grid_ok} grid-mismatch", flush=True)
    f, p_, r_, tm = rep(stat)
    print(f"\n=== PIIBench CROSS-DOMAIN — token-run F1 (seqeval semantics) ===")
    print(f"OVERALL  F1={f:.4f}  P={p_:.4f}  R={r_:.4f}  typeMatch={tm:.4f}  support={stat['g']}")
    print("  per source:")
    for s in sorted(by_src):
        f, p_, r_, tm = rep(by_src[s])
        print(f"  {s:<22s} F1={f:.4f} P={p_:.3f} R={r_:.3f} typeM={tm:.3f} n={by_src[s]['g']}")


if __name__ == "__main__":
    main()
