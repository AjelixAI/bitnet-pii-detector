#!/usr/bin/env python3
"""Cross-domain evaluation on PIIBench (pritesh-2711/pii-bench) prepared test split.

Reads data/test.jsonl records (tokens, labels[BIO canonical], source[, text]).
Reconstructs text deterministically as ' '.join(tokens); gold spans = B-X..I-X token
runs mapped to char offsets. Runs OUR model; compares:
  - exact-char detection F1 (type-agnostic)
  - overlap-based gold recall / pred precision
  - type-aware F1 with canonical->our-type mapping (where mapping exists)
Per-source breakdown. Usage: python eval_piibench.py [n_per_source_max]
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
O_IDX = len(PII_TYPES)

# canonical (PIIBench) type -> acceptable our-types (type-aware match)
MAP = {
    "PERSON": {"GIVENNAME", "SURNAME", "PERSON"},
    "PER": {"GIVENNAME", "SURNAME", "PERSON"},
    "GPE": {"CITY", "LOCATION"},
    "LOC": {"CITY", "STREET", "LOCATION"},
    "LOCATION": {"CITY", "STREET", "BUILDINGNUM", "ZIPCODE", "LOCATION"},
    "STREET": {"STREET", "CITY"},
    "CITY": {"CITY", "STREET"},
    "ZIP": {"ZIPCODE"},
    "ZIPCODE": {"ZIPCODE"},
    "EMAIL": {"EMAIL"},
    "EMAIL_ADDRESS": {"EMAIL"},
    "PHONE_NUMBER": {"TELEPHONENUM"},
    "TELEPHONE": {"TELEPHONENUM"},
    "PHONE": {"TELEPHONENUM"},
    "CREDIT_CARD": {"CREDITCARDNUMBER"},
    "CREDITCARD": {"CREDITCARDNUMBER"},
    "CREDIT_CARD_NUMBER": {"CREDITCARDNUMBER"},
    "USERNAME": {"USERNAME"},
    "PASSWORD": {"PASSWORD"},
    "DATE_OF_BIRTH": {"DATEOFBIRTH"},
    "DOB": {"DATEOFBIRTH"},
    "BIRTHDATE": {"DATEOFBIRTH"},
    "ID": {"IDCARDNUM", "SOCIALNUM", "TAXNUM", "DRIVERLICENSENUM", "ACCOUNTNUM"},
    "ID_NUMBER": {"IDCARDNUM", "SOCIALNUM", "TAXNUM", "DRIVERLICENSENUM", "ACCOUNTNUM"},
    "SSN": {"SOCIALNUM"},
    "SOCIAL_SECURITY_NUMBER": {"SOCIALNUM"},
    "TAX_ID": {"TAXNUM"},
    "DRIVER_LICENSE": {"DRIVERLICENSENUM"},
    "ACCOUNT_NUMBER": {"ACCOUNTNUM"},
    "IBAN": {"ACCOUNTNUM"},
    "ORG": set(),
    "ORGANIZATION": set(),
    "ADDRESS": {"STREET", "CITY", "BUILDINGNUM", "ZIPCODE"},
    "STREET_ADDRESS": {"STREET", "CITY", "BUILDINGNUM", "ZIPCODE"},
    "NAME": {"GIVENNAME", "SURNAME"},
    "COMPANY_NAME": set(),
    "FINANCIAL_ENTITY": {"ACCOUNTNUM"},
    "CRYPTO_ADDRESS": set(),
    "IP_ADDRESS": set(),
    "URL": set(),
    "TIME": set(),
    "DATE": set(),
    "MISC": set(),
}


def gold_spans(tokens, labels):
    """BIO token runs -> [(start_char, end_char, type)] over ' '.join(tokens)."""
    spans, cur, pos = [], None, 0
    for tok, lab in zip(tokens, labels):
        tlen = len(tok)
        if lab == "O" or lab.startswith("I-") and cur is None:
            if cur:
                spans.append(cur)
            cur = None
        elif lab.startswith("B-"):
            if cur:
                spans.append(cur)
            cur = (pos, pos + tlen, lab[2:])
        elif lab.startswith("I-") and cur:
            cur = (cur[0], pos + tlen, cur[2])
        else:
            if cur:
                spans.append(cur)
            cur = None
        pos += tlen + 1
    if cur:
        spans.append(cur)
    return spans


def pred_spans(tags, offs):
    """I-only runs -> [(start, end, our_type)] char spans (untrimmed)."""
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
    per_source = collections.Counter()
    with open(TEST) as f:
        for line in f:
            r = json.loads(line)
            per_source[r.get("source", "?")] += 1
    # sample cap per source
    sel, seen = [], collections.Counter()
    for line in open(TEST):
        r = json.loads(line)
        s = r.get("source", "?")
        if seen[s] >= MAX_PER_SOURCE:
            continue
        seen[s] += 1
        tokens, labels = r["tokens"], r["labels"]
        if not tokens or not labels:
            continue
        sel.append((tokens, labels, s))
    print(f"test records: {len(sel)} across {len(per_source)} sources", flush=True)
    for s, c in per_source.most_common():
        print(f"  {s}: {c}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForTokenClassification.from_pretrained(MODEL).cuda().eval()
    id2tag = {int(k): v for k, v in model.config.id2label.items()}

    stat = {"g": 0, "p": 0, "tp": 0, "ghit": 0, "phit": 0, "tmatch": 0}
    by_src = collections.defaultdict(lambda: {"g": 0, "p": 0, "tp": 0, "ghit": 0, "phit": 0, "tmatch": 0})

    with torch.no_grad():
        for bi in range(0, len(sel), BATCH):
            chunk = sel[bi:bi + BATCH]
            texts = [" ".join(t) for t, _, _ in chunk]
            enc = tok(texts, padding=True, truncation=True, max_length=SEQ)
            ids = torch.tensor(enc["input_ids"]).cuda()
            attn = torch.tensor(enc["attention_mask"]).cuda()
            logits = model(input_ids=ids, attention_mask=attn).logits.cpu()
            preds = logits.argmax(-1)
            for k, (tokens, labels, src) in enumerate(chunk):
                text = texts[k]
                gp = gold_spans(tokens, labels)  # [(s,e,type)]
                if not gp:
                    continue
                enc1 = tok(text, truncation=True, max_length=SEQ, return_offsets_mapping=True)
                offs = enc1["offset_mapping"]
                tags = [id2tag.get(x, "O") for x in preds[k][:len(offs)].tolist()]
                pp = pred_spans(tags, offs)
                pp = []
                for a, b, t in pred_spans(tags, offs):
                    while a < b and text[a].isspace():
                        a += 1
                    while b > a and text[b - 1].isspace():
                        b -= 1
                    if b > a:
                        pp.append((a, b, t))
                gd = {(s_, e_) for s_, e_, _ in gp}
                pd = {(a, b) for a, b, _ in pp}
                # exact + overlap
                tg = len(gd & pd)
                ghit = sum(1 for g in gd if any(g[0] < p[1] and p[0] < g[1] for p in pd))
                phit = sum(1 for p in pd if any(g[0] < p[1] and p[0] < g[1] for g in gd))
                # type-aware: gold span overlapping pred with mapped type
                tmatch = 0
                for gs_, ge_, gt_ in gp:
                    ok_types = MAP.get(gt_.upper(), set())
                    for a, b, t_ in pp:
                        if (gs_, ge_) == (a, b):
                            if t_ in ok_types:
                                tmatch += 1
                            break
                for c in (stat, by_src[src]):
                    c["g"] += len(gd)
                    c["p"] += len(pd)
                    c["tp"] += tg
                    c["ghit"] += ghit
                    c["phit"] += phit
                    c["tmatch"] += tmatch

    def rep(c):
        p_, r_ = c["tp"] / max(1, c["p"]), c["tp"] / max(1, c["g"])
        e = 2 * p_ * r_ / max(1e-9, p_ + r_)
        p2, r2 = c["phit"] / max(1, c["p"]), c["ghit"] / max(1, c["g"])
        o = 2 * p2 * r2 / max(1e-9, p2 + r2)
        tm = c["tmatch"] / max(1, c["g"])
        return e, o, p_, r_, p2, r2, tm

    print("\n=== PIIBench CROSS-DOMAIN (test split) ===")
    e, o, p_, r_, p2, r2, tm = rep(stat)
    print(f"OVERALL  exact-detF1={e:.4f} (P={p_:.4f} R={r_:.4f}) | overlapF1={o:.4f} (P={p2:.4f} R={r2:.4f}) | typeMatch={tm:.4f} | support={stat['g']}")
    print("  per source:")
    for s in sorted(by_src):
        e, o, p_, r_, p2, r2, tm = rep(by_src[s])
        print(f"  {s:<34s} exact={e:.4f} overlap={o:.4f} typeM={tm:.3f} n={by_src[s]['g']}")


if __name__ == "__main__":
    main()
