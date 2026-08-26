#!/usr/bin/env python3
"""PIIBench cross-domain evaluation — v3 (final).

Gold grid: subword tokens + BIO labels (test.jsonl, mBERT-style with ##).
Reconstruct gold char spans by decoding prefixes with bert-base-multilingual-cased
(the standard ##-detokenizer), which reproduces the original text spacing.
Model input: the record's real `text`. Compare char spans: exact + overlap,
type-agnostic; plus type-aware via canonical mapping.

Usage: python eval_piibench_v3.py [max_per_source]
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


def gold_char_spans(tokens, labels, bert):
    """decode prefix lengths for char offsets."""
    n = len(tokens)
    # authoritative: decode(["<s>"] + tokens) - decode(["<s>"]) ... use prefix decode
    prefix_lens = []
    L = 0
    for i in range(n):
        L = len(bert.decode(tokens[:i + 1]))
        prefix_lens.append(L)
    out, cur = [], None
    for i, lab in enumerate(labels[:n]):
        if lab == "O":
            if cur:
                out.append(cur)
            cur = None
        elif lab.startswith("B-"):
            if cur:
                out.append(cur)
            cur = (prefix_lens[i - 1] if i > 0 else 0, prefix_lens[i], lab[2:])
        elif lab.startswith("I-") and cur:
            cur = (cur[0], prefix_lens[i], cur[2])
        else:
            if cur:
                out.append(cur)
            cur = None
    if cur:
        out.append(cur)
    s = 0 if not out else None
    return out


def pred_char_spans(tags, offs):
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
    recs, seen = [], collections.Counter()
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

    stat = {"g": 0, "p": 0, "tp": 0, "gh": 0, "ph": 0, "tm": 0}
    by_src = collections.defaultdict(lambda: {"g": 0, "p": 0, "tp": 0, "gh": 0, "ph": 0, "tm": 0})
    bad_recon = 0

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
                try:
                    gp = gold_char_spans(r["tokens"], r["labels"], bert)
                except Exception:
                    bad_recon += 1
                    continue
                if not gp:
                    continue
                enc1 = tok(text, truncation=True, max_length=SEQ, return_offsets_mapping=True)
                tags = [id2tag.get(x, "O") for x in preds[k][:len(enc1["offset_mapping"])].tolist()]
                pp = pred_char_spans(tags, enc1["offset_mapping"])
                pp = [(a, b, t) for a, b, t in pp if b > a]
                gd = {(s_, e_) for s_, e_, _ in gp}
                pd = {(a, b) for a, b, _ in pp}
                gh = sum(1 for g in gd if any(g[0] < p[1] and p[0] < g[1] for p in pd))
                ph = sum(1 for p in pd if any(g[0] < p[1] and p[0] < g[1] for g in gd))
                tm = 0
                for gs_, ge_, gt_ in gp:
                    ok = MAP.get(gt_.upper(), set())
                    for a, b, t_ in pp:
                        if (gs_, ge_) == (a, b) and t_ in ok:
                            tm += 1
                            break
                src = r.get("source", "?")
                for c in (stat, by_src[src]):
                    c["g"] += len(gd)
                    c["p"] += len(pd)
                    c["tp"] += len(gd & pd)
                    c["gh"] += gh
                    c["ph"] += ph
                    c["tm"] += tm

    def rep(c):
        p_ = c["tp"] / max(1, c["p"])
        r_ = c["tp"] / max(1, c["g"])
        f = 2 * p_ * r_ / max(1e-9, p_ + r_)
        p2 = c["ph"] / max(1, c["p"])
        r2 = c["gh"] / max(1, c["g"])
        o = 2 * p2 * r2 / max(1e-9, p2 + r2)
        return f, p_, r_, o, p2, r2, c["tm"] / max(1, c["g"])

    print(f"recon failed rows: {bad_recon}", flush=True)
    f, p_, r_, o, p2, r2, tm = rep(stat)
    print(f"\n=== PIIBench CROSS-DOMAIN (char spans via ##-decode) ===")
    print(f"OVERALL  exactF1={f:.4f} (P={p_:.4f} R={r_:.4f}) | overlapF1={o:.4f} (P={p2:.4f} R={r2:.4f}) | typeM={tm:.4f} | support={stat['g']}")
    for s in sorted(by_src):
        f, p_, r_, o, p2, r2, tm = rep(by_src[s])
        print(f"  {s:<22s} exact={f:.4f} ovl={o:.4f} typeM={tm:.3f} n={by_src[s]['g']}")


if __name__ == "__main__":
    main()
