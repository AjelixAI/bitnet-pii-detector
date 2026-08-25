#!/usr/bin/env python3
# eval_eu_pii.py — honest span-level eval of the EuroBERT PII checkpoint on openpii val.
# Metrics: detection-tier F1 (found the span at all) + exact-type F1, per language and per type.
import argparse, json, glob, os, collections
import torch
from transformers import AutoTokenizer
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def decode_spans(logits_row, offsets, lmap_rev, seq_len, text):
    """BIO -> char spans. B starts a span, I extends same-label, O closes."""
    spans, cur = [], None
    for ti in range(min(seq_len, len(offsets))):
        a, b = offsets[ti]
        tag = lmap_rev.get(logits_row[ti], "O")
        if a == b or b == 0:  # special/pad token
            if cur:
                spans.append(cur); cur = None
            continue
        if tag.startswith("B-"):
            if cur: spans.append(cur)
            cur = {"start": a, "end": b, "type": tag[2:]}
        elif tag.startswith("I-") and cur and cur["type"] == tag[2:]:
            cur["end"] = b
        else:
            if cur: spans.append(cur); cur = None
    if cur: spans.append(cur)
    return spans

def score(pred, gold):
    pred_d = {(p["start"], p["end"]) for p in pred}
    gold_d = {(g["start"], g["end"]) for g in gold}
    tp_d = len(pred_d & gold_d); fp_d = len(pred_d - gold_d); fn_d = len(gold_d - pred_d)
    gm = {(g["start"], g["end"]): g["type"] for g in gold}
    used = set()
    tp_e = fp_e = 0
    for p in pred:
        k = (p["start"], p["end"])
        if k in gm and k not in used:
            used.add(k)
            if gm[k] == p["type"]: tp_e += 1
            else: fp_e += 1
        else: fp_e += 1
    fn_e = len(gold) - len(used)
    return (tp_d, fp_d, fn_d, tp_e, fp_e, fn_e)

def f1(tp, fp, fn):
    p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
    return p, r, (2 * p * r / max(1e-9, p + r))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/scratch/pii_corpus/eurobert_pii_eu.pt")
    ap.add_argument("--model", default="/root/models/EuroBERT-210m")
    ap.add_argument("--val", default="/scratch/pii_corpus/eu_shards")
    ap.add_argument("--languages", default="all")
    ap.add_argument("--max-rows", type=int, default=2000, help="per language, 0=all")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=48)
    args = ap.parse_args()

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    lmap = sd["labels"]; lmap_rev = {v: k for k, v in lmap.items()}
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    from eu_train import ROPE_INIT_FUNCTIONS  # noqa: shim applied at import
    from transformers import AutoModelForTokenClassification
    model = AutoModelForTokenClassification.from_pretrained(args.model, trust_remote_code=True, num_labels=len(lmap))
    model.load_state_dict(sd["model"]); model.to("cuda").eval()

    sel = args.languages.split(",") if args.languages != "all" else None
    files = sorted(glob.glob(os.path.join(args.val, "eu-val_*.jsonl")))
    agg = collections.Counter(); per_lang = {}; per_type = {}

    with torch.no_grad():
        for fp_ in files:
            lg = os.path.basename(fp_).split("_")[1].split(".")[0]
            if sel and lg not in sel: continue
            rows = [json.loads(l) for l in open(fp_)]
            if args.max_rows: rows = rows[: args.max_rows]
            lc = collections.Counter(); tc = collections.defaultdict(collections.Counter)
            for i in range(0, len(rows), args.batch):
                b = rows[i:i + args.batch]
                enc = tok([r["text"] for r in b], padding=True, truncation=True,
                          max_length=args.seq, return_offsets_mapping=True, return_tensors="pt")
                offs = enc.pop("offset_mapping")
                out = model(input_ids=enc["input_ids"].to("cuda"), attention_mask=enc["attention_mask"].to("cuda"))
                preds = out.logits.argmax(-1).cpu()
                for j, r in enumerate(b):
                    off = [(int(a), int(bb)) for a, bb in offs[j].tolist()]
                    pred = decode_spans(preds[j].tolist(), off, lmap_rev, len(off), r["text"])
                    m = score(pred, r["spans"])
                    for k, v in zip(["tp_d","fp_d","fn_d","tp_e","fp_e","fn_e"], m):
                        lc[k] += v; agg[k] += v
                    for g in r["spans"]:
                        tc[g["type"]]["gold"] += 1
                    for p_ in pred:
                        tc[p_["type"]]["pred"] += 1
                    for k in gm_keys(r["spans"]) & {(p["start"], p["end"]) for p in pred}:
                        t = {(g["start"], g["end"]): g["type"] for g in r["spans"]}[k]
                        tc[t]["hit"] += 1
            pd, rd, f1d = f1(lc["tp_d"], lc["fp_d"], lc["fn_d"])
            pe, re_, f1e = f1(lc["tp_e"], lc["fp_e"], lc["fn_e"])
            per_lang[lg] = (lc["tp_d"], lc["fp_d"], lc["fn_d"], lc["tp_e"], lc["fp_e"], lc["fn_e"], len(rows))
            for t_, c_ in tc.items():
                per_type[t_]["gold"] += c_["gold"]; per_type[t_]["pred"] += c_["pred"]; per_type[t_]["hit"] += c_["hit"]
            print(f"{lg}: det F1={f1d:.4f} (P{pd:.3f}/R{rd:.3f}) | exact F1={f1e:.4f} | n={len(rows)}", flush=True)

    pd, rd, f1d = f1(agg["tp_d"], agg["fp_d"], agg["fn_d"])
    pe, re_, f1e = f1(agg["tp_e"], agg["fp_e"], agg["fn_e"])
    print("\n" + "=" * 60)
    print(f"OVERALL: DETECTION F1={f1d:.4f} (P {pd:.4f} R {rd:.4f}) | EXACT-TYPE F1={f1e:.4f} (P {pe:.4f} R {re_:.4f})")
    print("=" * 60)
    print("\nPer-type gold/pred/hit:")
    for t in sorted(per_type, key=lambda x: per_type[x]["gold"], reverse=True):
        g = per_type[t]["gold"]; p_ = per_type[t]["pred"]; h = per_type[t]["hit"]
        prec = h / max(1, p_); rec = h / max(1, g)
        f_ = 2 * prec * rec / max(1e-9, prec + rec)
        print(f"  {t:22s} gold={g:7d} pred={p_:7d} hit={h:7d}  P={prec:.3f} R={rec:.3f} F1={f_:.4f}")

def gm_keys(spans):
    return {(g["start"], g["end"]) for g in spans}

if __name__ == "__main__":
    main()
