#!/usr/bin/env python3
# eval_gliner_pii.py — HONEST span-level F1 for the label-conditioned PII model.
#
# Evaluates the trained GlinerPII model on held-out data (the verifier-gated real/
# synthetic set, or the SPY-style OOD set). Exact-match: a predicted span is correct
# only if (start, end, type) matches a gold span. Reports type-aware and type-agnostic
# P/R/F1. This is the honest metric — NOT token-level.
#
# Run:
#   /root/venv/bin/python eval_gliner_pii.py --ckpt /scratch/pii_corpus/gliner_pii.pt \
#     --data /scratch/pii_corpus/pii_val.jsonl
import argparse, os, json
import torch
from tokenizers import Tokenizer
from bitnet_pretrain import EncoderConfig
from gliner_span import GlinerPII, label_text_to_char_ids


def decode_spans(start_logits, end_logits, offsets, label_names, tokenizer_offsets,
                 start_thresh=0.5, end_thresh=0.5, max_spans=32):
    """Decode (token,label) start/end logits -> list of {type,start,end,text}.
    For each label, find tokens with start_logit > thresh; for each, best end >= it
    with end_logit > thresh; reconstruct char span from token offsets."""
    B, S, L = start_logits.shape
    spans = []
    for li in range(L):
        s_logits = start_logits[0, :, li]
        starts = torch.nonzero(s_logits > start_thresh).flatten().tolist()
        for st in starts:
            e_logits = end_logits[0, st:, li]
            cand_ends = torch.nonzero(e_logits > end_thresh).flatten().tolist()
            if not cand_ends:
                continue
            et = st + cand_ends[0]
            # char offsets
            t_off = tokenizer_offsets
            if st >= len(t_off) or et >= len(t_off):
                continue
            cs, ce = t_off[st][0], t_off[et][1]
            spans.append({"type": label_names[li], "start": cs, "end": ce})
            if len(spans) >= max_spans:
                break
        if len(spans) >= max_spans:
            break
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    schema_types = set()
    for r in rows:
        for sp in r.get("spans", []):
            schema_types.add(sp["type"])
    from train_gliner_pii import LABEL_TEXTS, build_label_defs
    label_names = build_label_defs(schema_types)
    print(f"eval {len(rows)} rows, {len(label_names)} labels", flush=True)

    tok = Tokenizer.from_file("/root/pii_data/pretok/tokenizer.json")
    c = EncoderConfig(vocab_size=32000, hidden_size=640, num_layers=10,
                      num_heads=8, num_kv_heads=8, intermediate_size=2560,
                      max_seq_len=args.seq, dropout=0.1)
    model = GlinerPII(c).to("cuda")
    ck = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    model.eval()
    labels = label_text_to_char_ids([LABEL_TEXTS[n] for n in label_names]).to("cuda")

    tp = fp = fn = 0
    tp_s = fp_s = fn_s = 0
    for r in rows:
        text = r["text"]
        enc = tok.encode(text)
        ids = enc.ids[:args.seq]
        offs = enc.offsets[:args.seq]
        ids = ids + [0] * (args.seq - len(ids))
        x = torch.tensor([ids], dtype=torch.long, device="cuda")
        with torch.no_grad():
            sl, el = model(x, labels)
        pred = decode_spans(sl, el, offs, label_names, offs)
        # gold spans
        gold = [(sp["start"], sp["end"], sp["type"]) for sp in r.get("spans", [])]
        gold_set = {(s, e): t for s, e, t in gold}
        used = set()
        for p in pred:
            key = (p["start"], p["end"])
            if key in gold_set and key not in used:
                used.add(key)
                tp += 1
                if gold_set[key] == p["type"]:
                    tp_s += 1
                else:
                    fp_s += 1
            else:
                fp += 1; fp_s += 1
        for g in gold:
            key = (g[0], g[1])
            if key not in used:
                fn += 1; fn_s += 1

    P = tp / max(1, tp + fp); R = tp / max(1, tp + fn)
    F1 = 2 * P * R / max(1e-9, P + R)
    P2 = tp_s / max(1, tp_s + fp_s); R2 = tp_s / max(1, tp_s + fn_s)
    F12 = 2 * P2 * R2 / max(1e-9, P2 + R2)
    print(f"\n=== SPAN-LEVEL (exact: char span AND type) ===")
    print(f"tp={tp} fp={fp} fn={fn}")
    print(f"Precision={P:.4f} Recall={R:.4f} F1={F1:.4f}")
    print(f"\n=== SPAN-LEVEL (type-agnostic: char span only) ===")
    print(f"tp={tp_s} fp={fp_s} fn={fn_s}")
    print(f"Precision={P2:.4f} Recall={R2:.4f} F1={F12:.4f}")


if __name__ == "__main__":
    main()
