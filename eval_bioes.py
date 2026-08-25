#!/usr/bin/env python3
# eval_bioes.py — honest span-level F1 for the BIOES PII model.
#
# Decodes BIOES token predictions into char spans and reports:
#   * exact-type: (start,end,type) all match a gold span
#   * detection-tier (type-agnostic): (start,end) match (type ignored) — the
#     redaction-relevant metric used by the SOTA (Liquid AI).
#
# Run:
#   /root/venv/bin/python eval_bioes.py --ckpt /scratch/pii_corpus/bioes_pii.pt \
#     --data /scratch/pii_corpus/pii_val.jsonl
import argparse, os, json
import torch
from tokenizers import Tokenizer
from bitnet_pretrain import EncoderConfig
from bioes_pii import BIOESModel, build_type_names, make_label_names, spans_to_bioes


def decode_bioes(logits, enc_offsets, label_names, tmap, seq_len, text=None, min_conf=0.0):
    """Argmax BIOES per token -> char spans. text used to trim whitespace padding."""
    S = logits.shape[0]
    pred = logits.argmax(-1).cpu().numpy()
    spans = []
    cur = None  # (type,start_tok,end_tok)
    def close():
        nonlocal cur
        if cur is None:
            return
        typ, st, et = cur
        if st >= len(enc_offsets) or et >= len(enc_offsets):
            cur = None; return
        cs = enc_offsets[st][0]; ce = enc_offsets[et][1]
        # trim leading/trailing whitespace from char span (ByteLevel Ġ prefix on 1st token)
        while cs < ce and text[cs].isspace():
            cs += 1
        while ce > cs and text[ce-1].isspace():
            ce -= 1
        if ce > cs:
            spans.append({"type": typ, "start": int(cs), "end": int(ce)})
        cur = None
    for i in range(min(S, len(enc_offsets))):
        lab = label_names[pred[i]] if pred[i] < len(label_names) else "O"
        if enc_offsets[i][1] == 0:
            continue
        if lab == "O":
            close()
            continue
        # BIOES: B/I/E/S
        if lab.startswith("B-"):
            close()
            cur = (lab[2:], i, i)
        elif lab.startswith("I-"):
            if cur and cur[0] == lab[2:]:
                cur = (cur[0], cur[1], i)
            else:
                close(); cur = (lab[2:], i, i)
        elif lab.startswith("E-"):
            if cur and cur[0] == lab[2:]:
                cur = (cur[0], cur[1], i)
                close()
            else:
                close(); cur = (lab[2:], i, i); close()
        elif lab.startswith("S-"):
            close(); spans.append({"type": lab[2:], "start": int(enc_offsets[i][0]), "end": int(enc_offsets[i][1])})
    close()
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    type_names = build_type_names(rows)
    label_names, tmap = make_label_names(type_names)
    n_types = len(type_names)
    print(f"eval {len(rows)} rows, {n_types} types", flush=True)

    tok = Tokenizer.from_file("/root/pii_data/pretok_gpt/tokenizer.json")
    c = EncoderConfig(vocab_size=65000, hidden_size=1024, num_layers=16, num_heads=8,
                      num_kv_heads=8, intermediate_size=4096, max_seq_len=args.seq, dropout=0.1)
    model = BIOESModel(c, n_types).to("cuda")
    ck = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    model.eval()

    tp=0; fp=0; fn=0; tp_s=0; fp_s=0; fn_s=0
    for r in rows:
        text = r["text"]
        enc = tok.encode(text)
        ids = enc.ids[:args.seq] + [0]*(args.seq - len(enc.ids[:args.seq]))
        x = torch.tensor([ids], dtype=torch.long, device="cuda")
        with torch.no_grad():
            logits, _ = model(x)
        pred = decode_bioes(logits[0], enc.offsets[:args.seq], label_names, tmap, args.seq, text)
        gold = [(sp["start"], sp["end"]) for sp in r.get("spans", [])]
        gold_set = {(g[0], g[1]): sp["type"] for sp in r.get("spans", []) for g in [(sp["start"], sp["end"])]}
        used = set()
        for p in pred:
            key = (p["start"], p["end"])
            if key in gold_set and key not in used:
                used.add(key); tp += 1
                if gold_set[key] == p["type"]: tp_s += 1
                else: fp_s += 1
            else:
                fp += 1; fp_s += 1
        for g in gold:
            if g not in used:
                fn += 1; fn_s += 1
    P = tp/max(1,tp+fp); R = tp/max(1,tp+fn); F1 = 2*P*R/max(1e-9,P+R)
    P2 = tp_s/max(1,tp_s+fp_s); R2 = tp_s/max(1,tp_s+fn_s); F12 = 2*P2*R2/max(1e-9,P2+R2)
    print(f"\n=== EXACT-TYPE (span+type) ===")
    print(f"tp={tp} fp={fp} fn={fn}  P={P:.4f} R={R:.4f} F1={F1:.4f}")
    print(f"\n=== DETECTION-TIER (span only) [the redaction metric] ===")
    print(f"tp={tp_s} fp={fp_s} fn={fn_s}  P={P2:.4f} R={R2:.4f} F1={F12:.4f}")


if __name__ == "__main__":
    main()
