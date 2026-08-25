#!/usr/bin/env python3
"""Unified strict NER evaluation. One metric definition applied to every system:

  - decode: B opens span; I extends if same type; bare I does NOT open a span
    (standard NER). Whitespace-only boundary trim (GPT-2 tokenizers fold the leading
    space into token offsets; gold offsets don't include it).
  - detection (strict): predicted span == gold span exactly (start,end).
  - detection (overlap): any char overlap (reported separately, NOT a headline metric).
  - exact-type (strict): (start,end) match AND type equal.

Includes a train-row sanity check: a wired pipeline must reproduce ~1.0 on its own
training rows. If train F1 is low, the harness (not the model) is broken.
"""
import argparse, json, sys, os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

def rope_shim():
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    if "default" not in ROPE_INIT_FUNCTIONS:
        def _rope_default(config, device=None, seq_len=None):
            dim = config.hidden_size // config.num_attention_heads
            theta = getattr(config, "rope_theta", 10000.0)
            inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
            return inv_freq, 1.0
        ROPE_INIT_FUNCTIONS["default"] = _rope_default

def decode(logits, offsets, lmap_rev, text):
    cur, raw = None, []
    for ti, (a, b) in enumerate(offsets):
        if a == b or b == 0:  # special / pad
            if cur: raw.append(cur); cur = None
            continue
        tag = lmap_rev.get(logits[ti], "O")
        if tag.startswith("B-"):
            if cur: raw.append(cur)
            cur = [a, b, tag[2:]]
        elif tag.startswith("I-"):
            lab = tag[2:]
            if cur and cur[2] == lab: cur[1] = b
            else:
                if cur: raw.append(cur); cur = None  # bare I: no span (strict BIO)
        else:
            if cur: raw.append(cur); cur = None
    if cur: raw.append(cur)
    out = []
    for s, e, t in raw:
        while s < e and s < len(text) and text[s].isspace(): s += 1
        while e > s and e <= len(text) and text[e-1].isspace(): e -= 1
        if e > s: out.append((s, e, t))
    return out

def metric(m):
    tp, fp, fn = m
    p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
    f = 2 * p * r / max(1e-9, p + r)
    return p, r, f

def verdict(name, det, det_ov, ext, n):
    pd, rd, fd = metric(det); po, ro, fo = metric(det_ov); pe, re_, fe = metric(ext)
    print(f"{name:34s} STRICT det F1={fd:.4f} (P {pd:.3f} R {rd:.3f}) | overlap det F1={fo:.4f} (P {po:.3f} R {ro:.3f}) | STRICT exact-type F1={fe:.4f} | n={n}")

def eval_ours(ckpt, model_dir, rows):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    lmap_rev = {v: k for k, v in sd["labels"].items()}
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    rope_shim()
    m = AutoModelForTokenClassification.from_pretrained(model_dir, trust_remote_code=True, num_labels=len(lmap_rev))
    m.load_state_dict(sd["model"], strict=True); m.cuda().eval()
    det = [0,0,0]; det_ov = [0,0,0]; ext = [0,0,0]
    with torch.no_grad():
        for i in range(0, len(rows), 48):
            b = rows[i:i+48]
            enc = tok([r["text"] for r in b], padding=True, truncation=True, max_length=512,
                      return_offsets_mapping=True, return_tensors="pt")
            off = enc.pop("offset_mapping")
            lg = m(enc["input_ids"].cuda(), attention_mask=enc["attention_mask"].cuda()).logits.argmax(-1).cpu()
            for j, r in enumerate(b):
                offs = [(int(a), int(bb)) for a, bb in off[j].tolist()]
                pred = decode(lg[j].tolist(), offs, lmap_rev, r["text"])
                gset = {(g["start"], g["end"]): g["type"] for g in r["spans"]}
                ps = {(s, e): t for s, e, t in pred}
                used, used_ov = set(), set()
                for (s, e), t in ps.items():
                    if (s, e) in gset and (s, e) not in used:
                        used.add((s, e)); det[0] += 1
                        if gset[(s, e)] == t: ext[0] += 1
                        else: ext[1] += 1
                    else:
                        det[1] += 1; ext[1] += 1
                    hs = next((g for g in gset if g not in used_ov and max(s, g[0]) < min(e, g[1])), None)
                    if hs: used_ov.add(hs); det_ov[0] += 1
                    else: det_ov[1] += 1
                det[2] += len(gset) - len(used)
                ext[2] += len(gset) - len(used)
                det_ov[2] += len(gset) - len(used_ov)
    return det, det_ov, ext

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/scratch/pii_corpus/eurobert_final.pt")
    ap.add_argument("--model", default="/root/models/EuroBERT-210m")
    ap.add_argument("--val-file", default="/scratch/pii_corpus/eu_shards/eu-val_en.jsonl")
    ap.add_argument("--train-file", default="/scratch/pii_corpus/eu_shards/eu-train_en.jsonl")
    ap.add_argument("--rows", type=int, default=500)
    ap.add_argument("--train-sanity", type=int, default=200)
    args = ap.parse_args()

    val = [json.loads(l) for l in open(args.val_file)][:args.rows]
    trn = [json.loads(l) for l in open(args.train_file)][:args.train_sanity]

    print("=== SANITY: train rows (must be ~1.0 strict) ===", flush=True)
    d, ov, ex = eval_ours(args.ckpt, args.model, trn)
    verdict("EuroBERT-ours (TRAIN)", d, ov, ex, len(trn))
    print("=== VAL ===", flush=True)
    d, ov, ex = eval_ours(args.ckpt, args.model, val)
    verdict("EuroBERT-ours (val)", d, ov, ex, len(val))

if __name__ == "__main__":
    main()
