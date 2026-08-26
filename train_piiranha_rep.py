#!/usr/bin/env python3
"""train_piiranha_rep.py — EXACT replication of Piiranha-v1.

Base:       microsoft/mdeberta-v3-base (DeBERTa-v2 architecture)
Data:       ai4privacy/pii-masking-400k (325K train / 81K val)
Labels:     I-only (17 PII types + O = 18 labels)
            spans from privacy_mask (char offsets) → DeBERTa token labels
Hparams:    lr 5e-5, batch 128, 5 epochs, warmup 0.05, linear,
            Adam (0.9, 0.999), eps 1e-8, Native AMP
Tokenizing: DebertaV2Tokenizer, max length 256 (their context limit)
"""
import argparse, json, os, random, time, math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

PII_TYPES = ["ACCOUNTNUM", "BUILDINGNUM", "CITY", "CREDITCARDNUMBER", "DATEOFBIRTH",
             "DRIVERLICENSENUM", "EMAIL", "GIVENNAME", "IDCARDNUM", "PASSWORD",
             "SOCIALNUM", "STREET", "SURNAME", "TAXNUM", "TELEPHONENUM",
             "USERNAME", "ZIPCODE"]

# Piiranha's I-only label scheme: I-X for each type, then O
LABEL_NAMES = [f"I-{t}" for t in PII_TYPES] + ["O"]
LABEL_MAP = {l: i for i, l in enumerate(LABEL_NAMES)}
O_IDX = LABEL_MAP["O"]

def chars_to_ilabels(offsets, spans, seq_len):
    """Map char spans to I-only labels:
    token is I-X if its offset window overlaps ANY span of type X."""
    labels = [O_IDX] * seq_len
    for sp in spans:
        s, e, lab = sp["start"], sp["end"], sp["type"]
        i_idx = LABEL_MAP.get(f"I-{lab}")
        if i_idx is None: continue
        for ti, (a, b) in enumerate(offsets[:seq_len]):
            if b == 0 or a == b: continue
            if b <= s or a >= e: continue
            # overlap: mark I
            labels[ti] = i_idx
    return labels

def decode_ilabels(logits_row, offsets, text):
    """I-only decode: contiguous runs of same I-X = one span."""
    cur = None
    spans = []
    for ti, (a, b) in enumerate(offsets):
        if a == b or b == 0:
            if cur: spans.append(cur); cur = None
            continue
        tag = "O"
        p = logits_row[ti]
        if p < len(LABEL_NAMES):
            tag = LABEL_NAMES[p]
        if tag == "O":
            if cur: spans.append(cur); cur = None
        elif tag.startswith("I-"):
            t = tag[2:]
            if cur and cur["type"] == t:
                cur["end"] = b
            else:
                if cur: spans.append(cur)
                cur = {"start": a, "end": b, "type": t}
    if cur: spans.append(cur)
    # trim whitespace
    out = []
    for s in spans:
        while s["start"] < s["end"] and s["start"] < len(text) and text[s["start"]].isspace():
            s["start"] += 1
        while s["end"] > s["start"] and s["end"] <= len(text) and text[s["end"]-1].isspace():
            s["end"] -= 1
        if s["end"] > s["start"]:
            out.append(s)
    return out

class PIIDataset(Dataset):
    def __init__(self, rows, tok, max_len=256):
        self.rows = rows
        self.tok = tok
        self.max_len = max_len
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]

def collate(batch, tok, max_len):
    texts = [r["text"] for r in batch]
    enc = tok(texts, padding=False, truncation=True, max_length=max_len,
              return_offsets_mapping=True)
    ids, attn, labs = [], [], []
    for r, ii, oo in zip(batch, enc["input_ids"], enc["offset_mapping"]):
        lb = chars_to_ilabels(oo, r["spans"], len(ii))
        ids.append(ii); attn.append([1]*len(ii)); labs.append(lb)
    mx = max(len(x) for x in ids)
    def pad(xs, v): return torch.stack([torch.tensor(x + [v]*(mx-len(x))) for x in xs])
    return pad(ids, 0), pad(attn, 0), pad(labs, -100)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/models/mdeberta-v3-base")
    ap.add_argument("--data", default="/root/data/pii-masking-400k")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--save", default="/scratch/pii_corpus/piiranha_rep.pt")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--acc-steps", type=int, default=1)
    args = ap.parse_args()

    torch.manual_seed(42)
    from datasets import load_from_disk
    ds = load_from_disk(args.data)
    print(f"train: {len(ds['train'])} val: {len(ds['validation'])}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"tokenizer: {type(tok).__name__} | vocab {len(tok)}", flush=True)

    def convert(split):
        rows = []
        for r in ds[split]:
            text = r["source_text"]
            spans = []
            for sm in r.get("privacy_mask", []):
                lab = sm.get("label", "")
                if lab not in PII_TYPES: continue
                spans.append({"start": sm["start"], "end": sm["end"], "type": lab})
            if text and spans:
                rows.append({"text": text, "spans": spans})
        return rows

    print("Converting dataset...", flush=True)
    tr = convert("train")
    va = convert("validation")
    print(f"converted: {len(tr)} train, {len(va)} val", flush=True)

    model = AutoModelForTokenClassification.from_pretrained(
        args.model, trust_remote_code=True, num_labels=len(LABEL_NAMES))
    model = model.float()  # fp32 weights — required for GradScaler
    model.config.id2label = {i: l for i, l in enumerate(LABEL_NAMES)}
    model.config.label2id = {l: i for i, l in enumerate(LABEL_NAMES)}
    dev = "cuda"; model.to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M | labels: {len(LABEL_NAMES)}", flush=True)

    dl = DataLoader(PIIDataset(tr, tok, args.seq), batch_size=args.batch, shuffle=True,
                    collate_fn=lambda b: collate(b, tok, args.seq))
    va_dl = DataLoader(PIIDataset(va, tok, args.seq), batch_size=args.batch, shuffle=False,
                       collate_fn=lambda b: collate(b, tok, args.seq))

    # Adam (not AdamW!) with betas (0.9, 0.999), eps 1e-8 — EXACTLY Piiranha
    decay, nodecay = [], []
    for n_, p in model.named_parameters():
        (nodecay if ("norm" in n_ or "bias" in n_) else decay).append(p)
    opt = torch.optim.Adam([{"params": decay, "weight_decay": 0.01},
                            {"params": nodecay, "weight_decay": 0.0}],
                           lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    steps = len(dl) * args.epochs
    wu = int(args.warmup * steps)
    def lr_at(s): return args.lr * s / max(1, wu) if s < wu else args.lr * (1 - (s - wu) / max(1, steps - wu))

    best_loss, step = 1e9, 0
    t0 = time.time()
    scaler = torch.amp.GradScaler('cuda')
    for ep in range(args.epochs):
        model.train()
        for ids, attn, labs in dl:
            ids, attn, labs = ids.to(dev), attn.to(dev), labs.to(dev)
            for g in opt.param_groups: g["lr"] = lr_at(step)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(input_ids=ids, attention_mask=attn, labels=None)
                loss = F.cross_entropy(out.logits.view(-1, out.logits.shape[-1]),
                                       labs.view(-1), ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            if step % args.eval_every == 0:
                model.eval()
                vt = vn = 0
                with torch.no_grad():
                    for ids, attn, labs in va_dl:
                        ids, attn, labs = ids.to(dev), attn.to(dev), labs.to(dev)
                        with torch.autocast("cuda", dtype=torch.float16):
                            out = model(input_ids=ids, attention_mask=attn, labels=None)
                            vl = F.cross_entropy(out.logits.view(-1, out.logits.shape[-1]),
                                                 labs.view(-1), ignore_index=-100)
                        vt += vl.item(); vn += 1
                val = vt / max(1, vn)
                model.train()
                marker = ""
                if val < best_loss:
                    best_loss = val
                    torch.save({"model": model.state_dict(), "label_names": LABEL_NAMES}, args.save)
                    marker = " *BEST"
                print(f"ep{ep+1} step{step} loss={loss.item():.4f} val={val:.4f}{marker}", flush=True)
        print(f"epoch {ep+1} done {(time.time()-t0)/60:.0f}min", flush=True)
    print(f"BEST val loss {best_loss:.4f} -> {args.save}", flush=True)

if __name__ == "__main__":
    main()
