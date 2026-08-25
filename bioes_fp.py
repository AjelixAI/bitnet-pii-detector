#!/usr/bin/env python3
# bioes_pii.py — BIOES token-classification PII detector on the 1.58-bit encoder.
#
# This is the SOTA-validated architecture (Liquid AI LFM2.5-Encoder-PII-Detector):
# a strong MLM bidirectional encoder + a simple BIOES token-classification head.
# Labels: O=0; for each type t: B-t, I-t, E-t, S-t (1 + 4*num_types).
#
# Our 1.58-bit encoder (from bitnet_pretrain.BitnetEncoder) provides the backbone;
# the head is a per-token linear over BIOES labels. This avoids the GLiNER-style
# complexity and matches the reference that actually tops the PII benchmarks.
#
# Run:
#   /root/venv/bin/python bioes_pii.py --data train.jsonl --pretrain enc_100m.pt
import argparse, os, math, random, time, json
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from bitnet_pretrain import EncoderConfig
from fp_encoder import FPEncoder
from tokenizers import Tokenizer


class BIOESModel(nn.Module):
    """FULL-PRECISION encoder + BIOES token-classification head (control)."""
    def __init__(self, c, num_types):
        super().__init__()
        self.c = c
        self.encoder = FPEncoder(c)
        self.head = nn.Linear(c.hidden_size, 1 + 4 * num_types, bias=False)
        nn.init.normal_(self.head.weight, std=0.02)
    def forward(self, input_ids, labels=None, attention_mask=None):
        h = self.encoder(input_ids)             # [B,S,H]
        logits = self.head(h)                   # [B,S,1+4*T]
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1),
                                   ignore_index=-100)
        return logits, loss
    def param_count(self):
        return sum(p.numel() for p in self.parameters())
    def load_encoder(self, sd):
        self.encoder.load_bitnet_weights(sd)  # fp master weights, no quant


def build_type_names(rows):
    t = set()
    for r in rows:
        for sp in r.get("spans", []):
            t.add(sp["type"])
    return sorted(t)


def make_label_names(type_names):
    """O + 4*types (B/I/E/S per type). Returns (id->name, type->{B,I,E,S id})."""
    names = ["O"]
    tmap = {}
    for i, t in enumerate(type_names):
        b = 1 + 4*i
        tmap[t] = {"B": b, "I": b+1, "E": b+2, "S": b+3}
        names += [f"B-{t}", f"I-{t}", f"E-{t}", f"S-{t}"]
    return names, tmap


def spans_to_bioes(text, spans, enc_offsets, seq_len, tmap):
    """Convert char spans -> per-token BIOES ids. Handles multi-token and single-token."""
    labels = [-100] * seq_len
    # mark all valid (non-pad) tokens as O first
    n_tok = min(len(enc_offsets), seq_len)
    for i in range(n_tok):
        if enc_offsets[i][1] > 0:
            labels[i] = 0
    for sp in spans:
        t = sp["type"]
        if t not in tmap:
            continue
        s, e = sp["start"], sp["end"]
        # find ALL tokens overlapping the char span [s,e) — handles subword fragmentation
        tok_ids = [i for i in range(n_tok) if enc_offsets[i][1] > 0 and enc_offsets[i][0] < e and enc_offsets[i][1] > s]
        if not tok_ids:
            continue
        st, et = tok_ids[0], tok_ids[-1]
        if st == et:
            labels[st] = tmap[t]["S"]
        else:
            labels[st] = tmap[t]["B"]
            labels[et] = tmap[t]["E"]
            for i in range(st+1, et):
                labels[i] = tmap[t]["I"]
    return labels


def spans_to_binary(text, spans, enc_offsets, seq_len):
    """Convert char spans -> B-PII/I-PII/O (3 classes, binary detection).
    The OPF paper shows binary PII labels are far more data-efficient than per-class
    at small n (F1 0.634 vs 0.360 at n=100). This is the primary SOTA fine-tune pass."""
    labels = [-100] * seq_len
    n_tok = min(len(enc_offsets), seq_len)
    for i in range(n_tok):
        if enc_offsets[i][1] > 0:
            labels[i] = 0  # O
    for sp in spans:
        s, e = sp["start"], sp["end"]
        tok_ids = [i for i in range(n_tok) if enc_offsets[i][1] > 0 and enc_offsets[i][0] < e and enc_offsets[i][1] > s]
        if not tok_ids:
            continue
        st, et = tok_ids[0], tok_ids[-1]
        if st == et:
            labels[st] = 2  # S-PII (we use 2 for single; B=1,I=1,E=2 for the 3-class scheme)
        else:
            labels[st] = 1  # B-PII
            labels[et] = 2  # E-PII
            for i in range(st+1, et):
                labels[i] = 1  # I-PII
    return labels


class PIIDataset(Dataset):
    def __init__(self, rows, tok, seq_len, tmap, binary=False):
        self.rows = rows; self.tok = tok; self.seq_len = seq_len; self.tmap = tmap; self.binary = binary
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        text = r["text"]
        enc = self.tok.encode(text)
        ids = enc.ids[:self.seq_len]
        offsets = enc.offsets[:self.seq_len]
        ids = ids + [0] * (self.seq_len - len(ids))
        if self.binary:
            labels = spans_to_binary(text, r.get("spans", []), offsets, self.seq_len)
        else:
            labels = spans_to_bioes(text, r.get("spans", []), offsets, self.seq_len, self.tmap)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def collate(batch):
    ids = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    return ids, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pretrain", default="/scratch/pii_corpus/enc_100m.pt")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--entity-weight", type=float, default=1.0)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--vocab", type=int, default=65000)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--binary", action="store_true", help="binary B-PII/I-PII/O labels (SOTA data-efficient)")
    ap.add_argument("--use-wandb", action="store_true")
    ap.add_argument("--run-name", default="bioes-pii-350m")
    ap.add_argument("--save", default="/scratch/pii_corpus/bioes_pii.pt")
    args = ap.parse_args()

    torch.manual_seed(0); random.seed(0)
    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    random.shuffle(rows)
    split = int(len(rows) * 0.9)
    train_rows, val_rows = rows[:split], rows[split:]
    print(f"train={len(train_rows)} val={len(val_rows)}", flush=True)
    type_names = build_type_names(rows)
    label_names, tmap = make_label_names(type_names)
    n_types = len(type_names)
    if args.binary:
        # binary PII scheme: O=0, B-PII=1, E/S-PII=2  (3 classes)
        label_names = ["O", "B-PII", "E/S-PII"]
        tmap = None
        n_types = 0  # head uses 3 labels directly
    print(f"types={n_types} labels={3 if args.binary else 1+4*len(type_names) if type_names else 3}", flush=True)

    tok = Tokenizer.from_file("/root/pii_data/pretok_gpt/tokenizer.json")
    # match the pretrain config exactly (pretrain_mlm_bin uses heads=8, kv=8, ffn=hidden*4)
    c = EncoderConfig(vocab_size=args.vocab, hidden_size=args.hidden, num_layers=args.layers,
                      num_heads=8, num_kv_heads=8, intermediate_size=args.hidden*4,
                      max_seq_len=args.seq, dropout=args.dropout)
    head_types = 1 if args.binary else n_types  # head_types=1 -> 3 BIOES labels (1+4*? we special-case)
    model = BIOESModel(c, head_types).to("cuda")
    if args.binary:
        # override head to 3 classes (O, B-PII, E/S-PII) — keep on device
        model.head = nn.Linear(c.hidden_size, 3, bias=False).to("cuda")
        nn.init.normal_(model.head.weight, std=0.02)
    if args.pretrain and os.path.exists(args.pretrain):
        ck = torch.load(args.pretrain, map_location="cuda", weights_only=False)
        model.load_encoder(ck.get("model", ck))
        print("loaded pretrained 1.58-bit encoder", flush=True)
    print("params:", round(model.param_count()/1e6, 1), "M", flush=True)

    tr_ds = PIIDataset(train_rows, tok, args.seq, tmap, binary=args.binary)
    va_ds = PIIDataset(val_rows, tok, args.seq, tmap, binary=args.binary)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, collate_fn=collate, num_workers=2, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch, shuffle=False, collate_fn=collate, num_workers=2)

    # class weights (entity classes upweighted)
    n_labels = 3 if args.binary else 1 + 4*n_types
    cw = torch.ones(n_labels, device="cuda")
    cw[1:] = args.entity_weight
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98))
    n_total = len(tr_dl) * args.epochs
    warm = max(1, int(n_total * args.warmup_ratio))

    wandb = None
    if args.use_wandb:
        try:
            import wandb as _w
            try: os.environ["WANDB_API_KEY"] = open("/root/.wandb_key").read().strip()
            except Exception: pass
            wandb = _w.init(project="bitnet-1.58-pii-encoder", name=args.run_name, config={"params": model.param_count(), "types": n_types})
        except Exception as e:
            wandb = None; print("wandb skip:", e, flush=True)

    best = float("inf"); bad = 0; step = 0; t0 = time.time()
    for ep in range(args.epochs):
        model.train(); el = 0.0; cnt = 0
        for ids, labels in tr_dl:
            ids = ids.to("cuda"); labels = labels.to("cuda")
            logits, loss = model(ids, labels=labels)
            cw_t = cw
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), weight=cw_t, ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            lr = args.lr * (step/max(1,warm)) if step < warm else args.lr*0.5*(1+math.cos(math.pi*(step-warm)/max(1,n_total-warm)))
            for pg in opt.param_groups: pg["lr"] = lr
            opt.step(); opt.zero_grad()
            el += loss.item(); cnt += 1; step += 1
            if wandb: wandb.log({"loss": loss.item(), "lr": lr}, step=step)
        model.eval(); vl = 0.0; vc = 0
        with torch.no_grad():
            for ids, labels in va_dl:
                ids = ids.to("cuda"); labels = labels.to("cuda")
                logits, l = model(ids, labels=labels)
                vl += F.cross_entropy(logits.view(-1,logits.shape[-1]), labels.view(-1), weight=cw, ignore_index=-100).item(); vc += 1
        vavg = vl/max(1,vc)
        print(f"ep{ep+1} loss={el/max(1,cnt):.4f} val={vavg:.4f} {time.time()-t0:.0f}s", flush=True)
        if wandb: wandb.log({"val": vavg, "epoch": ep+1}, step=step)
        if vavg < best:
            best = vavg; bad = 0
            torch.save({"model": model.state_dict()}, args.save)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop ep{ep+1} (best {best:.4f})", flush=True)
                break
    print(f"BEST val {best:.4f} -> {args.save}", flush=True)
    if wandb:
        try: wandb.finish()
        except Exception: pass


if __name__ == "__main__":
    main()
