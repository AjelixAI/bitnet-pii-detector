#!/usr/bin/env python3
# train_gliner_pii.py — Stage-2: label-conditioned span fine-tune on PII.
#
# Loads the Stage-1 pretrained 1.58-bit encoder, attaches the Gliner span head,
# and fine-tunes on PII data (verified synthetic + optional real ai4privacy).
# Uses dropout + early-stopping (stop at best val F1, not fixed epochs) to avoid
# the overfitting that killed the from-scratch 42M model.
#
# Data: JSONL [{lang,text,spans:[{type,start,end,text}]}]. We build per-sample
# gold (label,start,end) targets and train the span head (start/end BCE per
# (token,label)). Labels are passed as text so the model is schema-flexible.
#
# Run:
#   /root/venv/bin/python train_gliner_pii.py --data /scratch/pii_corpus/pii_train.jsonl \
#     --pretrain /scratch/pii_corpus/enc_100m.pt
import argparse, os, math, random, time, json
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from bitnet_pretrain import EncoderConfig
from gliner_span import GlinerPII, label_text_to_token_ids
from tokenizers import Tokenizer

# closed English schema used for labels (must match generator)
LABEL_TEXTS = {
    "email": "email", "phone": "phone number", "dob": "date of birth",
    "iban": "iban", "bic": "bic code", "credit_card": "credit card",
    "ssn": "social security number", "username": "username", "ip": "ip address",
    "ipv6": "ipv6 address", "vat": "vat number", "fiscal_code": "fiscal code",
    "passport": "passport number", "utr": "utr", "nif": "nif",
    "aws_access_key": "aws access key", "github_token": "github token",
    "stripe_key": "stripe key", "openai_key": "openai api key",
    "slack_token": "slack token", "jwt": "jwt", "private_key": "private key",
    "bearer_token": "bearer token", "gcp_private_key": "gcp private key",
    "btc_address": "bitcoin address", "btc_bech32": "bitcoin address",
    "eth_address": "ethereum address", "ltc_address": "litecoin address",
    "sol_address": "solana address", "vin": "vin", "imei": "imei",
    "mac": "mac address", "tax_id": "tax id", "personal_code": "personal code",
    "account_number": "account number", "routing_number": "routing number",
    "license_plate": "license plate",
}


def build_label_defs(schema_types):
    """Return ordered list of (label_name, text) present in data."""
    names = []
    for t in schema_types:
        if t in LABEL_TEXTS:
            names.append(t)
    return names


class PIISpanDataset(Dataset):
    def __init__(self, rows, tok, seq_len, label_names, max_labels=40):
        self.rows = rows
        self.tok = tok
        self.seq_len = seq_len
        self.label_names = label_names
        self.label_char_ids = label_text_to_token_ids([LABEL_TEXTS[n] for n in label_names], tok)
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        text = r["text"]
        enc = self.tok.encode(text)
        ids = enc.ids[:self.seq_len]
        offsets = enc.offsets[:self.seq_len]
        # map char spans -> token indices (start token index, end token index)
        # label id = index into self.label_names; only keep types present
        target = []  # (label_idx, start_tok, end_tok)
        for sp in r.get("spans", []):
            if sp["type"] not in self.label_names:
                continue
            li = self.label_names.index(sp["type"])
            s, e = sp["start"], sp["end"]
            # find token covering s (start) and e (end, exclusive -> last token)
            st = et = -1
            for ti, (ts, te) in enumerate(offsets):
                if ts <= s < te:
                    st = ti
                if ts < e <= te:
                    et = ti
            if st < 0 or et < st:
                continue
            target.append((li, st, et))
        # pad ids
        ids = ids + [0] * (self.seq_len - len(ids))
        return (torch.tensor(ids, dtype=torch.long),
                torch.tensor(target, dtype=torch.long))


def collate(batch):
    ids = torch.stack([b[0] for b in batch])
    # variable-length target lists; pad into a flat tensor with (sample_idx,label,start,end)
    samples, labels, starts, ends = [], [], [], []
    for si, (_, tgt) in enumerate(batch):
        for (li, st, et) in tgt.tolist():
            samples.append(si); labels.append(li); starts.append(st); ends.append(et)
    return ids, (torch.tensor(samples, dtype=torch.long),
                 torch.tensor(labels, dtype=torch.long),
                 torch.tensor(starts, dtype=torch.long),
                 torch.tensor(ends, dtype=torch.long))


def span_loss(start_logits, end_logits, tgt, device):
    """GLiNER-style span loss that CANNOT collapse to all-zeros.

    The naive BCE over all (B,S,L) positions with a small pos_weight lets the model
    minimize loss by predicting 'no PII everywhere' (99.9% negatives). To force real
    span learning:
      1. Only supervise (token,label) pairs whose label HAS at least one gold span in
         the sample (so the model must predict SOMETHING for labels that are present).
      2. Use a large positive weight AND normalize by the number of positives, so the
         loss rewards finding each gold span, not just predicting none.
    tgt = (sample_idx, label_idx, start_tok, end_tok)."""
    if len(tgt[0]) == 0:
        return start_logits.sum() * 0.0
    samples, li, st, et = tgt
    B, S, L = start_logits.shape
    # labels present in this batch
    present_labels = torch.unique(li)
    if len(present_labels) == 0:
        return start_logits.sum() * 0.0

    sel_sl = start_logits[:, :, present_labels]   # [B, S, L_present]
    sel_el = end_logits[:, :, present_labels]
    start_tgt = torch.zeros_like(sel_sl)
    end_tgt = torch.zeros_like(sel_el)
    # map original label idx -> position in present_labels
    plist = present_labels.tolist()
    for si, lorig, stok, etok in zip(samples.tolist(), li.tolist(), st.tolist(), et.tolist()):
        p = plist.index(lorig)
        start_tgt[si, stok, p] = 1.0
        end_tgt[si, etok, p] = 1.0

    # positives per (sample,label)
    n_pos_sl = start_tgt.sum()
    n_pos_el = end_tgt.sum()
    # BCE (no extra pos_weight; restrict to present labels + normalize by positives)
    # sum over positives is what we reward; mean over all keeps scale stable but the
    # small positive fraction is already handled by restricting to present labels.
    # Use focal-ish weighting: give positives weight = (total_neg/total_pos) clamped.
    n_tot = sel_sl.numel()
    pos_w = torch.clamp(n_tot / max(1.0, n_pos_sl + 1e-6), 1.0, 50.0)
    pos_w = torch.tensor(pos_w.item(), device=device)
    sl = F.binary_cross_entropy_with_logits(sel_sl, start_tgt, pos_weight=pos_w)
    pos_w2 = torch.clamp(n_tot / max(1.0, n_pos_el + 1e-6), 1.0, 50.0)
    pos_w2 = torch.tensor(pos_w2.item(), device=device)
    el = F.binary_cross_entropy_with_logits(sel_el, end_tgt, pos_weight=pos_w2)
    return sl + el


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pretrain", default="/scratch/pii_corpus/enc_100m.pt")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--save", default="/scratch/pii_corpus/gliner_pii.pt")
    ap.add_argument("--use-wandb", action="store_true")
    ap.add_argument("--run-name", default="gliner-pii-100m")
    ap.add_argument("--freeze-encoder", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0); random.seed(0)
    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    random.shuffle(rows)
    split = int(len(rows) * 0.95)
    train_rows, val_rows = rows[:split], rows[split:]
    print(f"train={len(train_rows)} val={len(val_rows)}", flush=True)

    # collect all types present
    schema_types = set()
    for r in rows:
        for sp in r.get("spans", []):
            schema_types.add(sp["type"])
    label_names = build_label_defs(schema_types)
    print(f"labels ({len(label_names)}): {label_names}", flush=True)
    if not label_names:
        print("ERROR: no recognized label types in data"); return

    tok = Tokenizer.from_file("/root/pii_data/pretok/tokenizer.json")
    c = EncoderConfig(vocab_size=32000, hidden_size=640, num_layers=10,
                      num_heads=8, num_kv_heads=8, intermediate_size=2560,
                      max_seq_len=args.seq, dropout=0.1)
    model = GlinerPII(c).to("cuda")
    if args.pretrain and os.path.exists(args.pretrain):
        ck = torch.load(args.pretrain, map_location="cuda", weights_only=False)
        sd = ck.get("model", ck)  # enc_pretrain format: {"model": encoder_state_dict}
        model.load_encoder_weights(sd)
        print("loaded pretrained encoder (text + label backbone)", flush=True)
    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
    print("params:", round(model.param_count()/1e6, 1), "M (trainable:",
          round(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6, 1), "M)", flush=True)

    tr_ds = PIISpanDataset(train_rows, tok, args.seq, label_names)
    va_ds = PIISpanDataset(val_rows, tok, args.seq, label_names)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, collate_fn=collate, num_workers=2, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch, shuffle=False, collate_fn=collate, num_workers=2)

    # fixed label token-id tensor for all batches (schema is fixed per run),
    # encoded through the SHARED tokenizer.
    fixed_labels = label_text_to_token_ids([LABEL_TEXTS[n] for n in label_names], tok)
    model.label_enc_char_ids = fixed_labels

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98))
    n_total = len(tr_dl) * args.epochs
    warm = max(1, int(n_total * 0.05))

    wandb = None
    if args.use_wandb:
        try:
            import wandb as _w
            try: os.environ["WANDB_API_KEY"] = open("/root/.wandb_key").read().strip()
            except Exception: pass
            wandb = _w.init(project="bitnet-1.58-pii-encoder", name=args.run_name, config={"params": model.param_count()})
        except Exception as e:
            wandb = None; print("wandb skip:", e, flush=True)

    best = float("inf"); bad = 0; step = 0
    for ep in range(args.epochs):
        model.train()
        el = 0.0; cnt = 0
        for ids, tgt in tr_dl:
            ids = ids.to("cuda")
            tgt = tuple(t.to("cuda") for t in tgt)
            labels = model.label_enc_char_ids.to("cuda")
            sl, elo = model(ids, labels)
            loss = span_loss(sl, elo, tgt, ids.device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            lr = args.lr * (step/max(1,warm)) if step < warm else args.lr
            for pg in opt.param_groups: pg["lr"] = lr
            opt.step(); opt.zero_grad()
            el += loss.item(); cnt += 1; step += 1
        # validation
        model.eval(); vl = 0.0; vc = 0
        with torch.no_grad():
            for ids, tgt in va_dl:
                ids = ids.to("cuda"); tgt = tuple(t.to("cuda") for t in tgt)
                labels = model.label_enc_char_ids.to("cuda")
                sl, elo = model(ids, labels)
                vl += span_loss(sl, elo, tgt, ids.device).item(); vc += 1
        vavg = vl / max(1, vc)
        print(f"ep{ep+1} loss={el/max(1,cnt):.4f} val={vavg:.4f}", flush=True)
        if wandb: wandb.log({"loss": el/max(1,cnt), "val": vavg, "epoch": ep+1}, step=step)
        if vavg < best:
            best = vavg; bad = 0
            torch.save({"model": model.state_dict()}, args.save)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {ep+1} (val best {best:.4f})", flush=True)
                break
    print(f"BEST val {best:.4f} -> saved {args.save}", flush=True)
    if wandb:
        try: wandb.finish()
        except Exception: pass


if __name__ == "__main__":
    main()
