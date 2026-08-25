#!/usr/bin/env python3
# eu_train.py — fine-tune EuroBERT-210m as a multilingual PII token-classifier on
# the openpii-1m per-language shards (CC-BY-4.0). Standard BIO, offsets-derived labels.
#
# Recipe (from EuroBERT fine-tuning card): Adam β1 0.9 / β2 0.95 / eps 1e-5, wd 0.1,
# warmup ratio 0.1, linear LR decay; LR from their classification sweep (~2-6e-5).
import argparse, json, math, os, random, time, glob
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
# transformers>=5.14 dropped ROPE_INIT_FUNCTIONS["default"]. EuroBERT ships custom
# modeling code that still requires it. Register it before model load.
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
if "default" not in ROPE_INIT_FUNCTIONS:
    def _rope_default(config, device=None, seq_len=None):
        dim = config.hidden_size // config.num_attention_heads
        theta = getattr(config, "rope_theta", 10000.0)
        import torch as _t
        inv_freq = 1.0 / (theta ** (_t.arange(0, dim, 2, dtype=_t.int64, device=device).float() / dim))
        return inv_freq, 1.0
    ROPE_INIT_FUNCTIONS["default"] = _rope_default

O_IDX = 0
def build_labels(types):
    names = ["O"]
    for t in types:
        names += [f"B-{t}", f"I-{t}"]
    return {n: i for i, n in enumerate(names)}

def char_spans_to_bio(offsets, spans, seq_len, lmap):
    """offsets: list[(a,b)] per token (special tokens (0,0)). Label: B at first covered token,
    I on continuation. Subword coverage by overlap."""
    labels = [O_IDX] * seq_len
    for sp in spans:
        s, e, lab = sp["start"], sp["end"], sp["type"]
        b_idx, i_idx = lmap.get(f"B-{lab}"), lmap.get(f"I-{lab}")
        if b_idx is None:
            continue
        first = True
        for ti, (a, b) in enumerate(offsets[:seq_len]):
            if b == 0 or a == b:  # special/pad
                if not first:
                    break
                continue
            if b <= s:
                continue
            if a >= e:
                break
            labels[ti] = b_idx if first else i_idx
            first = False
    return labels

class ShardDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        return self.rows[i]

def collate(batch, tok, seq_len, lmap, pad_id):
    texts = [r["text"] for r in batch]
    enc = tok(texts, padding=False, truncation=True, max_length=seq_len,
              return_offsets_mapping=True)
    ids_batch, attn_batch, lab_batch = [], [], []
    for r, ids, offs in zip(batch, enc["input_ids"], enc["offset_mapping"]):
        labs = char_spans_to_bio(offs, r["spans"], len(ids), lmap)
        ids_batch.append(ids)
        attn_batch.append([1] * len(ids))
        lab_batch.append(labs)
    maxlen = max(len(x) for x in ids_batch)
    def pad(xs, val):
        return torch.stack([torch.tensor(x + [val] * (maxlen - len(x))) for x in xs])
    ids = pad(ids_batch, pad_id)
    attn = pad(attn_batch, 0)
    labs = pad(lab_batch, -100)
    return ids, attn, labs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/models/EuroBERT-210m")
    ap.add_argument("--shards", default="/scratch/pii_corpus/eu_shards")
    ap.add_argument("--languages", default="all")  # all | comma list
    ap.add_argument("--cap-per-lang", type=int, default=0)
    ap.add_argument("--cap-total", type=int, default=0)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3.6e-5)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--save", default="/scratch/pii_corpus/eurobert_pii.pt")
    ap.add_argument("--use-wandb", action="store_true")
    ap.add_argument("--run-name", default="eurobert-pii-eu")
    args = ap.parse_args()

    torch.manual_seed(0); random.seed(0)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # --- labels from train shards ---
    label_types = sorted(json.load(open(os.path.join(args.shards, "stats.json")))["labels"].keys())
    lmap = build_labels([t for t in label_types if t not in ("O",)])
    print(f"labels: {len(lmap)} ({len(label_types)} PII types)", flush=True)

    def load(split):
        files = sorted(glob.glob(os.path.join(args.shards, f"eu-{split}_*.jsonl")))
        sel = args.languages.split(",") if args.languages != "all" else None
        rows, per = [], {}
        for fp in files:
            lg = os.path.basename(fp).split("_")[1].split(".")[0]
            if sel and lg not in sel:
                continue
            n = 0
            with open(fp) as f:
                for line in f:
                    if args.cap_per_lang and n >= args.cap_per_lang:
                        break
                    r = json.loads(line)
                    r["spans"] = [s for s in r["spans"] if f"B-{s['type']}" in lmap]
                    rows.append(r); n += 1
            per[lg] = n
        random.shuffle(rows)
        if args.cap_total:
            rows = rows[: args.cap_total]
        print(f"{split}: {len(rows)} rows across {len(per)} langs | sample: {dict(list(per.items())[:5])}...", flush=True)
        return rows

    tr_rows = load("train")
    va_rows = load("val")

    model = AutoModelForTokenClassification.from_pretrained(
        args.model, trust_remote_code=True, num_labels=len(lmap))
    dev = "cuda"
    model.to(dev).to(torch.bfloat16) if hasattr(torch, "bfloat16") else model.to(dev)
    model.config.num_labels = len(lmap)
    nparams = sum(p.numel() for p in model.parameters())
    print("params:", round(nparams / 1e6, 1), "M", flush=True)

    wb = None
    if args.use_wandb:
        try:
            import wandb
            wandb.login(key=open("/root/.wandb_key").read().strip())
            wandb.init(project="eurobert-pii", name=args.run_name,
                       config={"params": nparams, "labels": len(lmap), "lr": args.lr,
                               "batch": args.batch, "seq": args.seq, "train_rows": len(tr_rows),
                               "languages": 23})
            wb = wandb
            print(f"wandb: tracking run '{args.run_name}'", flush=True)
        except Exception as e:
            print("wandb skip:", e, flush=True)

    ds = ShardDataset(tr_rows)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    collate_fn=lambda b: collate(b, tok, args.seq, lmap, pad_id), num_workers=8)
    vadl = DataLoader(ShardDataset(va_rows[:384]), batch_size=args.batch, shuffle=False,
                      collate_fn=lambda b: collate(b, tok, args.seq, lmap, pad_id), num_workers=8)

    decay, nodecay = [], []
    for n_, p in model.named_parameters():
        (nodecay if ("norm" in n_ or "bias" in n_) else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": nodecay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), eps=1e-5)
    steps = len(dl) * args.epochs
    wu = int(args.warmup * steps)
    def lr_at(s):
        if s < wu:
            return args.lr * s / max(1, wu)
        return args.lr * (1 - (s - wu) / max(1, steps - wu))

    best_val, bad_run, step, t0 = 1e9, 0, 0, time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for ids, attn, labs in dl:
            ids, attn, labs = ids.to(dev), attn.to(dev), labs.to(dev)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=ids, attention_mask=attn, labels=None)
                loss = F.cross_entropy(out.logits.view(-1, out.logits.shape[-1]),
                                       labs.view(-1), ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1; step += 1
            if step % args.eval_every == 0:
                model.eval()
                vt, vn = 0.0, 0
                with torch.no_grad():
                    for ids, attn, labs in vadl:
                        ids, attn, labs = ids.to(dev), attn.to(dev), labs.to(dev)
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            out = model(input_ids=ids, attention_mask=attn, labels=None)
                            vl = F.cross_entropy(out.logits.view(-1, out.logits.shape[-1]), labs.view(-1), ignore_index=-100)
                        vt += vl.item(); vn += 1
                v = vt / max(1, vn)
                model.train()
                marker = ""
                if v < best_val - 5e-3:  # 0.005 min-delta cuts churn on the small eval slice
                    best_val, bad_run = v, 0
                    torch.save({"model": model.state_dict(), "labels": lmap},
                               args.save)
                    marker = " *BEST"
                else:
                    bad_run += 1
                print(f"ep{ep} step{step} train={tot/max(1,nb):.4f} val={v:.4f}{marker}", flush=True)
                if wb:
                    wb.log({"step": step, "epoch": ep, "train_loss": round(tot / max(1, nb), 4),
                            "val": round(v, 4), "val_best": round(best_val, 4),
                            "lr": float(opt.param_groups[0]["lr"])})
                tot, nb = 0.0, 0
                if bad_run >= max(3, 2000 // args.eval_every):  # 2000-step patience (was 150!
                    print("early stop", flush=True)
                    print(f"BEST val {best_val} -> {args.save}", flush=True)
                    return
        print(f"ep{ep} done {(time.time()-t0)/60:.0f}min", flush=True)
    print(f"BEST val {best_val} -> {args.save}", flush=True)

if __name__ == "__main__":
    main()
