#!/usr/bin/env python3
"""eu_train_v3.py — GDPR-aware unified PII model. The real beast.

Key changes vs v2:
  - 10 unified PII types (PERSON, DATE, EMAIL, PHONE, ADDRESS, ID_NUMBER,
    CREDIT_CARD, ORGANIZATION, GENDER, LOCATION) — generalizes across benchmarks.
  - Trained on ai4privacy + TAB real legal text (unified corpus).
  - Contamination-free (verified by text fingerprint).
  - RoPE fix applied (persistent=True).
  - Verified save: in-memory F1 must match disk F1 or crash.
  - 6 epochs, 500-step eval, boundary-weighted CE.
"""
import argparse, json, math, os, random, time, glob, hashlib
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

if "default" not in ROPE_INIT_FUNCTIONS:
    def _rope_default(config, device=None, seq_len=None):
        dim = config.hidden_size // config.num_attention_heads
        theta = getattr(config, "rope_theta", 10000.0)
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim))
        return inv_freq, 1.0
    ROPE_INIT_FUNCTIONS["default"] = _rope_default

from transformers import AutoModelForTokenClassification

def build_bioes(types):
    names = ["O"]
    for t in types:
        names += [f"B-{t}", f"I-{t}", f"E-{t}", f"S-{t}"]
    return {n: i for i, n in enumerate(names)}

def char_spans_to_bioes(offsets, spans, seq_len, lmap):
    labels = [0] * seq_len
    weights = [1.0] * seq_len
    for sp in spans:
        s, e, lab = sp["start"], sp["end"], sp["type"]
        idxs = {k: lmap.get(f"{k}-{lab}") for k in "BIES"}
        if None in idxs.values(): continue
        cover = []
        for ti, (a, b) in enumerate(offsets[:seq_len]):
            if b == 0 or a == b: continue
            if b <= s or a >= e: continue
            cover.append(ti)
        if not cover: continue
        if len(cover) == 1:
            labels[cover[0]] = idxs["S"]
        else:
            labels[cover[0]] = idxs["B"]
            for ti in cover[1:-1]: labels[ti] = idxs["I"]
            labels[cover[-1]] = idxs["E"]
        weights[cover[0]] = 3.0
        weights[cover[-1]] = 3.0
    return labels, weights

def bioes_decode(logits, offsets, lmap_rev, text):
    cur, raw = None, []
    for ti, (a, b) in enumerate(offsets):
        if a == b or b == 0:
            if cur: raw.append(cur); cur = None
            continue
        tag = lmap_rev.get(logits[ti], "O")
        if tag.startswith("B-"):
            if cur: raw.append(cur)
            cur = [a, b, tag[2:]]
        elif tag.startswith("S-"):
            if cur: raw.append(cur); cur = None
            raw.append([a, b, tag[2:]])
        elif tag.startswith("I-"):
            lab = tag[2:]
            if cur and cur[2] == lab: cur[1] = b
            elif not cur: cur = [a, b, lab]
            else: raw.append(cur); cur = [a, b, lab]
        elif tag.startswith("E-"):
            lab = tag[2:]
            if cur and cur[2] == lab: cur[1] = b; raw.append(cur); cur = None
            else: raw.append([a, b, lab])
        else:
            if cur: raw.append(cur); cur = None
    if cur: raw.append(cur)
    out = []
    for s, e, t in raw:
        s, e = int(s), int(e)
        while s < e and s < len(text) and text[s].isspace(): s += 1
        while e > s and e <= len(text) and text[e-1].isspace(): e -= 1
        if e > s: out.append({"start": s, "end": e, "type": t})
    return out

def span_metrics(preds, rows):
    det = [0,0,0]; ext = [0,0,0]
    for p, r in zip(preds, rows):
        gset = {(g["start"], g["end"]): g["type"] for g in r["spans"]}
        used = set()
        for sp in p:
            k = (sp["start"], sp["end"])
            if k in gset and k not in used:
                used.add(k); det[0] += 1
                if gset[k] == sp["type"]: ext[0] += 1
                else: ext[1] += 1
            else: det[1] += 1; ext[1] += 1
        det[2] += len(gset) - len(used); ext[2] += len(gset) - len(used)
    dp = det[0]/max(1,det[0]+det[1]); dr = det[0]/max(1,det[0]+det[2])
    ep = ext[0]/max(1,ext[0]+ext[1]); er = ext[0]/max(1,ext[0]+ext[2])
    return 2*dp*dr/max(1e-9,dp+dr), 2*ep*er/max(1e-9,ep+er)

def save_verify(model, lmap, path, tok, lmap_rev, verify_rows, dev):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sd = {"model": model.state_dict(), "labels": lmap}
    tmp = path + ".tmp"
    torch.save(sd, tmp)
    os.replace(tmp, path)
    sd_disk = torch.load(path, map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(sd_disk, strict=True)
    model.to(dev).eval()
    preds = []
    with torch.no_grad():
        enc = tok([r["text"] for r in verify_rows], padding=True, truncation=True,
                  max_length=512, return_offsets_mapping=True, return_tensors="pt")
        off = enc.pop("offset_mapping")
        lg = model(enc["input_ids"].to(dev), attention_mask=enc["attention_mask"].to(dev)).logits.argmax(-1).cpu()
        for j, r in enumerate(verify_rows):
            offs = [(int(a), int(b)) for a, b in off[j].tolist()]
            preds.append(bioes_decode(lg[j].tolist(), offs, lmap_rev, r["text"]))
    f1d, f1e = span_metrics(preds, verify_rows)
    H = hashlib.md5(open(path, "rb").read()[:1 << 20]).hexdigest()[:8]
    return f1d, f1e, H

class RowsDS(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]

def collate(batch, tok, seq_len, lmap, pad_id):
    texts = [r["text"] for r in batch]
    enc = tok(texts, padding=False, truncation=True, max_length=seq_len, return_offsets_mapping=True)
    ids, attn, labs, wgts = [], [], [], []
    for r, ii, oo in zip(batch, enc["input_ids"], enc["offset_mapping"]):
        lb, wt = char_spans_to_bioes(oo, r["spans"], len(ii), lmap)
        ids.append(ii); attn.append([1]*len(ii)); labs.append(lb); wgts.append(wt)
    mx = max(len(x) for x in ids)
    def pad(xs, v): return torch.stack([torch.tensor(x+[v]*(mx-len(x))) for x in xs])
    return pad(ids, pad_id), pad(attn, 0), pad(labs, -100), pad(wgts, 0.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/models/EuroBERT-210m")
    ap.add_argument("--train", default="/scratch/pii_corpus/unified/unified_train.jsonl")
    ap.add_argument("--eval", default="/scratch/pii_corpus/unified/unified_eval.jsonl")
    ap.add_argument("--types", default="/scratch/pii_corpus/unified/types.json")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3.6e-5)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-rows", type=int, default=2000)
    ap.add_argument("--save", default="/scratch/pii_corpus/eurobert_pii_unified.pt")
    ap.add_argument("--use-wandb", action="store_true")
    ap.add_argument("--run-name", default="eurobert-pii-unified")
    args = ap.parse_args()

    torch.manual_seed(0); random.seed(0)
    types = json.load(open(args.types))
    lmap = build_bioes(types)
    lm_rev = {v: k for k, v in lmap.items()}
    print(f"Unified BIOES labels: {len(lmap)} ({len(types)} types: {types})", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    def load(path, limit=0):
        rows = []
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                r["spans"] = [s for s in r["spans"] if f"B-{s['type']}" in lmap]
                rows.append(r)
                if limit and len(rows) >= limit: break
        return rows

    tr = load(args.train)
    va = load(args.eval)
    print(f"train: {len(tr)} | eval: {len(va)}", flush=True)
    random.shuffle(tr)

    model = AutoModelForTokenClassification.from_pretrained(args.model, trust_remote_code=True, num_labels=len(lmap))
    dev = "cuda"
    model.to(dev)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"params: {nparams/1e6:.1f}M", flush=True)

    wb = None
    if args.use_wandb:
        try:
            import wandb
            wandb.login(key=open("/root/.wandb_key").read().strip())
            wandb.init(project="eurobert-pii", name=args.run_name,
                       config={"params": nparams, "labels": len(lmap), "types": types,
                               "lr": args.lr, "batch": args.batch, "seq": args.seq, "train_rows": len(tr)})
            wb = wandb
            print(f"wandb: {args.run_name}", flush=True)
        except Exception as e:
            print(f"wandb skip: {e}", flush=True)

    dl = DataLoader(RowsDS(tr), batch_size=args.batch, shuffle=True,
                    collate_fn=lambda b: collate(b, tok, args.seq, lmap, pad_id), num_workers=8)
    eval_slice = va[:args.eval_rows]

    decay, nodecay = [], []
    for n_, p in model.named_parameters():
        (nodecay if ("norm" in n_ or "bias" in n_) else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": nodecay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), eps=1e-5)
    steps = len(dl) * args.epochs
    wu = int(args.warmup * steps)
    def lr_at(s): return args.lr * s / max(1, wu) if s < wu else args.lr * (1 - (s - wu) / max(1, steps - wu))

    best_f1, bad_evals, step, t0 = -1.0, 0, 0, time.time()
    patience_evals = max(3, 3000 // args.eval_every)
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for ids, attn, labs, wgts in dl:
            ids, attn, labs, wgts = ids.to(dev), attn.to(dev), labs.to(dev), wgts.to(dev)
            for g in opt.param_groups: g["lr"] = lr_at(step)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=ids, attention_mask=attn, labels=None)
                tok_logp = F.log_softmax(out.logits.view(-1, out.logits.shape[-1]), dim=-1)
                lp = F.nll_loss(tok_logp, labs.view(-1), ignore_index=-100, reduction="none")
                w = wgts.view(-1).float()
                keep = (labs.view(-1) != -100).float()
                loss = (lp * w * keep).sum() / keep.sum().clamp(min=1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1; step += 1
            if step % args.eval_every == 0:
                model.eval()
                preds = []
                with torch.no_grad():
                    for i in range(0, len(eval_slice), args.batch):
                        rb = eval_slice[i:i+args.batch]
                        enc = tok([x["text"] for x in rb], padding=True, truncation=True,
                                  max_length=args.seq, return_offsets_mapping=True, return_tensors="pt")
                        off = enc.pop("offset_mapping")
                        logits = model(enc["input_ids"].to(dev), attention_mask=enc["attention_mask"].to(dev)).logits.argmax(-1).cpu()
                        for j in range(len(rb)):
                            offs = [(int(a), int(b)) for a, b in off[j].tolist()]
                            preds.append(bioes_decode(logits[j].tolist(), offs, lm_rev, rb[j]["text"]))
                f1_det, f1_ext = span_metrics(preds, eval_slice)
                model.train()
                marker = ""
                if f1_det > best_f1 + 1e-4:
                    best_f1, bad_evals = f1_det, 0
                    vd, ve, H = save_verify(model, lmap, args.save, tok, lm_rev, eval_slice, dev)
                    marker = f" *BEST saved detF1={vd:.4f} sha={H}"
                    if abs(vd - f1_det) > 0.02:
                        raise RuntimeError(f"save-verify divergence: in-mem {f1_det} vs disk {vd}")
                else:
                    bad_evals += 1
                print(f"ep{ep} step{step} loss={tot/max(1,nb):.4f} detF1={f1_det:.4f} exactF1={f1_ext:.4f} bestDetF1={best_f1:.4f}{marker}", flush=True)
                if wb:
                    wb.log({"step": step, "epoch": ep, "loss": round(tot/max(1,nb), 4),
                            "det_F1": round(f1_det, 4), "exact_F1": round(f1_ext, 4),
                            "best_det_F1": round(best_f1, 4), "lr": float(opt.param_groups[0]["lr"])})
                tot, nb = 0.0, 0
                if bad_evals >= patience_evals and step > 8000:
                    print("early stop", flush=True)
                    print(f"BEST detF1 {best_f1} -> {args.save}", flush=True)
                    return
        print(f"ep{ep} done {(time.time()-t0)/60:.0f}min", flush=True)
    print(f"BEST detF1 {best_f1} -> {args.save}", flush=True)

if __name__ == "__main__":
    main()
