#!/usr/bin/env python3
# pretrain_mlm_bin.py — Stage-1 MLM pretraining reading the pre-tokenized .bin corpus.
#
# Reads the sharded binary corpus (uint32 [len, ids...] per doc) produced by
# build_pretrain_corpus.py, streams docs, applies MLM masking, trains the 1.58-bit
# encoder. This decouples training from the (slow) HF dataset loader so the GPU
# is fed continuously.
#
# Uses the BitNet b1.58 recipe: warmup+cosine LR, WD=0.1, bf16 autocast, ternary
# on forward (STE), dropout. Logs every step to wandb when --use-wandb.
#
# Run:
#   /root/venv/bin/python pretrain_mlm_bin.py --bin /scratch/pii_corpus/pretok_corpus.bin
import argparse, os, math, random, time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from bitnet_pretrain import EncoderConfig, BitnetMLM


def mask_tokens(input_ids, mask_prob=0.15, mask_id=4, pad_id=0):
    labels = input_ids.clone()
    B, S = input_ids.shape
    non_pad = (input_ids != pad_id)
    do_mask = (torch.rand(B, S, device=input_ids.device) < mask_prob) & non_pad
    rand2 = torch.rand(B, S, device=input_ids.device)
    mask_pos = do_mask & (rand2 < 0.8)
    input_ids[mask_pos] = mask_id
    rand_pos = do_mask & (rand2 >= 0.8) & (rand2 < 0.9)
    input_ids[rand_pos] = torch.randint(0, mask_id, (B, S), device=input_ids.device)[rand_pos]
    labels[~do_mask] = -100
    return input_ids, labels


class BinDataset(Dataset):
    """Read uint32 var-length docs [len, ids...] from shards; return padded [S] tensor."""
    def __init__(self, pattern, seq_len, max_docs=None):
        self.seq_len = seq_len
        # expand shard glob (base + _shardN.bin)
        self.files = []
        base = pattern
        if os.path.exists(base):
            self.files.append(base)
        i = 1
        while True:
            p = base.replace(".bin", f"_shard{i}.bin")
            if os.path.exists(p):
                self.files.append(p)
                i += 1
            else:
                break
        # count docs + precompute byte offsets for random access
        self.docs = []  # (file_idx, byte_start, byte_len) for each doc
        import struct
        for fi, fpath in enumerate(self.files):
            data = open(fpath, "rb").read()
            arr = np.frombuffer(data, dtype=np.uint32)
            pos = 0
            while pos < len(arr):
                n = int(arr[pos])
                if n <= 0 or pos + 1 + n > len(arr):
                    break
                self.docs.append((fi, pos, n))
                pos += 1 + n
        if max_docs:
            self.docs = self.docs[:max_docs]
        # cache each file's raw bytes for fast access
        self._data = []
        for fpath in self.files:
            self._data.append(open(fpath, "rb").read())
        print(f"loaded {len(self.files)} shard(s), {len(self.docs)} docs", flush=True)

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, i):
        fi, pos, n = self.docs[i]
        arr = np.frombuffer(self._data[fi], dtype=np.uint32, offset=pos * 4, count=n)
        ids = arr.astype(np.int64)  # uint32 -> int64 (ids >= 2^31 possible? no, < 32k)
        # truncate/pad to seq_len
        out = np.zeros(self.seq_len, dtype=np.int64)
        m = min(n, self.seq_len)
        out[:m] = ids[:m]
        return torch.tensor(out, dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="/scratch/pii_corpus/pretok_corpus.bin")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=0)  # 0 = run full epochs
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--hidden", type=int, default=640)
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--mask-prob", type=float, default=0.15)
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--save", default="/scratch/pii_corpus/enc_100m.pt")
    ap.add_argument("--max-docs", type=int, default=0)
    ap.add_argument("--use-wandb", action="store_true")
    ap.add_argument("--run-name", default="bitnet-100m-encoder")
    args = ap.parse_args()

    torch.manual_seed(0); random.seed(0)
    dset = BinDataset(args.bin, args.seq, args.max_docs or None)
    dl = DataLoader(dset, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True)

    c = EncoderConfig(vocab_size=args.vocab, hidden_size=args.hidden, num_layers=args.layers,
                      num_heads=8, num_kv_heads=8, intermediate_size=args.hidden*4,
                      max_seq_len=args.seq, dropout=0.1)
    model = BitnetMLM(c).to("cuda")
    print("params:", round(model.param_count()/1e6, 2), "M", flush=True)

    wandb = None
    if args.use_wandb:
        try:
            import wandb as _w
            try:
                os.environ["WANDB_API_KEY"] = open("/root/.wandb_key").read().strip()
            except Exception:
                pass
            wandb = _w.init(project="bitnet-1.58-pii-encoder", name=args.run_name, config=asdict(c))
            print("wandb initialized", flush=True)
        except Exception as e:
            wandb = None; print("wandb skip:", e, flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98))
    n_batches = len(dl)
    n_total = args.max_steps or (n_batches * args.epochs)
    warm = max(1, int(n_total * 0.02))
    step = 0; t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        el = 0.0; cnt = 0
        for x in dl:
            if args.max_steps and step >= args.max_steps:
                break
            x = x.to("cuda")
            masked, labels = mask_tokens(x.clone(), args.mask_prob, 4, 0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, loss = model(masked, labels=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            lr = args.lr * (step/max(1,warm)) if step < warm else args.lr*0.5*(1+math.cos(math.pi*(step-warm)/max(1,n_total-warm)))
            for pg in opt.param_groups: pg["lr"] = lr
            opt.step(); opt.zero_grad()
            el += loss.item(); cnt += 1; step += 1
            if wandb is not None:
                wandb.log({"loss": loss.item(), "lr": lr, "step": step, "epoch": ep+1}, step=step)
            if step % 100 == 0:
                print(f"ep{ep+1} step{step} loss={el/max(1,cnt):.4f} lr={lr:.2e} {time.time()-t0:.0f}s", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        print(f"EPOCH {ep+1}/{args.epochs} avg_loss={el/max(1,cnt):.4f}", flush=True)
        if args.max_steps and step >= args.max_steps:
            break
    torch.save({"model": model.encoder.state_dict(), "config": asdict(c), "vocab": args.vocab}, args.save)
    print(f"SAVED encoder -> {args.save}", flush=True)
    if wandb is not None:
        try: wandb.finish()
        except Exception: pass


if __name__ == "__main__":
    main()
