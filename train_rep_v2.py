"""Piiranha replication v2 — H100: bf16 autocast, native batch 128, pre-tokenized data,
span-F1 evaluated every 500 steps, best-by-F1 checkpoint saving.

Exact Piiranha recipe: lr 5e-5, batch 128, 5 epochs, warmup 5%, linear decay,
Adam betas (0.9, 0.999) eps 1e-8, weight decay 0.0, seq 256.
"""
import argparse, time, os, sys, collections, random, importlib.util
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("tp", os.path.join(_HERE, "train_piiranha_rep.py"))
tp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tp)

from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_from_disk


def runs_from_labels(labels, offs):
    runs, cur = [], None
    for ti, (a, b) in enumerate(offs):
        if a == b or b == 0:
            if cur:
                runs.append(cur)
            cur = None
            continue
        p = labels[ti]
        if p == tp.O_IDX or p >= len(tp.PII_TYPES):
            if cur:
                runs.append(cur)
            cur = None
            continue
        t = tp.PII_TYPES[p]
        if cur and cur[2] == t and ti == cur[1] + 1:
            cur = (cur[0], ti, t)
        else:
            if cur:
                runs.append(cur)
            cur = (ti, ti, t)
    if cur:
        runs.append(cur)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/models/mdeberta-v3-base")
    ap.add_argument("--data", default="/root/data/pii-masking-400k")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-rows", type=int, default=1200, help="per language")
    ap.add_argument("--save", default="/root/piiranha_v2.pt")
    ap.add_argument("--use-wandb", action="store_true")
    ap.add_argument("--run-name", default="piiranha-v2-h100")
    args = ap.parse_args()

    torch.manual_seed(42)
    random.seed(42)
    dev = "cuda"
    ds = load_from_disk(args.data)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True,
                                        fix_mistral_regex=True)

    def convert(split):
        rows = []
        for r in ds[split]:
            text = r["source_text"]
            spans = [{"start": s["start"], "end": s["end"], "type": s["label"]}
                     for s in r["privacy_mask"] if s["label"] in tp.PII_TYPES]
            if text:
                rows.append({"text": text, "spans": spans, "lang": r.get("language", "?")})
        return rows

    print("converting...", flush=True)
    tr = convert("train")
    print(f"converted: {len(tr)} train, {len(ds['validation'])} val", flush=True)

    def pretok(rows):
        out = []
        B = 2048
        for i in range(0, len(rows), B):
            chunk = rows[i:i + B]
            enc = tok([r["text"] for r in chunk], padding=False, truncation=True,
                      max_length=args.seq, return_offsets_mapping=True)
            for r, ii, oo in zip(chunk, enc["input_ids"], enc["offset_mapping"]):
                out.append({"ids": ii, "offs": oo,
                            "labs": tp.chars_to_ilabels(oo, r["spans"], len(ii)),
                            "lang": r["lang"]})
        return out

    print("tokenizing train...", flush=True)
    tr = pretok(tr)
    print(f"train tokenized: {len(tr)}", flush=True)

    by_lang = collections.defaultdict(list)
    for i in range(len(ds["validation"])):
        by_lang[ds["validation"][i].get("language", "?")].append(i)
    eval_rows = []
    for lg, idxs in sorted(by_lang.items()):
        for i in idxs[:args.eval_rows]:
            r = ds["validation"][i]
            spans = [{"start": s["start"], "end": s["end"], "type": s["label"]}
                     for s in r["privacy_mask"] if s["label"] in tp.PII_TYPES]
            eval_rows.append({"text": r["source_text"], "spans": spans,
                              "lang": r.get("language", "?")})
    va_sel = pretok(eval_rows)
    print(f"eval rows: {len(va_sel)} (stratified, {len(by_lang)} languages)", flush=True)

    model = AutoModelForTokenClassification.from_pretrained(
        args.model, trust_remote_code=True, num_labels=len(tp.LABEL_NAMES)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                           eps=1e-8, weight_decay=args.wd)

    def pad_batch(items):
        mx = max(len(x["ids"]) for x in items)
        ids = torch.zeros(len(items), mx, dtype=torch.long)
        attn = torch.zeros(len(items), mx, dtype=torch.long)
        labs = torch.full((len(items), mx), -100, dtype=torch.long)
        for k, x in enumerate(items):
            n = len(x["ids"])
            ids[k, :n] = torch.tensor(x["ids"])
            attn[k, :n] = 1
            labs[k, :n] = torch.tensor(x["labs"])
        return ids, attn, labs

    def evaluate():
        model.eval()
        typed = {"g": 0, "p": 0, "tp": 0}
        det = {"g": 0, "p": 0, "tp": 0}
        lsum, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(va_sel), args.batch):
                chunk = va_sel[i:i + args.batch]
                ids, attn, labs = pad_batch(chunk)
                ids, attn, labs = ids.to(dev), attn.to(dev), labs.to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(input_ids=ids, attention_mask=attn)
                logits = out.logits.float()
                lsum += F.cross_entropy(logits.view(-1, len(tp.LABEL_NAMES)),
                                        labs.view(-1), ignore_index=-100).item()
                n += 1
                preds = logits.argmax(-1).cpu()
                for k, x in enumerate(chunk):
                    L = len(x["ids"])
                    g_runs = runs_from_labels(x["labs"][:L], x["offs"])
                    p_runs = runs_from_labels(preds[k][:L].tolist(), x["offs"])
                    gs, ps = set(g_runs), set(p_runs)
                    gd = {(a, b) for a, b, _ in g_runs}
                    pd = {(a, b) for a, b, _ in p_runs}
                    typed["g"] += len(gs); typed["p"] += len(ps); typed["tp"] += len(gs & ps)
                    det["g"] += len(gd); det["p"] += len(pd); det["tp"] += len(gd & pd)

        def f1(c):
            p = c["tp"] / max(1, c["p"])
            r = c["tp"] / max(1, c["g"])
            return 2 * p * r / max(1e-9, p + r), p, r

        tf1, tpm, trm = f1(typed)
        df1, dpm, drm = f1(det)
        model.train()
        return lsum / max(1, n), tf1, df1, (tpm, trm, dpm, drm)

    order = list(range(len(tr)))
    steps_total = (len(tr) // args.batch) * args.epochs
    warm = int(args.warmup * steps_total)
    best_f1 = -1.0
    step = 0
    wb = None
    if args.use_wandb:
        import wandb
        wandb.init(project="eurobert-pii", name=args.run_name, config=vars(args))
        wb = wandb
    print(f"steps: {steps_total} | warmup: {warm} | "
          f"params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)
    t0 = time.time()
    model.train()
    for ep in range(args.epochs):
        random.shuffle(order)
        for bi in range(0, len(order) - args.batch + 1, args.batch):
            items = [tr[j] for j in order[bi:bi + args.batch]]
            ids, attn, labs = pad_batch(items)
            ids, attn, labs = ids.to(dev), attn.to(dev), labs.to(dev)
            lr = (args.lr * step / max(1, warm) if step < warm else
                  args.lr * (1 - (step - warm) / max(1, steps_total - warm)))
            for g in opt.param_groups:
                g["lr"] = lr
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=ids, attention_mask=attn)
            loss = F.cross_entropy(out.logits.view(-1, len(tp.LABEL_NAMES)),
                                   labs.view(-1), ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % args.eval_every == 0:
                vl, tf1, df1, extra = evaluate()
                marker = ""
                if tf1 > best_f1:
                    best_f1 = tf1
                    torch.save({"model": model.state_dict(), "label_names": tp.LABEL_NAMES},
                               args.save)
                    marker = " *BEST"
                tpm, trm, dpm, drm = extra
                print(f"ep{ep+1} step{step} loss={loss.item():.4f} val={vl:.4f} "
                      f"spanF1={tf1:.4f} (P={tpm:.3f} R={trm:.3f}) detF1={df1:.4f} "
                      f"{(time.time()-t0)/60:.0f}min{marker}", flush=True)
                if wb:
                    wb.log({"step": step, "epoch": ep + 1, "loss": round(loss.item(), 4),
                            "val": round(vl, 4), "spanF1": round(tf1, 4),
                            "detF1": round(df1, 4), "best_spanF1": round(best_f1, 4),
                            "lr": lr})
    vl, tf1, df1, _ = evaluate()
    print(f"FINAL spanF1={tf1:.4f} detF1={df1:.4f} best_saved={best_f1:.4f}", flush=True)
    if wb:
        wb.finish()


if __name__ == "__main__":
    main()
