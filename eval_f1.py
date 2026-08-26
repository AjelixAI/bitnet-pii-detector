"""Standalone span-F1 eval for Piiranha-replication checkpoints.
Token-level entity F1 (exact token-run + type match), plus type-agnostic detection F1.
Usage: python eval_f1.py [ckpt_path]
"""
import sys, collections, importlib.util
import torch
import torch.nn.functional as F

sys.path.insert(0, "/root")
spec = importlib.util.spec_from_file_location("tp", "/root/train_piiranha_rep.py")
tp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tp)

from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_from_disk

MODEL = "/root/models/mdeberta-v3-base"
CKPT = sys.argv[1] if len(sys.argv) > 1 else "/root/piiranha_rep.pt"
SEQ = 256
PER_LANG = 2000
BATCH = 64


def runs_from_labels(labels, offs):
    """Contiguous token runs of same type = one entity (I-only scheme)."""
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
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, fix_mistral_regex=True)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL, trust_remote_code=True, num_labels=len(tp.LABEL_NAMES))
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.float().cuda().eval()
    print(f"loaded {CKPT}", flush=True)

    ds = load_from_disk("/root/data/pii-masking-400k")
    va = ds["validation"]
    by_lang = collections.defaultdict(list)
    for i in range(len(va)):
        by_lang[va[i].get("language", "?")].append(i)
    sel = []
    for lg, idxs in sorted(by_lang.items()):
        sel += idxs[:PER_LANG]

    # pre-tokenize eval set
    items = []
    for i in sel:
        r = va[i]
        spans = [{"start": s["start"], "end": s["end"], "type": s["label"]}
                 for s in r["privacy_mask"] if s["label"] in tp.PII_TYPES]
        enc = tok(r["source_text"], padding=False, truncation=True, max_length=SEQ,
                  return_offsets_mapping=True)
        items.append({"ids": enc["input_ids"], "offs": enc["offset_mapping"],
                      "labs": tp.chars_to_ilabels(enc["offset_mapping"], spans, len(enc["input_ids"])),
                      "lang": r.get("language", "?")})
    print(f"eval rows: {len(items)}", flush=True)

    def pad_batch(chunk):
        mx = max(len(x["ids"]) for x in chunk)
        ids = torch.zeros(len(chunk), mx, dtype=torch.long)
        attn = torch.zeros(len(chunk), mx, dtype=torch.long)
        for k, x in enumerate(chunk):
            n = len(x["ids"])
            ids[k, :n] = torch.tensor(x["ids"])
            attn[k, :n] = 1
        return ids, attn

    typed = {"g": 0, "p": 0, "tp": 0}
    det = {"g": 0, "p": 0, "tp": 0}
    per_lang = collections.defaultdict(lambda: {"g": 0, "p": 0, "tp": 0})
    lsum, n = 0.0, 0
    with torch.no_grad():
        for bi in range(0, len(items), BATCH):
            chunk = items[bi:bi + BATCH]
            ids, attn = pad_batch(chunk)
            ids, attn = ids.cuda(), attn.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=ids, attention_mask=attn)
            logits = out.logits.float()
            labs = torch.full(ids.shape, -100, dtype=torch.long)
            for k, x in enumerate(chunk):
                labs[k, :len(x["labs"])] = torch.tensor(x["labs"])
            lsum += F.cross_entropy(logits.view(-1, len(tp.LABEL_NAMES)),
                                    labs.view(-1).cuda(), ignore_index=-100).item()
            n += 1
            preds = logits.argmax(-1).cpu()
            for k, x in enumerate(chunk):
                L = len(x["ids"])
                g_runs = runs_from_labels(x["labs"][:L], x["offs"])
                p_runs = runs_from_labels(preds[k][:L].tolist(), x["offs"])
                gs, ps = set(g_runs), set(p_runs)
                gd = {(a, b) for a, b, _ in g_runs}
                pd = {(a, b) for a, b, _ in p_runs}
                for c in (typed, per_lang[x["lang"]]):
                    c["g"] += len(gs); c["p"] += len(ps); c["tp"] += len(gs & ps)
                det["g"] += len(gd); det["p"] += len(pd); det["tp"] += len(gd & pd)

    def f1(c):
        p = c["tp"] / max(1, c["p"])
        r = c["tp"] / max(1, c["g"])
        return 2 * p * r / max(1e-9, p + r), p, r

    print(f"val loss: {lsum / max(1, n):.4f}")
    tf1, tp_, tr_ = f1(typed)
    df1, dp_, dr_ = f1(det)
    print(f"SPAN F1 (type-aware):  {tf1:.4f}  (P={tp_:.4f} R={tr_:.4f})  support={typed['g']}")
    print(f"DET F1 (type-agnostic): {df1:.4f}  (P={dp_:.4f} R={dr_:.4f})")
    for lg in sorted(per_lang):
        c = per_lang[lg]
        f, p, r = f1(c)
        print(f"  {lg}: spanF1={f:.4f} (P={p:.3f} R={r:.3f}) support={c['g']}")


if __name__ == "__main__":
    main()
