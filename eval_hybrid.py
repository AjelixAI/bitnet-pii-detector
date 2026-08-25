#!/usr/bin/env python3
"""eval_hybrid.py — run detF1 / exactF1 on the final checkpoint A/B:
  - BIOES decode alone vs BIOES + hybrid (validators+cues).
  - Same strict span metric as eval_strict.py. Show both."""
import argparse, json, sys, torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
sys.path.insert(0, "/root/bitnet_train")
from eval_strict import rope_shim
import hybrid_decode

def eval_mode(ckpt, model_dir, rows, use_hybrid):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    lmap_rev = {v: k for k, v in sd["labels"].items()}
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    rope_shim()
    m = AutoModelForTokenClassification.from_pretrained(model_dir, trust_remote_code=True, num_labels=len(lmap_rev))
    m.load_state_dict(sd["model"], strict=True); m.cuda().eval()
    det=[0,0,0]; ext=[0,0,0]
    with torch.no_grad():
        for i in range(0, len(rows), 48):
            b = rows[i:i+48]
            enc = tok([r["text"] for r in b], padding=True, truncation=True, max_length=512,
                      return_offsets_mapping=True, return_tensors="pt")
            off = enc.pop("offset_mapping")
            lg = m(enc["input_ids"].cuda(), attention_mask=enc["attention_mask"].cuda()).logits.argmax(-1).cpu()
            for j, r in enumerate(b):
                offs = [(int(a), int(b)) for a, b in off[j].tolist()]
                preds = hybrid_decode.decode(r["text"], lg[j].tolist(), offs, lmap_rev, use_hybrid=use_hybrid)
                gset = {(g["start"], g["end"]): g["type"] for g in r["spans"]}
                used = set()
                for sp in preds:
                    k = (sp["start"], sp["end"])
                    if k in gset and k not in used:
                        used.add(k); det[0]+=1
                        ext[0] += 1 if gset[k] == sp["type"] else 0
                        ext[1] += 0 if gset[k] == sp["type"] else 1
                    else:
                        det[1]+=1; ext[1]+=1
                det[2] += len(gset) - len(used)
                ext[2] += len(gset) - len(used)
    def f1(v):
        p = v[0]/max(1,v[0]+v[1]); r = v[0]/max(1,v[0]+v[2])
        return 2*p*r/max(1e-9,p+r), p, r
    dd = f1(det); ee = f1(ext)
    print(f"  hybrid={use_hybrid}: STRICT det F1={dd[0]:.4f} P={dd[1]:.4f} R={dd[2]:.4f} | exact-type F1={ee[0]:.4f} P={ee[1]:.4f} R={ee[2]:.4f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/scratch/pii_corpus/eurobert_pii_v2.pt")
    ap.add_argument("--model", default="/root/models/EuroBERT-210m")
    ap.add_argument("--val-file", default="/scratch/pii_corpus/eu_shards/eu-val_en.jsonl")
    ap.add_argument("--rows", type=int, default=300)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.val_file)][:args.rows]
    print("=== A/B hybrid decode on same rows ===")
    eval_mode(args.ckpt, args.model, rows, use_hybrid=False)
    eval_mode(args.ckpt, args.model, rows, use_hybrid=True)

if __name__ == "__main__":
    main()
