#!/usr/bin/env python3
"""head_to_head.py — all 3 models, same data, same metric, one process.
Strict exact-span detection F1 + exact-type F1. No overlap credit.
"""
import torch, sys, json, time

# --- strict scoring (unit-tested: exact=1.0, partial=0.0) ---
def strict_score(preds, golds):
    tp_d = fp_d = fn_d = tp_e = fp_e = fn_e = 0
    for p, g in zip(preds, golds):
        gset = {(sp["start"], sp["end"]): sp["type"] for sp in g}
        used = set()
        for sp in p:
            k = (sp["start"], sp["end"])
            if k in gset and k not in used:
                used.add(k); tp_d += 1
                if gset[k] == sp["type"]: tp_e += 1
                else: fp_e += 1
            else:
                fp_d += 1; fp_e += 1
        fn_d += len(gset) - len(used)
        fn_e += len(gset) - len(used)
    def f1(tp, fp, fn):
        P = tp / max(1, tp + fp); R = tp / max(1, tp + fn)
        return P, R, 2 * P * R / max(1e-9, P + R)
    return (*f1(tp_d, fp_d, fn_d), *f1(tp_e, fp_e, fn_e), tp_d, fp_d, fn_d, tp_e, fp_e, fn_e)

def report(name, preds, golds):
    dp, dr, df, ep, er, ef, tp_d, fp_d, fn_d, tp_e, fp_e, fn_e = strict_score(preds, golds)
    print(f"{name:30s} det F1={df:.4f} (P={dp:.3f} R={dr:.3f}) | exact F1={ef:.4f} (P={ep:.3f} R={er:.3f}) | tp={tp_d} fp={fp_d} fn={fn_d}")
    return df, ef

# load val rows (en, 300 — same as all prior measurements)
rows = [json.loads(l) for l in open("/scratch/pii_corpus/eu_shards/eu-val_en.jsonl")][:300]
golds = [r["spans"] for r in rows]
print(f"=== HEAD-TO-HEAD: {len(rows)} EN val rows, strict exact-span ===\n")

# --- 1. PRESIDIO ---
print("Running Presidio...", flush=True)
t0 = time.time()
from presidio_analyzer import AnalyzerEngine
an = AnalyzerEngine()
presidio_preds = []
for r in rows:
    results = an.analyze(text=r["text"], language="en")
    presidio_preds.append([{"start": x.start, "end": x.end, "type": x.entity_type} for x in results])
report("Presidio", presidio_preds, golds)
print(f"  ({time.time()-t0:.1f}s)\n")

# --- 2. OURS (EuroBERT-210m BIOES) ---
print("Running EuroBERT-ours...", flush=True)
t0 = time.time()
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
if "default" not in ROPE_INIT_FUNCTIONS:
    def _d(config, device=None, seq_len=None):
        dim = config.hidden_size // config.num_attention_heads
        theta = getattr(config, "rope_theta", 10000.0)
        return 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)), 1.0
    ROPE_INIT_FUNCTIONS["default"] = _d
from transformers import AutoTokenizer, AutoModelForTokenClassification
sys.path.insert(0, "/root/bitnet_train")
from eu_train_v2 import bioes_decode, build_bioes

sd = torch.load("/scratch/pii_corpus/eurobert_verify.pt", map_location="cpu", weights_only=False)
lmap = sd["labels"]; lmap_rev = {v: k for k, v in lmap.items()}
tok = AutoTokenizer.from_pretrained("/root/models/EuroBERT-210m", trust_remote_code=True)
m = AutoModelForTokenClassification.from_pretrained("/root/models/EuroBERT-210m", trust_remote_code=True, num_labels=len(lmap))
m.load_state_dict(sd["model"], strict=True); m.cuda().eval()
ours_preds = []
with torch.no_grad():
    for i in range(0, len(rows), 48):
        b = rows[i:i+48]
        enc = tok([r["text"] for r in b], padding=True, truncation=True, max_length=512,
                  return_offsets_mapping=True, return_tensors="pt")
        off = enc.pop("offset_mapping")
        lg = m(enc["input_ids"].cuda(), attention_mask=enc["attention_mask"].cuda()).logits.argmax(-1).cpu()
        for j, r in enumerate(b):
            offs = [(int(a), int(bb)) for a, bb in off[j].tolist()]
            ours_preds.append(bioes_decode(lg[j].tolist(), offs, lmap_rev, r["text"]))
report("EuroBERT-ours (BIOES)", ours_preds, golds)
print(f"  ({time.time()-t0:.1f}s)\n")

# --- 3. LIQUIDAI LFM2.5 PII DETECTOR ---
print("Running LiquidAI LFM2.5...", flush=True)
t0 = time.time()
import importlib.util
spec = importlib.util.spec_from_file_location("pii_hybrid_decode", "/root/models/lfm_pii_detector/pii_hybrid_decode.py")
hd = importlib.util.module_from_spec(spec); spec.loader.exec_module(hd)
ltok = AutoTokenizer.from_pretrained("/root/models/lfm_pii_detector", trust_remote_code=True)
lmodel = AutoModelForTokenClassification.from_pretrained("/root/models/lfm_pii_detector", trust_remote_code=True).cuda().eval()
liquid_preds = []
with torch.no_grad():
    for r in rows:
        spans = hd.predict(r["text"], ltok, lmodel)
        liquid_preds.append([{"start": s["start"], "end": s["end"], "type": s["type"]} for s in spans])
report("LiquidAI LFM2.5 (hybrid)", liquid_preds, golds)
print(f"  ({time.time()-t0:.1f}s)\n")

print("=" * 80)
print("STRICT EXACT-SPAN DETECTION F1 (no overlap credit, no partial credit)")
print("=" * 80)
