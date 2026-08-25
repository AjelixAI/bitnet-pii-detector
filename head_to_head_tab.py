#!/usr/bin/env python3
"""head_to_head_tab.py — all 3 models on TAB (ECHR court cases), strict exact-span.
TAB = Text Anonymization Benchmark, 127 docs, 20,809 PII spans. Out-of-distribution
for all models (legal text, not synthetic PII training data).
"""
import torch, sys, json, time, importlib.util

def strict_score(preds, golds):
    tp_d = fp_d = fn_d = 0
    for p, g in zip(preds, golds):
        gset = {(sp["start"], sp["end"]) for sp in g}
        pset = {(sp["start"], sp["end"]) for sp in p}
        used = set()
        for sp in p:
            k = (sp["start"], sp["end"])
            if k in gset and k not in used:
                used.add(k); tp_d += 1
            else:
                fp_d += 1
        fn_d += len(gset) - len(used)
    P = tp_d / max(1, tp_d + fp_d); R = tp_d / max(1, tp_d + fn_d)
    return P, R, 2 * P * R / max(1e-9, P + R), tp_d, fp_d, fn_d

def report(name, preds, golds):
    P, R, F1, tp, fp, fn = strict_score(preds, golds)
    print(f"{name:30s} det F1={F1:.4f} (P={P:.3f} R={R:.3f}) | tp={tp} fp={fp} fn={fn}")
    return F1

# --- load TAB ---
tab_data = json.load(open("/scratch/pii_corpus/benchmarks/tab/echr_test.json"))
rows = []
for doc in tab_data:
    text = doc["text"]
    spans = []
    for annotator, layer in doc.get("annotations", {}).items():
        for em in layer.get("entity_mentions", []):
            if em.get("identifier_type") == "NO_MASK":
                continue
            s, e = em["start_offset"], em["end_offset"]
            if 0 <= s < e <= len(text):
                spans.append({"start": s, "end": e, "type": em.get("entity_type", "?")})
    if text and spans:
        rows.append({"text": text, "spans": spans})

print(f"=== TAB: {len(rows)} docs, {sum(len(r['spans']) for r in rows)} gold spans ===")
print(f"avg doc length: {sum(len(r['text']) for r in rows)//len(rows)} chars\n")

# --- 1. PRESIDIO ---
print("Running Presidio...", flush=True)
t0 = time.time()
from presidio_analyzer import AnalyzerEngine
an = AnalyzerEngine()
presidio_preds = []
for r in rows:
    results = an.analyze(text=r["text"], language="en")
    presidio_preds.append([{"start": x.start, "end": x.end, "type": x.entity_type} for x in results])
report("Presidio", presidio_preds, [r["spans"] for r in rows])
print(f"  ({time.time()-t0:.1f}s)\n")

# --- 2. OURS (EuroBERT-210m BIOES) ---
print("Running EuroBERT-ours...", flush=True)
t0 = time.time()
sys.path.insert(0, "/root/bitnet_train")
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
if "default" not in ROPE_INIT_FUNCTIONS:
    def _d(config, device=None, seq_len=None):
        dim = config.hidden_size // config.num_attention_heads
        theta = getattr(config, "rope_theta", 10000.0)
        return 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)), 1.0
    ROPE_INIT_FUNCTIONS["default"] = _d
from transformers import AutoTokenizer, AutoModelForTokenClassification
from eu_train_v2 import bioes_decode, build_bioes

sd = torch.load("/scratch/pii_corpus/eurobert_verify.pt", map_location="cpu", weights_only=False)
lmap = sd["labels"]; lmap_rev = {v: k for k, v in lmap.items()}
tok = AutoTokenizer.from_pretrained("/root/models/EuroBERT-210m", trust_remote_code=True)
m = AutoModelForTokenClassification.from_pretrained("/root/models/EuroBERT-210m", trust_remote_code=True, num_labels=len(lmap))
m.load_state_dict(sd["model"], strict=True); m.cuda().eval()

ours_preds = []
for r in rows:
    # TAB docs are long — process in 512-token windows sliding through the doc
    all_preds = []
    text = r["text"]
    # tokenize whole doc, slide in 512-token windows with 50-token overlap
    enc_full = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc_full["input_ids"]
    offs = enc_full["offset_mapping"]
    window = 510  # leave room for special tokens
    stride = 460
    for start in range(0, len(ids), stride):
        end = min(start + window, len(ids))
        cls_id = tok.cls_token_id if tok.cls_token_id is not None else tok.bos_token_id or 1
        sep_id = tok.sep_token_id if tok.sep_token_id is not None else tok.eos_token_id or 2
        chunk_ids = [cls_id] + ids[start:end] + [sep_id]
        chunk_offs = [(0, 0)] + offs[start:end] + [(0, 0)]
        x = torch.tensor([chunk_ids], dtype=torch.long).cuda()
        a = torch.ones_like(x).cuda()
        with torch.no_grad():
            lg = m(x, attention_mask=a).logits.argmax(-1)[0].cpu().tolist()
        preds = bioes_decode(lg, chunk_offs, lmap_rev, text)
        all_preds.extend(preds)
        if end >= len(ids):
            break
    # deduplicate overlapping predictions (keep first occurrence)
    seen = set()
    deduped = []
    for p in all_preds:
        k = (p["start"], p["end"])
        if k not in seen:
            seen.add(k); deduped.append(p)
    ours_preds.append(deduped)
report("EuroBERT-ours (BIOES)", ours_preds, [r["spans"] for r in rows])
print(f"  ({time.time()-t0:.1f}s)\n")

# --- 3. LIQUIDAI LFM2.5 PII DETECTOR ---
print("Running LiquidAI LFM2.5...", flush=True)
t0 = time.time()
spec = importlib.util.spec_from_file_location("pii_hybrid_decode", "/root/models/lfm_pii_detector/pii_hybrid_decode.py")
hd = importlib.util.module_from_spec(spec); spec.loader.exec_module(hd)
ltok = AutoTokenizer.from_pretrained("/root/models/lfm_pii_detector", trust_remote_code=True)
lmodel = AutoModelForTokenClassification.from_pretrained("/root/models/lfm_pii_detector", trust_remote_code=True).cuda().eval()
liquid_preds = []
with torch.no_grad():
    for r in rows:
        # LiquidAI handles 8k context, but very long docs may exceed — truncate
        text = r["text"][:32000]  # reasonable char limit
        spans = hd.predict(text, ltok, lmodel)
        liquid_preds.append([{"start": s["start"], "end": s["end"], "type": s["type"]} for s in spans])
report("LiquidAI LFM2.5 (hybrid)", liquid_preds, [r["spans"] for r in rows])
print(f"  ({time.time()-t0:.1f}s)\n")

print("=" * 80)
print("TAB BENCHMARK — STRICT EXACT-SPAN DETECTION F1 (out-of-distribution)")
print(f"TAB docs: {len(rows)} | gold spans: {sum(len(r['spans']) for r in rows)}")
print(f"LiquidAI card claims: 0.867 (partial-F1, 18-lang filtered)")
print("=" * 80)
