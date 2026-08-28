#!/usr/bin/env python3
"""Nervaluate benchmark of Presidio / authors' DeBERTa-base / AjelixAI EuroBERT-610m
on HF dataset hivetrace/pii-bench (multilingual multi-domain, char-span entities).

Metric: nervaluate span metrics. Primary = type-agnostic (all labels -> ENT) so it
measures pure span detection across systems with different taxonomies. Also computes
type-aware strict for the natural label space.

Usage: python eval_nervaluate.py [split] [max_docs]
"""
import collections, json, sys, torch
from datasets import load_dataset
from nervaluate import Evaluator
from transformers import AutoTokenizer, AutoModelForTokenClassification
from presidio_analyzer import AnalyzerEngine

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "domain"
MAXD = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
MAXLEN = 512

print(f"Loading hivetrace/pii-bench [{SPLIT}] ...", flush=True)
ds = load_dataset("hivetrace/pii-bench", streaming=False)
docs = ds[SPLIT]
if MAXD < len(docs):
    docs = docs.select(range(MAXD))
print(f"docs: {len(docs)}", flush=True)

GOLD = []
for r in docs:
    GOLD.append([{"label": "ENT", "start": e["start"], "end": e["end"]} for e in r["entities"]])
GOLD_TYPES = []
for r in docs:
    GOLD_TYPES.append([{"label": e["type"], "start": e["start"], "end": e["end"]} for e in r["entities"]])
TEXTS = [r["text"] for r in docs]


def to_ent(spans):
    return [{"label": "ENT", "start": a, "end": b} for a, b in spans]


# ---------- Presidio ----------
print("=== Presidio ===", flush=True)
engine = AnalyzerEngine()
def presidio_spans(text):
    try:
        res = engine.analyze(text=text, language="en")
        return [(x.start, x.end) for x in res]
    except Exception:
        return []
PRED_PRES = [to_ent(presidio_spans(t)) for t in TEXTS]

# ---------- neural systems ----------
def detect_byte_offsets(tokenizer):
    """Determine if offset_mapping is byte-based (Cyrillic 'а' = 2 bytes / 1 char)."""
    s = "аа"  # 2 Cyrillic chars = 4 bytes
    enc = tokenizer(s, return_offsets_mapping=True)
    offs = [o for o in enc["offset_mapping"] if o[0] != o[1]]
    return bool(offs and offs[0][1] >= 2)


def b2c(text, b):
    """byte offset -> char offset; clamps to a valid UTF-8 boundary."""
    raw = text.encode("utf-8")
    b = max(0, min(b, len(raw)))
    while b > 0 and b < len(raw) and (raw[b] & 0xC0) == 0x80:
        b -= 1
    return len(raw[:b].decode("utf-8"))


def neural_spans(text, tokenizer, model, id2label, byte):
    enc = tokenizer(text, truncation=True, max_length=MAXLEN, return_offsets_mapping=True)
    ids = torch.tensor([enc["input_ids"]]).cuda()
    attn = torch.tensor([enc["attention_mask"]]).cuda()
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=attn).logits
    preds = logits.argmax(-1)[0].cpu().tolist()
    offs = enc["offset_mapping"]
    spans, cur = [], None
    for i, tag in enumerate([id2label.get(x, "O") for x in preds[:len(offs)]]):
        a, b = offs[i]
        if a == b:
            if cur: spans.append(cur[0:3]); cur = None
            continue
        if tag == "O":
            if cur: spans.append(cur[0:3]); cur = None
        elif cur and cur[2] == tag[2:] and i == cur[3] + 1:
            cur = (cur[0], b, cur[2], i)
        else:
            if cur: spans.append(cur[0:3])
            cur = (a, b, tag[2:], i)
    if cur: spans.append(cur[0:3])
    raw = [(b2c(text, a), b2c(text, b)) for a, b, _ in spans] if byte else [(a, b) for a, b, _ in spans]
    return [trim(text, a, b) for a, b in raw]


def trim(text, a, b):
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    return a, b


def run_neural(name, model_dir, map_path=None):
    print(f"=== {name} ===", flush=True)
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)
    byte = detect_byte_offsets(tok)
    model = AutoModelForTokenClassification.from_pretrained(model_dir, trust_remote_code=False).cuda().eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()} if model.config.id2label else None
    if id2label is None and map_path:
        lm = json.load(open(map_path))
        labels = lm.get("labels", lm if isinstance(lm, list) else [])
        id2label = {i: l for i, l in enumerate(labels)}
    print(f"  offset_conv={'byte' if byte else 'char'}", flush=True)
    out = [to_ent(neural_spans(t, tok, model, id2label, byte)) for t in TEXTS]
    del model; torch.cuda.empty_cache()
    return out

PRED_DEB = run_neural("Authors DeBERTa-base", "/root/models/piibench-deberta-base",
                       "/root/models/piibench-deberta-base/label_mapping.json")
PRED_EUR = run_neural("AjelixAI EuroBERT-610m", "/root/pii-bench/models/best_model",
                      "/root/pii-bench/models/best_model/label_mapping.json")

# ---------- nervaluate evaluation ----------
def nerv_gold(pred, gold, tags):
    return Evaluator(gold, pred, tags=tags).evaluate()

def summarize(ev):
    st = ev["overall"]["strict"]; pt = ev["overall"]["partial"]
    return dict(strict=(st.precision, st.recall, st.f1), partial=(pt.precision, pt.recall, pt.f1))

def line(name, ev):
    s = summarize(ev)
    return (f"{name:<24} strictF={s['strict'][2]:.4f} (P{s['strict'][0]:.3f}/R{s['strict'][1]:.3f}) "
            f"| partial-F={s['partial'][2]:.4f}")

print("\n=== NERVALUATE (type-agnostic, all->ENT) ===")
print("=" * 88)
for name, pred in (("Presidio", PRED_PRES), ("DeBERTa-base", PRED_DEB), ("EuroBERT-610m", PRED_EUR)):
    ev = nerv_gold(pred, GOLD, ["ENT"])
    print(line(name, ev))

# per-domain breakdown (type-agnostic)
print("\n=== per-domain strict-micro F1 (type-agnostic) ===")
doms = collections.defaultdict(lambda: {"g": [], "pd": [], "pe": []})
for i, r in enumerate(docs):
    d = r["domain"]
    if not GOLD[i]:
        continue  # nervaluate can't index empty-entity docs; drop from per-domain view
    doms[d]["g"].append(GOLD[i]); doms[d]["pd"].append(PRED_DEB[i]); doms[d]["pe"].append(PRED_EUR[i])
print(f"{'domain':<14}{'DeBERTa':>12}{'EuroBERT':>12}")
for d in sorted(doms):
    g = doms[d]["g"]; 
    def micro(ev): return ev["overall"]["strict"].f1
    evd = nerv_gold(doms[d]["pd"], g, ["ENT"]); eve = nerv_gold(doms[d]["pe"], g, ["ENT"])
    print(f"{d:<14}{micro(evd):>12.4f}{micro(eve):>12.4f}")

# type-aware for our system (natural labels) - strict macro
print("\n=== type-aware strict (natural labels) ===")
tagnames = sorted({e["label"] for g in GOLD_TYPES for e in g} | {e["label"] for p in PRED_EUR for e in p} | {e["label"] for p in PRED_DEB for e in p})
# rebuilding pred with labels for DeBERTa/EuroBERT (reuse neural_spans with labels)
def neural_spans_labelled(text, tokenizer, model, id2label, byte):
    enc = tokenizer(text, truncation=True, max_length=MAXLEN, return_offsets_mapping=True)
    ids = torch.tensor([enc["input_ids"]]).cuda(); attn = torch.tensor([enc["attention_mask"]]).cuda()
    with torch.no_grad(): logits = model(input_ids=ids, attention_mask=attn).logits
    preds = logits.argmax(-1)[0].cpu().tolist(); offs = enc["offset_mapping"]
    spans, cur = [], None
    for i, tag in enumerate([id2label.get(x, "O") for x in preds[:len(offs)]]):
        a, b = offs[i]
        if a == b:
            if cur: spans.append((cur[0], cur[1], cur[2])); cur = None
            continue
        if tag == "O":
            if cur: spans.append((cur[0], cur[1], cur[2])); cur = None
        elif cur and cur[2] == tag[2:] and i == cur[3] + 1:
            cur = (cur[0], b, cur[2], i)
        else:
            if cur: spans.append((cur[0], cur[1], cur[2]))
            cur = (a, b, tag[2:], i)
    if cur: spans.append((cur[0], cur[1], cur[2]))
    if byte:
        raw = [(b2c(text, a), b2c(text, b), t) for a, b, t in spans]
    else:
        raw = [(a, b, t) for a, b, t in spans]
    return [{"label": t, "start": trim(text, a, b)[0], "end": trim(text, a, b)[1]} for a, b, t in raw]

# reuse loaded models for type-aware (expensive otherwise) - just report ours + DeBERTa with raw labels
tok_d = AutoTokenizer.from_pretrained("/root/models/piibench-deberta-base", trust_remote_code=False)
m_d = AutoModelForTokenClassification.from_pretrained("/root/models/piibench-deberta-base", trust_remote_code=False).cuda().eval()
tok_e = AutoTokenizer.from_pretrained("/root/pii-bench/models/best_model", trust_remote_code=False)
m_e = AutoModelForTokenClassification.from_pretrained("/root/pii-bench/models/best_model", trust_remote_code=False).cuda().eval()
id2_d = {int(k): v for k, v in m_d.config.id2label.items()}
id2_e = {int(k): v for k, v in m_e.config.id2label.items()}
byte_d = detect_byte_offsets(tok_d); byte_e = detect_byte_offsets(tok_e)
PRED_DEB_T = [neural_spans_labelled(t, tok_d, m_d, id2_d, byte_d) for t in TEXTS]
PRED_EUR_T = [neural_spans_labelled(t, tok_e, m_e, id2_e, byte_e) for t in TEXTS]
for name, pred in (("DeBERTa-base", PRED_DEB_T), ("EuroBERT-610m", PRED_EUR_T)):
    ev = Evaluator(GOLD_TYPES, pred, tags=tagnames).evaluate()
    print(line(name, ev))
del m_d, m_e; torch.cuda.empty_cache()
print("\nDONE")
