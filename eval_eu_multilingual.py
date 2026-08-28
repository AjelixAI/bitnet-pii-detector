#!/usr/bin/env python3
"""Per-EU-language PII detection comparison on PIIBench ai4privacy test rows.

Systems compared (type-agnostic detection F1 on the mBERT gold grid):
  1. Presidio (rule-based)          -> presidio_analyzer
  2. Authors' DeBERTa-base          -> Pritesh-2711/piibench-deberta-base (prev. best 0.6455 overall)
  3. AjelixAI EuroBERT-610m         -> /root/pii-bench/models/best_model (ours, 0.6754 overall)

Language mapping: authoritative from ai4privacy parquet (source_text -> language);
language detection (langdetect) validated against it, used as fallback for any unmatched row.
Gold spans: PIIBench mBERT grid (tokens/labels). Cross-grid alignment: bert-base-multilingual-cased offsets.
Metric: seqeval token-run precision/recall/F1, type-agnostic (a predicted PII region matches if its
token-index run exactly equals a gold PII run; taxonomies differ across systems so type-aware would be unfair).

Usage: python eval_eu_multilingual.py [max_rows_per_lang]
"""
import json, os, collections, sys
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import pyarrow.parquet as pq
from langdetect import detect as ld_detect

MAX_PER_LANG = int(sys.argv[1]) if len(sys.argv) > 1 else 10**9
EU_LANGS = ("de", "fr", "it", "es", "nl")
BERT_ALIGN = "bert-base-multilingual-cased"
SEQ = 256

# ---- authoritative language map from ai4privacy parquet ----
print("Building language map ...", flush=True)
t = pq.read_table("/root/data/pii400k_train.parquet", columns=["source_text", "language"])
st = t.column("source_text").to_pylist()
lang = t.column("language").to_pylist()
LANG_MAP = {s: l for s, l in zip(st, lang)}

# ---- load gold rows from test.jsonl (ai4privacy EU rows) ----
rows = []
seen = collections.Counter()
for line in open("/root/pii-bench/data/test.jsonl"):
    r = json.loads(line)
    if r.get("source") not in ("ai4privacy_400k", "ai4privacy_300k"):
        continue
    text = r.get("text") or " ".join(r.get("tokens") or [])
    lab = LANG_MAP.get(text)
    if lab is None:
        try:
            lab = ld_detect(text)
        except Exception:
            lab = "?"
    if lab not in EU_LANGS:
        continue
    if seen[lab] >= MAX_PER_LANG:
        continue
    seen[lab] += 1
    if not r.get("tokens") or not r.get("labels"):
        continue
    rows.append({"text": text, "tokens": r["tokens"], "labels": r["labels"], "lang": lab})
print(f"EU test rows: {len(rows)}  langs={dict(seen)}", flush=True)


def gold_runs(labels):
    runs, cur = [], None
    for i, l in enumerate(labels):
        if l == "O":
            if cur:
                runs.append((cur[0], cur[1]))
            cur = None
        elif cur and i == cur[1] + 1 and cur[2] == l[2:]:
            cur = (cur[0], i, cur[2])
        elif l.startswith("B-"):
            if cur:
                runs.append((cur[0], cur[1]))
            cur = (i, i, l[2:])
    if cur:
        runs.append((cur[0], cur[1]))
    return runs


# precompute gold runs per row (list of (start_idx,end_idx))
GOLD = []
for r in rows:
    GOLD.append(gold_runs(r["labels"]))


def align_spans(text, spans, bert, grid_len):
    """char spans -> token-index runs on the gold grid (truncated to grid_len)."""
    enc = bert(text, return_offsets_mapping=True, add_special_tokens=False,
               truncation=True, max_length=SEQ)
    offs = enc["offset_mapping"]
    n = min(len(offs), grid_len)
    merged = []
    for (a, b) in spans:
        if b <= a:
            continue
        idx = [i for i in range(n) if offs[i][1] > a and offs[i][0] < b]
        if idx:
            merged.append((idx[0], idx[-1]))
    merged.sort()
    runs = []
    if merged:
        s, e = merged[0]
        for a2, b2 in merged[1:]:
            if a2 <= e + 1:
                e = max(e, b2)
            else:
                runs.append((s, e))
                s, e = a2, b2
        runs.append((s, e))
    return [run for run in runs if run[1] < grid_len]


def rep(stats):
    g, p, tp = stats["g"], stats["p"], stats["tp"]
    pr = tp / max(1, p); rc = tp / max(1, g)
    f = 2 * pr * rc / max(1e-9, pr + rc)
    opr = stats["ovl_p"] / max(1, p); orc = stats["ovl_g"] / max(1, g)
    of = 2 * opr * orc / max(1e-9, opr + orc)
    return pr, rc, f, opr, orc, of


def add(stats, g, p, tp, ovg, ovp):
    stats["g"] += g; stats["p"] += p; stats["tp"] += tp
    stats["ovl_g"] += ovg; stats["ovl_p"] += ovp


def overlaps(g_runs, p_runs):
    ovg = sum(1 for gr in g_runs if any(gr[0] < pr[1] and pr[0] < gr[1] for pr in p_runs))
    ovp = sum(1 for pr in p_runs if any(gr[0] < pr[1] and pr[0] < gr[1] for gr in g_runs))
    return ovg, ovp


def evaluate_row(r, grid_len, bert, detect_fn, stats):
    text = r["text"]
    spans = detect_fn(text)
    p_runs = align_spans(text, spans, bert, grid_len)
    g_runs = [run for run in GOLD_f if run[1] < grid_len]
    ovg, ovp = overlaps(g_runs, p_runs)
    add(stats, len(g_runs), len(p_runs), len(set(g_runs) & set(p_runs)), ovg, ovp)


# pre-filter gold per row for truncation; GOLD_f recompute inside using grid_len
# Use bert alignment tokenizer
bert = AutoTokenizer.from_pretrained(BERT_ALIGN, trust_remote_code=True)

lang_stats = {l: {"g": 0, "p": 0, "tp": 0, "ovl_g": 0, "ovl_p": 0} for l in EU_LANGS}
total = {"g": 0, "p": 0, "tp": 0, "ovl_g": 0, "ovl_p": 0}


# ---------- System 1: Presidio ----------
print("=== Presidio ===", flush=True)
from presidio_analyzer import AnalyzerEngine
presidio_engine = AnalyzerEngine()


def detect_presidio(text):
    res = presidio_engine.analyze(text=text, language="en")
    return [(x.start, x.end) for x in res]


for GOLD_f, r in zip(GOLD, rows):
    grid_len = len(bert(r["text"], truncation=True, max_length=SEQ, add_special_tokens=False)["input_ids"])
    evaluate_row(r, grid_len, bert, detect_presidio, lang_stats[r["lang"]])


# re-accumulate per-lang from scratch was fine; total now double-counts, keep per-lang only
print("Presidio done", flush=True)


# ---------- Systems 2 & 3: neural ----------
def neural_spans_factory(model, tokenizer, id2label):
    def detect(text):
        enc = tokenizer(text, truncation=True, max_length=SEQ, return_offsets_mapping=True)
        ids = torch.tensor([enc["input_ids"]]).cuda()
        attn = torch.tensor([enc["attention_mask"]]).cuda()
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=attn).logits
        preds = logits.argmax(-1)[0].cpu().tolist()[:len(enc["offset_mapping"])]
        offs = enc["offset_mapping"]
        spans, cur = [], None
        for i, tag in enumerate([id2label.get(x, "O") for x in preds]):
            a, b = offs[i]
            if a == b:
                if cur:
                    spans.append((cur[0], cur[1], cur[2], cur[3]))
                cur = None
                continue
            if tag == "O":
                if cur:
                    spans.append((cur[0], cur[1], cur[2], cur[3]))
                cur = None
            elif cur and cur[2] == tag[2:] and i == cur[3] + 1:
                cur = (cur[0], b, tag[2:], i)
            else:
                if cur:
                    spans.append((cur[0], cur[1], cur[2], cur[3]))
                cur = (a, b, tag[2:], i)
        if cur:
            spans.append((cur[0], cur[1], cur[2], cur[3]))
        return [(a, b) for a, b, _, _ in spans]
    return detect


def run_neural(name, model_dir, label_map_path=None):
    print(f"=== {name} ===", flush=True)
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)
    model = AutoModelForTokenClassification.from_pretrained(model_dir, trust_remote_code=False).cuda().eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()} if model.config.id2label else None
    if id2label is None and label_map_path:
        lm = json.load(open(label_map_path))
        id2label = {i: l for i, l in enumerate(lm["labels"] or lm)}
    detect = neural_spans_factory(model, tok, id2label)
    # detect needs bert for alignment; recompute per lang
    ls = collections.defaultdict(lambda: {"g": 0, "p": 0, "tp": 0, "ovl_g": 0, "ovl_p": 0})
    for i, (GOLD_f, r) in enumerate(zip(GOLD, rows)):
        grid_len = len(bert(r["text"], truncation=True, max_length=SEQ, add_special_tokens=False)["input_ids"])
        spans = detect(r["text"])
        p_runs = align_spans(r["text"], spans, bert, grid_len)
        g_runs = [run for run in GOLD_f if run[1] < grid_len]
        if i == 0:
            print(f"  [dbg {name}] text={r['text'][:40]!r} gold={len(g_runs)} spans={len(spans)} grid_len={grid_len} p_runs={len(p_runs)}", flush=True)
        ovg, ovp = overlaps(g_runs, p_runs)
        ls[r["lang"]]["g"] += len(g_runs); ls[r["lang"]]["p"] += len(p_runs)
        ls[r["lang"]]["tp"] += len(set(g_runs) & set(p_runs))
        ls[r["lang"]]["ovl_g"] += ovg; ls[r["lang"]]["ovl_p"] += ovp
    del model
    torch.cuda.empty_cache()
    return ls


ls_deberta = run_neural("Authors DeBERTa-base", "/root/models/piibench-deberta-base",
                        "/root/models/piibench-deberta-base/label_mapping.json")
ls_eurobert = run_neural("AjelixAI EuroBERT-610m", "/root/pii-bench/models/best_model")

# ---------- report ----------
def ovl_f1(s):
    return rep(s)[5]
def ex_f1(s):
    return rep(s)[2]

print("\n=== PER-EU-LANGUAGE detection F1 (type-agnostic, mBERT grid) ===")
print("\n--- SPAN-OVERLAP F1 (robust detection) ---")
print(f"{'lang':<6}{'Presidio':>14}{'DeBERTa':>14}{'EuroBERT':>14}")
for l in EU_LANGS:
    print(f"{l:<6}{ovl_f1(lang_stats[l]):>14.4f}{ovl_f1(ls_deberta[l]):>14.4f}{ovl_f1(ls_eurobert[l]):>14.4f}")
def macro(ls, idx):
    return sum([rep(ls[l])[idx] for l in EU_LANGS]) / len(EU_LANGS)
print(f"{'macro':<6}{macro(lang_stats,5):>14.4f}{macro(ls_deberta,5):>14.4f}{macro(ls_eurobert,5):>14.4f}")

print("\n--- EXACT token-run F1 ---")
print(f"{'lang':<6}{'Presidio':>14}{'DeBERTa':>14}{'EuroBERT':>14}")
for l in EU_LANGS:
    print(f"{l:<6}{ex_f1(lang_stats[l]):>14.4f}{ex_f1(ls_deberta[l]):>14.4f}{ex_f1(ls_eurobert[l]):>14.4f}")
print(f"{'macro':<6}{macro(lang_stats,2):>14.4f}{macro(ls_deberta,2):>14.4f}{macro(ls_eurobert,2):>14.4f}")

print("\nEuroBERT per-lang (overlap) P/R/F:")
for l in EU_LANGS:
    pr, rc, f, opr, orc, of = rep(ls_eurobert[l])
    print(f"  {l}: P={opr:.3f} R={orc:.3f} OvF={of:.3f} | exactF={f:.3f} (gold={ls_eurobert[l]['g']})")
print("DeBERTa per-lang (overlap) P/R/F:")
for l in EU_LANGS:
    pr, rc, f, opr, orc, of = rep(ls_deberta[l])
    print(f"  {l}: P={opr:.3f} R={orc:.3f} OvF={of:.3f} | exactF={f:.3f} (gold={ls_deberta[l]['g']})")
print("Presidio per-lang (overlap):")
for l in EU_LANGS:
    pr, rc, f, opr, orc, of = rep(lang_stats[l])
    print(f"  {l}: P={opr:.3f} R={orc:.3f} OvF={of:.3f} (gold={lang_stats[l]['g']})")
