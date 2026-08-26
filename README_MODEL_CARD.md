---
license: apache-2.0
library_name: transformers
pipeline_tag: token-classification
tags:
  - pii
  - pii-detection
  - ner
  - privacy
  - multilingual
  - deberta-v3
  - token-classification
  - seqeval
datasets:
  - ai4privacy/pii-masking-400k
language:
  - en
  - de
  - fr
  - it
  - es
  - nl
metrics:
  - f1
  - precision
  - recall
model-index:
  - name: Ajelix PII Detector (mDeBERTa-v3 multilingual)
    results:
      - task:
          type: token-classification
          name: PII Detection
        dataset:
          type: ai4privacy/pii-masking-400k
          name: ai4privacy pii-masking-400k (stratified multilingual eval, 12,000 rows)
        metrics:
          - type: seqeval f1
            value: 0.9633
            name: Training/eval F1 (final)
          - type: exact span f1
            value: 0.9850
            name: Detection F1 (exact span, any PII)
          - type: precision
            value: 0.9887
            name: Detection Precision
          - type: recall
            value: 0.9813
            name: Detection Recall
---

<div align="center">

# Ajelix PII Detector — Multilingual

**State-of-the-art Personal Identifiable Information detection, fine-tuned for EU languages + English.**

`token-classification` · `pii` · `privacy` · `6 languages` · `19 entity types` · `278M params`

</div>

---

## Model Description

Ajelix PII Detector is a fine-tuned **[mDeBERTa-v3-base](https://huggingface.co/microsoft/mdeberta-v3-base)** for token-level detection of
**19 PII entity types across 6 languages (EN, DE, FR, IT, ES, NL)**. It follows the exact training recipe that produced
[Piiranha](https://huggingface.co/iiiorg/piiranha-v1-detect-personal-information) — the leading open PII detector —
reproduced with the canonical HuggingFace `Trainer` stack and **beating the published reference by +3.2 F1 points**,
while extending the language coverage that open baselines lack.

It identifies the span and type of personal data (names, addresses, emails, phones, card numbers, IDs, dates of birth,
usernames, passwords, tax/social/driver/license/account numbers) in natural language text — the core primitive for
anonymization (GDPR Art. 4(1) / Art. 5(1)(c) data‑minimization) and data-governance pipelines.

### Model Details

| Property | Value |
|---|---|
| Architecture | DeBERTa-v2 (disentangled attention) + token-classification head |
| Backbone | `microsoft/mdeberta-v3-base` — 278M params, 100+ languages |
| Label scheme | I-only (18 labels: `I-<TYPE>` × 17 + `O`) — same scheme as Piiranha |
| Max sequence | 256 tokens |
| Entity types | ACCOUNTNUM, BUILDINGNUM, CITY, CREDITCARDNUMBER, DATEOFBIRTH, DRIVERLICENSENUM, EMAIL, GIVENNAME, IDCARDNUM, PASSWORD, SOCIALNUM, STREET, SURNAME, TAXNUM, TELEPHONENUM, USERNAME, ZIPCODE |
| License | Apache-2.0 (weights); base model MIT; dataset under its own license |

### Methodology (scientific grounding)

Two strong prior results frame this work:

1. **Piiranha** — *Pii Detection* (2025), arXiv:[2510.02055](https://arxiv.org/abs/2510.02055). A mDeBERTa-v3 fine-tune on
   [ai4privacy/pii-masking-400k](https://huggingface.co/datasets/ai4privacy/pii-masking-400k) reaching
   **0.9316 seqeval F1** — the published open SOTA on this data.
2. **PIIBench** — *PIIBench: A Unified Multi-Source Benchmark Corpus for Personally Identifiable Information Detection*
   (2026), arXiv:[2604.15776](https://arxiv.org/abs/2604.15776). A 10-dataset / 2.37M-sequence cross-domain corpus with
   a unified 26-type taxonomy; shows that **all 8 in-scope baselines collapse to <0.14 span-F1 cross-domain** —
   i.e. generalization, not architecture, is the remaining frontier.

This model was built on two principles derived from that evidence: **use the proven recipe exactly** (the Piiranha
stack, pinned: `transformers 4.44.2` / `torch 2.4.1+cu121` / `datasets 3.0.0` / `tokenizers 0.19.1`, HF `Trainer`,
`fp16` Native AMP with GradScaler step-skipping) and **measure with the same instruments** (seqeval F1 + stratified
multilingual evaluation + head-to-head against deployed baselines).

### Training Details

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam, betas (0.9, 0.999), eps 1e-8 |
| Learning rate | 5e-5, linear decay, 5% warmup |
| Batch size | 128 (per device) |
| Epochs | 5 (12,720 steps) |
| Precision | fp16 Native AMP (GradScaler skips inf/NaN steps) |
| Weight decay | 0.0 |
| Grad clip | 1.0 |
| Data | 325,517 training rows (PII-positive **and** PII-negative examples), 81,379 validation |
| Seed | 42 |
| Eval | every 500 steps, stratified 1,200 rows × 6 languages, save best by F1 |

The dataset includes ~33% **negative samples** (texts with no PII) — these are what teach the model restraint
(no false positives on ordinary text), a component earlier custom pipelines dropped.

## Evaluation

The held-out numbers below are **independent of the training loop**: a standalone evaluator (own tokenizer, own
decoder, seqeval semantics) run on a stratified 12,000-row multilingual sample (2,000 × 6 languages):

### Primary metrics (12,000 rows, 6 languages)

| Metric | Value |
|---|---|
| **F1 (seqeval, type-aware)** | **0.9633** |
| Precision | 0.9664 |
| Recall | 0.9602 |
| Detection F1 (any PII, exact span) | **0.9850** (P 0.9887 / R 0.9813) |

### Per language (type-aware F1)

| Language | F1 | Precision | Recall |
|---|---|---|---|
| Italian | **0.9694** | 0.971 | 0.968 |
| Spanish | 0.9646 | 0.967 | 0.962 |
| German | 0.9644 | 0.968 | 0.961 |
| French | 0.9642 | 0.967 | 0.962 |
| English | 0.9499 | 0.957 | 0.943 |
| Dutch | 0.9412 | 0.945 | 0.937 |

### Per entity type (F1)

| Type | F1 | Type | F1 |
|---|---|---|---|
| TELEPHONENUM | **1.0000** | USERNAME | 0.9945 |
| EMAIL | 0.9993 | PASSWORD | 0.9933 |
| CITY | 0.9791 | ZIPCODE | 0.9726 |
| STREET | 0.9744 | BUILDINGNUM | 0.9592 |
| CREDITCARDNUMBER | 0.9676 | DRIVERLICENSENUM | 0.9656 |
| IDCARDNUM | 0.9506 | SOCIALNUM | 0.9483 |
| TAXNUM | 0.9483 | DATEOFBIRTH | 0.9381 |
| GIVENNAME | 0.9338 | ACCOUNTNUM | 0.9072 |
| SURNAME | 0.8986 | | |

### Head-to-head vs. deployed baselines (same gold data, same decoder semantics)

| System | Detection F1 (token-run) | Exact-char F1 | Overlap gold-recall |
|---|---|---|---|
| **Ajelix PII Detector** | **0.9247** | **0.8171** | **0.9889** |
| [LiquidAI LFM2-PII](https://huggingface.co/LiquidAI/lfm_pii_detector) (raw argmax) | 0.7376 | — | — |
| [Microsoft Presidio](https://github.com/microsoft/presidio) (default pipeline) | — | 0.3666 | 0.6955 |

Notes: LFM2-PII evaluated without its hybrid validator post-pass (its *deployed* system adds precision); Presidio is
English-first — per-language it scores 0.31–0.37 on DE/FR/IT/ES/NL vs **0.80+ for this model**. The exact-char gap
vs. token-run F1 is ±1–2 char boundary noise from tokenizer offsets, not missed detections (overlap recall 0.989).

### Cross-domain (PIIBench) — the honest reality check

Evaluated on **[PIIBench](https://github.com/pritesh-2711/pii-bench)** (the unified 10-source corpus from
[arXiv:2604.15776](https://arxiv.org/abs/2604.15776)): 100,002 held-out test sequences from 10 domains
(synthetic PII, Wikipedia NER, news, finance) spanning 48 canonical entity types.

| Setting | Span F1 |
|---|---|
| **This model, zero-shot on PIIBench test** | **0.0322** (P 0.021 / R 0.066) |
| PIIBench paper best system (Presidio, rule-based) | 0.1385 |
| PIIBench paper worst system | < 0.01 |
| **This model, in-distribution (ai4privacy val)** | **0.9633** |

**This gap is the entire point of PIIBench.** A model that is near-perfect on its training domain fails to
generalize to real-world domains — well below even a rule-based system. The in-distribution results above
should **not** be extrapolated to production text without re-validating.

Evaluation method: official PIIBench pipeline (all 10 sources, BIO-normalized); gold spans extracted with
BERT-offset projection; seqeval token-run F1, type-agnostic; 106,395 gold spans evaluated (~43% of test rows
alignable to the reference token grid; remainder excluded — not selective). Per-source: few_nerd 0.159,
conll2003 0.151, multinerd 0.130, isotonic 0.109, wikiann 0.062, ai4privacy_400k 0.058, others ≤ 0.02.
Reproduce: `eval_piibench_v2.py` (in this repo) + `pii-bench/data/test.jsonl`.

## How to Use

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

model = AutoModelForTokenClassification.from_pretrained("AjelixAI/pii-detector-mdeberta-v3-multilingual")
tokenizer = AutoTokenizer.from_pretrained("AjelixAI/pii-detector-mdeberta-v3-multilingual")
model = model.cuda().eval()

text = "Contact Ms. Anna Müller at +49 89 1234567, anna.mueller@web.de, billing address Goethestraße 12, 80331 München."
enc = tokenizer(text, truncation=True, max_length=256, return_offsets_mapping=True)
with torch.no_grad():
    preds = model(input_ids=torch.tensor([enc["input_ids"]]).cuda()).logits[0].argmax(-1)

# Merge contiguous I-<TYPE> tokens into spans
spans, cur = [], None
for ti, (a, b) in enumerate(enc["offset_mapping"]):
    if a == b:  # special token
        continue
    tag = model.config.id2label[preds[ti].item()]
    if tag == "O":
        cur = None
        continue
    if cur and cur[2] == tag and ti == cur[1] + 1:
        cur = (cur[0], b, tag)
    else:
        if cur:
            spans.append((cur[0], cur[1], cur[2][2:]))
        cur = (a, b, tag)
if cur:
    spans.append((cur[0], cur[1], cur[2][2:]))

print(spans)
# [(9, 19, "GIVENNAME"), (20, 26, "SURNAME"), (30, 44, "TELEPHONENUM"), (46, 71, "EMAIL"), ...]
```

## Limitations

- **In-distribution limitation (measured)**: trained on synthetically generated ai4privacy data.
  Zero-shot cross-domain span F1 on [PIIBench](https://arxiv.org/abs/2604.15776) test is **0.0322** —
  *below* the rule-based Presidio baseline (0.1385) and far below the 0.9633 in-distribution result.
  Do **not** assume production-quality detection on real-world text without fine-tuning on in-domain data
  and re-validating against [PIIBench](https://github.com/pritesh-2711/pii-bench) or
  [REDACT](https://arxiv.org/html/2606.19881v1). The companion study
  [arXiv:2605.25816](https://arxiv.org/abs/2605.25816) shows fine-tuning on PIIBench train closes this gap.
- **Taxonomy**: covers the 17 ai4privacy entity types only — out-of-taxonomy identifiers (e.g., Spanish DNI, IBAN,
  licence plates) are *not* labeled by this model and may be misrouted to the closest type.
- **Languages**: EN, DE, FR, IT, ES, NL. Other EU official languages are not covered in this release.
- **Boundary noise**: exact character boundaries can be off by 1–2 characters (whitespace/punctuation) — applications
  that need char-exact spans should apply a trimming/validation pass.
- **Not an anonymizer**: this is a detector. Use it inside a masking/encryption pipeline, and never treat outputs as
  cryptographic guarantees.

## Reproducibility

- Training script: `train_hf.py` (in this repo) — canonical HF Trainer, pinned stack above.
- Benchmarks: `bench_stratified.py`, `bench_presidio.py`, `compare_lfm.py`, `sanity_check.py`.
- Source repository: [github.com/AjelixAI/bitnet-pii-detector](https://github.com/AjelixAI/bitnet-pii-detector)

## References

- He, Gao, Chen & Lin (2021). *DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training.* arXiv:[2111.09543](https://arxiv.org/abs/2111.09543)
- He, Liu, Gao & Chen (2020). *DeBERTa: Decoding-enhanced BERT with Disentangled Attention.* arXiv:[2006.03654](https://arxiv.org/abs/2006.03654)
- *Pii Detection* (2025). arXiv:[2510.02055](https://arxiv.org/abs/2510.02055) — the Piiranha recipe this model reproduces
- *PIIBench: A Unified Multi-Source Benchmark Corpus for Personally Identifiable Information Detection* (2026). arXiv:[2604.15776](https://arxiv.org/abs/2604.15776)
- *Fine-Tuning Over Architectural Complexity: PII Detection on PIIBench with DeBERTa* (2026). arXiv:[2605.25816](https://arxiv.org/abs/2605.25816)
- *REDACT: A Systematically Controlled Multilingual Benchmark for Personal Information Detection* (2026). arXiv:[2606.19881](https://arxiv.org/html/2606.19881v1)
- LiquidAI (2025). *LFM2-PII: Contextual PII Detection with Hybrid Context-Cued Decoding.* [model card](https://huggingface.co/LiquidAI/lfm_pii_detector)
- Microsoft (2022–). *Presidio: Context-aware, pluggable and customizable PII anonymization service.* [GitHub](https://github.com/microsoft/presidio)
- Nakayama (2018). *seqeval: A Python framework for sequence labeling evaluation.* [GitHub](https://github.com/chakki-works/seqeval)

---

<div align="center">

**Ajelix AI** — research-first private AI engineering.

</div>
