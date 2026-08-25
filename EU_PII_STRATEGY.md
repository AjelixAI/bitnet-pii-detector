# EU PII Detection Strategy: Beating Presidio at Scale

**Status:** Research & strategy map — prerequisites researched, architecture & data paths defined.
**Goal:** strongest EU-focused PII detector for GDPR-grade anonymization, beating Presidio and existing tools on benchmarks.
**Constraint change:** dropping 1.58-bit as the differentiator; we need the **strongest, deployable** model, not the smallest.

---

## 1. The IP landscape (what we're up against & where we win)

### The current SOTA small PII model: LiquidAI `LFM2.5-Encoder-350-PII-Detector`
| Benchmark | Their score | Notes |
|---|---|---|
| SPY | 0.428 | detection 0.509 |
| Gretel | 0.880 | detection 0.885 |
| TAB (EU court cases) | 0.867 | their top benchmark |
| ai4privacy | 0.715 | WEAK — their training data |
| Nemotron | 0.855 | |
| **MAPA (GDPR/EU anonymization)** | **0.236** | **their main VULNERABILITY** |
| **How** | hybrid model+rules, custom decode | 40 types, 16 languages |

**The IP lever:** **MAPA 0.236 is catastrophically low** (EU legal anonymization is their demonstrable weakness). TAB (0.867) is decent but EU-focused. This is our entry wedge: **if we build a model that dominates on MAPA + TAB and beats Presidio's known weaknesses, we are "the GDPR model"**.

### Presidio (the incumbent to beat)
Microsoft `presidio-analyzer` — the standard GDPR/PII tool the map must beat:
- **Hybrid**: regex + checksums + context/recognizers, but only ~50 regexes and no true generalization.
- **Weaknesses**: brittle on culturally-variable PII (names, orgs, addresses across EU locales), virtually no EU multilingual depth (only en/de/es/fr/it), and a known false-positive/negative profile the research shows clearly.

---

## 2. What model is the right base? (source-verified)

### Architecture decision: LFM-recipe, not 1.58-bit
The strongest path = a **small bidirectional encoder on the LFM architecture**, because it's both SOTA-competitive and fast. But the critical insight is *how* we get there:

| Base choice | Reality | Verdict |
|---|---|---|
| LiquidAI ready encoder (LFM2.5-Encoder-350M) | license `lfm1.0` **restricts commercial use** → **NO** for our API | reject |
| Our own LFM-style encoder (their recipe) | Rebuild the LFM2-to-encoder conversion ourselves | **viable** |
| Off-the-shelf multilingual open encoder | BERT-base/ModernBERT multilingual → train-on-PII | **baseline/comparison** |
| EuroBERT (`EuroBERT-350M`) | **European multilingual, 8k context, 27 languages, arxiv SOTA** | **strong candidate** |
| DeBERTa-v3-base (your ref) | English-only (160GB English corpus), 513 context | **rejected — wrong for EU multilingual** |

### Source-verified fact: how SOTA encoders are actually built
**LiquidAI's exact recipe** (from `lfm2-5-encoders` blog + model card):
1. Start from a **pretrained causal LFM2 decoder** (230M/350M).
2. **Convert to bidirectional**: replace causal attention mask with bidirectional, make short convolutions non-causal (symmetric center padding).
3. Train **masked-language objective at 30% masking** (denser than BERT's 15% — per arxiv 2305.x).
4. Two-phase: short-context (1k) general pretraining → long-context adaptation (8k) for legal/multilingual competence.
5. **Fine-tune token-classification head** on 40 PII types / 16 languages.
6. **Ship custom hybrid decode** (`pii_hybrid_decode.py` + `context_cued.py`) — model + **rule-based validators** (checksums) + **context-cuer** (the secret to their quality).

**Our playbook = the same recipe, but on multilingual EU data.**

---

## 3. What data do we need & how do we get it?

### The EU PII data challenge
The critical bottleneck is **multilingual EU-resistant PII training data** — the entire model quality rests on it.

### Recommended data stack (all accessible)

| Corpus | Why it matters | Access |
|---|---|---|
| **ai4privacy `pii-masking-openpii-1.5m`** | **1.6M examples, 30 languages, 19 PII classes** (~98.3% labeled-annotation accuracy per QA) | HF download (academic; contact for commercial license) |
| **MAPA** | the EU anonymization benchmark LiquidAI fails (0.236 our entry wedge) | public GitHub (CMP) |
| **TAB (Text Anonymization Benchmark)** | ECHR court cases, 1,268 docs, EU legal & privacy — arxiv 2202.00443 | public |
| Nemotron | PII-heavy synthetic | public |
| **Own proven data generators** | Our vetted `pii_generator_full.py` + `verify_pii.py`, per-EU-locale | we have |
| **Per-EU-language NER corpora** | CoNLL-2002/2003, GermEval, CrossNER-EU, NaijaNER-EU, etc. | public |

**Key EU needs (what multilingual PII actually looks like):**
- **de** (umlauts ß, IBAN format, street name conventions, date orders)
- **fr** (accents, postal codes "75xxx", phone prefixes "+33")
- **it/es/pt/pl/nl** (diacritics, IDs format variants, place names)
- **uk/ch devant-garde** (already in ai4privacy)

### Data acquisition plan (concrete)
1. **Pull ai4privacy openpii-1.5m** — the base multilingual training corpus (30 languages incl. our EU target set).
2. **Pull MAPA + TAB** (the evaluation benchmarks) — these are public and define the eval target.
3. **Pull Nemotron** — the PII-rich supplemental corpus.
4. **Build our own EU-locale PII test matrix** — per-country formats (IBAN, national IDs, postal codes, date formats, name structures with diacritics) and **gril-by-locale decision targets** from multilingual legal text.
5. **Generate additional disabiguating samples** using our existing verifier pipeline, but with **EU-locale PII templates** (`+49/33/351` prefixes, German names with umlauts, EU address forms, EU legal contexts).

---

## 4. The evaluation roadmap (how we prove it's better)

### Benchmarks we must beat (source-verified, the actual targets):
- **MAPA** — LiquidAI 0.236, Piiranha 0.946 — EU legal anonymization. THIS is our wedge.
- **TAB (ECHR)** — LiquidAI 0.867, Griffy 0.880 — EU legal text.
- **ai4privacy** — LiquidAI 0.715 — the base multilingual corpus.
- **Presidio** — must beat it. Its weaknesses: culturally-variable PII, shallow multilingual.

### The seametrics map (how we evaluate "we beat Presidio"):
1. **Exact-type F1** per-language per-type (German names vs Italian names vs Dutch addresses).
2. **Detection-tier F1** (did we find the span at all — the privacy-relevant metric).
3. **Recall-first weighting** (GDPR-correct: missing PII is a legal liability).
4. **Per-benchmark idiosyncrasies** we MUST replicate: MAPA's date-as-DoB labeling convention (per LiquidAI) — handle via our separate decode fix.

---

## 5. The technical plan (how we actually build it)

### Phase A: The multilingual encoder
Two viable paths (we should build both, benchmark):

**Path 1 (build-from-recipe):** Rebuild a **multilingual LFM-style encoder** using their exact recipe converted from an open multilingual decoder (e.g. init from Qwen3-8B/Llama3-1B multilingual, convert to bidirectional via bidirectional attention + non-causal convs + 30% LM). Effort: medium, IP-safe, fast.

**Path 2 (baseline):** Use an **existing open multilingual encoder** (BERT-base-Multilingual or **EuroBERT if truly open/license**) — fine-tune token-classification head directly. This is the reference baseline.

### Phase B: The PII head & decode — the real differentiator
**The strong insight is the DECODE**, not the base encoder. Our decode stack must include:
1. **Hybrid rule-based validators** (Luhn, IBAN checksum, EU postal-format regexes, date formatters, phone prefixes per-country) — the validators the OPF model proves best on.
2. **Context cuer** (`context_cued.py`) — the keyword-context trigger ("my name is", "Dr. Schmidt lives at", "IBAN:") that boosts detection near labeled-context cues.
3. **Threshold calibration per-language+per-type** — tunable for GDPR recall-optimization.

### Phase C: The multilingual data assembly
- Convert ai4privacy 1.5M + our generated per-locale samples into ONE unified token-classification set with **per-language tags** ("de"/"fr"/"it"/"es").
- Weights: higher for EU languages and harder types (org/person).

### Phase D: Train + evaluate + ship
- Fine-tune, benchmark per-MAPA/TAB/ai4privacy.
- If we beat MAPA: we are the EU-GDPR model.
- Deploy as an `Rust`/Python API (the model stays ~350M-1B, capable of CPU).

---

## 6. Honest risks & gaps

| Risk | Reality | Mitigation |
|---|---|---|
| DATA AVAILABILITY | ai4privacy 1.5m access may need a commercial license | contact them; build our own EU generator as insurance |
| **ARCHITECTURE TIME** | Building a multilingual LFM encoder from scratch is nontrivial | start with existing multilingual encoder (BERT-base-Multi) for the baseline first |
| Presidio integration | Presidio evolves; we need to freeze the version we benchmark | pinned version, exact reproducible benchmarks |
| GDPR liability | We are building something that processes personal data | must run on-prem/private infra (we know this already) |
| MAPA's quirk | MAPA's date-as-DoB labeling penalizes correct models | handle via separate decode configuration |

---

## 7. Recommended next actions (concrete)

**A. Validate data access.** Confirm `ai4privacy openpii-1.5m` is accessible (academic license is immediate; commercial needs contact). Download + inspect the EU-language subset.

**B. Establish the baseline.** Start with the simplest viable baseline: `BERT-base-Multilingual` (or EuroBERT if license is permissive) fined-tuned for token-classification on ai4privacy EU subset. Get our honest MAPA/TAB/ai4privacy baseline numbers.

**C. Build the strong path.** Rebuild the multilingual LFM-encoder recipe on an open decoder backbone, with the hybrid decode (validators + context-cuer). This is the competitive model.

**D. Benchmark rigorously.** Freeze Presidio version, build a clean per-benchmark evaluator, and produce per-language/per-type breakdowns. **Target: PII detection-tier F1 + exact-F1 that beats MAPA >0.4, TAB >0.87, ai4privacy >0.75.**

**E. EU deployment.** Package as an on-prem Python/Rust API — CPU-friendly, GDPR-hosted, the enterprise sales motion.

---

## 8. Source records (verified)

- **LFM architecture**: `LFM2 Technical Report` (arxiv:2511.23404) — hybrid gated conv + GQA, hardware-in-the-loop NAS, 10T token pretraining, distillation. The encoders derive from this by conversion.
- **LiquidAI recipe**: `lfm2-5-encoders` blog + model card — causal→bidirectional conversion, 30% masking, 8k context, hybrid decode (`pii_hybrid_decode.py`+`context_cued.py`).
- **MAPA**: arxiv/tab-benchmark — ECHR court cases, EU legal anonymization.
- **EuroBERT**: arxiv 2503.05500 — multilingual European encoders (27 languages, 8k context).
- **Presidio**: GitHub: `microsoft/presidio` — regex+model hybrid, ~50 recognizers, en/de/es/fr/it.
- **deberta-v3-base**: arxiv 2111.09543 — English-only 86M RDT encoder.
