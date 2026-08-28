# PIIBench Results — AjelixAI EuroBERT-610m PII Detector

## Full held-out test (100,002 records, corrected 82-entity taxonomy, seqeval span+type)
| Metric | Value |
|---|---|
| **Overall F1** | **0.6754** |
| Overall Precision | 0.6191 |
| Overall Recall | 0.7430 |
| per-type macro-F1 | 0.8783 |
| weighted-F1 | 0.6877 |
| support | 580,736 spans |

## Comparison (authors' `run_streaming_model_benchmark.py`, identical harness)
| System | F1 |
|---|---|
| **EuroBERT-610m (fine-tuned, this work)** | **0.6754** |
| Direct fine-tuned DeBERTa-base | 0.6455 |
| Source-conditioned Hierarchical DeBERTa | 0.5894 |
| Curriculum-enabled hierarchical | 0.2772 |
| SpanMarker BERT | 0.1723 |
| Presidio (rule-based) | 0.1385 |

## Notes
- Beats the prior best (0.6455) by +0.030 on the corrected-taxonomy full test.
- macro-F1 0.878: excellent across the 82 types; IBAN, BIC, SSN, TAX_ID, NATIONAL_ID,
  MEDICAL_RECORD_NUMBER, PHONE, URL all > 0.95.
- Weak types (all synthetic-template or confusable): FINANCIAL_ENTITY 0.346 (115,924 support),
  IP_ADDRESS 0.484, USERNAME 0.500, TIME 0.448, MISC 0.602, EMAIL 0.670.
- Precision (0.619) below recall (0.743): weighted-CE O-weight (0.1) over-fires.
  Next lever: unweighted CE / raised O-weight should lift precision.

## Reproduce
- Data: `pii-bench/data/` (source-stratified 80/10/10)
- Model: EuroBERT/EuroBERT-610m, `--max-length 256 --batch-size 64 --lr 2.8e-5
  --warmup-ratio 0.1 --weight-decay 0.1 --epochs 3 --max-steps 37500`
- Eval: `run_streaming_model_benchmark.py --test-file ./data/test.jsonl
  --model-path ./models/best_model --chunk-size 5000 --batch-size 16`

## Per-EU-language comparison (PIIBench, type-agnostic span-overlap F1)
Evaluation on ai4privacy EU-language test rows (de/fr/it/es/nl), 800 rows/lang (~40% of EU test),
gold spans on the mBERT grid; systems run with the same alignment + seqeval span-overlap metric.

| lang | Presidio | DeBERTa-base | EuroBERT-610m |
|---|---|---|---|
| de | 0.337 | 0.534 | **0.610** |
| fr | 0.408 | 0.556 | **0.641** |
| it | 0.345 | 0.521 | **0.592** |
| es | 0.348 | 0.495 | **0.531** |
| nl | 0.351 | 0.489 | **0.543** |
| macro | 0.358 | 0.519 | **0.584** |

EuroBERT-610m is best on all 5 EU languages (+0.065 macro over DeBERTa-base, +0.226 over Presidio).
Note: exact token-run F1 is lower for EuroBERT (0.172 vs DeBERTa 0.255) due to tokenizer boundary
differences on the mBERT grid; overlap is the correct cross-system detection metric.
Reproduce: `eval_eu_multilingual.py [max_rows_per_lang]` on the PIIBench server.
