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
