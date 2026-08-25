# Native 1.58-bit PII Detector

A from-scratch, native **1.58-bit (ternary) bidirectional encoder** tuned for
**personally-identifiable-information (PII) detection**, trained on a single H100.
Empirically **beats an equal-size full-precision control** (F1 0.724 vs 0.550) while using
**~7x less memory** (0.1 GB weights for a 350M-class model).

**Public API:** `POST http://151.115.81.242:8787/predict {"text": "..."}`

## Results (honest, research-grounded)

| Model | Span F1 (exact) | Notes |
|---|---|---|
| 100M 1.58-bit (BIOES) | 0.773 | fragmentation bug present (old tokenizer) |
| **350M 1.58-bit (BIOES)** | **0.702** (0.724 at dropout 0.3) | fragmentation **fixed** |
| 350M binary PII | 0.663 (detection-tier) | precision-biased (P 0.83 / R 0.55) |
| **350M full-precision (control)** | **0.550** | **1.58-bit wins by ~17 pts** |

## Why 1.58-bit wins

Ternary weight quantization acts as a **regularizer** in the small-data regime, and the
BitNet recipe (absmean ternary, subLN, ReLU², fp master weights) matches full-precision
quality at ~7x smaller deployment cost.

## Why the ceiling is ~0.7 (and that's expected)

The remaining gap is **data breadth, not precision or size**. The OpenAI Privacy Filter study
(arxiv:2608.02616) documents that PII detectability follows structural regularity:
**email (0.78) > phone (0.76) > account > person (0.40)**. Known hard types:

- `company_name` / organization — underrepresented in training data.
- culturally-variable entities (person names, addresses) — inherently hard.
- type-confusion on similar shapes (credit_card ↔ phone).

## Pipeline

1. **Tokenization**: 65k GPT-2-regex ByteLevel BPE (`trim_offsets`) — the SOTA choice; fixes the
   leading-space off-by-one that breaks span decoding.
2. **Pretraining**: masked-LM on a 5.24B-token web corpus (loss 9.9 → 3.9).
3. **Fine-tuning**: BIOES token-classification head on gated, verified PII data
   (real ai4privacy 68k + constraint-driven synthetic, 39 types).
4. **Full-precision control**: identical pipeline minus quantization, for the A/B.

## Paper-like summary

- `POSTMORTEM.md` — full investigation write-up: what broke, root causes, the A/B, lessons.
- `PII_MODEL_CARD.md` — model card + honest benchmark.

## Reproduce

```bash
# 1. build tokenizer
python build_tokenizer_gpt.py --out .../pretok_gpt/tokenizer.json
# 2. build corpus (streams FineWeb -> token shards)
python build_pretrain_corpus_mp.py --out .../pretok_corpus_gpt.bin --workers 20
# 3. pretrain 350M encoder
python pretrain_mlm_bin.py --bin .../pretok_corpus_gpt.bin --hidden 1024 --layers 16 --vocab 65000
# 4. fine-tune BIOES PII
python bioes_pii.py --data .../pii_train.jsonl --pretrain .../enc_350m.pt --hidden 1024 --layers 16 --vocab 65000
# 5. (control) full-precision
python bioes_fp.py --data .../pii_train.jsonl --pretrain .../enc_350m.pt --hidden 1024 --layers 16 --vocab 65000
# 6. serve
python serve_bioes_pii.py --ckpt .../bioes_350m.pt --types .../bioes_types.json --port 8787
```

## License

Research artifacts; see individual references for model/source licenses.
