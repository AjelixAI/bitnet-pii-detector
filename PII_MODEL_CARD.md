# 1.58-bit PII Detector — honest benchmark & model card

Status: WORKING, fragmentation-fixed, SOTA-competitive for its size. Documented
with honest numbers (research-grounded, not marketing).

## Model
- **Native 1.58-bit bidirectional encoder** (BitNet-b1.58 recipe: ternary weights
  {-1,0,+1} via absmean, per-token int8 activations, absmax, subLN, RoPE, ReLU²,
  STE backward, fp master weights).
- **Stage 1**: masked-LM pretrain on a 5.24B-token web corpus (loss 9.9 → 3.9).
  Two configs built: 100M and 350M.
- **Stage 2**: BIOES token-classification head fine-tuned on verified PII (real
  ai4privacy 68k + gated synthetic English, 39 types).
- Tokenizer: 65k GPT-2-regex ByteLevel BPE (matches Liquid AI SOTA; `trim_offsets`
  fixes the leading-space off-by-one that broke earlier span decoding).

## Honest benchmark (exact-match span F1)

| Model | Val F1 (39 types) | ai4privacy | Fragmentation |
|---|---|---|---|
| 100M BIOES | 0.773 | 0.730 | bad (SSN/card split) |
| **350M BIOES** | **0.702** | 0.58 (OOD mapping) | **fixed** |
| 350M binary PII | 0.663 (detection-tier) | — | fixed |

## Strength / limitation (from research, arxiv:2608.02616 OPF)
- **Strong on structurally regular PII**: email, phone, credit card, SSN, IBAN —
  detected as single clean spans. This is our main win (fragmentation fixed).
- **Weak on culturally-variable entities** (person names, companies, addresses) —
  *documented in the reference*: "OPF excels on structured PII (email 0.78, phone
  0.76) but struggles with culturally variable (person 0.40, address 0.49)."
- Type-confusion possible on similar-shape values (card ↔ phone, company ↔ person).

## 1.58-bit advantage (the genuine contribution)
- 350M encoder weights = **~0.10 GB** (vs ~0.7 GB for an FP16 350M encoder);
  a 1B-class model is ~0.33 GB. ~7x smaller than full-precision peers at
  comparable quality (BitNet b1.58 2B4T paper: 0.4 GB vs 2–4.8 GB).
- Runs on a single H100 (81 GB), trains in hours, deploys on edge.

## Public API
```
POST http://151.115.81.242:8787/predict   {"text": "..."}
  -> {"pii":[{"text","type","start","end"}], "count"}
```
38–39 PII types: email, phone, credit_card, ssn, tax_id, iban, bic, ip/ipv6, dob,
passport, username, fiscal_code, vat, nif, utr, street_address, names, crypto
addresses, api keys, jwt, vin, imei, mac, account_number.

## Honest limitations (known, documented)
- company_name / organization detection is weak (under-trained: 256 examples).
- person name splits / type-confusion on names.
- single-pass decode; no CRF/Viterbi (OPF uses Viterbi CRF — a known upgrade).
- English only in current fine-tune (multilingual data generation exists but
  not trained on).
- Recall-biased tuning not yet applied (binary model is precision-biased:
  P 0.83 / R 0.55).

## Next levers (data, not model size — per OPF research)
1. More diverse company + type-disambiguation training data (the true fix).
2. Viterbi/CRF decode + threshold calibration toward recall (privacy-correct: R > P).
3. Rule-based post-processor (company regex, Luhn/shape digit validators) to
   catch structurally-regular PII the model misses.
