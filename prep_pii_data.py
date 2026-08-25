#!/usr/bin/env python3
# prep_pii_data.py — combine synthetic (gated) + real ai4privacy into one fine-tune set.
#
# Writes {lang,text,spans:[{type,start,end,text}]} JSONL, split into train/val.
# Synthetic rows already have spans. Real rows get char spans by locating each
# entity string in source_text (greedy, non-overlapping). Only identifier types
# in the closed schema are kept.
import json, os, argparse, random

# map real ai4privacy category -> our closed schema type
REAL_CAT_MAP = {
    "user_name": "username", "street_address": "street_address", "first_name": "first_name",
    "last_name": "last_name", "email": "email", "phone_number": "phone", "date_of_birth": "dob",
    "ssn": "ssn", "ip": "ip", "ipv6": "ipv6", "credit_card_number": "credit_card",
    "license_plate": "license_plate", "account_number": "account_number",
    "bank_routing_number": "routing_number", "national_id": "national_id",
    "tax_id": "tax_id", "certificate_license_number": "license_number",
    "health_plan_beneficiary_number": "health_plan_beneficiary_number",
    "device_identifier": "device_identifier", "biometric_identifier": "biometric_identifier",
    "unique_identifier": "unique_identifier", "company_name": "company_name",
    "ipv4": "ip", "swift_bic": "bic", "cvv": "cvv", "pin": "pin",
}

# keep only types the fine-tune schema knows
KNOWN_TYPES = set([
    "email","phone","dob","iban","bic","credit_card","ssn","username","ip","ipv6",
    "vat","fiscal_code","passport","utr","nif","tax_id","personal_code","vin","imei","mac",
    "aws_access_key","github_token","stripe_key","openai_key","slack_token","jwt",
    "private_key","bearer_token","gcp_private_key","btc_address","btc_bech32",
    "eth_address","ltc_address","sol_address","account_number","routing_number",
    "license_plate","first_name","last_name","street_address","city","postal_code",
    "country","state","license_number","national_id","health_plan_beneficiary_number",
    "device_identifier","biometric_identifier","unique_identifier","company_name","cvv","pin",
])


def real_row_to_span_row(text, entities, seq_cap=1000):
    """Build {text,spans} from source_text + entities (entity text + category)."""
    spans = []
    used = set()
    for ent in entities:
        etext = ent.get("entity", "")
        cat = ent.get("category", "")
        typ = REAL_CAT_MAP.get(cat, cat)
        if typ not in KNOWN_TYPES:
            continue
        # locate entity text (first unused occurrence)
        pos = 0
        found = None
        while True:
            idx = text.find(etext, pos)
            if idx < 0:
                break
            if idx not in used and (idx, idx + len(etext)) not in used:
                found = (idx, idx + len(etext))
                break
            pos = idx + 1
        if found is None:
            continue
        used.add(found)
        spans.append({"type": typ, "start": found[0], "end": found[1], "text": etext})
    if not spans:
        return None
    return {"lang": "en", "text": text, "spans": spans}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", default="/scratch/pii_corpus/pii_en.jsonl")
    ap.add_argument("--real-limit", type=int, default=4000)
    ap.add_argument("--out", default="/scratch/pii_corpus/pii_train.jsonl")
    ap.add_argument("--out-val", default="/scratch/pii_corpus/pii_val.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    rows = []
    # synthetic
    for line in open(args.synthetic):
        if not line.strip():
            continue
        r = json.loads(line)
        filtered = [s for s in r["spans"] if s["type"] in KNOWN_TYPES]
        if filtered:
            rows.append({"lang": "en", "text": r["text"], "spans": filtered})
    print("synthetic rows:", len(rows))

    # real (from validation split; we carve our own train/val here for clean OOD-ish)
    from datasets import load_dataset
    ds = load_dataset("automated-analytics/ai4privacy-pii-masking-en-v1-ner", split="validation")
    n_real = 0
    for i in range(min(args.real_limit, len(ds))):
        s = ds[i]
        text = s["source_text"] or s["target_text"] or " ".join(s["tokens"])
        rr = real_row_to_span_row(text, s["entities"])
        if rr:
            rows.append(rr)
            n_real += 1
    print("real rows added:", n_real, "total:", len(rows))

    random.shuffle(rows)
    split = int(len(rows) * 0.9)
    train, val = rows[:split], rows[split:]
    with open(args.out, "w") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")
    with open(args.out_val, "w") as f:
        for r in val:
            f.write(json.dumps(r) + "\n")
    print(f"train={len(train)} val={len(val)} -> {args.out}, {args.out_val}")


if __name__ == "__main__":
    main()
