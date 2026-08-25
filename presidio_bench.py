#!/usr/bin/env python3
"""Head-to-head: Presidio (rule + spacy NER) vs gold on openpii val, and numbers to
compare against our EuroBERT fine-tune. AI4P labels map to Presidio entity names as:
  PERSON->GIVENNAME/SURNAME/TITLE, EMAIL_ADDRESS->EMAIL, PHONE_NUMBER->TELEPHONENUM,
  LOCATION->CITY/STREET/ZIPCODE..., CREDIT_CARD,... etc. Detection-tier partially mapped.
"""
import argparse, json, glob, os, collections
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

AI4P_TO_PRESIDIO = {
    "GIVENNAME": "PERSON", "SURNAME": "PERSON", "TITLE": "PERSON",
    "EMAIL": "EMAIL_ADDRESS", "TELEPHONENUM": "PHONE_NUMBER",
    "CITY": "LOCATION", "STREET": "LOCATION", "ZIPCODE": "LOCATION",
    "BUILDINGNUM": "LOCATION",
    "DATE": "DATE_TIME", "AGE": "DATE_TIME",
    "IDCARDNUM": "IN_PAN", "DRIVERLICENSENUM": "IN_PAN",
    "TAXNUM": "IT_FISCAL_CODE" if False else "US_TAX_ID",
    "SOCIALNUM": "US_SSN", "CREDITCARDNUMBER": "CREDIT_CARD",
    "SEX": "_SPAN_ONLY", "GENDER": "_SPAN_ONLY",
    "PASSPORTNUM": "IN_PAN",
}

def spans_overlap(a_start, a_end, b_start, b_end):
    return max(a_start, b_start) < min(a_end, b_end)

def run_presidio_rows(analyzer, rows, seq_len=0):
    """Returns per-row predicted spans: [{start,end,type}]"""
    out = []
    for r in rows:
        text = r["text"] if not seq_len else r["text"][:seq_len]
        results = analyzer.analyze(text=text, language="en")
        out.append([{"start": x.start, "end": x.end, "type": x.entity_type} for x in results])
    return out

def score_dtts(preds, gold):
    d_tp = d_fp = d_fn = 0; e_tp = e_fp = e_fn = 0
    for g, p in zip(gold, preds):
        gspans, pspans = g["spans"], p
        gset = {(sp["start"], sp["end"]): sp for sp in gspans}
        used = set()
        for pp in pspans:
            hit = None
            for (gs, ge), gsp in gset.items():
                if (gs, ge) in used:
                    continue
                if spans_overlap(pp["start"], pp["end"], gs, ge):
                    hit = (gs, ge); break
            if hit:
                used.add(hit); d_tp += 1
                gsp = gset[hit]
                want = AI4P_TO_PRESIDIO.get(gsp["type"])
                if want and want != "_SPAN_ONLY" and pp["type"] == want:
                    e_tp += 1
                else:
                    e_fp += 1
            else:
                d_fp += 1; e_fp += 1
        remaining = len(gset) - len(used)
        d_fn += remaining; e_fn += remaining
    d_p = d_tp / max(1, d_tp + d_fp); d_r = d_tp / max(1, d_tp + d_fn)
    e_p = e_tp / max(1, e_tp + e_fp); e_r = e_tp / max(1, e_tp + e_fn)
    return (d_tp, d_fp, d_fn, e_tp, e_fp, e_fn,
            2 * d_p * d_r / max(1e-9, d_p + d_r),
            2 * e_p * e_r / max(1e-9, e_p + e_r))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-shard-glob", default="/scratch/pii_corpus/eu_shards/eu-val_en.jsonl")
    ap.add_argument("--max-rows", type=int, default=1000)
    ap.add_argument("--model-ckpt", default="/scratch/pii_corpus/eurobert_pii_eu.pt")
    ap.add_argument("--model", default="/root/models/EuroBERT-210m")
    ap.add_argument("--skip-presidio", action="store_true")
    ap.add_argument("--skip-model", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.val_shard_glob)]
    if args.max_rows: rows = rows[: args.max_rows]
    print(f"ai4privacy val en rows: {len(rows)} (avg spans {sum(len(r['spans']) for r in rows)/len(rows):.1f})")

    if not args.skip_presidio:
        analyzer = AnalyzerEngine()
        preds = run_presidio_rows(analyzer, rows)
        m = score_dtts(preds, rows)
        print(f"Presidio:  detection F1={m[6]:.4f} (P {m[3+0]/(max(1,m[3]+m[4])):.3f} R {m[3]/(max(1,m[3]+m[5])):.3f})  | exact-type F1={m[7]:.4f}")

    if not args.skip_model and os.path.exists(args.model_ckpt):
        import torch
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        sys_path = "/root/bitnet_train"
        import sys; sys.path.insert(0, sys_path)
        import eu_train  # noqa: F401 applies RoPE shim for trust_remote_code
        from eval_eu_pii import decode_spans, score
        sd = torch.load(args.model_ckpt, map_location="cpu", weights_only=False)
        lmap_rev = {v: k for k, v in sd["labels"].items()}
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForTokenClassification.from_pretrained(args.model, trust_remote_code=True, num_labels=len(lmap_rev))
        model.load_state_dict(sd["model"]); model.to("cuda").eval()
        all_dtp = all_dfp = all_dfn = all_etp = all_efp = all_efn = 0
        with torch.no_grad():
            for i in range(0, len(rows), 48):
                batch = rows[i:i+48]
                enc = tok([b["text"] for b in batch], padding=True, truncation=True, max_length=512,
                          return_offsets_mapping=True, return_tensors="pt")
                off = enc.pop("offset_mapping")
                logits = model(enc["input_ids"].cuda(), attention_mask=enc["attention_mask"].cuda()).logits.argmax(-1).cpu()
                for j, row in enumerate(batch):
                    offs = [(int(a), int(b)) for a, b in off[j].tolist()]
                    pred = decode_spans(logits[j].tolist(), offs, lmap_rev, len(offs), row["text"])
        mm = score(pred, row["spans"])
        all_dtp+=mm[0]; all_dfp+=mm[1]; all_dfn+=mm[2]
        all_etp+=mm[3]; all_efp+=mm[4]; all_efn+=mm[5]
    # model gets the SAME overlap-based detection metric as Presidio
    dtp2=dfp2=dfn2=0
    with torch.no_grad():
        for i in range(0, len(rows), 48):
            batch = rows[i:i+48]
            enc = tok([b["text"] for b in batch], padding=True, truncation=True, max_length=512,
                      return_offsets_mapping=True, return_tensors="pt")
            off = enc.pop("offset_mapping")
            logits = model(enc["input_ids"].cuda(), attention_mask=enc["attention_mask"].cuda()).logits.argmax(-1).cpu()
            for j, row in enumerate(batch):
                offs = [(int(a), int(b)) for a, b in off[j].tolist()]
                pred = decode_spans(logits[j].tolist(), offs, lmap_rev, len(offs), row["text"])
                g=[(s_["start"],s_["end"]) for s_ in row["spans"]]; p=[(x["start"],x["end"]) for x in pred]
                used=set()
                for ps_,pe_ in p:
                    hit=next(((gs,ge) for gs,ge in g if (gs,ge) not in used and spans_overlap(gs,ge,ps_,pe_)), None)
                    if hit: used.add(hit); dtp2+=1
                    else: dfp2+=1
                dfn2+=len(g)-len(used)
    dp2=dtp2/max(1,dtp2+dfp2); dr2=dtp2/max(1,dtp2+dfn2)
    dp = all_dtp/max(1,all_dtp+all_dfp); dr = all_dtp/max(1,all_dtp+all_dfn)
    ep = all_etp/max(1,all_etp+all_efp); er = all_etp/max(1,all_etp+all_efn)
    print(f"EuroBERT:  detection(exact-span) F1={2*dp*dr/max(1e-9,dp+dr):.4f} | detection(overlap) F1={2*dp2*dr2/max(1e-9,dp2+dr2):.4f} (P {dp2:.3f} R {dr2:.3f}) | exact-type F1={2*ep*er/max(1e-9,ep+er):.4f}")

if __name__ == "__main__":
    main()
