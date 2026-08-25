#!/usr/bin/env python3
# eval_binary.py — evaluate the binary PII detector (detection-tier F1).
#
# The binary model predicts O / B-PII / I-PII / E-PII (3 classes). It answers:
# "did we find the PII span at all" (the redaction-relevant metric), regardless of
# the fine-grained type. This is what the SOTA (OPF) emphasizes: recall-biased,
# which is correct for privacy (missing PII worse than over-redacting).
import argparse, json
import torch
from tokenizers import Tokenizer
from bitnet_pretrain import EncoderConfig
from bioes_pii import BIOESModel


def decode_binary(logits, enc_offsets, seq_len):
    """Decode O/B/I/E (3-class) -> list of {start,end} char spans.
    Merges ALL contiguous non-O tokens into one span (B starts, I/E/S extends,
    O closes). This reconstructs an atomic PII span even when the tokenizer
    fragments it into subwords."""
    S = logits.shape[0]
    pred = logits.argmax(-1).cpu().numpy()
    spans = []
    cur = None
    for i in range(min(S, len(enc_offsets))):
        c = int(pred[i])
        if c == 0:
            if cur:
                spans.append(cur); cur = None
            continue
        # any non-O token (B/I/E/S) belongs to a PII span
        if cur is None:
            cur = [enc_offsets[i][0], enc_offsets[i][1]]
        else:
            cur[1] = max(cur[1], enc_offsets[i][1])
    if cur:
        spans.append(cur)
    # trim whitespace from each span
    out = []
    for s, e in spans:
        out.append((s, e))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    if args.limit: rows = rows[:args.limit]
    tok = Tokenizer.from_file("/root/pii_data/pretok_gpt/tokenizer.json")
    c = EncoderConfig(vocab_size=65000, hidden_size=1024, num_layers=16, num_heads=8,
                      num_kv_heads=8, intermediate_size=4096, max_seq_len=args.seq, dropout=0.1)
    model = BIOESModel(c, 1).to("cuda")
    model.head = torch.nn.Linear(c.hidden_size, 3, bias=False).to("cuda")
    ck = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    sd = ck.get("model", ck)
    # if head is 5-wide (from BIOESModel(c,1)), strip to 3
    if sd["head.weight"].shape[0] == 5:
        sd["head.weight"] = sd["head.weight"][:3]
    model.load_state_dict(sd)
    model.eval()
    tp=fp=fn=0
    for r in rows:
        text = r["text"]; enc = tok.encode(text)
        ids = enc.ids[:args.seq] + [0]*(args.seq - len(enc.ids[:args.seq]))
        x = torch.tensor([ids], dtype=torch.long, device="cuda")
        with torch.no_grad(): logits, _ = model(x)
        pred = decode_binary(logits[0], enc.offsets[:args.seq], args.seq)
        gold = set((sp["start"], sp["end"]) for sp in r.get("spans", []))
        used = set()
        for s, e in pred:
            # match: predict covers gold start (overlap) -> count as found
            matched = False
            for gs, ge in gold:
                if gs >= s and ge <= e:
                    matched = True; break
            if matched and (s, e) not in used:
                used.add((s, e)); tp += 1
            else:
                fp += 1
        fn += len([g for g in gold if g not in used])
    P = tp/max(1,tp+fp); R = tp/max(1,tp+fn); F1 = 2*P*R/max(1e-9,P+R)
    print(f"eval {len(rows)} rows")
    print(f"=== DETECTION-TIER (span found, type ignored) ===")
    print(f"tp={tp} fp={fp} fn={fn}  P={P:.4f} R={R:.4f} F1={F1:.4f}")


if __name__ == "__main__":
    main()
