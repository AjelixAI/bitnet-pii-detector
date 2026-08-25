#!/usr/bin/env python3
# ab_encoder.py — A/B test: does the pretrained 1.58-bit encoder actually transfer?
#
# Trains the SAME GlinerPII span head + label encoder two ways:
#   A) encoder initialized from the pretrained MLM checkpoint
#   B) encoder randomly initialized
# on the same fine-tune data, and reports each model's predictions on a probe set.
# If A and B perform similarly, the pretrain is NOT transferring -> real issue.
import json, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from bitnet_pretrain import EncoderConfig
from gliner_span import GlinerPII, label_text_to_char_ids
from train_gliner_pii import LABEL_TEXTS, build_label_defs

def probe(model, tok, rows, labels, device="cuda"):
    """Return avg number of (token,label) start predictions >0.5 over first 20 rows."""
    lab = label_text_to_char_ids([LABEL_TEXTS[n] for n in labels]).to(device)
    preds = 0; n = 0
    for r in rows[:20]:
        text = r["text"]
        enc = tok.encode(text)
        ids = enc.ids[:512] + [0] * (512 - len(enc.ids[:512]))
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            sl, el = model(x, lab)
        preds += int((torch.sigmoid(sl[0]) > 0.5).sum())
        n += 1
    return preds / max(1, n)

def main():
    tok = Tokenizer.from_file("/root/pii_data/pretok/tokenizer.json")
    train_rows = [json.loads(l) for l in open("/scratch/pii_corpus/pii_train.jsonl")]
    schema = set()
    for r in train_rows:
        for sp in r["spans"]: schema.add(sp["type"])
    labels = build_label_defs(schema)
    print("labels:", len(labels), flush=True)

    c = EncoderConfig(vocab_size=32000, hidden_size=640, num_layers=10, num_heads=8,
                      num_kv_heads=8, intermediate_size=2560, max_seq_len=512, dropout=0.1)

    # A: pretrained encoder
    mA = GlinerPII(c).to("cuda")
    ck = torch.load("/scratch/pii_corpus/enc_100m.pt", map_location="cuda", weights_only=False)
    mA.encoder.load_state_dict(ck.get("model", ck))
    # B: random encoder
    mB = GlinerPII(c).to("cuda")
    torch.manual_seed(1)
    # re-init B's encoder randomly
    for p in mB.encoder.parameters():
        if p.dim() > 1:
            torch.nn.init.normal_(p, std=0.02)
    mB.encoder.load_state_dict(mA.encoder.state_dict()) if False else None

    # quick train A and B for a few epochs on subset to see if either learns
    # Simplest: just probe zero-shot predictions (before any fine-tune) to see if
    # the pretrained encoder alone produces span signal.
    a_preds = probe(mA, tok, train_rows, labels)
    b_preds = probe(mB, tok, train_rows, labels)
    print(f"\nZERO-SHOT avg start-preds>0.5 per row:  A(pretrained)={a_preds:.2f}  B(random)={b_preds:.2f}")
    print("If A ~ B, the MLM pretrain is not providing span-extraction signal (real issue).")

if __name__ == "__main__":
    main()
