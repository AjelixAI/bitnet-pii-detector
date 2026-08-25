#!/usr/bin/env python3
# eval_real_bioes.py — evaluate the trained BIOES model on REAL ai4privacy val.
#
# Uses the FIXED 38-type schema the model was trained with (saved at fine-tune time),
# converts real ai4privacy entities to char spans, and measures exact + detection F1.
import json, torch, argparse
from tokenizers import Tokenizer
from bitnet_pretrain import EncoderConfig
from bioes_pii import BIOESModel, build_type_names, make_label_names
from eval_bioes import decode_bioes

# The 38 types the model was trained on (from the verified synthetic + real prep).
TRAIN_TYPES = None  # loaded from meta.json if present

def load_types(path="/scratch/pii_corpus/bioes_types.json"):
    try:
        return json.load(open(path))
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/scratch/pii_corpus/bioes_pii.pt")
    ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--types", default="/scratch/pii_corpus/bioes_types.json")
    args = ap.parse_args()
    types = load_types(args.types)
    if not types:
        print("ERROR: need bioes_types.json"); return
    names, tmap = make_label_names(types)
    from datasets import load_dataset
    ds = load_dataset("automated-analytics/ai4privacy-pii-masking-en-v1-ner", split="validation")
    tok = Tokenizer.from_file("/root/pii_data/pretok_gpt/tokenizer.json")
    c = EncoderConfig(vocab_size=65000, hidden_size=1024, num_layers=16, num_heads=8,
                      num_kv_heads=8, intermediate_size=4096, max_seq_len=512, dropout=0.1)
    m = BIOESModel(c, len(types)).to("cuda")
    m.load_state_dict(torch.load(args.ckpt, map_location="cuda", weights_only=False)["model"])
    m.eval()
    tp=fp=fn=0; tp_s=fp_s=fn_s=0
    for i in range(min(args.limit, len(ds))):
        s = ds[i]; text = s["source_text"] or " ".join(s["tokens"])
        spans = []
        for ent in s["entities"]:
            et = ent.get("entity",""); 
            idx = text.find(et)
            if idx >= 0:
                spans.append({"start":idx,"end":idx+len(et)})
        enc = tok.encode(text); ids = enc.ids[:512]+[0]*(512-len(enc.ids[:512]))
        x = torch.tensor([ids], dtype=torch.long, device="cuda")
        with torch.no_grad(): logits,_ = m(x)
        pred = decode_bioes(logits[0], enc.offsets[:512], names, tmap, 512, text)
        gold = set((sp["start"],sp["end"]) for sp in spans)
        used=set()
        for p in pred:
            k=(p["start"],p["end"])
            if k in gold and k not in used: tp+=1; used.add(k)
            else: fp+=1
        fn += len(gold - used)
    P=tp/max(1,tp+fp); R=tp/max(1,tp+fn); F1=2*P*R/max(1e-9,P+R)
    print(f"REAL ai4privacy val ({min(args.limit,len(ds))} rows): tp={tp} fp={fp} fn={fn}")
    print(f"  Precision={P:.3f} Recall={R:.3f} F1={F1:.3f}")

if __name__ == "__main__":
    main()
