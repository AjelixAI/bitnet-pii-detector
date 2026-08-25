#!/usr/bin/env python3
# serve_bioes_pii.py — public HTTP API for the 0.73-F1 1.58-bit PII detector.
#
# Loads the trained BIOES PII model (bioes_pii.pt) and serves span detection.
# POST /predict  {"text": "..."}  -> {pii:[{type,start,end,text,conf}], count}
#
# Run:  /root/venv/bin/python serve_bioes_pii.py --ckpt /scratch/pii_corpus/bioes_pii.pt --port 8787
import sys, os, argparse, json
import torch
from fastapi import FastAPI, Request
import uvicorn
from tokenizers import Tokenizer
from bitnet_pretrain import EncoderConfig
from bioes_pii import BIOESModel, make_label_names
from eval_bioes import decode_bioes


def load_types(path):
    return json.load(open(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/scratch/pii_corpus/bioes_350m.pt")
    ap.add_argument("--types", default="/scratch/pii_corpus/bioes_types.json")
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()
    types = load_types(args.types)
    label_names, tmap = make_label_names(types)
    tok = Tokenizer.from_file("/root/pii_data/pretok_gpt/tokenizer.json")
    c = EncoderConfig(vocab_size=65000, hidden_size=1024, num_layers=16, num_heads=8,
                      num_kv_heads=8, intermediate_size=4096, max_seq_len=512, dropout=0.1)
    model = BIOESModel(c, len(types)).to("cuda")
    model.load_state_dict(torch.load(args.ckpt, map_location="cuda", weights_only=False)["model"])
    model.eval()
    print(f"API loaded: {len(types)} types, params={round(model.param_count()/1e6,1)}M", flush=True)

    def detect(text):
        enc = tok.encode(text)
        ids = enc.ids[:512] + [0]*(512 - len(enc.ids[:512]))
        x = torch.tensor([ids], dtype=torch.long, device="cuda")
        with torch.no_grad():
            logits, _ = model(x)
        spans = decode_bioes(logits[0], enc.offsets[:512], label_names, tmap, 512, text)
        return spans

    app = FastAPI(title="PII Detector (1.58-bit native encoder)")

    @app.post("/predict")
    async def predict(req: Request):
        body = await req.json()
        text = body.get("text", "")
        if not text:
            return {"error": "provide 'text'"}
        spans = detect(text)
        out = []
        for s in spans:
            out.append({"text": text[s["start"]:s["end"]], "type": s["type"],
                        "start": s["start"], "end": s["end"]})
        return {"pii": out, "count": len(out)}

    @app.get("/health")
    async def health():
        return {"ok": True, "types": types}

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
