#!/usr/bin/env python3
"""AjelixAI PII Detector (EuroBERT-610m) — FastAPI inference service.

GET /            browser UI (paste text -> PII spans)
POST /predict    {"text": str}  -> {"spans": [{type,start,end,text,score}], "n": int}
POST /batch      {"texts": [str]} -> {"results": [...]}
GET /health      ok
"""
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, JSONResponse
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_DIR = "/root/pii-bench/models/best_model"
MAXLEN = 512

print("Loading model ...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=False)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR, trust_remote_code=False).cuda().eval()
id2label = {int(k): v for k, v in model.config.id2label.items()}


def detect_byte_offsets(tok):
    s = "аа"
    enc = tok(s, return_offsets_mapping=True)
    offs = [o for o in enc["offset_mapping"] if o[0] != o[1]]
    return bool(offs and offs[0][1] >= 2)


BYTE = detect_byte_offsets(tokenizer)
print(f"offset_conv={'byte' if BYTE else 'char'} | labels={len(id2label)}", flush=True)


def b2c(text, b):
    raw = text.encode("utf-8")
    b = max(0, min(b, len(raw)))
    while b > 0 and b < len(raw) and (raw[b] & 0xC0) == 0x80:
        b -= 1
    return len(raw[:b].decode("utf-8"))


import string as _string

# Characters that may legitimately appear *inside* an entity of each type.
# Used to bridge fragmented spans (e.g. an email split by subword tokenization)
# where the model predicted O on a middle sub-token.
ALNUM = _string.ascii_letters + _string.digits
BRIDGE = {
    "EMAIL": set(ALNUM + "@._-+#"),
    "IBAN": set(ALNUM + " "),
    "BANK_CARD_NUMBER": set(ALNUM + " "),
    "CREDIT_CARD": set(ALNUM + " "),
    "CREDIT_DEBIT_CARD": set(ALNUM + " "),
    "PHONE_NUMBER": set("0123456789+ ()-/."),
    "PHONE": set("0123456789+ ()-/."),
    "FAX_NUMBER": set("0123456789+ ()-/."),
    "URL": set(ALNUM + ":/?#[]@!$&'()*+,;=.-_%"),
    "HTTP_COOKIE": set(ALNUM + "=;@.-_"),
    "USERNAME": set(ALNUM + "._-@"),
    "IPV4": set("0123456789."),
    "IPV6": set("0123456789abcdefABCDEF:"),
    "IP_ADDRESS": set("0123456789."),
    "TAX_ID": set(ALNUM + "-"),
    "SSN": set("0123456789-"),
    "PASSPORT_NUMBER": set(ALNUM),
    "DRIVER_LICENSE_NUMBER": set(ALNUM),
    "BIC": set(ALNUM),
    "SWIFT_BIC": set(ALNUM),
    "COORDINATE": set("0123456789.,- "),
}
# Types we do NOT bridge (semantic, language-dependent): keep fragments separate.
NO_BRIDGE = {"NAME", "PERSON", "FIRST_NAME", "LAST_NAME", "GIVENNAME", "SURNAME",
             "ADDRESS", "STREET_ADDRESS", "LOC", "CITY", "COMPANY", "COMPANY_NAME",
             "ORG", "MISC", "OCCUPATION", "JOB", "DATE", "TIME", "FINANCIAL_ENTITY",
             "AGE", "GENDER", "ORGANIZATION"}


import re as _re

# Well-defined formats: anchor each predicted span to the full regex match it sits in.
FORMAT_RE = {
    "EMAIL": _re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    "IBAN": _re.compile(r"\b[A-Z]{2}[0-9]{2}(?:[ ]?[0-9]{1,4}){3,9}\b"),
    "BANK_CARD_NUMBER": _re.compile(r"\b(?:\d[ \-]?){12,19}\b"),
    "CREDIT_CARD": _re.compile(r"\b(?:\d[ \-]?){12,19}\b"),
    "CREDIT_DEBIT_CARD": _re.compile(r"\b(?:\d[ \-]?){12,19}\b"),
    "PHONE_NUMBER": _re.compile(r"(?:\+\d{1,4}[\s\-]?)?(?:\(\d{1,5}\)[\s\-]?)?\d[\d\s().\-]{4,}\d"),
    "PHONE": _re.compile(r"(?:\+\d{1,4}[\s\-]?)?(?:\(\d{1,5}\)[\s\-]?)?\d[\d\s().\-]{4,}\d"),
    "BIC": _re.compile(r"\b[A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b"),
    "SWIFT_BIC": _re.compile(r"\b[A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b"),
    "IPV4": _re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"),
    "URL": _re.compile(r"(?:https?://|www\.)[^\s\"'<>]+"),
}


def expand_format(sp, text):
    rx = FORMAT_RE.get(sp["type"])
    if rx is None:
        return sp
    s, e = sp["start"], sp["end"]
    best = None
    for m in rx.finditer(text):
        if m.start() <= s < m.end():
            if best is None or m.end() > best.end():
                best = m
    if best:
        sp["start"], sp["end"], sp["text"] = best.start(), best.end(), best.group()
    return sp


def merge_spans(spans, text):
    """Merge adjacent same-type spans whose intervening gap is filled by entity-valid chars."""
    spans = [expand_format(sp, text) for sp in spans]
    # drop overlapping dupes (prefer the longer), then remerge
    spans.sort(key=lambda x: (x["start"], -x["end"]))
    ded = []
    for sp in spans:
        if ded and sp["start"] < ded[-1]["end"] and sp["type"] == ded[-1]["type"]:
            if sp["end"] > ded[-1]["end"]:
                ded[-1]["start"], ded[-1]["end"], ded[-1]["text"] = sp["start"], sp["end"], sp["text"]
            continue
        ded.append(sp)
    out = []
    for sp in sorted(ded, key=lambda x: (x["start"], -x["end"])):
        if out and out[-1]["type"] == sp["type"] and out[-1]["end"] < sp["start"]:
            gap = text[out[-1]["end"]:sp["start"]]
            allow = BRIDGE.get(sp["type"])
            if allow is not None and gap and all(c in allow for c in gap) and len(gap) <= 40:
                out[-1]["end"] = sp["end"]
                out[-1]["text"] = text[out[-1]["start"]:sp["end"]]
                out[-1]["score"] = min(out[-1]["score"], sp["score"])
                continue
        out.append(sp)
    return out


# Higher precedence first: a region claimed by one format type can't be re-typed by a
# lower-precedence one (e.g. phone must not swallow IBAN/card digit sequences).
FORMAT_PRECEDENCE = ["IBAN", "BANK_CARD_NUMBER", "CREDIT_CARD", "CREDIT_DEBIT_CARD",
                     "EMAIL", "IPV4", "URL", "BIC", "SWIFT_BIC", "PHONE_NUMBER", "PHONE"]


def format_pass(spans, text):
    """Detect format-defined entities by regex (highest-precedence first); add any the
    model missed, and re-type model spans that are actually clean format entities."""
    out = list(spans)
    claimed = []
    for t in FORMAT_PRECEDENCE:
        rx = FORMAT_RE.get(t)
        if rx is None:
            continue
        for m in rx.finditer(text):
            a, b = m.start(), m.end()
            # skip regions already claimed by a higher-precedence format
            if any(cs < b and a < ce for (cs, ce) in claimed):
                continue
            hit = None
            for sp in out:
                if sp["start"] < b and a < sp["end"]:
                    hit = sp
                    break
            # trim leading/trailing punctuation that isn't part of the entity
            while a < b and text[a] in "([\"']":
                a += 1
            while b > a and text[b - 1] in ".,;:!?)]\"'":
                b -= 1
            if b <= a:
                continue
            if hit is not None:
                hit["type"], hit["start"], hit["end"], hit["text"] = t, a, b, text[a:b]
            else:
                out.append({"type": t, "start": a, "end": b, "text": text[a:b], "score": 0.99})
            claimed.append((a, b))
    return out


def dedupe(spans):
    seen = set(); res = []
    for sp in sorted(spans, key=lambda x: (x["start"], -x["end"])):
        key = (sp["type"], sp["start"], sp["end"])
        if key in seen:
            continue
        seen.add(key)
        res.append(sp)
    return res


def extract(text):
    enc = tokenizer(text, truncation=True, max_length=MAXLEN, return_offsets_mapping=True)
    ids = torch.tensor([enc["input_ids"]]).cuda()
    attn = torch.tensor([enc["attention_mask"]]).cuda()
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=attn).logits[0]
    probs = torch.softmax(logits, -1).cpu()
    preds = logits.argmax(-1).cpu().tolist()
    offs = enc["offset_mapping"]
    spans, cur = [], None
    for i, pid in enumerate(preds[:len(offs)]):
        tag = id2label.get(pid, "O"); a, b = offs[i]
        if a == b:
            if cur: spans.append(cur); cur = None
            continue
        if tag == "O":
            if cur: spans.append(cur); cur = None
        elif cur and cur[2] == tag[2:] and i == cur[3] + 1:
            cur = (cur[0], b, cur[2], i, max(cur[4], float(probs[i, pid])))
        else:
            if cur: spans.append(cur)
            cur = (a, b, tag[2:], i, float(probs[i, pid]))
    if cur: spans.append(cur)
    conv = []
    for a, b, t, _, sc in spans:
        ca, cb = (b2c(text, a), b2c(text, b)) if BYTE else (a, b)
        while ca < cb and text[ca].isspace():
            ca += 1
        while cb > ca and text[cb - 1].isspace():
            cb -= 1
        if cb > ca:
            conv.append({"type": t, "start": ca, "end": cb, "text": text[ca:cb], "score": round(sc, 4)})
    return dedupe(format_pass(merge_spans(conv, text), text))


app = FastAPI(title="AjelixAI PII Detector — EuroBERT-610m")


class TextReq(BaseModel):
    text: str
    max_length: int = MAXLEN


class BatchReq(BaseModel):
    texts: list[str]


@app.get("/health")
def health():
    return {"status": "ok", "labels": len(id2label)}


@app.post("/predict")
def predict(req: TextReq):
    try:
        spans = extract(req.text)
        return {"spans": spans, "n": len(spans), "text": req.text}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/batch")
def batch(req: BatchReq):
    return {"results": [{"spans": extract(t)} for t in req.texts]}


HTML = """<!doctype html><html><head><meta charset="utf-8"><title>AjelixAI PII Detector</title>
<style>body{font-family:system-ui;max-width:760px;margin:2rem auto;padding:0 1rem}
textarea{width:100%;height:120px;font-size:14px}
button{padding:.6rem 1.4rem;font-size:15px;cursor:pointer}
table{border-collapse:collapse;width:100%;margin-top:1rem}
th,td{border:1px solid #ccc;padding:6px 8px;text-align:left;font-size:14px}
th{background:#f0f0f0}.tag{background:#eef;padding:2px 6px;border-radius:4px}</style></head><body>
<h2>AjelixAI PII Detector — EuroBERT-610m</h2>
<p>Paste text, detect PII. (Multilingual; best on EN/EU languages.)</p>
<textarea id=text placeholder="Paste text here...">Oye Alliyah, ¿has recibido la tarjeta de débito de Banco Mare Nostrum?</textarea><br>
<button onclick=run()>Detect PII</button> <span id=status></span>
<div id=out></div>
<script>
async function run(){
  const t=document.getElementById('text').value; const s=document.getElementById('status');
  s.textContent='...'; document.getElementById('out').innerHTML='';
  try{
    const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:t})});
    const d=await r.json(); s.textContent=d.n+' PII span(s) detected';
    let h='<table><tr><th>Type</th><th>Text</th><th>Pos</th><th>Conf</th></tr>';
    for(const x of d.spans){h+=`<tr><td><span class=tag>${x.type}</span></td><td><b>${x.text}</b></td><td>${x.start}-${x.end}</td><td>${x.score}</td></tr>`}
    h+='</table>'; document.getElementById('out').innerHTML=h;
  }catch(e){s.textContent='error: '+e}
}
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def ui():
    return HTML


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
