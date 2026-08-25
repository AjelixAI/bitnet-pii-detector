#!/usr/bin/env python3
"""hybrid_decode.py — decode post-processing on top of BIOES logits.

Three stages:
  1) legal-sequence BIOES collapse (B/I/E/S/O) — no fragment spoons.
  2) validators on structurally regular types (email/Luhn/CC/zip/ID/passport/date/phone);
     demote failures unless a context cue bumps them.
  3) context cues — language-agnostic cue phrases ("my name is", "iban:", "card", "call me"...).

Deterministic; no extra model, no gradient.
"""
import re

# Validators
RE_EMAIL = re.compile(r"^[\w.+\-]+@[\w\-]+(\.[\w\-]+)+$")
RE_PHONE = re.compile(r"^\+?\d[\d\s()./-]{4,}\d$")
RE_CC = re.compile(r"^\d[\d\s-]{11,17}\d$")
RE_ZIP = re.compile(r"^\d{3,10}([\- ]\d{1,5})?$")
RE_IDCARD = re.compile(r"^[A-Z0-9]{5,20}$", re.I)
RE_PASSPORT = re.compile(r"^[A-Z0-9]{6,12}$", re.I)
RE_DATE_DMY = re.compile(r"^\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}$")
RE_DATE_ISO = re.compile(r"^\d{4}[/-]\d{2}[/-]\d{2}")
RE_DATE_MON = re.compile(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}", re.I)

_EU_TZ = re.compile(r"^(0?[1-9]|1[0-9]|2[0-4])\.(0?[1-9]|1[0-2])\.\d{2,4}$")  # EU date dd.mm.yyyy

VALIDATORS = {
    "EMAIL": RE_EMAIL,
    "TELEPHONENUM": RE_PHONE,
    "CREDITCARDNUMBER": RE_CC,
    "ZIPCODE": lambda s: bool(RE_ZIP.match(s)),
    "IDCARDNUM": lambda s: len(s) >= 5 and bool(RE_IDCARD.match(s)),
    "PASSPORTNUM": RE_PASSPORT,
    "DRIVERLICENSENUM": lambda s: bool(re.match(r"^[A-Z0-9\-]{4,16}$", s, re.I)),
    "DATE": lambda s: bool(RE_DATE_DMY.match(s) or RE_DATE_ISO.match(s) or RE_DATE_MON.match(s) or _EU_TZ.match(s)),
}

def luhn_ok(surface):
    ds = [int(c) for c in surface if c.isdigit()]
    if len(ds) < 12:
        return False
    s = 0
    for i, d in enumerate(reversed(ds)):
        if i % 2 == 1:
            d *= 2
            if d > 9: d -= 9
        s += d
    return s % 10 == 0

def valid_span(t, surface):
    t = t.upper()
    fn = VALIDATORS.get(t)
    if t == "CREDITCARDNUMBER":
        return bool(RE_CC.match(surface)) and luhn_ok(surface)
    if fn is None:
        return True
    return bool(fn.match(surface)) if callable(fn) else fn(surface)


# Context cues (look ±80 chars around span for a cue phrase)
_CUES = {
    "GIVENNAME": [r"\bname\b", r"\bcalled\b", r"\bi am\b", r"\bmein name\b", r"\bich bin\b", r"\bm[aä]dchenname\b", r"\bsoy\b"],
    "SURNAME": [r"\bsurname\b", r"\blast name\b", r"\bfamily name\b", r"\bnachname\b", r"\bapellido\b"],
    "PERSON": [r"\bdr\.", r"\bmr\.", r"\bmrs\.", r"\bms\.", r"\bmadam\b", r"\bherr\b", r"\bfrau\b", r"\bher\b", r"\bseñor\b", r"\bprofesor\b"],
    "EMAIL": [r"\be-?mail\b", r"\bescribe\b", r"\bwrite\b", r"\bmailen\b", r"@"],
    "TELEPHONENUM": [r"\bcall\b", r"\bphone\b", r"\btel\b", r"\bmobile\b", r"\btelefon\b", r"\banrufen\b", r"\bellamar\b"],
    "DATE": [r"\bborn\b", r"\bbirthday\b", r"\bdate\b", r"\bgeboortedatum\b", r"\bgeboren\b", r"\bnacido\b", r"\bn[ée]\b", r"\bhalaman\b"],
    "STREET": [r"\baddress\b", r"\bstreet\b", r"\broad\b", r"\bdirección\b", r"\badress\b", r"\bavenue\b"],
    "CREDITCARDNUMBER": [r"\bcard\b", r"\bcredit\b", r"\bvisa\b", r"\bmastercard\b", r"\bamex\b", r"\bkarte\b"],
    "ZIPCODE": [r"\bzip\b", r"\bpostal\b", r"\bpostcode\b", r"\bcodigo\b", r"\bplz\b"],
    "IDCARDNUM": [r"\bid\b", r"\bausweis\b", r"\bidcard\b", r"\bdni\b", r"\bnie\b"],
    "TAXNUM": [r"\btax\b", r"\bsteuer\b", r"\bhacienda\b"],
}
_C = {k: [re.compile(p, re.I) for p in v] for k, v in _CUES.items()}

def context_hit(t, text, span):
    pats = _C.get(t.upper())
    if not pats:
        return False
    lo = max(0, span["start"] - 90); hi = min(len(text), span["end"] + 90)
    win = text[lo:hi]
    return any(p.search(win) for p in pats)


# BIOES decode
def decode_bioes(logits, offsets, lmap_rev, text):
    cur, raw = None, []
    for ti, (a, b) in enumerate(offsets):
        if a == b or b == 0:
            if cur: raw.append(cur); cur = None
            continue
        tag = lmap_rev.get(logits[ti], "O")
        if tag.startswith("B-"):
            if cur: raw.append(cur)
            cur = [a, b, tag[2:]]
        elif tag.startswith("S-"):
            if cur: raw.append(cur); cur = None
            raw.append([a, b, tag[2:]])
        elif tag.startswith("I-"):
            lab = tag[2:]
            if cur and cur[2] == lab:
                cur[1] = b
            elif not cur:
                cur = [a, b, lab]
            else:
                raw.append(cur); cur = [a, b, lab]
        elif tag.startswith("E-"):
            lab = tag[2:]
            if cur and cur[2] == lab:
                cur[1] = b; raw.append(cur); cur = None
            else:
                raw.append([a, b, lab])
        else:
            if cur: raw.append(cur); cur = None
    if cur: raw.append(cur)
    out = []
    for s, e, t in raw:
        s, e = int(s), int(e)
        while s < e and s < len(text) and text[s].isspace(): s += 1
        while e > s and e <= len(text) and text[e-1].isspace(): e -= 1
        if e > s:
            out.append({"start": s, "end": e, "type": t})
    return out


def decode(text, logits, offsets, lmap_rev, use_hybrid=True):
    spans = decode_bioes(logits, offsets, lmap_rev, text)
    if not use_hybrid:
        return spans
    kept = []
    for sp in spans:
        surface = text[sp["start"]:sp["end"]]
        if valid_span(sp["type"], surface):
            kept.append(sp)
            continue
        if context_hit(sp["type"], text, sp):
            kept.append(sp)
    return kept
