#!/usr/bin/env python3
"""hybrid_decode.py — post-processing layer on top of BIOES logits.

Three stages:
  1) Legal-sequence BIOES decode (B/I/E/S/O) — no fragments.
  2) Validators on structurally-regular PII types (email/Luhn/IBAN/IP/etc.);
     a candidate failing its validator is dropped UNLESS a context cue rescues it.
  3) Context cues — multilingual cue phrases near the span ("my name is",
     "IBAN:", "credit card", "call me at", etc.) that rescue borderline spans.

Covers the PIIBench 26-type canonical taxonomy (arxiv 2604.15776).
Deterministic; no extra model, no gradient.
"""
import re

# ═══════════════════════════════════════════════════════════════
# VALIDATORS — regex/checksum for each structurally-regular type
# ═══════════════════════════════════════════════════════════════

RE_EMAIL = re.compile(r"^[\w.+\-]+@[\w\-]+(\.[\w\-]+)+$")
RE_PHONE = re.compile(r"^\+?\d[\d\s()./-]{4,}\d$")
RE_CC = re.compile(r"^\d[\d\s-]{11,17}\d$")
RE_IBAN = re.compile(r"^[A-Z]{2}\d{2}[A-Z0-9]{1,30}$")
RE_BIC = re.compile(r"^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$")
RE_IPv4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
RE_IPv6 = re.compile(r"^[0-9a-fA-F:]{2,39}$")
RE_MAC = re.compile(r"^([0-9a-fA-F]{2}[:-]){5}[0-9a-fA-F]{2}$")
RE_URL = re.compile(r"^(https?://)?[\w\-]+(\.[\w\-]+)+(/[\w\-./?=&%#]*)?$")
RE_USERNAME = re.compile(r"^[\w.\-]{3,30}$")
RE_PASSWORD = re.compile(r"^[\S]{4,}$")
RE_SSN = re.compile(r"^\d{3}-?\d{2}-?\d{4}$")
RE_TAX_ID = re.compile(r"^\d{2}-?\d{7}$|^IT[A-Z0-9]{11}$|^DE\d{9,11}$")
RE_DATE_DMY = re.compile(r"^\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}$")
RE_DATE_ISO = re.compile(r"^\d{4}[/-]\d{2}[/-]\d{2}")
RE_DATE_MON = re.compile(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}", re.I)
RE_TIME = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?\s*(am|pm|AM|PM)?$")
RE_CRYPTO = re.compile(r"^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$|^0x[a-fA-F0-9]{40}$|^bc1[a-z0-9]{39,59}$")
RE_PIN = re.compile(r"^\d{3,8}$")
RE_VEHICLE = re.compile(r"^[A-Z0-9]{5,8}[A-Z]$")  # VIN-like
RE_ACCOUNT = re.compile(r"^\d{6,20}$")
RE_ROUTING = re.compile(r"^\d{9}$")
RE_AMOUNT = re.compile(r"^[\$€£¥]\s?\d[\d,.]*$|^\d[\d,.]*\s?(EUR|USD|GBP|JPY|€|\$|£)$")
RE_CURRENCY = re.compile(r"^(USD|EUR|GBP|JPY|CHF|\$|€|£|¥)$", re.I)
RE_ZIP = re.compile(r"^\d{3,10}([\- ]\d{1,5})?$")

VALIDATORS = {
    "EMAIL": RE_EMAIL,
    "PHONE": RE_PHONE,
    "CREDIT_CARD": None,  # needs Luhn (handled below)
    "IBAN": RE_IBAN,
    "BIC": RE_BIC,
    "IP_ADDRESS": lambda s: bool(RE_IPv4.match(s) or RE_IPv6.match(s) or RE_MAC.match(s)),
    "URL": RE_URL,
    "USERNAME": RE_USERNAME,
    "PASSWORD": RE_PASSWORD,
    "SSN": RE_SSN,
    "TAX_ID": RE_TAX_ID,
    "DATE": lambda s: bool(RE_DATE_DMY.match(s) or RE_DATE_ISO.match(s) or RE_DATE_MON.match(s)),
    "TIME": RE_TIME,
    "CRYPTO_ADDRESS": RE_CRYPTO,
    "PIN": RE_PIN,
    "VEHICLE": RE_VEHICLE,
    "ACCOUNT_NUMBER": RE_ACCOUNT,
    "ROUTING_NUMBER": RE_ROUTING,
    "ADDRESS": lambda s: bool(RE_ZIP.match(s)) or len(s) > 3,
    "AMOUNT": RE_AMOUNT,
    "CURRENCY": RE_CURRENCY,
}

def luhn_ok(surface):
    ds = [int(c) for c in surface if c.isdigit()]
    if len(ds) < 12: return False
    s = 0
    for i, d in enumerate(reversed(ds)):
        if i % 2 == 1:
            d *= 2
            if d > 9: d -= 9
        s += d
    return s % 10 == 0

def iban_checksum(surface):
    """Validate IBAN using the mod-97 algorithm."""
    s = re.sub(r"\s+", "", surface).upper()
    if len(s) < 5 or not RE_IBAN.match(s): return False
    rearr = s[4:] + s[:4]
    nums = ""
    for ch in rearr:
        if ch.isdigit(): nums += ch
        elif ch.isalpha(): nums += str(int(ch, 36) - 9 + 9)  # A=10
        else: return False
    try: return int(nums) % 97 == 1
    except: return False

def valid_span(ptype, surface):
    t = ptype.upper()
    # Credit card: Luhn check
    if t == "CREDIT_CARD":
        return bool(RE_CC.match(surface)) and luhn_ok(surface)
    # IBAN: mod-97 checksum
    if t == "IBAN":
        return iban_checksum(surface)
    fn = VALIDATORS.get(t)
    if fn is None:
        return True  # PERSON, ORG, LOC, MISC, JOB — no regex, accept model
    return bool(fn.match(surface)) if hasattr(fn, "match") else bool(fn(surface))


# ═══════════════════════════════════════════════════════════════
# CONTEXT CUES — multilingual phrases near spans that rescue borderline preds
# (±80 char window). Privacy-correct: recall bias — better to over-detect.
# ═══════════════════════════════════════════════════════════════

_CUES = {
    "PERSON": [r"\bname\b", r"\bcalled\b", r"\bi am\b", r"\bdr\.", r"\bmr\.", r"\bmrs\.", r"\bms\.",
               r"\bher\s", r"\bhis\s", r"\bich bin\b", r"\bmein name\b", r"\bherr\b", r"\bfrau\b",
               r"\bseñor\b", r"\bseñora\b", r"\bsoy\b", r"\bje suis\b", r"\bmadame\b", r"\bmonsieur\b"],
    "EMAIL": [r"\be-?mail\b", r"\bwrite\b", r"\bescribe\b", r"@", r"\bmailto\b"],
    "PHONE": [r"\bcall\b", r"\bphone\b", r"\btel\b", r"\bmobile\b", r"\btelefon\b", r"\btéléphone\b",
              r"\bteléfono\b", r"\brufnummer\b", r"\banrufen\b", r"\bcontact\b"],
    "DATE": [r"\bborn\b", r"\bbirthday\b", r"\bdate\b", r"\bgeboren\b", r"\bnacido\b", r"\bné\b",
             r"\bdatum\b", r"\bsince\b", r"\buntil\b"],
    "ADDRESS": [r"\baddress\b", r"\bstreet\b", r"\broad\b", r"\bdirección\b", r"\badress\b", r"\bplz\b",
                r"\bpostal\b", r"\bzip\b"],
    "CREDIT_CARD": [r"\bcard\b", r"\bcredit\b", r"\bvisa\b", r"\bmastercard\b", r"\bamex\b", r"\bkarte\b"],
    "IBAN": [r"\biban\b", r"\bbank\b", r"\bkonto\b", r"\bcompte\b", r"\bcuenta\b"],
    "SSN": [r"\bssn\b", r"\bsocial security\b", r"\bsozialversichert\b"],
    "TAX_ID": [r"\btax\b", r"\bsteuer\b", r"\bfiscal\b"],
    "IP_ADDRESS": [r"\bip\b", r"\baddress\b", r"\bnetwork\b", r"\bserver\b"],
    "URL": [r"\burl\b", r"\bwebsite\b", r"\blink\b", r"\bhttp\b"],
    "USERNAME": [r"\busername\b", r"\blogin\b", r"\buser\b", r"\baccount\b", r"\bbenutzer\b"],
    "PASSWORD": [r"\bpassword\b", r"\bpasswort\b", r"\bsecret\b", r"\bcredential\b"],
    "ORG": [r"\bcompany\b", r"\borganization\b", r"\bfirma\b", r"\bempresa\b", r"\bsociété\b"],
    "CRYPTO_ADDRESS": [r"\bwallet\b", r"\bbtc\b", r"\beth\b", r"\bbitcoin\b", r"\bethereum\b"],
    "ACCOUNT_NUMBER": [r"\baccount\b", r"\bkonto\b", r"\bcompte\b"],
    "PIN": [r"\bpin\b", r"\bcode\b", r"\bpasscode\b"],
    "TIME": [r"\btime\b", r"\bat \d", r"\bo'clock\b"],
    "AMOUNT": [r"\bamount\b", r"\bsum\b", r"\btotal\b", r"\bprice\b"],
    "CURRENCY": [r"\bcurrency\b", r"\b\$", r"\b€"],
}
_C = {k: [re.compile(p, re.I) for p in v] for k, v in _CUES.items()}

def context_hit(ptype, text, span):
    pats = _C.get(ptype.upper())
    if not pats: return False
    lo = max(0, span["start"] - 80); hi = min(len(text), span["end"] + 80)
    win = text[lo:hi]
    return any(p.search(win) for p in pats)


# ═══════════════════════════════════════════════════════════════
# BIOES DECODE (legal sequence enforcement)
# ═══════════════════════════════════════════════════════════════

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
            if cur and cur[2] == lab: cur[1] = b
            elif not cur: cur = [a, b, lab]
            else: raw.append(cur); cur = [a, b, lab]
        elif tag.startswith("E-"):
            lab = tag[2:]
            if cur and cur[2] == lab: cur[1] = b; raw.append(cur); cur = None
            else: raw.append([a, b, lab])
        else:
            if cur: raw.append(cur); cur = None
    if cur: raw.append(cur)
    out = []
    for s, e, t in raw:
        s, e = int(s), int(e)
        while s < e and s < len(text) and text[s].isspace(): s += 1
        while e > s and e <= len(text) and text[e-1].isspace(): e -= 1
        if e > s: out.append({"start": s, "end": e, "type": t})
    return out


def decode(text, logits, offsets, lmap_rev, use_hybrid=True):
    """BIOES decode → spans, then hybrid filter (validators + context cues)."""
    spans = decode_bioes(logits, offsets, lmap_rev, text)
    if not use_hybrid:
        return spans
    kept = []
    for sp in spans:
        surface = text[sp["start"]:sp["end"]]
        if valid_span(sp["type"], surface):
            kept.append(sp)
            continue
        # validator failed → rescue via context cue (recall bias for privacy)
        if context_hit(sp["type"], text, sp):
            kept.append(sp)
    return kept
