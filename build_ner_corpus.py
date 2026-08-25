#!/usr/bin/env python3
# build_ner_corpus.py — convert Wikineural (BIO token labels) to a char-span corpus.
#
# Wikineural gives tokenized text + integer BIO ner_tags. We join the tokens and
# re-tokenize with the SHARED pretrain tokenizer, mapping BIO entity spans to char
# spans. This trains the span head on GENERAL entities (PER/ORG/LOC/...), teaching
# the extraction MECHANISM before PII fine-tune.
#
# Output: JSONL {lang,text,spans:[{type,start,end,text}]}, types as coarse labels.
import argparse, json, os, random
from tokenizers import Tokenizer
from datasets import load_dataset

# Wikineural follows CoNLL-2003 BIO: 4 types, each B=(odd) I=(even).
#   1=B-PER, 2=I-PER, 3=B-ORG, 4=I-ORG, 5=B-LOC, 6=I-LOC, 7=B-MISC, 8=I-MISC
WIKI_BTYPE = {1: "person", 3: "organization", 5: "location", 7: "misc"}
# an I-tag (2,4,6,8) continues the entity whose B-tag is (tag-1)
WIKI_ITYPE_PAIR = {2: "person", 4: "organization", 6: "location", 8: "misc"}


def bio_to_char_spans(text, tokens, ner_tags, tok, max_seq=512):
    """Join tokens, re-tokenize with SHARED tokenizer, map BIO entity tokens to char spans."""
    joined = " ".join(tokens)
    # build char offsets of each token in joined
    dt_off = []
    pos = 0
    for t in tokens:
        st = joined.find(t, pos)
        if st < 0:
            st = pos
        dt_off.append((st, st + len(t)))
        pos = st + len(t)
    # group BIO into entities: a B-tag starts, following I-tags of the same type continue.
    entities = []  # (type, start_char, end_char)
    cur = None
    for i, tag in enumerate(ner_tags):
        if i >= len(dt_off):
            break
        if tag == 0:
            if cur:
                entities.append(cur); cur = None
            continue
        st, en = dt_off[i]
        typ = WIKI_BTYPE.get(tag) or WIKI_ITYPE_PAIR.get(tag)
        if typ is None:
            if cur:
                entities.append(cur); cur = None
            continue
        if tag in WIKI_ITYPE_PAIR:
            # I-tag: continue if same type is open, else start fresh (malformed)
            if cur and cur[0] == typ:
                cur = (typ, cur[1], en)
            else:
                if cur:
                    entities.append(cur)
                cur = (typ, st, en)
        else:
            # B-tag: start new entity
            if cur:
                entities.append(cur)
            cur = (typ, st, en)
    if cur:
        entities.append(cur)
    # re-tokenize joined with our tokenizer so char offsets align with model input
    enc = tok.encode(joined)
    ids = enc.ids[:max_seq]
    offsets = enc.offsets[:max_seq]
    spans = []
    for (typ, cs, ce) in entities:
        if cs >= ce or ce > len(joined):
            continue
        # clamp to tokenized region
        spans.append({"type": typ, "start": cs, "end": ce, "text": joined[cs:ce]})
    return joined[:max_seq], spans, ids, offsets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/scratch/pii_corpus/ner_corpus.jsonl")
    ap.add_argument("--max-docs", type=int, default=20000)
    ap.add_argument("--tok", default="/root/pii_data/pretok/tokenizer.json")
    args = ap.parse_args()
    tok = Tokenizer.from_file(args.tok)
    ds = load_dataset("Babelscape/wikineural", split="train_en", streaming=True)
    out = open(args.out, "w")
    n = 0; with_span = 0
    for d in ds:
        if n >= args.max_docs:
            break
        tokens = d["tokens"]; ner_tags = d["ner_tags"]
        if not tokens:
            continue
        # join (Wikineural tokens may be space-joined already; safe to re-join)
        text, spans, ids, offsets = bio_to_char_spans(" ".join(tokens), tokens, ner_tags, tok)
        if len(text.strip()) < 20:
            n += 1; continue
        # keep only rows with >=1 non-O entity
        valid = [s for s in spans if s["start"] < s["end"] and s["end"] <= len(text)]
        if valid:
            out.write(json.dumps({"lang": "en", "text": text, "spans": valid}) + "\n")
            with_span += 1
        n += 1
        if n % 5000 == 0:
            print(f"processed {n} docs, {with_span} with spans", flush=True)
    out.close()
    print(f"DONE: {n} docs, {with_span} with spans -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
