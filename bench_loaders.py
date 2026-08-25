#!/usr/bin/env python3
"""Benchmark loaders turning external anonymization corpora into {text, spans} eval rows.

Supports:
  TAB  — NorskRegnesentral/text-anonymisation-benchmark (ECHR JSON, entity_mentions)
  MAPA — FrancophonIA/MAPA_Anonymization_package (EU legal .txt + annotated token .tsv)

Both yield our uniform format: {"text", "spans": [{start, end, type}]}.
"""

import glob, json, os, collections


# ---------- TAB (ECHR anonymisation benchmark) ----------

def load_tab(path):
    """Load TAB echr_*.json -> list of {text, spans}.

    Only spans whose identifier_type marks them as masking candidates count:
    TAB distinguishes DIRECT / QUASI identifiers (should be masked) from NO_MASK
    (generic terms an earlier annotator stage already generalised). We exclude NO_MASK.
    """
    data = json.load(open(path))
    out = []
    for doc in data:
        text = doc["text"]
        spans = []
        annots = doc.get("annotations", {})
        for _annotator, layer in annots.items():
            for em in layer.get("entity_mentions", []):
                itype = em.get("identifier_type", "DIRECT")
                if itype == "NO_MASK":
                    continue
                s, e = em["start_offset"], em["end_offset"]
                if 0 <= s < e <= len(text):
                    spans.append({"start": s, "end": e, "type": em.get("entity_type", "?")})
        if text:
            out.append({"text": text, "spans": spans, "doc_id": doc.get("doc_id")})
    return out


# ---------- MAPA (EU legal multilingual anonymization package) ----------

def load_mapa(base_dir, langs=None):
    """Pair each source .txt with its annotated .tsv -> {text, spans}.

    TSV rows: `1-3<TAB>9-17<TAB>TRIBUNAL<TAB>col4<TAB>col5`. Attribute-bearing
    columns mark entities; identical labels merge into one span when consecutive.
    The package provides annotation in the fifth column in this repo layout.
    """
    src_dir = os.path.join(base_dir, "data/ANNOTATED_DATA/EUR_LEX/source")
    ann_dir = os.path.join(base_dir, "data/ANNOTATED_DATA/EUR_LEX/annotated/full_dataset")
    out = []
    for txt_path in sorted(glob.glob(os.path.join(src_dir, "*.txt"))):
        name = os.path.basename(txt_path)
        if langs and not any(name.endswith(f"_{lg}.txt") for lg in langs):
            continue
        tsv_path = os.path.join(ann_dir, name.replace(".txt", ".tsv"))
        if not os.path.exists(tsv_path):
            continue
        text = open(txt_path).read()
        spans = _mapa_tsv_to_spans(open(tsv_path).read(), text)
        out.append({"text": text, "spans": spans, "doc_id": name})
    return out


def _mapa_tsv_to_spans(tsv, text):
    """Parse MAPA token-level TSV. Non-'_' entity columns mark PII tokens;
    consecutive tokens with same label merge into a char span."""
    spans = []
    cur = None  # (label, end)
    for line in tsv.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        try:
            offsets = parts[1].split("-")
            s, e = int(offsets[0]), int(offsets[1])
        except Exception:
            continue
        label = "_"
        for c in parts[3:]:
            if c and c != "_":
                label = c
                break
        if label == "_":
            if cur:
                spans.append({"start": cur[1], "end": cur[2], "type": cur[0]})
                cur = None
            continue
        if cur and cur[0] == label and s <= cur[2] + 1:
            cur = (label, cur[1], e)  # extends
        else:
            if cur:
                spans.append({"start": cur[1], "end": cur[2], "type": label})
            if not cur or cur[0] != label:
                cur = (label, s, e)
            else:
                cur = (label, s, e)
    if cur:
        spans.append({"start": cur[1], "end": cur[2], "type": cur[0]})
    # bounds check
    return [sp for sp in spans if 0 <= sp["start"] < sp["end"] <= len(text)]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tab", default="tab/echr_test.json")
    ap.add_argument("--mapa", default="mapa", help="MAPA package dir (contains data/ANNOTATED_DATA/...)")
    ap.add_argument("--outdir", default="converted")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    tab = load_tab(args.tab)
    json.dump(tab, open(os.path.join(args.outdir, "tab.jsonl").replace("tab.jsonl", "tab.json"), "w"))
    mapa = load_mapa(args.mapa)
    json.dump(mapa, open(os.path.join(args.outdir, "mapa.json"), "w"))
    print("tab docs:", len(tab), "spans:", sum(len(d["spans"]) for d in tab))
    print("mapa docs:", len(mapa), "spans:", sum(len(d["spans"]) for d in mapa))
    lg = collections.Counter(d["doc_id"].rsplit("_", 1)[1].split(".")[0] for d in mapa)
    print("mapa langs:", dict(lg))
