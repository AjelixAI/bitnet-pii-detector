#!/usr/bin/env python3
"""Smoke test: run the saved best model on curated real-style multilingual texts,
print detected PII spans with types. Visual sanity proof.
"""
import collections
import sys

from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "/root/piiranha_hf/best"
PII_TYPES = ["ACCOUNTNUM", "BUILDINGNUM", "CITY", "CREDITCARDNUMBER", "DATEOFBIRTH",
             "DRIVERLICENSENUM", "EMAIL", "GIVENNAME", "IDCARDNUM", "PASSWORD",
             "SOCIALNUM", "STREET", "SURNAME", "TAXNUM", "TELEPHONENUM",
             "USERNAME", "ZIPCODE"]
LABEL_NAMES = [f"I-{t}" for t in PII_TYPES] + ["O"]
O_IDX = LABEL_NAMES.index("O")

TEXTS = [
    ("EN", "John Smith can be reached at john.smith@acme.com or +1 202 555 0199. "
           "His credit card 4111 1111 1111 1111 expires 12/27. "
           "Home: 742 Evergreen Terrace, Springfield, IL 62701."),
    ("DE", "Die Kundin Anna Müller, geb. am 14.03.1990, wohnt in der Goethestraße 12, "
           "80331 München. Kontakt: +49 89 1234567, anna.mueller@web.de. "
           "Kontonummer DE89370400440532013000."),
    ("FR", "M. Jean Dupont, né le 5 juin 1985, réside au 14 rue de la Paix, 75002 Paris. "
           "Tél : +33 1 42 68 53 00, jean.dupont@orange.fr. Carte : 4970 1000 1234 5678."),
    ("IT", "Il signor Mario Rossi, nato il 22/01/1978, vive in Via Roma 10, 20121 Milano. "
           "Telefono: +39 02 1234567, mario.rossi@libero.it. Codice fiscale RSSMRA78A01H501U."),
    ("ES", "La señora María García, nacida el 10/09/1992, reside en Calle Mayor 5, 28013 Madrid. "
           "Teléfono: +34 91 123 4567, maria.garcia@gmail.com. DNI 12345678Z."),
    ("NL", "Mevrouw Emma de Vries, geboren op 12-04-1988, woont op Keizersgracht 123, "
           "1015 CJ Amsterdam. Telefoon: +31 20 123 4567, emma.devries@ziggo.nl. "
           "IBAN NL91 ABNA 0417 1643 00."),
    ("EN-negative", "The server was restarted during scheduled maintenance. "
                    "Build 4.2.1 passed all 128 tests. The deployment completed successfully."),
    ("EN-tricky", "Meeting Monday at 10:00 with support. Order #A-1042 shipped. "
                  "Please contact the helpdesk."),
]


def decode_spans(logits_row, offsets, text):
    spans, cur = [], None
    for ti, (a, b) in enumerate(offsets):
        if a == b or b == 0:
            if cur:
                spans.append(cur)
            cur = None
            continue
        tag = LABEL_NAMES[logits_row[ti]] if logits_row[ti] < len(LABEL_NAMES) else "O"
        if tag == "O":
            if cur:
                spans.append(cur)
            cur = None
        else:
            t = tag[2:]
            if cur and cur[2] == t:
                cur = (cur[0], b, t)
            else:
                if cur:
                    spans.append(cur)
                cur = (a, b, t)
    if cur:
        spans.append(cur)
    return spans


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR).cuda().eval()
    print(f"model: {MODEL_DIR} | labels: {len(LABEL_NAMES)}", flush=True)
    for lg, text in TEXTS:
        enc = tok(text, truncation=True, max_length=256, return_offsets_mapping=True)
        ids = torch.tensor([enc["input_ids"]]).cuda()
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0].cpu().argmax(-1).tolist()
        spans = decode_spans(logits, enc["offset_mapping"], text)
        print(f"\n[{lg}] {text}")
        if spans:
            for a, b, t in spans:
                print(f"   {'!' if t else ''}{t:<18s} '{text[a:b]}'")
        else:
            print("   (no PII detected)")
    print("\nSMOKE TEST DONE")


if __name__ == "__main__":
    main()
