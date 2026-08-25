#!/usr/bin/env python3
"""unified_taxonomy.py — GDPR-aware unified PII taxonomy.

Covers GDPR Article 4 (personal data definition) + Article 9 (special categories).
Maps across ai4privacy (19 types), TAB (6 types), MAPA, and LiquidAI's 40-type schema.

16 unified types — coarse enough to generalize across benchmarks, granular enough
for GDPR compliance reporting (Art 9 special categories explicitly separated).

=== UNIFIED GDPR-AWARE PII TAXONOMY ===

IDENTIFIERS (Art 4(1) — directly identifiable)
  PERSON            names, titles, given/surname
  ID_NUMBER         passport, national ID, driver license, SSN, tax, case codes
  DIGITAL_ID        IP address, MAC, username, URL, device ID, user ID

CONTACT (Art 4(1) — contact information)
  EMAIL             email addresses
  PHONE             telephone/fax/mobile
  ADDRESS           street, building, city, zip, postal, GPS

FINANCIAL (Art 4(1) — financial identifiers)
  CREDIT_CARD       credit/debit card numbers
  BANK_ACCOUNT      IBAN, bank account, SWIFT/BIC, crypto wallet

DEMOGRAPHIC (Art 4(1) + partial Art 9)
  GENDER            gender, sex, biological sex
  AGE               age, date of birth (Art 9-adjacent: DoB is sensitive when linked)
  DATE              general dates (not DoB)

ORGANIZATIONAL
  ORGANIZATION      companies, institutions, trade unions (Art 9 if trade union)
  LOCATION          geographic places not covered by ADDRESS

ARTICLE 9 SPECIAL CATEGORIES (explicitly protected by GDPR)
  HEALTH            medical conditions, medications, medical record numbers, health status
  SPECIAL_CATEGORY religion, political opinions, sexual orientation, race/ethnicity
  CREDENTIAL        API keys, passwords, private keys, JWTs (not Art 9 but sensitive)
"""

UNIFIED_TYPES = [
    # Identifiers
    "PERSON",
    "ID_NUMBER",
    "DIGITAL_ID",
    # Contact
    "EMAIL",
    "PHONE",
    "ADDRESS",
    # Financial
    "CREDIT_CARD",
    "BANK_ACCOUNT",
    # Demographic
    "GENDER",
    "AGE",
    "DATE",
    # Organizational
    "ORGANIZATION",
    "LOCATION",
    # Article 9 Special Categories
    "HEALTH",
    "SPECIAL_CATEGORY",
    "CREDENTIAL",
]

# ── ai4privacy 19 types → unified ──
AI4P_MAP = {
    "GIVENNAME": "PERSON",
    "SURNAME": "PERSON",
    "TITLE": "PERSON",
    "DATE": "DATE",
    "AGE": "AGE",
    "EMAIL": "EMAIL",
    "TELEPHONENUM": "PHONE",
    "STREET": "ADDRESS",
    "BUILDINGNUM": "ADDRESS",
    "CITY": "ADDRESS",
    "ZIPCODE": "ADDRESS",
    "IDCARDNUM": "ID_NUMBER",
    "PASSPORTNUM": "ID_NUMBER",
    "DRIVERLICENSENUM": "ID_NUMBER",
    "SOCIALNUM": "ID_NUMBER",
    "TAXNUM": "ID_NUMBER",
    "CREDITCARDNUMBER": "CREDIT_CARD",
    "GENDER": "GENDER",
    "SEX": "GENDER",
}

# ── TAB 6 types → unified ──
TAB_MAP = {
    "PERSON": "PERSON",
    "DATETIME": "DATE",
    "CODE": "ID_NUMBER",
    "ORG": "ORGANIZATION",
    "LOCATION": "LOCATION",
    "MISC": "LOCATION",
}

# ── LiquidAI 40 types → unified (for eval comparison) ──
LIQUID_MAP = {
    "identity.person_name": "PERSON",
    "identity.ssn": "ID_NUMBER",
    "identity.national_id": "ID_NUMBER",
    "identity.passport": "ID_NUMBER",
    "identity.drivers_license": "ID_NUMBER",
    "identity.date_of_birth": "AGE",
    "identity.tax_id": "ID_NUMBER",
    "contact.email": "EMAIL",
    "contact.phone": "PHONE",
    "contact.address": "ADDRESS",
    "contact.postal_code": "ADDRESS",
    "contact.ip_address": "DIGITAL_ID",
    "financial.credit_card": "CREDIT_CARD",
    "financial.iban": "BANK_ACCOUNT",
    "financial.bank_account": "BANK_ACCOUNT",
    "financial.swift_bic": "BANK_ACCOUNT",
    "financial.crypto_wallet": "BANK_ACCOUNT",
    "financial.amount": "DATE",  # not ideal but amounts aren't PII per se
    "credential.api_key": "CREDENTIAL",
    "credential.password": "CREDENTIAL",
    "credential.private_key": "CREDENTIAL",
    "credential.jwt": "CREDENTIAL",
    "credential.connection_string": "CREDENTIAL",
    "developer.login_credentials": "CREDENTIAL",
    "online.username": "DIGITAL_ID",
    "online.url": "DIGITAL_ID",
    "device.mac_address": "DIGITAL_ID",
    "device.imei": "DIGITAL_ID",
    "developer.device_id": "DIGITAL_ID",
    "location.gps_coordinates": "ADDRESS",
    "healthcare.medical_record": "HEALTH",
    "healthcare.condition": "HEALTH",
    "healthcare.medication": "HEALTH",
    "healthcare.health_plan_id": "HEALTH",
    "org.company_name": "ORGANIZATION",
    "special.religion": "SPECIAL_CATEGORY",
    "special.political": "SPECIAL_CATEGORY",
    "special.orientation": "SPECIAL_CATEGORY",
    "special.health_status": "HEALTH",
    "legal.case_number": "ID_NUMBER",
}

# ── MAPA types → unified ──
MAPA_MAP = {
    "PERSON": "PERSON",
    "DATE": "DATE",
    "ORG": "ORGANIZATION",
    "ORGANIZATION": "ORGANIZATION",
    "LOCATION": "LOCATION",
    "ADDRESS": "ADDRESS",
    "CODE": "ID_NUMBER",
    "PHONE": "PHONE",
    "EMAIL": "EMAIL",
}


def remap_spans(spans, source_map):
    """Remap span types from source taxonomy to unified. Drops unmapped types."""
    out = []
    for s in spans:
        # try exact match, then uppercase
        utype = source_map.get(s["type"]) or source_map.get(s["type"].upper())
        if utype:
            out.append({"start": s["start"], "end": s["end"], "type": utype})
    return out


def merge_adjacent(spans, text, gap=2):
    """Merge adjacent/overlapping spans of the same unified type.
    PERSON 'Mr' + PERSON 'Galip' + PERSON 'Yalman' → PERSON 'Mr Galip Yalman'.
    gap=2 allows merging across single spaces/punctuation between same-type spans."""
    if not spans:
        return []
    spans = sorted(spans, key=lambda s: s["start"])
    merged = [dict(spans[0])]
    for s in spans[1:]:
        last = merged[-1]
        if s["type"] == last["type"] and s["start"] <= last["end"] + gap:
            last["end"] = max(last["end"], s["end"])
        else:
            merged.append(dict(s))
    return merged


def normalize_for_eval(spans, source_map, text):
    """Full pipeline: remap types → merge adjacent same-type spans."""
    remapped = remap_spans(spans, source_map)
    return merge_adjacent(remapped, text)


# ── GDPR compliance metadata ──
GDPR_ARTICLE_9_TYPES = {"HEALTH", "SPECIAL_CATEGORY", "CREDENTIAL"}
GDPR_DIRECT_IDENTIFIERS = {"PERSON", "ID_NUMBER", "EMAIL", "PHONE"}
GDPR_CONTACT = {"EMAIL", "PHONE", "ADDRESS"}
GDPR_FINANCIAL = {"CREDIT_CARD", "BANK_ACCOUNT"}

def get_gdpr_category(unified_type):
    """Returns GDPR category for a unified type (for compliance reporting)."""
    if unified_type in GDPR_ARTICLE_9_TYPES:
        return "Article 9 (special category)"
    if unified_type in GDPR_DIRECT_IDENTIFIERS:
        return "Article 4(1) (direct identifier)"
    if unified_type in GDPR_CONTACT:
        return "Article 4(1) (contact data)"
    if unified_type in GDPR_FINANCIAL:
        return "Article 4(1) (financial data)"
    return "Article 4(1) (personal data)"


if __name__ == "__main__":
    print(f"Unified GDPR-aware taxonomy: {len(UNIFIED_TYPES)} types")
    print("\n=== TYPE → GDPR CATEGORY ===")
    for t in UNIFIED_TYPES:
        cat = get_gdpr_category(t)
        marker = " ⚠️" if "Article 9" in cat else ""
        print(f"  {t:20s} → {cat}{marker}")
    print(f"\nai4privacy mappable: {len(AI4P_MAP)}/19 types")
    print(f"TAB mappable: {len(TAB_MAP)}/6 types")
    print(f"LiquidAI mappable: {len(LIQUID_MAP)}/40 types")
    print(f"MAPA mappable: {len(MAPA_MAP)}/9 types")
    print(f"\nArticle 9 special categories: {GDPR_ARTICLE_9_TYPES}")
