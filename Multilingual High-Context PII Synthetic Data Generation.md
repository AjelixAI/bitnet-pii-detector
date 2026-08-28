# Framework: Multilingual High-Context PII Synthetic Data Generation

## 1. Objective
Generate a massive, highly diverse, realistic, and "messy" synthetic dataset of up to 8,000 tokens per document. The dataset must span European languages (e.g., German, French, Polish, Spanish, Italian) and blend complex medical, financial, and legal domains. All synthetic PII entities must be explicitly wrapped in tracking XML tags for programmatic extraction.

## 2. Global Execution Rules
*   **Target Length:** Emulate an extensive, long-form dossier reaching up to **8,000 tokens** per execution.
*   **Structural Integrity:** Maintain strict linguistic grammar, regional colloquialisms, formatting, and structural cohesion across the entire length.
*   **Context Quality:** Avoid generic fillers. Content must read like genuine, highly detailed enterprise, medical, and financial records.
*   **Formatting:** Deliver raw text with specific inline XML tags wrapping every single target entity. Do not output JSON or explanation blocks outside the requested text.

---

## 3. Input Seed Generation Matrix
For each execution, the orchestration pipeline will pass a unique **Seed Configuration Matrix**. You must adapt your vocabulary, tone, and entity injection to match this matrix exactly.

### Target Domains & Sub-Contexts
*   **Medical:** Clinical intake forms, longitudinal physician notes, ICD-10 diagnostic entries, prescription histories, specialized treatment logs, clinical trial consent forms.
*   **Financial/Legal:** Multi-currency invoice breakdowns, EU bank transfers (IBAN, BIC/SWIFT), bankruptcy declarations, insurance billing disputes, tax audit threats (*Steuernummer* / *INSEE* / *Codice Fiscale*), data privacy non-compliance notices.

### Persona, Tone, and "Messiness" Modes
*   **The Shorthand Clinician:** Fragmented sentences, heavy abbreviations, missing punctuation, dense jargon.
*   **The Frustrated Customer:** Highly conversational, erratic capitalization, regional slang, run-on sentences, angry threats.
*   **The Multi-Lingual Expat:** Dominant language mixed with loan words or administrative terms from another EU state (e.g., a Polish speaker navigating a German hospital).

---

## 4. Required PII Tagging Taxonomy
Every piece of synthetic sensitive information must be wrapped in its corresponding XML tag. Do not invent tags outside this schema:

```xml
<NAME>First and last name variations relative to the target locale</NAME>
<DOB>Date of birth using regional standard formatting (e.g., DD.MM.YYYY)</DOB>
<PHONE>International and local phone configurations</PHONE>
<EMAIL>Realistic generated emails matching personas</EMAIL>
<ADDRESS>Complete localized address blocks (Street, City, Postal Code, Country)</ADDRESS>
<GOV_ID>Country-specific tax/national identifiers (e.g., INSEE, Steuernummer, PESEL)</GOV_ID>
<IBAN>Validly structured International Bank Account Numbers</IBAN>
<SWIFT>BIC/SWIFT codes matching the designated financial institutions</SWIFT>
<MEDICAL_ID>Patient insurance policy numbers, health card IDs</MEDICAL_ID>
<DIAGNOSIS>Explicit medical conditions, diseases, or mental health conditions</DIAGNOSIS>
<FINANCIAL_VALUE>Specific debt totals, invoice lines, account balances, salary details</FINANCIAL_VALUE>
```

---

## 5. Sequential Section-by-Section Prompting Flow
To prevent context drift, hallucination, or premature truncation across the 8k context window, execute the document generation in four sequential, cumulative phases.

### Phase 1: Structural Blueprint Allocation
**Prompt:**
> You are an expert data generation agent specializing in regulatory compliance. Review the following Seed Matrix:
> * **Language:** [Insert Language]
> * **Domain Fusion:** [Insert e.g., Medical Negligence + Bankruptcy Appeal]
> * **Tone/Messiness:** [Insert Tone Profile]
> 
> Outline a comprehensive 8,000-token document breakdown consisting of 4 distinct sections. Ensure the narrative flows logically from an administrative ingestion form to deep clinical logs, financial ledgers, and final legal correspondence. Provide ONLY the outline structure. Do not generate the text yet.

### Phase 2: Administrative and Ingestion Text Generation (Tokens 0 - 2,000)
**Prompt:**
> Generate Section 1 based on the approved outline. This section must represent highly structured administrative documents, clinical intake notes, and verification sheets. 
> * Inject high-density tracking entities (`<NAME>`, `<DOB>`, `<ADDRESS>`, `<MEDICAL_ID>`, `<GOV_ID>`).
> * Follow the language rules and tone profile provided in the seed.
> * Output ONLY the text of Section 1 with strict XML tracking tags applied. Do not truncate.

### Phase 3: Unstructured Context and Deep Clinical Log (Tokens 2,000 - 5,500)
**Prompt:**
> Read the existing context from Section 1. Now, generate Section 2 (Deep Clinical Narrative and Diagnostic Evaluation). 
> * Expand heavily on complex paragraphs, multi-paragraph case notes, treatment workflows, and historical medical logs.
> * Maintain a dense, messy, multi-layered text environment with heavy variations of `<DIAGNOSIS>`.
> * Integrate conversational feedback or doctor shorthand reflecting the chosen persona.
> * Append your response directly to the previous output.

### Phase 4: Financial Ledger, Disputes, and Legal Escalation (Tokens 5,500 - 8,000)
**Prompt:**
> Read the cumulative text from Section 1 and Section 2. Now, complete the document by generating Section 3 (Detailed Financial Breakdown) and Section 4 (Legal Disputes/Escalation Emails).
> * Weave in heavy financial datasets, detailed invoice itemization, asset sheets, and explicit bank wiring data (`<IBAN>`, `<SWIFT>`, `<FINANCIAL_VALUE>`).
> * Transition into the "Messy/Frustrated" or formal legal threat tone. Inject typos, intense spacing irregularities, and grammar errors natural to an individual typing under stress or using a localized regional keyboard.
> * Finalize the output by emitting the absolute end-of-document marker. Do not truncate or summarize.

---

## 6. Output Quality Assertions (Guardrails)
*   **No Clean LLM Bias:** Do not write flawless, textbook paragraphs unless explicitly instructed by a formal tone setting. Human inputs are flawed, erratic, and full of text anomalies.
*   **Strict XML Isolation:** Ensure every open XML tag has a matching closing tag. Never let a tag break across a token boundary (e.g., `<NA ME>` is invalid).
*   **Context Continuity:** If a patient is named `<NAME>Marc Dubois</NAME>` in Phase 2, they must retain the exact same name or a logical, intended variation (e.g., *Monsieur Dubois*) throughout the rest of the 8,000-token generation.
