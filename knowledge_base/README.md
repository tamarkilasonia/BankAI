# BankAI knowledge base (MOCK)

These documents are the only source the assistant may use for policy, fee and
product questions (retrieval-augmented generation). They describe a **fictional
bank**. All fees, rates, limits, phone numbers and procedures are illustrative
MOCK values - not the terms of any real bank.

Rules for editing:
* One topic per file; each `##` section must make sense on its own (sections are
  the retrieval chunks).
* Product rates and limits must match `PRODUCTS` in `app/database/seed.py` -
  a test fails if they drift apart.
* After editing, re-index: `python -m scripts.ingest_knowledge_base`.
