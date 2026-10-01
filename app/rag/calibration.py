"""Questions used to calibrate and regression-test retrieval.

``IN_SCOPE`` questions are answered by the knowledge base (with the document that
answers them); ``OUT_OF_SCOPE`` questions are not. A good similarity threshold
keeps every in-scope question above it and every out-of-scope one below it -
``scripts/calibrate_rag.py`` measures exactly that for a given embedder.
"""

IN_SCOPE: tuple[tuple[str, str], ...] = (
    ("What are the fees for international transfers?", "international_transfers.md"),
    ("How much does it cost to withdraw cash from another bank's ATM?", "fees.md"),
    ("How do I block a lost card?", "cards.md"),
    ("What is the daily ATM withdrawal limit?", "payment_limits.md"),
    ("What is the interest rate on the auto loan?", "loans.md"),
    ("What down payment do I need for a mortgage?", "mortgages.md"),
    ("Can I take my money out of a term deposit early?", "deposits_savings.md"),
    ("Will the bank ever ask for my PIN?", "security.md"),
    ("How long does an international transfer take?", "international_transfers.md"),
    ("How do I close my account?", "account_management.md"),
    ("How much is a replacement card?", "cards.md"),
    ("What happens when my account is frozen?", "account_management.md"),
    ("How are exchange rates set?", "exchange_rates.md"),
    ("Is there a fee to repay my consumer loan early?", "loans.md"),
    ("What does a transfer to another bank in Georgia cost?", "transfers.md"),
    ("What interest does the savings account pay?", "deposits_savings.md"),
)

OUT_OF_SCOPE: tuple[str, ...] = (
    "What is the bank's policy on the fictional service XYZ?",
    "Do you offer cryptocurrency trading?",
    "What's the weather in Tbilisi tomorrow?",
    "Can I buy stocks through the app?",
    "Do you offer pet insurance?",
    "Who won the football match yesterday?",
)
