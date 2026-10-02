"""LLM evaluation: run the agent on a labelled dataset and measure it.

    dataset (JSONL) ─► runner ─► BankingAssistant (frozen MOCK bank) ─► reply
                         │                                               │
                         └─► SQL oracle (expected values) ─► scorers ◄───┘
                                                               │
                                       metrics + report ◄──────┴──► evaluation_runs / _results

* ``schemas``  - what a case looks like and what a result records.
* ``dataset``  - loading and versioning the cases.
* ``oracle``   - the ground truth, computed at run time with hand-written SQL.
* ``scoring``  - one case, one reply -> per-metric scores and failure reasons.
* ``metrics``  - aggregate results into the report numbers (never typed in).
* ``runner``   - orchestration: sandbox bank, agent, multi-turn cases.
* ``store``    - persistence of runs and results.
* ``report``   - text and Markdown reports.
"""
