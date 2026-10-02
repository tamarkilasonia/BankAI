"""The safety layer: checks that run around the agent, independent of the LLM.

    customer message ──► input guard ──► agent ──► output guard ──► customer
                         (policy.py)               (output_guard.py)

* ``policy``        - WHAT is not allowed: categories, detection rules, refusal texts.
* ``normalization`` - undoes disguises (homoglyphs, zero-width characters, leetspeak,
                      base64/hex/ROT13...) so the rules see what the text really says.
* ``input_guard``   - applies the policy to customer messages and to untrusted content
                      (retrieved documents, data from tools: indirect prompt injection).
* ``output_guard``  - checks every answer for leaked prompts, secrets, other customers'
                      identifiers and claims that money was moved.
* ``actions``       - high-risk actions: what was asked for, and the human-in-the-loop
                      confirmation that only ever records a MOCK request.
* ``injection_suite`` - the automated prompt-injection test set and its PASS/FAIL report.

None of this is the security boundary - authorization in the tool layer is (see
``app/services/authorization.py``). The guards reduce risk, give consistent
refusals, and make attacks visible in logs and metrics.
"""
