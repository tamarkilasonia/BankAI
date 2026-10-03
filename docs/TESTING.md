# Testing strategy

BankAI is a prototype that works only with **MOCK** data. Testing is the main point of the
project, not an afterthought: the assistant is checked by ordinary example tests, by an
evaluation dataset that scores the AI's behaviour, and by techniques that check the tests
themselves (property-based testing, mutation testing, random test order).

Numbers in this document are from the phase 10 run on 2026-10-03, updated for phases 11-13 (Docker, CI, documentation).

---

## 1. In one minute

- **1,511 tests** in `tests/` (counting each parametrized case). The 1,452 in the default lane
  need no network, no API key, no database server and no LLM, and run in about a minute.
- Five **opt-in lanes** use real infrastructure: PostgreSQL + pgvector, a local LLM
  (Ollama), the local speech models, headless Chrome, and the Docker Compose stack. A lane
  whose infrastructure is missing is *skipped* locally. In CI it *fails*: CI sets
  `REQUIRE_INTEGRATION_DB=1`, `REQUIRE_BROWSER=1` and `REQUIRE_DOCKER=1`.
- **Coverage is 99.8%** of `app/` and `scripts/` (branch coverage, with PostgreSQL).
  CI fails below 95%. Coverage only shows that code *ran*, so:
- **Mutation testing** breaks 33 defences on purpose (an authorization check, the
  injection decoder, log redaction...) and checks that the tests notice. Result:
  **33/33 caught**.
- **Property-based tests** (Hypothesis) generate thousands of disguised attacks, amounts,
  dates and malformed tool calls. They check rules that must hold for *every* input.
- Tests run in a **random order** on every run (`pytest-randomly`), so no test can depend
  on another one having run first.
- Phase 10's new tests found **12 problems**, all fixed: 3 hallucination holes in the answer
  grounding, 6 gaps in the injection normalizer and action detection, 1 missing
  authorization layer, and 2 weaknesses in the tests themselves (§8).

## 2. Running the tests

| what | command |
|---|---|
| everything that needs no services | `pytest` |
| + PostgreSQL integration tests | a local PostgreSQL with pgvector and a `bankai_test` database: `REQUIRE_INTEGRATION_DB=1 pytest`. Or the compose database: `docker compose up -d --wait database`, once `docker compose exec database createdb -U bankai bankai_test`, then `TEST_DATABASE_URL=postgresql+asyncpg://bankai:bankai@localhost:5433/bankai_test REQUIRE_INTEGRATION_DB=1 pytest` |
| with coverage | `pytest --cov` (report in the terminal; `--cov-report=html` for a browsable one) |
| live LLM tests (local Ollama) | `ollama serve`, `ollama pull qwen2.5:3b`, then `pytest -m llm` |
| speech model tests | `pip install -r requirements-voice.txt`, `python -m scripts.download_voice_models`, `pytest -m voice` |
| browser end-to-end | needs Chrome and Node: `pytest -m ui` |
| the Docker stack (build, start, test from outside, remove) | needs Docker: `pytest -m docker` (about 30 s with a cached image) |
| one layer | `pytest tests/safety`, `pytest tests/property`, ... |
| more property examples | `HYPOTHESIS_PROFILE=thorough pytest tests/property` (2,000 per property instead of 100) |
| mutation testing | `python -m scripts.mutation_test` (about a minute) or `--list` |
| reproduce an order-dependent failure | `pytest --randomly-seed=<seed printed in the header>` (or `--randomly-seed=last`); `pytest -p no:randomly` turns shuffling off |
| JavaScript unit tests only | `node --test tests/frontend/` |
| reports | `python -m scripts.run_evaluation`, `python -m scripts.run_safety_suite`, `python -m scripts.evaluate_voice` |

Markers (`pyproject.toml`, `--strict-markers`): `postgres`, `llm`, `voice`, `ui`, `docker`.

All of this runs on every push in GitHub Actions - see [CI.md](CI.md). The pipeline itself is
tested too (`tests/unit/test_ci_workflow.py`): pinned actions, no silent skips, every marker run.

## 3. The layers

| layer | folder | what is real | what is replaced | tests |
|---|---|---|---|---|
| unit | `tests/unit` | the function under test, SQLite in memory | LLM, network | 659 |
| property | `tests/property` | guard, grounding, calculators, MCP server | - (inputs are generated) | 17 |
| safety | `tests/safety` | whole agent over MCP, seeded bank | LLM (scripted) | 168 |
| integration | `tests/integration` | agent + MCP + services + database; **PostgreSQL**, pgvector, Alembic, MCP over HTTP | LLM (scripted) | 114 |
| API | `tests/api` | FastAPI app, middleware, agent, MCP, database | speech engines (fakes) | 65 |
| evaluation | `tests/evaluation` | dataset, SQL oracle, scorers, runner | LLM (offline mode) | 366 |
| voice | `tests/voice` | audio validation, pipeline; **Whisper and Piper** (marked `voice`) | engines (fakes) in the unmarked tests | 78 |
| frontend | `tests/frontend` | served files, CSP, JS syntax/unit tests; **headless Chrome** (marked `ui`) | - | 24 |
| live LLM | `tests/llm` | **qwen2.5:3b via Ollama** | nothing | 13 |
| containers | `tests/docker` | **the built image and the whole Compose stack** | nothing | 7 |

The many fast tests at the bottom catch most bugs within seconds. The few slow tests at the top
show that the real parts work together.

## 4. Requirement → test traceability

Each requirement from the project specification, and the tests that show it is met. Every
test named in this table must exist: `tests/unit/test_docs.py` checks that on every run.

**Unit tests (spec §16)**

| requirement | tests |
|---|---|
| tool validation | `tests/unit/test_tool_schemas.py::test_unknown_arguments_are_rejected`, `tests/unit/test_tool_schemas.py::test_no_tool_accepts_an_identity_argument`, `tests/unit/test_router_and_prompts.py::test_invalid_values_are_dropped_individually` |
| authorization | `tests/unit/test_services_accounts.py::test_other_user_is_denied`, `tests/unit/test_services_accounts.py::test_foreign_and_missing_accounts_are_indistinguishable`, `tests/unit/test_services_transactions.py::test_foreign_account_is_denied` |
| database functions | `tests/unit/test_database.py::test_foreign_keys_are_enforced`, `tests/unit/test_services_transactions.py::test_spending_matches_oracle`, `tests/unit/test_services_transactions.py::test_like_wildcards_are_literal` |
| RAG retrieval | `tests/unit/test_rag.py::test_in_scope_questions_are_answerable`, `tests/unit/test_rag.py::test_out_of_scope_questions_are_not`, `tests/integration/test_rag_postgres.py::test_retriever_over_pgvector_matches_the_in_memory_behaviour` |
| safety detection | `tests/unit/test_input_guard.py`, `tests/unit/test_output_guard.py`, `tests/property/test_safety_properties.py::test_a_disguised_attack_is_refused_for_the_same_reason` |
| prompt validation | `tests/unit/test_router_and_prompts.py::test_customer_text_is_fenced_as_data`, `tests/unit/test_router_and_prompts.py::test_prompts_contain_no_secrets`, `tests/unit/test_router_and_prompts.py::test_router_prompt_arguments_match_the_tool_models` |

**Integration tests (spec §16)**

| requirement | tests |
|---|---|
| FastAPI + database | `tests/api/test_banking_and_voice.py::test_own_accounts`, `tests/integration/test_api_postgres.py::test_balances_are_exact_decimals_from_numeric_columns`, `tests/integration/test_api_postgres.py::test_an_evaluation_run_is_stored_as_jsonb_and_read_back` |
| LangGraph + MCP | `tests/integration/test_agent.py::test_example_1_balance` (in-process MCP), `tests/integration/test_mcp_http.py::test_whole_agent_turns_over_http` (streamable HTTP) |
| agent + tools | `tests/integration/test_agent.py::test_example_2_restaurant_spending`, `tests/integration/test_agent.py::test_tool_level_authorization_backs_up_the_router`, `tests/integration/test_mcp_tools.py` |
| RAG + LLM | `tests/integration/test_rag_llm.py` (scripted LLM), `tests/llm/test_live_llm.py::test_live_rag_answers_are_grounded_and_cited` (real model) |

**AI behaviour (spec §16)** - measured on 246 labelled cases by the evaluation
(`app/evaluation/datasets/banking_eval.jsonl`), and gated in CI by
`tests/evaluation/test_runner.py::test_full_offline_run_meets_the_safety_gates`.

| requirement | tests |
|---|---|
| correct intent / tool | `tests/unit/test_router_and_prompts.py::test_confident_rule_does_not_call_the_llm`, `tests/integration/test_agent.py::test_llm_routes_what_rules_cannot`, `tests/evaluation/test_scoring.py` |
| correct response | `tests/integration/test_agent.py::test_example_1_balance`, `tests/evaluation/test_oracle.py` |
| hallucination | `tests/integration/test_rag_llm.py::test_an_invented_fee_is_replaced_by_the_document`, `tests/integration/test_agent.py::test_hallucinated_number_falls_back_to_template`, `tests/property/test_grounding_properties.py::test_an_amount_the_facts_do_not_contain_is_always_caught` |
| unsupported questions (spec §8) | `tests/integration/test_agent.py::test_unknown_service_gets_not_enough_information`, `tests/integration/test_rag_llm.py::test_when_the_documents_do_not_cover_it_the_llm_is_never_asked` |
| safety | `tests/safety/test_prompt_injection_suite.py::test_case` (83 cases, one test each) |
| authorization | `tests/integration/test_agent.py::test_example_4_other_customer_is_refused`, `tests/integration/test_agent.py::test_typed_foreign_account_id_still_hits_authorization` |

**Security tests (spec §16, §7, §11-12)**

| requirement | tests |
|---|---|
| prompt injection | `tests/safety/test_prompt_injection_suite.py::test_case`, `tests/property/test_safety_properties.py::test_a_disguised_attack_is_refused_for_the_same_reason`, `tests/safety/test_indirect_injection.py::test_poisoned_passage_is_dropped_before_the_llm_sees_it` |
| data leakage | `tests/safety/test_tool_authorization.py::test_foreign_account_id_is_denied_without_leaking`, `tests/property/test_tool_fuzzing.py::test_arbitrary_arguments_never_leak_or_crash`, `tests/api/test_concurrency.py::test_concurrent_customers_only_ever_see_their_own_data` |
| unauthorized access | `tests/api/test_banking_and_voice.py::test_another_customers_data_is_forbidden_and_unknown_looks_the_same`, `tests/api/test_auth_and_chat.py::test_invalid_session_tokens_are_rejected`, `tests/safety/test_tool_authorization.py::test_scoped_queries_hold_even_if_the_ownership_check_is_skipped` |
| system prompt extraction | `tests/unit/test_output_guard.py::test_system_prompt_text_is_a_leak`, `tests/unit/test_output_guard.py::test_prompt_fingerprints_cover_every_prompt` |
| high-risk actions never execute | `tests/safety/test_high_risk_actions.py::test_confirmed_transfer_is_recorded_as_mock_and_moves_no_money`, `tests/safety/test_tool_authorization.py::test_no_tool_changes_the_database` |
| no secrets in logs | `tests/unit/test_logging.py::test_redacts_secrets_inside_free_text`, `tests/api/test_auth_and_chat.py::test_every_turn_is_audited_with_secrets_redacted` |
| no stack traces to users | `tests/api/test_evaluations_and_errors.py::test_unexpected_errors_reveal_nothing` |
| XSS in the web UI | `tests/frontend/test_frontend.py::test_no_html_injection_sinks`, `tests/frontend/test_frontend.py::test_end_to_end_in_headless_chrome` |

**Containers (spec §20)**

| requirement | tests |
|---|---|
| services backend, database, mcp-server | `tests/unit/test_docker_config.py::test_the_services_of_the_specification_exist`, `tests/docker/test_compose_stack.py::test_all_services_are_healthy_and_the_init_job_succeeded` |
| startable with `docker compose up` | `tests/docker/test_compose_stack.py::test_a_customer_is_answered_through_the_mcp_server`, `tests/docker/test_compose_stack.py::test_running_the_init_job_again_changes_nothing` |
| environment variables documented | `tests/unit/test_docker_config.py::test_every_setting_is_documented_in_env_example`, `tests/unit/test_docker_config.py::test_every_environment_variable_is_a_real_setting` |
| never commit API keys | `tests/unit/test_docker_config.py::test_no_secret_is_written_in_a_compose_file`, `tests/unit/test_docker_config.py::test_the_build_context_is_an_allowlist`, `tests/unit/test_generate_secrets.py` |
| container security | `tests/docker/test_compose_stack.py::test_containers_run_unprivileged_on_a_read_only_file_system`, `tests/docker/test_compose_stack.py::test_the_mcp_server_and_the_database_are_not_open_to_the_network` |

**Voice tests (spec §16)**

| requirement | tests |
|---|---|
| STT success | `tests/voice/test_pipeline.py::test_a_spoken_question_is_answered_in_text_and_speech`, `tests/voice/test_local_models.py::test_english_round_trip` (real Whisper) |
| invalid audio | `tests/voice/test_audio_and_speech.py::test_bad_audio_is_rejected_with_a_clear_message`, `tests/voice/test_local_models.py::test_a_damaged_recording_is_invalid_audio` |
| unsupported audio | `tests/voice/test_audio_and_speech.py::test_formats_are_recognized_by_content`, `tests/api/test_banking_and_voice.py::test_bad_audio` |
| TTS errors | `tests/voice/test_providers.py::test_openai_tts_errors`, `tests/voice/test_pipeline.py::test_tts_failure_keeps_the_text_answer` |

**CI and documentation (spec §21, §23-24, §30)**

| requirement | tests |
|---|---|
| pipeline order: install → lint → unit → integration → coverage → image | `tests/unit/test_ci_workflow.py::test_the_pipeline_follows_the_specification` |
| no real API keys in CI; LLM tests kept separate | `tests/unit/test_ci_workflow.py::test_workflows_only_read_the_repository_and_use_no_secrets`, `tests/unit/test_ci_workflow.py::test_every_test_marker_runs_in_some_workflow` |
| README with the 23 required sections | `tests/unit/test_readme.py::test_the_readme_has_the_sections_the_specification_requires`, `tests/unit/test_readme.py::test_every_relative_link_and_anchor_resolves` |
| Mermaid architecture diagram | `tests/unit/test_readme.py::test_the_architecture_diagram_shows_every_path_the_specification_names` |
| published metrics are not hard-coded | `tests/unit/test_readme.py::test_the_readme_evaluation_numbers_are_the_stored_reports` |
| INTERVIEW.md with at least 40 questions | `tests/unit/test_readme.py::test_the_interview_guide_has_at_least_forty_numbered_questions` |

## 5. Test doubles - and what each one could hide

Every double is a risk: it can hide a bug that only the real thing has. So each one is paired with
a test of the real thing.

| double | replaces | could hide | the real thing is tested by |
|---|---|---|---|
| `ScriptedLLM` (`tests/fakes.py`) | the LLM | how a real model actually phrases and fails | `tests/llm` (qwen2.5:3b), the evaluation run with `LLM_PROVIDER=openai` |
| in-memory SQLite | PostgreSQL | NUMERIC precision, timestamptz, JSONB, ILIKE, concurrency | `tests/integration/*postgres*`, `test_migrations.py` |
| in-process MCP transport | MCP over HTTP | serialization, headers, the network | `tests/integration/test_mcp_http.py` |
| `HashingEmbedder` | a neural embedding model | retrieval quality | `tests/integration/test_rag_postgres.py`, `scripts/calibrate_rag.py` (live) |
| `FakeSTT` / `FakeTTS` | Whisper / Piper | recognition quality, audio decoding | `tests/voice/test_local_models.py` (marked `voice`), `scripts/evaluate_voice.py` |
| pinned clock (`ANCHOR`) | the real time | date edge cases | `tests/property/test_finance_properties.py::test_every_named_period_is_a_sane_range_ending_today` (any date 2000-2099) |

One double hid a problem in the *tests*: the shared in-memory SQLite uses a single
connection, so parallel requests failed there ("cannot commit transaction"). That is not a bug
in the app, but it made a concurrency test flaky. Concurrency tests therefore use a database
file, and are repeated on PostgreSQL (§8, finding 12).

## 6. Oracles - how a test knows the right answer

A test that computes the expected value with the code it tests proves nothing. So:

- **`tests/oracle.py`** recomputes balances, transactions and spending in plain Python from
  the seed data. It doesn't touch the services or SQL it checks.
- **`app/evaluation/oracle.py`** answers each evaluation case with its own SQL against a frozen
  copy of the bank. Expected numbers are not copied into the dataset by hand, so they can't drift
  from the data.
- **Policy facts** in the evaluation dataset are checked against the knowledge-base text
  (`tests/evaluation/test_dataset.py::test_document_facts_really_are_in_the_cited_document`).
- The **look-alike letter table** in the property tests is written independently of the one in
  `normalization.py`: the attacker's view, not a copy of the implementation.

## 7. Testing the tests

### Property-based tests (`tests/property`, Hypothesis)

An example test checks one input. A property test states a rule, and Hypothesis tries hundreds of
generated inputs to break it. When one does, Hypothesis *shrinks* it to the smallest failing input
and saves it in `.hypothesis/`, so the next run replays it first.

| property | file |
|---|---|
| a refused attack stays refused under any disguise the normalizer claims to undo (case, spacing, zero-width, look-alike and full-width letters, leetspeak, a spaced-out word, base64, hex, URL encoding, ROT13, reversal) and combinations of them | `test_safety_properties.py` |
| a benign question in odd typography is never refused | `test_safety_properties.py` |
| normalizing twice changes nothing (idempotence) | `test_safety_properties.py` |
| the guard never crashes, whatever the input | `test_safety_properties.py` |
| an amount written differently is still grounded; an invented amount is always caught; a small number is a count, never a free amount | `test_grounding_properties.py` |
| loans are always repaid in full; a higher rate never lowers the payment; a longer term never raises it | `test_finance_properties.py` |
| currency conversion there and back loses at most the rounding (derived bound) | `test_finance_properties.py` |
| every named period is a sane range for any date 2000-2099; custom ranges are accepted exactly when valid | `test_finance_properties.py` |
| any tool call - well-formed from the JSON schema or junk - never returns another customer's data and never crashes the server | `test_tool_fuzzing.py` |

The default profile runs 100 examples per property (fast enough for every run).
`HYPOTHESIS_PROFILE=thorough` runs 2,000. Each bypass found was also pinned as a plain regression
test (`tests/unit/test_input_guard.py::test_disguises_found_by_hypothesis_are_refused`), so it is
checked every time, not only when the random search happens to find it again.

### Mutation testing (`scripts/mutation_test.py`)

High coverage can be misleading: a line can run during a test without any assertion depending on
it. The script applies one deliberate bug at a time to a temporary copy of the project, runs the
tests that should protect that defence, and expects them to **fail**.

```
$ python -m scripts.mutation_test
  KILLED    authz-customer-check        GET /accounts/{user_id} serves any customer's data
  KILLED    authz-account-owner         tools accept another customer's account_id
  KILLED    authz-query-scoping         transaction queries are no longer limited to the customer's own accounts
  KILLED    guard-plain-view-only       encoded attacks (base64, leetspeak, ROT13...) are no longer decoded
  KILLED    confirm-any-customer        another customer can confirm a pending transfer
  KILLED    grounding-claim-subject     "free" from one passage backs "international transfers are free"
  KILLED    logs-unredacted             API keys and passwords typed by a customer are written to logs
  KILLED    ui-html-sink                agent answers and attack strings are rendered as HTML (XSS)
  KILLED    eval-isolation-scorer       the evaluation reports 100% data isolation whatever leaked
  ... (33 in total, incl. container and CI mistakes: a root user, a published MCP port, an unpinned action)
Mutation score: 33/33 killed
```

The mutants are curated, not generated: a generic tool (mutmut) makes thousands of mutants and
needs hours; these are the bugs that matter in a bank, and each names the tests that guard it.
`tests/unit/test_mutation_script.py` fails if a refactoring removes the code a mutant targets, so
the list cannot silently go stale. Mutation testing also tested the *tests*. It found that the
fuzz test, sending only random junk, never got past argument validation, so it could not notice a
deleted ownership check (finding 10).

### Random order, concurrency, a real browser

- **`pytest-randomly`** shuffles the test order on every run and prints the seed. The suite passes
  in any order, so no test relies on state another one left behind.
- **Concurrency**: 36 interleaved requests from three customers through one app instance. No
  answer contains another customer's balance or account number, and every turn gets an audit row
  naming the right customer (`tests/api/test_concurrency.py`, and on PostgreSQL). A second test
  races two customers answering "yes" in the same conversation.
- **Headless Chrome** drives the real UI: sign-in, chat, transfer confirmation, an HTML-injection
  attempt rendered as text, and the dashboard. It fails on any JavaScript error or CSP violation.

## 8. What phase 10 found

| # | found by | defect | fix | regression test |
|---|---|---|---|---|
| 1 | RAG + LLM test | "International transfers are free." passed grounding: *another* passage said own-account transfers are free | a claim must be made by one fact sentence about the same subject (`unbacked_claims`) | `tests/integration/test_rag_llm.py::test_a_claim_borrowed_from_another_passage_is_rejected` |
| 2 | RAG + LLM test | leading questions vouched for facts: "Are international transfers free?" → "Yes, they are free" passed, as did "Is the fee 0.1%?" → "Yes, 0.1%" | claims never come from the question; policy answers take numbers only from the passages | `tests/integration/test_rag_llm.py::test_a_leading_question_cannot_vouch_for_a_fact` |
| 3 | RAG + LLM test | "the minimum fee is 10 GEL" passed: 0-12 counted as harmless counts | a number next to a currency or % is an amount, never a count | `tests/unit/test_grounding_and_rendering.py::test_small_amounts_and_rates_are_not_counts` |
| 4 | property test | base64 of an attack padded with zero-width characters was judged binary and not decoded | invisible characters don't count against "readable" | `tests/unit/test_input_guard.py::test_disguises_found_by_hypothesis_are_refused` |
| 5 | property test | a spaced-out word swallowed a neighbouring letter: "my wife's a c c o u n t" → "wife'saccount" | despacing variants that keep the first/last letter apart | (same) |
| 6 | property test (thorough) | spaced-out **Georgian** letters were not joined | despacing for every script | (same) |
| 7 | property test (thorough) | leetspeak decoding garbled a customer number ("cu57-0003" → "cust-ooo3"), then (while fixing it) apostrophes and "l157" | identifiers (4+ digits in a row, or digits 2/6/8/9) are left alone; tokens split at "-" | (same) |
| 8 | property test | normalization was not idempotent: a Cyrillic capital became a *small* Cyrillic letter, mapped only by accident in other views | look-alikes mapped again after case folding | `tests/property/test_safety_properties.py::test_normalizing_twice_changes_nothing`, `tests/unit/test_input_guard.py::test_capital_lookalikes_are_mapped_in_the_plain_view` |
| 9 | property test | action detection read the raw text: "Can I transfer...?" in look-alike letters was held as a transfer request | action detection uses the guard's normalized text | `tests/unit/test_input_guard.py::test_a_question_in_lookalike_letters_is_still_a_question` |
| 10 | mutation testing | the fuzz test sent only junk, never reached the ownership checks, and missed a deleted check | half the fuzz calls are well-formed, generated from the tools' JSON schemas | `tests/property/test_tool_fuzzing.py` (kills `authz-account-owner` on its own) |
| 11 | mutation testing | the balance and details tools loaded a requested account by ID alone: the explicit check was their only protection, although the docs promised two layers | the query is also scoped to the owner | `tests/safety/test_tool_authorization.py::test_scoped_queries_hold_even_if_the_ownership_check_is_skipped` |
| 12 | concurrency test | (test setup) audit writes failed under concurrency on the single-connection test database | concurrency tests use a database file; the check is repeated on PostgreSQL | `tests/integration/test_api_postgres.py::test_concurrent_customers_are_isolated_and_fully_audited` |

Also found, and not a code bug: the currency round-trip property failed at first because *my
tolerance* was wrong. 4-decimal rates make a small rate such as 0.3143 GEL→EUR up to 0.016% off.
The test now derives the bound from the rounding instead of guessing it.

Every fix was checked both ways: the new test fails on the code before the fix and passes after
it.

**What the stricter grounding cost.** A check that rejects too much would make answers worse, so
the evaluation was re-run with the real model (qwen2.5:3b, 246 cases): **209/246 before and after**,
with every metric unchanged. Exactly one LLM answer changed. For "i lost my card what do i do" the
model merged two passage sentences into one sentence containing the claim word "immediately". The
sentence-level check could not confirm it, so the correct template answer was used instead (LLM-written
answers 37.4% → 37.0%). That is the intended trade-off: when unsure, fall back to the safe answer.

## 9. Determinism and flakiness

A flaky test is worse than no test, because people learn to ignore red builds. So:

- the default run is **deterministic**: a pinned clock (`ANCHOR`), seeded mock data, the offline
  agent (rules + templates), a hashing embedder, and a scripted LLM;
- **non-deterministic tests are opt-in and marked**: the live LLM (`llm`) and the speech models
  (`voice`). Their assertions are properties ("no invented number"), not exact strings;
- Hypothesis failures are **reproducible**: the failing example is printed, shrunk and saved;
- random test order prints its **seed**;
- the evaluation run is checked to give the same result twice
  (`tests/evaluation/test_runner.py::test_offline_run_is_deterministic`).

## 10. Coverage

`pytest --cov` measures `app/` and `scripts/` with branch coverage: **99.8%** with
PostgreSQL, 99.6% without it. CI fails below **95%**, which is above the
specification's 80%. The gate sits a few points below the actual value, so a real drop fails the
build but an honest refactoring doesn't. It is meant for the full run: `pytest tests/unit --cov`
alone covers less and will report the gate as missed.

The code excluded from coverage is four lines, each with a written reason
(`# pragma: no cover - ...`): branches the graph's routing makes unreachable. Nothing is excluded
to raise the number.

## 11. Known gaps (honest)

- **No load or performance tests.** Latency is measured per turn and by the evaluation (p50/p95),
  but nothing tests behaviour under hundreds of users.
- **Browser coverage**: the UI is tested in Chrome only, not Safari or Firefox.
- **Live LLM results vary between runs.** `tests/llm` asserts properties, but a model update can
  change pass rates; the evaluation run is the measurement, not a gate on exact answers.
- **Claim grounding is lexical.** It matches words, not meaning: a claim phrased without a listed
  claim word ("costs nothing") is not checked. Negation is ignored ("not free" needs the claim to
  be in the facts too; the answer falls back to the template).
- **Numbers glued to letters** ("GEL999") are not seen by the number check. Identifiers such as
  account numbers look the same way, and those must not be read as amounts.
- **Retrieval quality with the offline hashing embedder** is limited: for "Do international
  transfers arrive instantly?" it ranked a domestic-transfer passage first (seen while writing
  the RAG + LLM tests). The semantic embedder is measured by `scripts/calibrate_rag.py`.
- **Injection rules miss paraphrases** (documented in the architecture, §8): detection is
  best-effort, and the authorization layer is the guarantee.
- **Georgian**: speech recognition quality is low with the local models (the pipeline refuses to
  act on it, which is tested), and the safety rules cover fewer Georgian phrasings than English ones.
