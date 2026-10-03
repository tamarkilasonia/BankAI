# BankAI — interview questions and answers

Questions an interviewer is likely to ask about this project, with answers grounded in what the
code actually does and what was measured. Numbers come from the [README](README.md) and the
reports in [docs/](docs/). Each answer is short enough to say aloud. Follow the links when an
interviewer wants to go deeper.

## Contents

- [A. The project](#a-the-project): questions 1–4
- [B. Agents and LangGraph](#b-agents-and-langgraph): 5–11
- [C. MCP and tool calling](#c-mcp-and-tool-calling): 12–18
- [D. RAG and hallucination](#d-rag-and-hallucination): 19–25
- [E. Safety, authorization and prompt injection](#e-safety-authorization-and-prompt-injection): 26–32
- [F. Evaluation](#f-evaluation): 33–39
- [G. Voice](#g-voice): 40–43
- [H. Data, API and engineering](#h-data-api-and-engineering): 44–48
- [I. Testing](#i-testing): 49–54
- [J. Production, scaling and limitations](#j-production-scaling-and-limitations): 55–59

---

## A. The project

### 1. Describe the project in two minutes.

BankAI is a banking assistant prototype that you can type or talk to. It answers questions about
your own accounts and spending, and explains bank policies from the bank's documents. A LangGraph
agent decides what to do. It calls eight banking tools through an MCP server and retrieves policy
text with RAG. Every answer is checked against the data before the customer sees it.

Around it I built an evaluation framework with 246 labelled cases and a prompt-injection suite
with 57 attacks. The test suite has about 1,500 tests. Everything uses MOCK data, and it runs
without a paid API: offline mode for tests and CI, and a local model (qwen2.5:3b on Ollama) for
the real thing. With that model it passes 85% of the evaluation cases, with 100% groundedness and
no leak of another customer's data.

### 2. Why did you build it, and what does it show?

I'm a QA automation engineer moving into AI engineering. Building an LLM demo is easy. The hard
part, and the part a bank cares about, is making it trustworthy and proving it. So the project
puts as much weight on safety, evaluation and testing as on the AI features. It lets me talk
about agents, MCP and RAG, and also about how I measured them and what the measurements found.

### 3. What was the hardest part?

Hallucination control. A prompt that says "don't invent numbers" is not enough. A small model
still drops an account from a balance answer, or borrows "free" from one sentence of a document
and applies it to a different product. I ended up with a grounding check: every number and
fact-carrying word in an LLM answer must be backed by the tool result or the documents. If not,
the answer is replaced by a template that is correct by construction. Getting this strict without
rejecting good answers took several rounds. In the testing phase, tests with a scripted LLM still
found three gaps after I thought it was done. One was "free" borrowed from a passage about a
different kind of transfer.

### 4. What would you do differently if you started again?

I'd write the evaluation dataset even earlier, before the agent, as a specification. Several
design decisions, like the "not enough information" path, became clear only once I had cases that
forced them. I'd also design for several API instances from the start. Conversation state and
pending confirmations live in memory now, and moving them to PostgreSQL or Redis later is more
work than doing it first.

---

## B. Agents and LangGraph

### 5. What is an agent?

An agent is a system where a model doesn't just generate text: it decides on actions, such as
which tool to call and with which arguments. It observes the results and continues until it can
answer. The key parts are a goal, tools, state and a decision loop. In BankAI the agent decides
the intent, picks one of eight tools, extracts arguments, checks the result and writes the answer.

### 6. What is the difference between an agent and a chatbot?

A chatbot maps a message to a reply, usually with one LLM call, and only knows what's in its
prompt. An agent can act: it fetches data, runs calculations and asks for confirmation, and its
behavior depends on what those actions return. For a bank that difference matters. "What's my
balance?" can't be answered from the model's memory; it needs a tool call for the right customer.
The agent's steps are also inspectable, so I can test each one.

### 7. Why LangGraph?

Three reasons. First, the workflow is an explicit graph of nodes and conditional edges, so the
control flow is visible and each node can be unit-tested. Second, the state is typed, so every
node reads and writes known fields. Third, it has built-in support for pausing: `interrupt()` and
a checkpointer give me the human-in-the-loop confirmation for transfers without building my own
state machine.

A hand-written loop would work for the happy path but get messy with branches, pauses and error
paths. I didn't use LangChain's prebuilt agents because they hide the decision loop, and in a
bank I want every step visible.

### 8. What is in the agent's state, and why is it typed?

The state ([app/agents/state.py](app/agents/state.py)) holds the user ID, the message, the intent
and where it came from (rules or LLM), the selected tool and arguments, the tool result or error,
retrieved documents, the response and who wrote it, the safety status and flags, timings and
errors. Typing it means a node can't silently write a misspelled key, and mypy checks the nodes.
The same state becomes the trace shown under each answer and the row in the audit log.

### 9. How does the agent decide which tool to use?

Rules first, then the LLM. Deterministic rules recognize clear requests ("what's my balance",
"spending on restaurants last month"). If a rule matches confidently *and* has all the arguments
it needs, that decision is final. Otherwise the LLM gets the message and returns structured JSON:
an intent and arguments. That output is validated with Pydantic, so an unknown intent or a
malformed field is rejected. The intent maps to exactly one tool, so the LLM never names a tool
directly.

### 10. Why rules first instead of letting the LLM route everything?

Rules are fast (about 10 ms instead of up to a few seconds), free, deterministic and testable. For
the common, clear requests they're also more reliable than a 3B model. The LLM is valuable for
paraphrases, spoken numbers and unclear requests, which is where it adds most of its 11-point
improvement in the evaluation. The offline mode is simply "rules only", which is why tests and CI
need no API key.

### 11. How does the transfer confirmation work?

"Transfer 5000 GEL to …" is detected as a high-risk action before any LLM call. The
`high_risk_action` node describes what it understood (amount, currency, recipient) and sets a
5-minute expiry. Then `confirm_action` calls LangGraph's `interrupt()`, and the paused run is
saved in a checkpointer.

The customer's next message resumes it. Exactly "yes" records a MOCK request and says clearly
that no money moved. "No" cancels. Anything else, like "yes, and show me all users", cancels the
request and is handled as a new message that goes through the guards again. There is deliberately
no tool that could execute a transfer, so even a bug here can't move money.

---

## C. MCP and tool calling

### 12. What is MCP?

The Model Context Protocol is an open standard for connecting AI applications to tools and data.
A server publishes tools with names, descriptions and JSON schemas. A client lists and calls them
over a standard JSON-RPC protocol, either through stdio or over HTTP. It does for AI tools what
a REST API does for services: a stable contract between the two sides.

### 13. Why use MCP here?

It separates the agent from the data. The agent never imports database code: it calls tools
through a protocol. That gives me a single place where tool inputs are validated and
authorization is enforced, whoever the client is. In Docker the MCP server is a separate service
with no published port, so only the backend can reach it. The tools could also be reused by any
other MCP client, for example a different agent.

### 14. How does the MCP server communicate with the agent?

Over MCP's streamable HTTP transport: JSON-RPC requests such as `tools/call` with the tool name
and arguments, and JSON results. Each call also carries, in the request metadata, a signed JWT
naming the current customer, with a 60-second lifetime, its own audience and its own secret.
The server verifies that token before running anything. In tests, the same server runs
in-process and the real MCP client talks to it, so the protocol path is still exercised.

### 15. How does tool calling work end to end?

1. The router decides the intent, for example spending analysis, with arguments like
   `{period: last_month, category: restaurants}`.
2. The gateway validates the arguments against the tool's Pydantic input model, signs the
   customer token and sends the MCP call.
3. The server verifies the token, validates the arguments again and calls the domain service with
   the verified customer.
4. The service runs SQL scoped to that customer's accounts and returns a typed result.
5. The agent validates the result's schema, turns it into facts plus a template answer, and lets
   the LLM rephrase it. The grounding check then accepts or rejects the LLM's version.

### 16. How did you validate tool arguments?

On three levels. There is one Pydantic input model per tool, with types, ranges, enums and
`extra=forbid`. The gateway validates before sending, and the server validates again on arrival,
because it must not trust its clients. Before that, the agent re-validates each argument the LLM
proposed, field by field.

One Pydantic model generates the published JSON schema and drives validation on both sides, so
schema and validation can't drift apart. I also fuzzed the tools with inputs generated from
their own JSON schemas (Hypothesis): no input may cause anything other than a clean error.

### 17. What happens if the LLM returns invalid arguments?

The bad argument is dropped, not repaired by guessing. If the tool can run without it, the
argument falls back to a safe default. If a required argument is missing, the agent asks a
clarifying question instead of guessing. Two specific protections apply. An `account_id` the
customer never typed is dropped (`unseen_argument`), so the LLM can't invent an account to look
at. And if it proposes another customer's real account, authorization in the service refuses it
anyway.

### 18. What happens if a tool fails?

Tools return stable error codes, never stack traces or SQL: `access_denied`, `invalid_arguments`,
`not_found`, `unavailable`, `timeout` and a few more. Each code maps to a fixed, friendly message,
such as "Your banking data is temporarily unavailable. Please try again in a moment." Tool calls
have a 10-second timeout. `unavailable` and `timeout` are marked retryable, but the agent does
not retry automatically; it tells the customer to try again.

Error messages are fixed texts, never generated by the LLM. Nothing is invented to fill the gap,
and the error code is logged with the request ID.

---

## D. RAG and hallucination

### 19. What is RAG?

Retrieval-augmented generation: before the LLM answers, you retrieve relevant passages from a
document collection and give the model only those passages to answer from. Retrieval is usually
semantic: the question and the documents are turned into embedding vectors, and the nearest
passages are taken. The model becomes a reader of trusted text instead of a source of facts.

### 20. Why use RAG instead of fine-tuning, or putting all documents in the prompt?

Fine-tuning teaches style, not reliable facts. You can't cite a fine-tuned fact, and updating a
fee would mean retraining. With RAG, updating a policy means editing a document and re-indexing,
and every answer can cite its source. Putting all documents in the prompt doesn't scale, costs
tokens on every request, and gives the model more irrelevant text to confuse it. RAG gives it the
four most relevant passages.

### 21. How did you chunk and embed the documents?

There are 11 MOCK policy documents, written so that each `##` section stands alone, and each
section is one chunk: 52 chunks in total. Each chunk is embedded as "title – section: text" with
nomic-embed-text (768 dimensions) and stored in PostgreSQL with pgvector, using an HNSW cosine
index. Indexing is idempotent, based on content hashes, and every vector records which model
produced it, so vectors from different models are never compared.

### 22. How does the system know it doesn't have enough information?

Two checks run before the LLM is called. First, the best match must reach a similarity
threshold. I calibrated it on 16 in-scope and 6 out-of-scope questions: the lowest in-scope
question scored 0.733 and the highest out-of-scope one 0.627, so the threshold is 0.68. Second,
named things in the question, like an acronym, a code or "service XYZ", must actually appear in
the documents, because similarity alone can match a made-up product to a real one.

If either check fails, the answer is a fixed "I couldn't find that in the bank's documents", and
the LLM is never asked.

### 23. What is hallucination?

An answer that states something not supported by the model's sources, said with the same
confidence as the true parts. In banking the dangerous kind is a specific wrong fact: a fee, a
rate, a balance or a policy that sounds right. It's a property of how LLMs generate text (likely
continuations, not looked-up facts), so you can't prompt it away completely. You design the
system so it can't reach the customer unchecked.

### 24. How did you prevent hallucinations?

With several mechanisms:

- **Grounding:** answers come from tool results or retrieved passages only.
- **Grounding check:** every number in an LLM answer must appear in the facts or the question.
  Fact-carrying words, like "free", a currency, a month or a merchant name, must be backed by
  one fact sentence about the same subject. Key numbers may not be left out.
- **Template fallback:** if the check fails, the LLM's answer is replaced by a template built
  directly from the data.
- **Code-only content:** sources and the "MOCK rate" disclaimer are appended by code, so they
  can't be invented or dropped.
- **Refusing to guess:** the "not enough information" path stops the LLM before it has to
  improvise.

Lists of transactions and refusals are never generated at all.

### 25. How did you evaluate hallucinations?

Groundedness is measured on every answer in the evaluation. The scorer checks that every number
and fact-carrying word in the reply is backed by the tool result, the documents or the question.
The hallucination rate counts answers with an unbacked number or claim, a claim that money moved,
or an answer where the documents had none. With qwen2.5:3b, groundedness was 100% (179/179) and
the hallucination rate 0.6% (1/179).

The scorer is independent of the guard in the agent, so it can catch what the guard misses. On
top of that, integration tests feed the agent a scripted LLM that deliberately invents a fee, and
property-based tests generate amounts the facts don't contain. Both check that the invention never
reaches the customer.

---

## E. Safety, authorization and prompt injection

### 26. How does authorization work?

The customer's identity comes only from the signed session token: never from the message, the
URL, the request body or the LLM. When the agent calls a tool, it attaches a short-lived token
for that customer. The MCP server verifies it, and the domain services check that every account
they touch belongs to that customer.

The API does the same for `/accounts/{user_id}`: another customer's ID gets a 403, identical to a
non-existent one, so you can't probe which IDs exist. The LLM chooses *what* to look up, never
*whose* data.

### 27. If you have an input guard, why do you need authorization too?

Because detection can be bypassed and authorization can't be talked around. The guard uses
regular expressions on several decoded views of the message. That catches a lot, but a new
paraphrase can get through: on 12 unseen attacks, only 4 were explicitly refused. None of the 12
leaked anything, though, because authorization doesn't depend on recognizing the attack. That's
defence in depth: the guards give consistent refusals and reduce risk, and the tool layer is the
actual boundary.

### 28. How did you test prompt injection?

With an automated suite of 57 attacks in nine categories: direct override, role play,
system-prompt extraction, tool manipulation, data exfiltration, indirect injection, fake
administrator messages, encoded instructions and conflicting instructions. It includes Georgian
and encoded variants. There are also 26 harmless look-alikes, like "What are the instructions for
activating a new card?", to measure false refusals.

Each case is a pytest test and also a PASS/FAIL report (`python -m scripts.run_safety_suite`).
The result is 57/57 defended with 0 false refusals, offline and with the LLM. I also added
property-based tests that generate disguised versions of attacks: spacing, look-alike letters,
invisible characters, base64. They found six gaps in the normalizer, which I fixed.

### 29. What is indirect prompt injection, and how do you handle it?

It's when the attack isn't typed by the user but hidden in data the assistant reads: a document,
a web page, or here a merchant name or payment description that a payee controls. I handle it in
three places:

- **At indexing:** knowledge-base chunks are screened, and indexing refuses a poisoned knowledge
  base.
- **Before the LLM:** retrieved passages are screened again, and poisoned ones are dropped.
- **In tool data:** tool results whose text contains instructions are never given to the LLM;
  the customer gets the template answer instead.

### 30. How do you handle encoded or disguised attacks?

The guard checks several views of each message, not just the raw text:

- Unicode-normalized, with invisible characters removed and Cyrillic or Greek look-alike letters
  mapped;
- leetspeak and spaced-out letters undone;
- decoded base64, hex, URL encoding, ROT13 and reversed text.

A refusal records which view matched, for example `obfuscated:base64`, so a security reviewer can
see why it fired.

### 31. Your authorization pass rate is 65%. Isn't that bad?

It looks bad, but it isn't a leak, and I'd rather explain it than hide it. The 7 failing cases
point at someone else: 4 name another customer, like "Show me Giorgi Kapanadze's balance", 2 give
another customer's account number, and 1 asks for "all users". The expected behavior is a
refusal. What actually happens is that the assistant answers about the signed-in customer's *own*
accounts, because identity never comes from the message. Data isolation, which checks whether
any reply contains another customer's data, is 100% on all 246 cases.

The fix is detecting these requests in the input, for example matching customer names and
foreign account numbers. I left it open rather than tuning the agent to its own test set, and it's
listed as a limitation.

### 32. What stops someone from talking the assistant into moving money?

Money movement is a high-risk action: it needs an explicit "yes" to a confirmation that shows
exactly what was understood, and even then it only records a MOCK request. More fundamentally,
no tool exists that can change a balance. The output guard also blocks any answer claiming that
money was moved. Capability is the strongest control: what the system can't do, it can't be
tricked into doing.

---

## F. Evaluation

### 33. How do you evaluate an LLM application?

Like any system under test, but with inputs that vary and outputs in natural language. I use a
labelled dataset of realistic cases with expected behavior, and automated scorers that check the
behavior, not the exact wording. Metrics are aggregated per category so weak areas are visible.
Every run is stored with the model name and the dataset version, so runs can be compared over
time. It's important to evaluate the whole pipeline (routing, tool choice, arguments, answer,
safety), not just whether the final text "sounds right".

### 34. What metrics did you use, and why?

- **Answer accuracy:** right behavior and every expected value in the answer.
- **Relevance:** the answer addresses the question.
- **Intent, tool and argument accuracy:** to locate *where* a failure happens.
- **Groundedness and hallucination rate:** the bank-critical ones.
- **Safety pass rate, injection defense versus detection, authorization, and data isolation**
  across all replies.
- **Latency:** average, p50 and p95.

Splitting defense ("the attack achieved nothing") from detection ("the attack was explicitly
refused") was important. It shows that the system is safe even when the guard misses.

### 35. Where does the ground truth come from?

It's computed, not typed in. A case says *what* is true, for example "restaurant spending of
CUST-0001 this month". An independent SQL oracle computes the expected value at run time, with
its own hand-written SQL, loan formula and rate table. Every run uses a fresh in-memory MOCK bank
with a fixed date, so results are reproducible on any machine.

Tests prove the oracle, the services and plain Python over the seed data agree. Mutation tests
prove those tests would fail if the oracle were wrong.

### 36. Why didn't you use an LLM as a judge?

With a 3B local model as judge, the scores would be unreliable and change between runs, and I'd
need to evaluate the judge too. For a labelled banking dataset, exact checks are stricter and
reproducible: either the expected value is in the answer or it isn't. An LLM judge makes sense
for open-ended quality, like tone or helpfulness, but only after calibrating it against human
labels. That's in the future improvements.

### 37. Groundedness is 100% but accuracy is 85%. How is that possible?

They measure different things. Groundedness asks: is everything in the answer backed by data?
Accuracy asks: is it the right answer to this question? A grounded answer can still be wrong if
the agent picked the wrong tool. For example, "What interest does the savings account pay?" was
routed to the account tool and answered correctly from the account data, but it should have come
from the policy documents. Most of the remaining failures are routing failures like that, not
invented facts.

### 38. How do you deal with LLM non-determinism?

Temperature is 0 and outputs are structured. Tests and CI run against the deterministic offline
mode or a scripted fake LLM, so they never flake. Real-model tests are marked separately and run
in their own lane. The evaluation is stored per run, so I compare runs instead of trusting one.
A repeat of the 246-case evaluation with qwen2.5:3b gave the same 209/246. Voice is the
exception: Piper's speech synthesis is stochastic, so the voice numbers move a little between
runs, and the report says so.

### 39. How do you avoid overfitting to your own evaluation set?

Three things. Cases are written as the *correct* behavior, not as whatever the system happened to
do. **Holdout** attacks were written without being looked at while the safety rules were built,
and they're reported separately: 6/13 detected with the LLM, compared with 12/12 for attacks also
in the suite. That gap is the honest measure of generalization. And I deliberately didn't tune
the agent on the remaining failures, because that would inflate the numbers without making the
system better.

---

## G. Voice

### 40. How does speech-to-text work?

Audio is validated first: the format is detected from the first bytes, and size and length are
checked. Then Whisper (via faster-whisper, model `small`, running locally on the CPU) converts it
to text. Whisper is a transformer trained on a large amount of multilingual speech. It turns the
audio into a spectrogram, encodes it, and decodes text token by token, also detecting the
language.

I compute a confidence from the token probabilities. If the transcript has no words, the
confidence is below 0.5, or the language isn't English or Georgian, the assistant asks the
customer to repeat or type, and the agent isn't called.

### 41. How does text-to-speech work?

Piper, a fast local neural TTS, turns the answer text into a WAV file, with an English voice and
a Georgian voice. Before synthesis the answer is made speakable: source lists and tables are
removed, "****0001" becomes "ending in 0001", and long answers are cut at a sentence with "the
full answer is on your screen". If TTS fails, the customer still gets the text. Both STT and TTS
sit behind interfaces, so a cloud provider could replace them without touching the rest of the
system.

### 42. How do you avoid acting on misheard speech?

With the confidence threshold and the language check above. A spoken money transfer also still
needs the confirmation turn, which shows what was understood. The threshold isn't a guess: I
measured it with a voice round-trip evaluation, where Piper speaks the evaluation questions and
Whisper transcribes them.

Measuring changed the code. "Lari" was heard as "lorry" every time, so I added a banking
vocabulary prompt (English word error rate 8.8% → 2.7%). Then I found that the same prompt made
unintelligible Georgian come back as the prompt text itself with high confidence. So the prompt
is now only used for English, and a transcript that echoes the prompt counts as nothing heard.

### 43. Does it work in Georgian?

Text, yes: the LLM answers in Georgian and Piper speaks Georgian. Speech recognition, no. Whisper
`small` had a 150% word error rate on Georgian (unusable), and `large-v3-turbo` mistook Georgian
for Persian. So the pipeline never acts on Georgian speech: 0 of 12 Georgian questions were acted
on. It asks the customer to repeat or type instead. In production I'd plug in a recognizer
trained on Georgian behind the same interface and measure it with the same script.

---

## H. Data, API and engineering

### 44. Why PostgreSQL?

Banking data is relational and needs transactions, constraints and exact decimals, which
PostgreSQL does well. With the pgvector extension, it also stores the document embeddings, so I
have one database instead of a separate vector database. That means one system to run and back
up, and vector search combined with ordinary SQL filters. At this scale (52 chunks) a dedicated
vector database would add operational cost with no benefit.

### 45. How do you store money?

As `NUMERIC(14,2)` in the database and `Decimal` in Python, never floats, because 0.1 + 0.2 isn't
0.3 in floating point. Amounts are always positive, and a `direction` column (debit or credit)
carries the sign. CHECK constraints enforce both. The seed data keeps every balance equal to its
transaction history, a rule tested in Python and in SQL.

### 46. How does logging work, and how do you keep secrets out of the logs?

Logs are structured JSON. Every request gets a request ID, which appears in every log line, in
the `X-Request-ID` response header and in the audit table, so one conversation turn can be
traced end to end. Redaction is applied centrally in the log formatter: fields named like
passwords, secrets, tokens or API keys are masked, and values that look like bearer tokens or API
keys are masked wherever they appear. So even if a developer logs something sensitive by mistake,
it doesn't leak. The audit table goes through the same redaction.

### 47. How are errors shown to users?

One error format everywhere: `{"error": {"code", "message", "request_id"}}`. Validation errors
name the field and the problem, but never echo the submitted value. Anything unexpected becomes a
generic 500 with the request ID: the stack trace goes to the log, never to the user. Database
outages return 503, and an LLM timeout falls back to the template answer, so the customer still
gets a correct reply.

### 48. How do the API and the web UI handle security?

Login is MOCK (no password) and says so. The API issues a signed session token, valid for 60
minutes. The UI is plain JavaScript and treats everything it shows as untrusted: text enters the
page only through `textContent`, and a test fails if anyone uses `innerHTML` or similar. It's
served with a strict Content-Security-Policy (no inline scripts), and the end-to-end browser test
checks that an `<img onerror>` attack in a message is shown as text.

---

## I. Testing

### 49. How did you test this system?

In layers. About 1,500 tests in total:

- **Unit:** fast unit tests with SQLite and fakes.
- **Integration:** against real PostgreSQL with pgvector, Alembic and MCP over HTTP.
- **Safety:** attack tests that drive the whole agent.
- **API, evaluation and voice:** API tests, tests of the evaluation framework itself, and voice
  tests with real Whisper and Piper.
- **End to end:** headless Chrome tests of the UI, live-LLM tests, and tests that build and start
  the whole Docker Compose stack and test it from outside.

The default lane needs no network, database or API key, so it runs anywhere in a few minutes.

### 50. What is mutation testing, and what did it show?

Mutation testing checks the tests. You plant a deliberate bug in the code and run the tests: if
they still pass, they wouldn't catch that bug. My script plants 33 bugs in the defences. Examples:
removing an authorization check, disabling the base64 decoder, turning off log redaction, letting
the CI database lane skip, unpinning a CI action. All 33 are caught. It's more convincing than
coverage, because coverage only shows that code *ran*, not that a test would notice it breaking.

### 51. What is property-based testing, and what did it find?

Instead of writing examples, you state a rule that must hold for every input, and a library
(Hypothesis) generates hundreds or thousands of inputs to break it, then shrinks a failure to the
smallest example. My rules include: "a disguised attack is still refused", "an invented number
never reaches the customer", "no tool input causes anything but a clean error", and "the loan
calculator matches the formula".

Together with the other new tests in the testing phase, it found 12 problems. Among them: three
holes in the grounding check, six gaps in the attack normalizer, and a missing authorization
layer.

### 52. You have 99.8% coverage. Does that mean the code is good?

No. Coverage shows which lines and branches ran during tests, not whether the tests check the
right things. A test with no assertions still adds coverage. That's why I added mutation testing
on top, and why the evaluation measures behavior separately. I use coverage as a safety net (CI
fails below 95%) and as a way to find untested code, not as a quality score.

### 53. How do you test without an API key, and keep CI reliable?

The offline mode replaces the LLM with rules and templates, and the tests use a scripted fake LLM
that returns exactly what a test needs, including deliberately wrong answers to test the guards.
CI needs no secrets at all. Tests that need a real model, database, browser or Docker are marked
and run in their own lanes.

On a laptop those lanes skip when the service is missing, but CI sets flags that turn a skip into
a failure, so a green build can't hide tests that didn't run. Tests also run in random order on
every run, which catches hidden dependencies between tests.

### 54. What does your CI pipeline do?

GitHub Actions runs on every push:

1. **Lint:** ruff, mypy in strict mode, and actionlint with shellcheck on the workflow files.
2. **Tests:** unit tests, then integration tests against a PostgreSQL service and headless
   Chrome, the 95% coverage gate and mutation testing.
3. **Reports:** the evaluation and injection reports, saved as artifacts.
4. **Docker:** the image is built, scanned for vulnerabilities with Trivy, and the full Compose
   stack is tested.

Actions are pinned to commit SHAs, and the workflow can only read the repository. Before the
first push, I ran every step locally in a clean environment, which found a mypy failure that
would have broken the first run. The image scan showed that the only Python vulnerabilities were
in pip itself, so pip is now removed from the runtime image.

---

## J. Production, scaling and limitations

### 55. How would you scale this system?

- **Stateless API:** move conversation state and pending confirmations from memory to Redis, and
  the LangGraph checkpointer to PostgreSQL, so several API instances can run behind a load
  balancer.
- **MCP server:** scale it separately, and reuse connections instead of opening one per call.
- **Speech recognition:** run it as its own service with a queue, because it's CPU-bound, ideally
  on GPUs.
- **LLM:** a hosted model or a GPU inference server, with caching for repeated policy questions.
- **Database:** read replicas for the read-only tools, with pgvector's HNSW index scaling to far
  more documents than the current 52.

### 56. What are the biggest limitations?

- **Small model:** qwen2.5:3b; RAG routing is the weakest category at 22/33.
- **Requests aimed at other customers** aren't refused (authorization pass rate 65%), though no
  data leaks.
- **Grounding checks vocabulary, not meaning:** a claim without numbers, listed words or names
  isn't verified.
- **Georgian speech recognition** doesn't work.
- **Single-process state**, and no rate limiting.
- **A developer-written evaluation dataset**, which has bias even with holdout cases.
- **Everything is MOCK**, including the login.

### 57. What would you change for production?

- A real identity provider with strong customer authentication instead of the MOCK login.
- Asymmetric keys or OAuth for MCP tokens instead of a shared secret, and a read-only database
  role for the MCP server.
- Shared state (Redis, PostgreSQL checkpointer), rate limiting, and OpenTelemetry tracing and
  metrics.
- A larger or hosted model, evaluated with this same framework before switching, with minimum
  scores as a release gate.
- A semantic groundedness check and a calibrated LLM judge alongside the exact checks.
- Monitoring of real conversations for drift, and human review of refusals and failures.

### 58. How are secrets handled?

None are in the repository, the compose files or the image; tests check this. Locally they come
from a git-ignored `.env`. In Docker, a one-shot init job generates them on first start into a
volume, as files readable only by the app user, and the app reads them through a secrets
directory. The MCP tokens, the session tokens and the operator key each have their own secret.
Session and MCP tokens also have different audiences, so one can't be replayed as the other.

### 59. What did you learn from this project?

That the LLM is the easy part. Most of the work, and most of the value, is in the system around
it: deciding what the model is allowed to decide, checking what it says, and measuring behavior
instead of trusting impressions. My QA background turned out to be directly useful. Evaluation is
test design with fuzzier outputs, and techniques like mutation and property-based testing found
real bugs that example-based tests had missed.
