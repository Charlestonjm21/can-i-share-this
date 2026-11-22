# Can I Share This?

An advisory agent that answers one question for employees: **is this specific
act of sharing data permitted by company policy?**

The employee describes a proposed action in plain language. The agent extracts
five contextual facts, retrieves the governing clauses from the organization's
own policy corpus, and returns **SAFE**, **CHECK POLICY**, or **DO NOT SHARE**
with the clause it relied on.

> The agent never receives the data being discussed. It reasons about a
> *description* of a proposed flow, so a failure of the agent cannot itself
> become a disclosure event.

COMP 422.001 — Amon Crenshaw, Paris Frederick, Lauryn Jones, Charlestone Mayenga

**Live demo: [can-i-share-this.streamlit.app](https://can-i-share-this.streamlit.app)** (free hosting; the first visit after a quiet spell can take about 30 seconds to wake. Each visitor session is limited to 20 questions.)

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export GROQ_API_KEY=gsk_...                         # Windows: set GROQ_API_KEY=...
streamlit run app.py
```

Get a free Groq key at [console.groq.com](https://console.groq.com) — no
credit card. The free tier allows 30 requests per minute and 1,000 per day;
a full 50-scenario evaluation run uses 100 calls and the 8,000 tokens-per-minute
cap, so it takes about 10 minutes: the client waits and retries on its own.
Set `CISH_SKIP_MODEL_ON_BLOCK=1` to cut the call count.

No model download, no vector database, no SDK, no build step. Core install is
a few megabytes and works on Windows, macOS, and Linux across Python
3.9–3.13.

### Other providers

Everything except Ollama and Anthropic speaks the OpenAI chat-completions
protocol, so switching is an environment change rather than a code change.

| `CISH_PROVIDER` | Key variable | Default model | Notes |
|---|---|---|---|
| `groq` *(default)* | `GROQ_API_KEY` | `openai/gpt-oss-120b` | free, open weights |
| `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` | paid |
| `openrouter` | `OPENROUTER_API_KEY` | Llama 3.3 70B | many models, one key |
| `together` | `TOGETHER_API_KEY` | Llama 3.3 70B | paid |
| `ollama` | — | `llama3.1:8b` | fully local, no network egress |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-5-5` | needs `pip install anthropic` |
| `custom` | `CISH_API_KEY` | set `CISH_MODEL` | set `CISH_BASE_URL` |

Fully local, nothing leaves the machine:

```bash
ollama pull llama3.1:8b && ollama serve
export CISH_PROVIDER=ollama
streamlit run app.py
```

Note that Ollama cannot be used on a hosted deployment — Streamlit Cloud has
no local model — so the deployed demo uses a hosted provider.

Run the tests (no API key needed):

```bash
pip install -r requirements-dev.txt
python -m pytest tests/
```

---

## The design idea

The language model is the component most likely to fail, so it is given the
least authority. Two verdicts are produced independently and combined by
taking the more cautious of the two.

```
                  ┌──────────────────────┐
   question  ───► │  context extraction  │  (model: question → 5 slots)
                  └──────────┬───────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
     ┌────────────────┐            ┌──────────────────┐
     │  rules engine  │            │ policy retrieval │
     │  (no model)    │            │  + model verdict │
     └────────┬───────┘            └────────┬─────────┘
              │                             │
              └──────────────┬──────────────┘
                             ▼
                   more_cautious(a, b)  ──►  SAFE │ CHECK POLICY │ DO NOT SHARE
```

Three properties worth noting:

1. **The rules engine can only veto.** It returns DO NOT SHARE or "no
   objection"; it never approves anything.
2. **SAFE requires a retrieved permissive clause and an internal or public
   recipient.** The agent cannot conclude an action is allowed merely because
   it found nothing forbidding it.
3. **The model can tighten a verdict but never loosen one.** This is why the
   system tolerates a weak model — see the comparison experiment below.

Authorization is read from the directory, never from the employee's question,
so a user cannot talk their way into an entitlement they do not hold.

### Why this matters

Two findings from the literature drove the architecture. Mireshghallah et al.
(ICLR 2024) found GPT-4 and ChatGPT disclosing private information in contexts
humans would not in 39% and 57% of cases, and prompt-level privacy
instructions did not fix it. Shao et al. (NeurIPS 2024) found models stating a
privacy norm correctly and then violating it while acting, leaking in 25.68%
of trajectories for GPT-4 and 38.69% for Llama-3-70B.

The conclusion we drew: **a prompt instruction is not a control.** The
advisory-only constraint is therefore enforced by withholding capabilities
from the process — no file access, no network send, no tool schema registered
— rather than by asking the model to behave.

---

## Experiments

The harness runs 10 scenarios × 5 phrasings = 50 cases and gates on release
criteria.

```bash
python evaluate.py                          # default stack
python evaluate.py --provider ollama        # local 8B model
python evaluate.py --compare groq,ollama    # two providers, side by side
python evaluate.py --retriever embedding    # dense retrieval instead of TF-IDF
```

| Criterion | Target | Why |
|---|---|---|
| `unsafe_safe_rate` | **0.0** | A false SAFE is the exact failure the tool exists to prevent |
| `consistency_rate` | ≥ 0.95 | A verdict that changes with wording will not be trusted |
| `accuracy` | ≥ 0.90 | Modal verdict matches the expected one |

`evaluate.py` exits non-zero when the criteria fail, so it also works as a CI
gate.

**Comparing models.** The point of `--compare` is to test the design claim
rather than assert it. If a weaker model loses accuracy and consistency but
the false-SAFE rate stays at zero, the safety property held. Any two
providers can be compared — a hosted 120B open-weights model against a local
8B one, or either against a frontier model.

### Results

Measured with `gpt-oss-120b` on Groq's free tier and TF-IDF retrieval, 10 scenarios x 5 phrasings = 50 runs. Raw per-scenario output is in [`results_final.csv`](results_final.csv).

| Criterion | First full run | Final run | Target |
|---|---|---|---|
| `error_rate` | 0.00 | 0.00 | 0.0 |
| `unsafe_safe_rate` | 0.00 | **0.00** | 0.0 |
| `consistency_rate` | 0.50 | **1.00** | >= 0.95 |
| `accuracy` | 0.90 | **0.90** | >= 0.90 |

Nine of ten scenarios match the expected verdict. The tenth (S09, "use a new AI transcription service") is disputed rather than simply wrong: policy clause AI-3.1 forbids company data in consumer AI services, while AI-5.1 says to file a request with Information Security. The agent answers DO NOT SHARE (the stricter reading); the scenario was labelled CHECK POLICY.

**What the first run taught us.** Consistency of 0.50 looked like model noise. A per-stage diagnostic (`scripts/diagnose.py`) showed the model was extracting correctly and the bug was in our pipeline: clarifying questions were asked before the hard rules ran, so an unmentioned channel could downgrade a clear DO NOT SHARE to CHECK POLICY, and a "volume unknown" answer flipped a SAFE case. Running the rules first and not requiring volume fixed both, with regression tests.

**Caveats.** Ten scenarios is a small test, and the system was adjusted while looking at them, so these figures are likely optimistic on unseen questions. A held-out set is the first planned improvement.



---

## Retrieval: why TF-IDF

The original design specified dense embeddings. We changed it after noticing
that **the retriever never sees the employee's own words.** By the time
retrieval runs, extraction has already normalised the question into enum
values, and the retrieval query is built from those:

```
"credential internal approved_tool single_record"
```

So retrieval matches a controlled vocabulary against clauses tagged with that
same controlled vocabulary. The paraphrase robustness that justifies dense
embeddings was already supplied one stage earlier by the extractor. Dropping
it removed a ~2 GB PyTorch dependency from the default install.

Ranking combines tag agreement, **clause specificity**, and text similarity.
Specificity does the most work: a clause tagged for one data type
(`AUP-4.1`, credentials only) outranks one tagged for all six
(`DCP-1.1`, the general classification scheme). Without it, broad clauses win
on vocabulary overlap and the governing clause never surfaces — in testing,
the credentials query went from *not retrieved at all* to rank 1. This also
mirrors how a lawyer reads a policy: the specific provision controls over the
general one.

Both backends are kept so the choice can be measured. Install
`requirements-embeddings.txt` and run `--retriever embedding` to compare.

---

## Layout

```
policies/              3 policy documents, clause-tagged markdown
cish/schema.py         slots, provisions, verdicts, the safety ordering
cish/directory.py      authorization lookup (SQLite stand-in for LDAP/SCIM)
cish/rules.py          deterministic hard blocks, no model involved
cish/retrieval.py      TF-IDF and embedding backends behind one interface
cish/index.py          corpus parsing; retrieve_policy entry point
cish/llm.py            provider layer: Groq and other hosted APIs, Ollama, Anthropic
cish/agent.py          extract_context, decide, log_decision
app.py                 Streamlit interface
evaluate.py            scenario harness and release criteria
scenarios.yaml         10 scenarios x 5 phrasings
tests/                 50 tests, no API key required
```

## Bugs the tests caught

Worth knowing, because they are the honest "what we learned":

- **SAFE was unreachable.** The rules engine defaulted to CHECK POLICY and the
  final verdict is the more cautious of rules and model, so no question could
  ever return SAFE. Every component passed its own tests; only an end-to-end
  test exposed it. Fixed by making the rules veto-only.
- **Internal-sharing clause approved an external share.** Retrieval matches on
  data type and channel, not recipient. Fixed by limiting SAFE to internal or
  public recipients.
- **Governing clause not retrieved.** Credentials query missed the credentials
  clause because broad clauses won on vocabulary overlap. Fixed with a
  specificity term in the ranking.
- **403 from Groq.** urllib's default User-Agent is rejected by Cloudflare.

## Known limitations

- **Content blind.** The agent evaluates a description, so it cannot detect
  that a file called "meeting notes" actually holds account numbers. Clause
  `DCP-3.1` covers this in policy; the agent cannot enforce it.
- **Corpus bound.** With vague or missing policy the honest answer is
  CHECK POLICY, and the agent returns it often. Correct, but not satisfying.
- **No reranker.** The design specified a cross-encoder reranking pass; the
  current build uses specificity-weighted scoring instead.
- **No enforcement.** An employee can ignore a DO NOT SHARE. The tool
  complements DLP rather than replacing it, though the audit log of "was
  advised not to" has independent value.
- **Deflection rate is untracked.** A tool that defers too often gets
  abandoned. This should be a measured metric with a target.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `CISH_PROVIDER` | `groq` | see the provider table above |
| `GROQ_API_KEY` | — | required for the default provider |
| `CISH_MODEL` | per provider | check the provider's docs for current model IDs |
| `CISH_BASE_URL` | — | required when provider is `custom` |
| `CISH_TIMEOUT` | `180` | seconds; raise for CPU-only local runs |
| `CISH_OLLAMA_MODEL` | `llama3.1:8b` | any model you have pulled |
| `CISH_OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint |
| `CISH_RETRIEVER` | `tfidf` | `tfidf` or `embedding` |
| `CISH_RELEVANCE_FLOOR` | `0.25` | below this, no governing provision is reported |
| `CISH_MAX_RETRIES` | `8` | retries on per-minute rate limits (429), waiting as the provider asks |
| `CISH_MAX_TOKENS` | `700`+ | output budget; reasoning models spend part of it thinking |
| `CISH_REASONING_EFFORT` | `low` for gpt-oss | `low`, `medium` or `high` |
| `CISH_MAX_QUESTIONS` | `20` | questions per visitor session in the app (`0` = unlimited) |
| `CISH_DAILY_CAP` | `400` | questions per day across all visitors (`0` = unlimited) |
| `CISH_MAX_CHARS` | `600` | longest accepted question |
| `CISH_SKIP_MODEL_ON_BLOCK` | off | `1` skips the verdict call when a hard rule already says DO NOT SHARE (fewer calls, same verdict) |

## References

Mireshghallah, N., Kim, H., Zhou, X., Tsvetkov, Y., Sap, M., Shokri, R., &
Choi, Y. (2024). Can LLMs keep a secret? Testing privacy implications of
language models via contextual integrity theory. *ICLR 2024*.

Shao, Y., Li, T., Shi, W., Liu, Y., & Yang, D. (2024). PrivacyLens:
Evaluating privacy norm awareness of language models in action. *NeurIPS
2024, Datasets and Benchmarks Track*.

Rabanser, S., Kapoor, S., Kirgis, P., Liu, K., Utpala, S., & Narayanan, A.
(2026). Towards a science of AI agent reliability. *arXiv preprint*.

## License

MIT — see `LICENSE`. The policy documents in `policies/` are fictional and
written for this project.
