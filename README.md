# Autonomous Data Analyst

Upload a table. It profiles the data, decides what to predict, trains and ranks
models, measures what actually drove the predictions, judges its own output
against thresholds, and — when the data can't support the question — says so
instead of shipping a confident answer.

That last part is the point. Most AutoML demos always produce a result. This
one will tell you your data isn't good enough.

<!-- SCREENSHOT: a completed run page. The trace, SHAP chart and quality
     checks in one frame is the strongest single image this project has. -->
![A completed analysis run](docs/screenshot-run.png)

## Tested against real data

Seven public datasets. Four classes of silent failure found and fixed —
each needing a different check, none catchable by unit tests alone:

| Finding | What exposed it |
|---|---|
| A column restating the target (Titanic `alive` → false perfect score) | Per-feature mutual information |
| A target summing its own features (taxis, R² 0.997 from arithmetic) | Linear reconstruction of the target |
| Missingness misattributed to a column's value (planets) | Explicit missingness indicators |
| Attribution lost in report formatting | Reading the output |

Every fix is documented in `docs/PHASE_NOTES.md` with the measurement that
prompted it.
---

## What it does

```
CSV  ──►  Profile  ──►  Plan  ──►  Train  ──►  Explain  ──►  Reflect  ──►  Report
         (no LLM)      (LLM)     (no LLM)     (no LLM)       (LLM)        (LLM)
                                                   │             │
                                                   └──── retry ◄─┘
```

**Profile** — deterministic statistics: semantic column types, null rates,
cardinality, IQR outliers, skew, correlations, data-quality warnings, and a
ranked shortlist of plausible prediction targets. No model involved, so it is
reproducible and costs nothing.

**Plan** — an LLM reads the profile and chooses a target and task type. Its
choice is validated against the real column list; a hallucinated column falls
back to the deterministic shortlist rather than failing the run.

**Train** — scikit-learn pipelines with per-type imputation, scaling and
one-hot encoding. Two candidate models per task, ranked. Identifier, constant
and free-text columns are excluded automatically, with reasons recorded.

**Explain** — SHAP `TreeExplainer` for ensembles, permutation importance
otherwise. One-hot columns are folded back to their source column, because a
reader wants to know that `contract` mattered, not `cat__contract_Two year`.

**Reflect** — thresholds (not an LLM) decide whether the result is weak. Only
then is a model asked what to do about it, and it may drop inert features,
change target, or abandon. The retry loop is bounded three ways.

**Report** — a plain-language summary grounded in the measured importances,
explicitly forbidden from speculating about causes.

## Run it

Requires Docker and a free API key from
[Google AI Studio](https://aistudio.google.com/apikey) or
[Groq](https://console.groq.com/keys). Neither needs a credit card.

```bash
git clone <your-repo-url> && cd autonomous-data-analyst
cp .env.example .env          # then paste your key into GOOGLE_API_KEY
docker compose up -d --build
docker compose exec api alembic upgrade head
```

Open **http://localhost:5173**. There's a deliberately messy CSV in
`sample_data/` to try it on.

```bash
docker compose exec api pytest       # 115 tests
curl localhost:8000/api/v1/llm/check # confirms the model is reachable
```

## Design decisions

**The LLM is used in three places, not everywhere.** Profiling, training and
explanation are pure Python. Token spend is therefore flat regardless of
dataset size, and the expensive analysis is fully testable without API quota.
Roughly 100 of the 115 tests never make a network call.

**Deterministic layers produce measurements; LLM layers produce judgements.**
Keeping the boundary sharp means a hallucinated plan produces a validation
error rather than a corrupt pipeline.

**Every model suggestion is validated before use.** Target columns are checked
against the real schema, proposed feature exclusions are filtered to columns
that exist, and a revision that would drop every remaining feature is refused.
Each guard has a test that feeds it a bad suggestion.

**The retry loop is bounded three independent ways:** a hard round cap, a
required improvement margin on the *gating* metric, and detection of retries
that change nothing. An unbounded loop is the characteristic failure of
agentic systems, so it is the most heavily tested part of the codebase.

**Grounding beats prohibition.** Early versions invented causes — claiming
missing values had "hindered predictive ability" when the column was noise, or
that 482 rows was "too small" when the sample-size check had passed. Telling
the model not to speculate did not work. Handing it the specific facts it was
speculating about — measured importances, and the list of checks that
*passed* — did.

**No DataFrames in graph state.** LangGraph checkpoints state on every
transition. Datasets and fitted models live on disk; state carries IDs and
paths. The explain step reconstructs its split from a fixed seed rather than
receiving it.

## Stack

| Layer | Choice |
|---|---|
| API | FastAPI, async SQLAlchemy 2.0, Alembic |
| Agents | LangGraph, LangSmith tracing |
| LLM | Gemini, Groq or local Ollama — swappable with one env var |
| ML | scikit-learn, SHAP |
| Data | PostgreSQL 16, Redis 7 |
| UI | React 18, TypeScript, Tailwind v4, Recharts |
| Infra | Docker Compose, GitHub Actions |

Everything runs on free tiers.

## Layout

```
backend/app/
  core/       config, structured logging, LLM provider factory
  db/         models, async session
  services/   profiling, storage, training, explain, quality, analysis
  agents/     state, prompt compression, nodes, graph
  api/        routes
frontend/src/
  api.ts        typed client mirroring the FastAPI models
  components/   RunTrace, Verdict, ImportanceChart
  pages/        Projects, ProjectDetail, RunDetail
sample_data/    a deliberately messy CSV
docs/           design notes and decisions per phase
```

## Known limits

- Analysis runs synchronously in the request. Fine for tens of MB, wrong for
  gigabytes — a background worker is the next piece of work.
- No authentication. `api/deps.get_current_user` is the seam where it lands,
  and it refuses to serve if `ENVIRONMENT=production`.
- Datetime columns are excluded from features rather than engineered.
- Graph checkpointing is in-memory, so an interrupted run cannot resume.

## Notes

`docs/PHASE_NOTES.md` records the reasoning behind each decision, including
the bugs that shaped them: the async lazy-load that only appeared through the
HTTP layer, the semantic type that routed string columns into a median
imputer, and the gate metric that disagreed with the stop condition for two
full rounds before anyone noticed.
