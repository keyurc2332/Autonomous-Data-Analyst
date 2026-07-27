# Phase notes

## Phase 1 — what was built and why

**config.py** uses `pydantic-settings` with `@lru_cache` on the accessor.
Two database URLs are exposed: the async `asyncpg` one for the app, and a
sync `psycopg` one for Alembic. Alembic's autogenerate is much simpler with
a sync driver, and mixing them is a common source of confusion.

**models.py** — the enum type objects (`RunStatusType`, etc.) are module-level
singletons on purpose. Constructing `Enum(RunStatus, name="run_status")` twice
makes Alembic emit two `CREATE TYPE run_status` statements and the migration
fails on the second. `Report.status` and `AgentRun.status` share one object.

`AgentRun.parent_run_id` is self-referential so a graph run can own its node
runs, giving you a run tree that mirrors the LangGraph execution.

**llm.py** — `get_llm` is cached, and imports the provider package lazily
inside each builder. Lazy import means a missing `langchain-groq` doesn't
break startup when you're using Google.

**Deliberately deferred:** authentication (Phase 2 — `User` table exists but
there is no auth flow yet), the Python executor sandbox (Phase 4 — the
hardest security problem in the project), and any LangGraph code (Phase 3).

## Open decisions for Phase 2

1. Upload validation: reject on size, extension, and a pandas dry-parse of the
   first N rows before accepting the file.
2. Content hashing for dedupe — `Dataset.content_hash` is already there.
3. Whether profiling runs synchronously (fine up to a few MB) or goes straight
   to the worker. Recommend sync first, move it later if it hurts.

---

## Phase 2 — dataset service

**No LLM anywhere in this layer.** Agents produce opinions; this layer produces
measurements. Keeping them separate means the profiler is fully testable
without API quota, and the agents in Phase 3 get facts they did not invent.

### The JSON coercion trap (`profiling._py`)

The single most likely thing to break a `profile` write. Two distinct problems
share one fix:

1. `np.int64` / `np.float64` are not JSON-serialisable. `json.dumps` raises
   `TypeError`, and asyncpg rejects them for a JSONB parameter.
2. `NaN` and `±Inf` serialise to the literals `NaN` / `Infinity`, which are
   invalid JSON. Postgres JSONB refuses them outright.

Every value entering the profile passes through `_py()`. `test_profiling.py`
asserts the whole profile survives `json.dumps(..., allow_nan=False)`.

### Why profiling runs in a threadpool

pandas is synchronous and CPU-bound. Calling `profile_file` directly from an
async handler blocks the event loop, stalling every other request for the
duration. `anyio.to_thread.run_sync` moves it off the loop. This is a stopgap:
Phase 4 moves profiling into the arq worker alongside training.

### Identifier detection uses a ratio, not equality

The first version required `n_unique == n_rows`. A single duplicated row then
reclassified a primary key as a numeric feature, and it appeared in the
target-candidate list. `IDENTIFIER_UNIQUE_RATIO = 0.98` fixes it, and
`test_identifier_survives_duplicate_rows` pins the behaviour.

### Path traversal

`storage.resolve()` checks containment with `is_relative_to`. Without it a
crafted `storage_path` like `../../etc/passwd` escapes the storage root.
Covered by `test_path_traversal_is_blocked`.

### Uploads are streamed, never read whole

`await upload.read()` with no argument materialises the entire file in memory,
so one large upload can kill the process regardless of `MAX_UPLOAD_MB`. We read
in 1 MiB chunks, hashing as we go, and abort the moment the limit is crossed.
Partial files are unlinked on any failure.

### `api/deps.get_current_user` is the auth seam

It resolves a single development user, created on first use, and raises if
`ENVIRONMENT=production`. Every route already depends on a `User`, so adding
real JWT auth later changes this one function and nothing else.

### Known data trap, deferred to the Cleaning agent

pandas reads the literal string `None` in a CSV as `NaN`. A categorical column
with a legitimate "None" category silently becomes binary-plus-nulls. The
Cleaning agent needs an explicit `keep_default_na` strategy per column; the
profiler currently reports what pandas saw, which is the honest answer.

## Open decisions for Phase 3

1. Where the LangGraph checkpointer lives — Postgres (durable, replayable) or
   Redis (fast, ephemeral). Postgres is probably right given `agent_runs`.
2. Whether the Planner picks the target or the user confirms it. The profile
   already ships ranked `target_candidates` to support either.
3. Streaming: SSE is simpler than WebSockets and sufficient for one-way
   progress updates.

---

## Phase 3 — LangGraph vertical slice

    planner --> training --> summary --> END
       |            |
       +------------+--> END (on error)

### The layer boundary that matters

The Planner decides **what** to model. `services/training.py` decides **how**
and does it. Nothing in the training layer touches an LLM, so it is fully
unit-testable without API quota, and a hallucinated plan produces a validation
error rather than a corrupt pipeline.

### No DataFrames in graph state

`agents/state.py` carries IDs, a relative storage path, and the profile dict.
LangGraph serialises state on every node transition; a DataFrame in state
means re-serialising the dataset repeatedly and unusable checkpoints. Nodes
load what they need and discard it.

### The hallucination guard

`planner_node` validates `target_column` against the actual column set before
anything downstream sees it. On a mismatch it falls back to the deterministic
top `target_candidate` rather than failing the run, and records
`planner:corrected` in `steps` so the substitution is visible rather than
silent. Same for a dead provider (`planner:fallback`).

This is why Phase 2 shipped `target_candidates`: the heuristic is the safety
net under the model.

### Prompt compression

`agents/prompting.summarise_profile` renders the profile down to a compact
digest. A raw profile for a 60-column dataset is tens of thousands of tokens;
on a rate-limited free tier that is the difference between working and a 429
on every call. A test asserts the digest stays under 8000 characters.

### Failure policy per node

- Planner fails -> fall back to the heuristic target, continue.
- Training fails -> end the run. There is nothing to summarise.
- Summary fails -> synthesise a plain-text summary from the metrics and
  succeed anyway. The numbers are the deliverable; the narrative is cosmetic.

### Bug found by the graph tests

`select_features` routed every `binary` column to the numeric pipeline. But
"binary" covers a 0/1 integer column *and* a two-value string column like
"A"/"B" -- the latter reached `SimpleImputer(strategy="median")` as strings and
every model failed to fit. Routing now keys off the recorded dtype, not the
semantic label. The sample data hid this because `churn` was already integer.

### Checkpointer: MemorySaver, deliberately

A durable Postgres checkpointer only earns its keep once runs are long-lived
and resumable, which arrives with the arq worker in Phase 4. Results are
already durable -- `agent_runs` and `experiments` are written by
`services/analysis.py` independently of the graph's own checkpoint. Phase 3
also adds no new dependencies, so no rebuild is needed.

### `GET /llm/check`

One trivial LLM call, reporting provider, model, latency, and the exact
exception type on failure. It isolates credential problems from graph
problems: when a run misbehaves you know immediately which half to look at.

## Open decisions for Phase 4

1. arq worker + job ids; profiling and training both move off the request path.
2. Postgres checkpointer, once runs can be interrupted and resumed.
3. SHAP: `TreeExplainer` for the forest, sampled to a few thousand rows.
4. Whether the Reflection agent re-plans on a weak result, and its stop condition.

---

## Phase 4a — explainability

    planner -> training -> explain -> summary -> END

### Why explain before summary

The Phase 3 summary claimed missing satisfaction values "likely hindered the
model's predictive ability". That column is random noise in the sample data;
its missingness hindered nothing. The model reached past its evidence into
plausible-sounding causation.

Two fixes, both needed:

1. The summary node now receives **measured** importances and is told that
   any column not listed did not meaningfully drive predictions.
2. The prompt explicitly forbids explaining *why* a result is weak, and
   forbids claiming a column helped or harmed unless the figures show it.

Grounding beats instruction on its own. Telling a model not to speculate
works far better when you also hand it the facts it was speculating about.

### Two tiers, chosen automatically

- **SHAP TreeExplainer** for tree ensembles: exact and fast.
- **Permutation importance** otherwise: model-agnostic, sklearn-only,
  measured on held-out data rather than training impurity.

If `shap` is ever missing or fails, the code degrades to permutation
importance rather than failing. `test_broken_pipeline_degrades_instead_of_raising`
pins that, and it caught a real gap -- the estimator probe originally sat
outside the try block and could raise.

### One-hot aggregation

Importances are summed back to source columns: a stakeholder wants to know
that `contract` matters, not `cat__contract_Month-to-month`. Matching is
longest-name-first so `plan_type_A` resolves to `plan_type`, not `plan`.

### Models are artifacts now

A fitted Pipeline cannot go in graph state -- not serialisable, and state is
checkpointed on every transition. The training node writes it to
`storage/models/{run_id}/{model}.joblib` and passes the path, the same
discipline datasets already follow. This finally populates
`Experiment.artifact_path`.

### Reproducing the split without carrying DataFrames

`prepare_split()` is seeded, so the explain node reconstructs byte-identical
train/test sets rather than receiving them through state. Determinism is what
makes that safe, and there is a test asserting repeated runs match.

## Remaining

- [ ] Reflection agent: re-plan when the result is weak, now that it has
      measured importances to reason about rather than guesses.
- [ ] arq worker + job ids; profiling and training off the request path.
- [ ] Postgres checkpointer, once runs are interruptible.
- [ ] React frontend, SSE progress.
- [ ] PDF report.

---

## Phase 4b — reflection loop

                        +--------------- retry ---------------+
                        v                                     |
        planner -> training -> explain -> reflect -------------+
            |          |                     |
            |          |                     +--> summary -> END
            +----------+--> END (on error)

### The decision to reflect is deterministic

`services/quality.assess()` applies thresholds -- ROC-AUC below 0.65, F1
below 0.60, R2 below 0.30, fewer than 200 training rows. Only when it returns
"weak" is the LLM consulted, and then only about *what to do*, never about
whether something is wrong.

Two reasons. Asking a model "is this good?" is non-reproducible. And a good
run should cost zero extra tokens; `test_strong_result_never_calls_the_reflection_llm`
asserts exactly that.

### The loop is bounded three ways

Any one of these ends it:

1. **Round cap** -- `MAX_REFLECTION_ROUNDS = 2`, so at most three attempts.
2. **Improvement margin** -- a retry must beat the incumbent by 0.02. Without
   a margin the loop churns on noise-level differences.
3. **No-op detection** -- a retry proposing no actual change is converted to
   accept, because it would re-run at the identical score forever.

Plus the revision guard rejects unknown columns and refuses to exclude every
remaining feature, which would make the next round raise rather than retry.

Each has a test. An unbounded retry loop is the real failure mode of an
agentic system, and it is the thing worth over-testing.

### Abandon is a first-class outcome

The prompt explicitly prefers abandoning over a retry the model does not
expect to help, and says so: "a weak result honestly reported is more useful
than three rounds of churn". An agent that admits the data cannot answer the
question is more valuable than one that keeps trying, and harder to build.

### Retries are visible in the schema

Each training round becomes a child `AgentRun` linked by `parent_run_id` --
the self-referential column defined in Phase 1 for precisely this. The run
tree now mirrors the graph's real execution, retries included.

### A real behaviour observed while testing

On the noise dataset, round 1 scored 0.5247 and round 2 scored 0.5374 after
dropping a feature. The improvement stop fired -- 0.0127 is below the 0.02
margin -- and the loop ended rather than continuing to chase noise. That is
the mechanism doing its job on data it had never seen.

## Remaining

- [ ] arq worker + job ids; profiling and training off the request path.
- [ ] Postgres checkpointer, once runs are interruptible.
- [ ] React frontend, SSE progress.
- [ ] PDF report.

---

## Phase 5 — the interface

### Design brief

The distinctive thing about this system is not the modelling, it is that it
**shows its working**: the step trace, which checks passed and which failed,
the retry that did not help, the honest verdict. So the interface is built as
an instrument readout rather than a dashboard.

- **Palette is semantic, not decorative.** Teal = verified, amber = weak,
  rose = failed. This application exists to issue verdicts; colour carries
  meaning or it is not used.
- **Type**: IBM Plex Sans for structure, IBM Plex Mono for every measured
  value. A number the system computed should never look like prose the system
  wrote, so all metrics, column names and step labels are monospaced.
- **Signature element**: the run trace. The retry is a genuine loop in the
  graph, so it is drawn as a loop with a returning bracket. Flattening it to a
  straight list would misrepresent execution. Rounds are numbered because the
  order carries information, not for decoration.

### Quality checks show passes as well as failures

The same reason the reflection prompt does: stating only what is broken
invites invented causes. A reader deserves to see that the sample-size check
passed, not just that ROC-AUC failed.

### Dev-time proxy instead of CORS juggling

Vite proxies `/api` to the `api` service, so the browser makes same-origin
requests. No base URL in the client, and CORS config stops mattering in
development.

### Two container gotchas worth remembering

1. The `web` service mounts `./frontend` for hot reload, which would shadow
   the image's `node_modules`. An anonymous volume at `/app/node_modules`
   preserves them; without it vite will not start on a fresh clone.
2. `usePolling` is set in the vite watcher. Filesystem events do not
   propagate reliably through Docker bind mounts on Windows, so hot reload
   silently stops working without it.

## Remaining

- [ ] arq worker + job ids; training off the request path, SSE progress.
- [ ] Postgres checkpointer, once runs are interruptible.
- [ ] PDF report.

---

## Field testing on real datasets

Four public datasets, chosen to stress different paths. Two real bugs and one
architectural limit surfaced in about ten minutes -- a better return than any
amount of additional feature work.

### titanic: target leakage, reported as "Strong"

Seaborn's Titanic carries both `survived` (0/1) and `alive` (yes/no) -- the
same variable twice. The model used `alive`, scored a perfect 1.0000, and the
quality check called it **strong**. A system that confidently reports a
meaningless model is worse than one that fails loudly.

Pearson correlation could not see it: both columns are categorical, and the
profiler's collinearity check only covers numeric pairs. The fix is normalised
mutual information between each feature and the target, which is type-agnostic.
Above 0.95 the feature is **removed before training**, not merely flagged, and
the run page shows it above the summary.

Titanic now scores ROC-AUC 0.866 -- an honest model where there was a perfect
and worthless one.

### titanic: booleans were being thrown away

`adult_male` and `alone` were dropped as `unhandled type 'boolean'`.
`select_features` had no branch for them. After the fix `adult_male` is the
single strongest predictor, which is exactly what the historical record says.

### tips: worked as designed

R2 0.4387 passed the variance threshold, the sample-size check failed at 195
rows, verdict weak, and reflection abandoned with the right reasoning: adding
data would help more than any model change. No intervention needed.

### diamonds: one heavy run took the whole API offline

Not slow -- unresponsive. Listing projects timed out and the UI rendered "No
projects yet", which reads as data loss.

Cause: `RandomForest` with unbounded depth on 16,000 rows grows ~3.7M nodes at
depth 39. SHAP `TreeExplainer` scales with **leaves squared**, so 50 rows took
88 seconds and the 500-row cap implied roughly 15 minutes. That work is
CPU-bound Python, so despite running in a worker thread the GIL starved the
event loop.

Measured trade-off:

| max_depth | nodes | R2 | SHAP (100 rows) |
|---|---|---|---|
| None | 3,690,590 | 0.9793 | > 10 min |
| 20 | 3,049,454 | 0.9794 | > 10 min |
| 16 | 1,874,394 | 0.9794 | > 10 min |
| **12** | **653,372** | **0.9780** | **12s** |

Three changes:

1. `MAX_TREE_DEPTH = 12`, and fewer trees above 10k rows. Unbounded depth was
   overfitting anyway; 0.0013 of R2 for a 5.6x smaller model is an obvious trade.
2. SHAP rows are **budgeted against node count** rather than fixed, with a hard
   ceiling above which permutation importance is used instead. Cost has to be
   predicted, not discovered.
3. Tables above 20,000 rows are sampled, with the sampling shown in the UI.

Diamonds end to end: **20 seconds, R2 0.9779**, down from over 15 minutes.

This is the background worker justified by evidence rather than assertion. The
sampling and depth caps are the right defaults regardless, but a heavy job
still belongs off the request path.

### Orphaned runs

A run row is written before the graph executes, so a crash mid-run leaves
`status=running` forever. Startup now closes out any survivor -- nothing can
legitimately be in flight when the process has just started.

---

## Cleaning

The profiler had been reporting 107 duplicate rows in Titanic and 146 in
diamonds, and then models trained on them anyway. Detection without action.

`services/cleaning.py` runs as the **first graph node**, before planning, so
the planner sees the data that will actually be modelled. It writes a cleaned
copy to `storage/cleaned/{run_id}.csv`; the original upload is never modified.

Four rules, applied in this order because each depends on the last:

1. **Trim whitespace** — `' Male'` and `'Male'` are one category, not two.
2. **Normalise placeholders** — `N/A`, `unknown`, `-`, `?` become real nulls.
   Left as text they are learned as categories and the null rate reads as zero.
3. **Drop empty columns** — only detectable after step 2.
4. **Drop exact duplicates** — they inflate apparent sample size and can place
   identical rows in both train and test, overstating every score.

No LLM. Every rule has a correct answer, and each action is recorded with its
reason so the report can state exactly what changed. Imputation stays in the
sklearn pipeline, where it is fitted on training data only; doing it here would
leak test-set statistics.

### A pandas 3.0 trap

The first version checked `dtype != object` to find text columns. **pandas 3.0
made `str` the default dtype for text** where 2.x used `object`, so the check
skipped every text column and cleaning silently did nothing at all — no error,
no warning, just no effect. Detection is now by exclusion of numeric, datetime,
bool and timedelta types.

Worth generalising: a type check written against one library version can fail
open rather than loud. `test_text_columns_detected_regardless_of_pandas_dtype`
pins all three dtype spellings.

---

## Chat with dataset

The one place in the system where the model genuinely **chooses** what to do.
Everywhere else the graph fixes the sequence; here the model decides which
tools to call, in what order, and when it has enough to answer.

### There is deliberately no "run this code" tool

A model that can execute arbitrary Python or SQL against a user's data is a
much larger security problem than this feature is worth. Instead there is a
fixed vocabulary of seven verified operations — overview, describe column, top
values, aggregate, count rows, correlation, latest analysis — each with a
Pydantic schema the model must satisfy.

`test_hallucinated_tool_cannot_execute` feeds it a call to `ExecutePython`
with `rm -rf /` as the argument and asserts the dispatcher refuses it.

### Tool errors go back to the model, not up the stack

A bad column name returns `{"error": "There is no column called 'profit'.
Available columns: ..."}`, which the model reads and recovers from. Errors are
written to be *useful to their reader*, which happens to be a language model.
Column names are also matched case-insensitively, because models routinely
lowercase them.

### The loop is bounded

Four tool rounds maximum. On exhaustion the model is asked to answer from what
it has rather than returning nothing. Same discipline as the reflection loop:
an unbounded agent loop is the characteristic failure of these systems.

### Grounding, again

The system prompt says: never state a number you have not obtained from a
tool. This is the fourth place in the project where the fix for confident
invention was to remove the need for it — the model does not estimate because
it has a cheap way to know.

### `ConversationMessage` finally used

Defined in Phase 1, empty until now. History is replayed into each turn, so
follow-up questions work.

---

## Executive PDF report

`GET /api/v1/analysis/{run_id}/report.pdf`

### Ordering is the design

Verdict first, then what the model found, then how far to trust it, then how it
was produced, then limits. A report that buries its caveats behind its numbers
is worse than no report, because it lends unearned confidence to whatever the
reader saw first. `test_verdict_appears_before_any_metric` asserts the ordering
holds by checking character positions in the extracted text.

A weak verdict states plainly: "This model is not reliable enough for
decisions." Leakage, when present, appears before the methodology rather than
in a footnote.

### ReportLab, not matplotlib

Charts use ReportLab's own `HorizontalBarChart`. That avoids a heavy dependency
and an image round-trip, and keeps the whole report vector.

Two ReportLab traps worth recording:

1. **Plain strings in table cells do not wrap.** They run past the column and
   are clipped at the page margin. Anything potentially long must be a
   `Paragraph`. This was visible only on rendering the page to an image —
   text extraction showed the content as present, because it *was* present,
   just drawn outside the visible area.
2. Unicode subscripts and superscripts render as black boxes in the built-in
   fonts. Avoided entirely.

### Every value is printed on its bar

A reader should not have to estimate a magnitude from bar length.

## Remaining

- [ ] Background worker; training off the request path.
- [ ] Datetime feature engineering — date columns are currently dropped.
- [ ] Group collinear features before ranking importance. On diamonds, `y`
      outranked `carat` and `x` was called inert, though all three measure the
      same thing. The profiler detects the collinearity; nothing downstream
      uses it.

---

## Second field test: additive leakage

Three more public datasets (seaborn taxis, mpg, planets). One found a class of
bug the first round could not.

### taxis: the target was a sum of its own features

`total` predicted at R2 **0.9969**, top drivers `fare`, `tip`, `tolls`. In a
taxi dataset `total` *is* fare + tip + tolls. The model learned addition.

Both existing defences missed it:

- **Per-feature mutual information** checks columns one at a time. No single
  component restates `total`, so every check passed.
- **The implausibility warning** fired only at 0.999. 0.9969 slipped under.

The fix is a linear reconstruction test: fit a linear model of the target on
all numeric features. If it reconstructs the target almost exactly, the
relationship is arithmetic, not predictive. Measured separation:

| dataset | linear R2 | verdict |
|---|---|---|
| taxis (`total` = fare + tip + tolls) | 0.994 | **derived** |
| diamonds (`price`) | 0.854 | legitimate |
| mpg | 0.824 | legitimate |
| tips | 0.481 | legitimate |

Threshold 0.99, and the coefficients are reported: values near 1.0 are the
signature of a sum. The implausibility net was also tightened from 0.999 to
0.98 as a backstop, diamonds at 0.978 being the highest legitimate score
observed.

**Reported, not auto-removed.** Unlike a single restated column, it is
ambiguous which component should go — that is the user's decision. The quality
gate marks the run weak so reflection can propose an exclusion.

### The general lesson

The first leakage fix was per-feature; this one is combinatorial. Both were
found by running real data through and asking whether a good-looking score was
believable. **A high score is a hypothesis about the model; it can equally be
a hypothesis about the data.**

### Two gaps confirmed, not fixed

- `pickup` and `dropoff` datetimes were dropped as unhandled, as were
  `pickup_zone` and `dropoff_zone` at 194 and 203 categories.
- mpg's `name` column ("chevrolet chevelle malibu") was dropped at 305
  categories, discarding the manufacturer, which is real signal. Extracting a
  first token would recover it.

Both are feature-engineering gaps rather than defects.

---

## Informative missingness

**planets** came back `strong` at f1 0.939 with `mass` ranked second — a column
the profiler had flagged as 50.4% missing. A half-imputed column driving
predictions is worth interrogating.

Measured missing rate of `mass` by detection method:

| method | missing | n |
|---|---|---|
| Imaging | 100% | 38 |
| Microlensing | 100% | 23 |
| Transit | 99.7% | 397 |
| Radial Velocity | 7.8% | 553 |

`NMI(mass_is_missing, method) = 0.62`. "Was mass measured?" almost perfectly
separates the two largest classes. Median imputation fills half the rows with
an identical value, the model learns *that*, and SHAP attributes the effect to
the column's **value** — when the real signal is its **absence**.

Not strictly leakage: in deployment you would also know whether mass had been
measured. But the attribution was wrong, and a report that says "mass drove
predictions" when it means "whether mass was recorded drove predictions" is
misleading in a way that matters.

Fix: `SimpleImputer(add_indicator=True)`. The model receives an explicit
indicator, SHAP scores it separately, and `_source_column` deliberately does
**not** merge it back — "the value of mass" and "whether mass was recorded"
are different facts, and merging them would hide the thing the indicator was
added to reveal. A non-gating quality check fires when such an indicator ranks
in the top five, warning that the model will not transfer if missingness
reflects collection method rather than the subject.

This is also better modelling: the indicator is a real feature, and titanic's
f1 rose slightly once `age`'s missingness became explicit.

**Known limitation.** Only SHAP surfaces the indicator, because it scores
encoded features. Permutation importance shuffles the original column with its
missingness included and attributes the whole effect to the parent — which is
defensible, but means the check fires for tree models and not linear ones. A
test writing the assertion against `best_model` exposed this, since logistic
regression won on perfectly separable data.

### The pattern across all three leakage findings

Each was found by asking whether a good-looking number was *believable*, and
each needed a different check:

1. `alive` restating `survived` — per-feature mutual information.
2. `total` summing fare + tip + tolls — linear reconstruction of the target.
3. `mass` standing in for its own absence — an explicit missingness indicator.

No single test would have caught all three. What generalises is not the checks
but the habit: **treat a high score as a hypothesis about the data, not a
result.**


### The indicator worked, then the summary undid it

With indicators enabled, planets ranked `mass (was missing)` first — and the
summary reported that **"mass" was the strongest driver**. The exact
misattribution the indicator existed to prevent, reintroduced one layer later.

Cause: facts were formatted as `mass (was missing) (0.0334)`. Two bracketed
groups, and the model read the first as the column name. Now names are quoted
(`"mass (was missing)" = 0.0334`), the prompt instructs that they be used
verbatim, and when any indicator ranks in the top five an explicit note states
that the two are different findings.

Worth noting as a class of bug: a correct fix at one layer can be silently
reversed by formatting at the next. The data was right; the presentation of it
to the model was not.

---

## CI packaging failure

`pip install -e ".[dev]"` failed in GitHub Actions with:

> Multiple top-level packages discovered in a flat-layout: ['app', 'alembic']

setuptools saw `app/`, `alembic/` and `tests/` side by side and refused to
guess which was the distribution.

**Docker hid it.** The Dockerfile copies `pyproject.toml` before the source so
dependency installation caches across builds — which means nothing was present
to be ambiguous when pip ran. CI checks out the whole tree first and fails
immediately. A build that only works because of the order files happen to
arrive is a build that works by accident.

Fixed with an explicit `[build-system]` block and:

```toml
[tool.setuptools.packages.find]
include = ["app*"]
```

The lint step was also failing on accumulated warnings. Eight were real and
fixed: a nested conditional, a `try/except/pass` that should be
`contextlib.suppress`, an unused local, and several over-long lines.

`UP042` (`class X(str, Enum)` should be `StrEnum`) is exempted with a stated
reason rather than silenced: SQLAlchemy and Pydantic both rely on the member
values being plain strings, and `StrEnum` changes what `str(member)` returns,
which several comparisons depend on. An ignore with a justification is a
decision; an ignore without one is a shrug.


### The linter was unpinned

The next CI run failed on `UP017` — a rule that had not existed when the code
was written. `ruff>=0.8.0` resolves to the newest release at run time, so a
new ruff version adds rules and the build breaks with no change on our side.
The same code passed locally on ruff 0.16.0 and failed in CI minutes later.

Pinned to `ruff==0.16.0`. A linter is a build input and gets an exact version
like any other dependency; "latest" is not a version.

The flagged code was changed anyway, since `datetime.UTC` is the better
spelling on Python 3.11+.


### And then a packaging mistake of my own

CI failed again on `UP037` in `app/db/models.py`. The pin was correct and ruff
was the right version; the file simply had never received the fix.

`ruff check --fix` had already corrected it locally as part of a bulk autofix,
but the change set shipped afterwards listed only the files edited by hand.
The autofix touched files nobody was tracking.

The lesson is about change sets, not linting: **after a bulk automated edit,
ship the whole directory, not the files you remember touching.** `git status`
would have shown it immediately — a subset assembled from memory will not.

Removing the quotes is safe here: `from __future__ import annotations` already
makes every annotation a string at runtime, and SQLAlchemy resolves
relationship targets lazily at mapper-configuration time. Verified by calling
`configure_mappers()` explicitly rather than assuming.
