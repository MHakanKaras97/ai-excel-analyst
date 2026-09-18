# AI Excel Analyst

An AI-powered Excel analytics and executive reporting application.

## Project Status

🚧 In development — V0.1 through V0.7 (multi-file comparison, Q&A, and
charts) are complete.

## Goal

The goal of this project is to build an application that can:

- Read Excel and CSV files
- Validate and clean data
- Calculate KPIs and statistics
- Detect trends and anomalies
- Generate data visualizations
- Use an LLM to interpret analytical results
- Answer natural-language questions about the data, grounded in deterministic results
- Generate executive summaries
- Generate PowerPoint reports

## Planned Technology

- Python
- Pandas
- NumPy
- OpenPyXL
- Plotly
- Streamlit
- LLM API
- python-pptx
- Pytest
- Git / GitHub

## Architecture

High-level pipeline:

```
Excel(s)
 ↓
Data Understanding + Normalization (per file)
 ↓
Deterministic Analytics + Anomaly Detection (per file)
 ↓
AI Analyst Core (interprets only — never calculates)
 ↓
Charts / Q&A / Insights
 ↓
Multi-file / Report / PPT
```

### Q&A Architecture (V0.5 — complete)

```
User Question
 ↓
Q&A Interpreter (ONE LLM call)
 ↓
Structured Intent
 ↓
Deterministic Q&A Engine
 ↓
Grounded Result
 ↓
Deterministic Answer
 ↓
Streamlit
```

**Architectural principle:** the LLM interprets the question. Python resolves,
calculates/looks up, grounds, and answers. The LLM does not generate numeric
answers.

### V0.5 Q&A Progress

- [x] Prompt builder (`src/qa_prompt_builder.py`)
- [x] Structured intent interpreter (`src/qa_interpreter.py`)
- [x] Deterministic column resolution (`src/qa_engine.py`)
- [x] Deterministic period resolution (`src/qa_engine.py`)
- [x] Intent dispatch (`src/qa_engine.py`)
- [x] Grounded result generation (`src/qa_engine.py`)
- [x] Deterministic answer templates (`src/qa_answer.py`)
- [x] End-to-end integration test (`tests/test_qa_pipeline.py`)
- [x] Streamlit Q&A integration (`app.py`)

### Supported Q&A Intents

The Q&A interpreter classifies each question into exactly one of:

- `period_value`
- `period_extremum`
- `period_change`
- `column_stat`
- `anomaly_check`
- `missing_values`
- `unsupported`

The interpreter makes exactly one LLM call per question and returns only a
structured intent (intent, metric, and free-text hints) — it does not answer
the question or compute any value itself.

### Deterministic Resolution

Once the LLM returns a structured intent, hint resolution against the actual
dataset is entirely deterministic (`src/qa_engine.py`):

- Case-insensitive exact column matching
- Conservative fuzzy column matching using Python's stdlib `difflib` (no
  external fuzzy-matching dependency)
- Explicit `ambiguous_column` handling — multiple plausible matches are
  surfaced, never silently guessed
- Exact period matching (`YYYY-MM`)
- Month/year normalization (e.g. "March 2024", "Mar 2024", "03 2024")
- Explicit `ambiguous_period` handling — a bare month matching multiple years
  is surfaced, never defaulted to the latest or first match
- No silent guessing at any resolution step

### Chart Architecture (V0.6 — complete)

```
Chart Request
 ↓
Chart Intent Interpreter (ONE LLM call)
 ↓
Structured Intent
 ↓
Deterministic Chart Engine (qa_chart_engine.py)
 ↓
Chart Specification
 ↓
Chart Builder (chart_builder.py) — existing single-file chart functions
 ↓
Plotly Figure
 ↓
Streamlit
```

The LLM classifies the chart request and extracts hints only; the chart
data itself is always read from already-computed deterministic analytics.
The LLM never generates Plotly code or chart data.

### Multi-File Architecture (V0.7 — complete)

V0.7 adds an **additive** multi-file layer on top of the existing
single-file pipeline. Each uploaded file becomes an independent `DataFile`
record (`file_id`, `filename`, `display_name`, `role`, `raw_df`,
`value_column`, `period_column`, `analysis`), analyzed with the same
deterministic `analyze_file()` used for the single-file dashboard — no
separate multi-file analytics engine exists.

```
File A, File B (each already analyzed independently via analyze_file())
 ↓
Deterministic File Resolution (role or display name — exact match only)
 ↓
Deterministic Multi-File Comparison Engine (multi_file_comparison.py)
 ↓
Comparison Result (fully-keyed, JSON-safe)
 ↓
Deterministic Answer / Chart Spec / AI Comparison Insight
```

**File resolution is exact-match only** — by an explicitly assigned role
(`previous`, `current`, `reference`, `comparison`) or by exact display
name. There is no fuzzy matching, no inference from upload order, and the
internal `file_id` is never sent to the LLM — only roles and display names
are, as plain contextual labels (mirroring how column names are already
passed as untrusted context, never resolved by the LLM).

**Supported comparison types** (`multi_file_comparison.py`):

- `file_value_comparison` — previous/current scalar value + absolute/percentage change
- `file_period_comparison` — same, for a specific period
- `file_trend_comparison` — trend label per file (not forced into a scalar change)
- `file_anomaly_comparison` — anomaly counts/details per file (not forced into a scalar change)

Each comparison type returns the fields that actually make sense for it;
type-specific detail lives in an `extra` dict rather than distorting the
common previous/current/absolute_change/percentage_change shape.

#### Multi-File Q&A (V0.7.5)

```
Question → Multi-File Q&A Interpreter (ONE LLM call) → Structured Intent
 → Deterministic Multi-File Comparison Engine → Comparison Result
 → Deterministic Answer
```

The interpreter (`multi_file_qa_interpreter.py`) classifies a question into
`file_value_comparison` / `file_period_comparison` / `file_trend_comparison`
/ `file_anomaly_comparison` / `unsupported` and extracts only free-text
hints (`column_hint`, `metric`, `period_hint`, `from_file_hint`,
`to_file_hint`) — never a value, a change, or a file identity. All
resolution and arithmetic happen afterward, deterministically.

#### Multi-File Charts (V0.7.6)

```
Question → Chart Intent Interpreter (ONE LLM call) → Structured Intent
 → Deterministic File Resolution → Deterministic Comparison
 → Chart Specification → Existing Chart Builder → Plotly Figure → Streamlit
```

V0.7 supports exactly one multi-file chart type: `file_comparison_chart`, a
previous-vs-current bar chart for a selected column/metric. It reuses the
existing single-file chart builder/renderer verbatim by reshaping the
comparison into the same `numeric_summary` chart specification shape — no
new chart-rendering code was added. Trend charts, anomaly visualizations,
row-diffing, and auto chart recommendation across files are not part of
V0.7 (see Limitations).

#### AI Comparison Insight (V0.7.4)

Reuses the existing `AIProvider`/`interpret()` machinery with an alternate
prompt builder (`comparison_prompt_builder.py`) and a dedicated payload
builder (`comparison_ai_payload.py`) that sends only the deterministic
Comparison Result — never raw DataFrames or workbook contents. The same
numeric-hallucination grounding used for single-file insights applies here
unchanged.

#### Streamlit Integration (V0.7.7)

- Multiple files can be uploaded at once; each gets an independent
  `file_id` (never derived from filename, so duplicate filenames don't
  collide).
- With 2+ files uploaded, a role-assignment table appears (assign
  `previous`/`current`/`reference`/`comparison` per file) alongside
  per-file "Remove" and a "Clear All Files" action.
- Per-file results (AI insight, Q&A answer, chart spec) are namespaced by
  `file_id` in `session_state`, so switching files or removing one never
  leaks a stale result onto a different file.
- Removing or clearing files also invalidates any stored multi-file
  comparison state, since a stored comparison may have involved the
  removed file.
- File uploads are tracked by a persistent "seen upload id" set (not just
  currently-registered files) so that removing or clearing a file does not
  cause it to silently reappear as a new `DataFile` on the very next
  Streamlit rerun (the `file_uploader` widget otherwise keeps holding the
  same bytes).

## Testing

Full test suite: **665 passed, 2 deselected** (a `pytest.ini` marker
excludes network-dependent Gemini integration tests by default).

Focused test suites:

| Module | Tests |
|---|---|
| V0.5 Q&A (`qa_prompt_builder`, `qa_interpreter`, `qa_engine`, `qa_answer`, pipeline, Streamlit) | 120 passed |
| V0.6 Charts (chart intent, `qa_chart_engine`, `qa_chart_renderer`, pipeline, Streamlit) | 143 passed |
| V0.7.1-2 `file_analysis` (per-file extraction) | 21 passed |
| V0.7.3 `multi_file_comparison` (deterministic comparison engine) | 50 passed |
| V0.7.4 AI comparison insight (`comparison_ai_payload`, `comparison_prompt_builder`, `ai_interpreter` extensions) | 56 passed |
| V0.7.5 Multi-file Q&A (`multi_file_qa_prompt_builder`, `multi_file_qa_interpreter`, `qa_answer` comparison answers) | 32 passed |
| V0.7.6 Multi-file charts (`multi_file_chart_intent_interpreter`, `build_multi_file_chart_spec`) | 20 passed |
| V0.7.1/7/8 Multi-file Streamlit (AppTest: registration, roles, remove/clear, comparison Q&A/chart/insight, stale-result prevention) | 28 passed |

Run the full suite from the repo root with `pytest`.

## Principles

- Deterministic calculations are performed by Python, not the LLM
- The LLM is used for interpretation, not calculation
- Ambiguity is surfaced rather than silently guessed
- Existing working modules are reused rather than unnecessarily refactored
- Provider abstraction (`AIProvider`) keeps the AI layer replaceable
- Tests are provider/network independent wherever possible

## Limitations

**Multi-file comparison (V0.7) is intentionally narrow:**

- Comparisons are strictly **pairwise** — exactly two files per comparison.
  There is no path to compare three or more files at once.
- File resolution is **exact match only** on an explicitly assigned role or
  exact display name. There is no fuzzy file matching and no automatic
  role inference (e.g. from upload order, filename patterns, or dates).
- Only `file_comparison_chart` (a previous-vs-current scalar bar chart) is
  supported for multi-file charting. There is no multi-file trend chart,
  no anomaly visualization across files, and no auto chart recommendation.
- There is no row-level diffing between files and no semantic/fuzzy column
  alignment across files — a column hint is resolved independently, and
  identically, within each file via the existing single-file column
  resolution rules.
- There is no unit conversion; values are compared as-is.
- The `reference`/`comparison` roles are accepted by the resolver but are
  not exercised by any built-out comparison flow beyond `previous`/
  `current` — only `previous`/`current` are treated as primary in the UI
  and in the interpreters' prompts.
- The AI layer never generates chart data, Plotly code, or numeric
  comparison values — it only classifies requests and extracts hints; the
  same numeric-hallucination grounding used elsewhere still applies to the
  AI Comparison Insight text.
- With 2 or more files uploaded, the single-file dashboard (trend
  selection, single-file Q&A/charts/insight) is not shown — only the
  multi-file comparison view is. Switch back to a single upload (Clear All
  Files, then re-upload one file) to use the single-file dashboard.

## Development Roadmap

| Version | Milestone | Status |
|---|---|---|
| V0.1 | Excel Loading | COMPLETE |
| V0.2 | Data Understanding / Profiling | COMPLETE |
| V0.3 | Automatic Visualization Engine | COMPLETE |
| V0.4 | Anomaly Detection | COMPLETE |
| V0.5 | Q&A | COMPLETE |
| V0.6 | Q&A-driven Dynamic Charts | COMPLETE |
| V0.7 | Multi-file Comparison (Q&A, charts, AI insight) | COMPLETE |
| V0.8 | Advanced multi-file analytics, PowerPoint Generator | PLANNED |
| V1.0 | Integration / Polish | PLANNED |

**Deferred to V0.8+:** row-level diffing, semantic/fuzzy column matching
across files, automatic role inference, richer cross-file visualization
(trend/anomaly charts across files, auto-recommended charts), unit
conversion, forecasting, advanced anomaly analysis, persistent/
database-backed storage, RAG, agentic workflows, M365 integration,
advanced auth, and production deployment infrastructure.
