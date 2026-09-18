# AI Excel Analyst

> Excel/CSV analytics where every number comes from Python and every AI
> explanation is checked against it — not a spreadsheet chatbot that
> guesses at numbers.

## Project Status

**V0.8 Complete — Trusted & Intelligent Analytics**

V0.1 through V0.8 are implemented and tested: deterministic Excel/CSV
analytics, grounded natural-language Q&A and charts, multi-file
comparison, semantic schema detection, evidence-based AI interpretation,
forecasting baselines, anomaly investigation, and an offline evaluation
framework. **866 tests passing, 0 failed.** This is an actively developed
portfolio project, not a hosted production service — see [Limitations](#limitations).

## Overview

AI Excel Analyst is a Streamlit application that lets you upload one or
more Excel/CSV files and get:

- Deterministic profiling, cleaning, and statistical analytics
- Trend, period-comparison, and anomaly detection
- Natural-language Q&A and chart requests, answered from computed results
- Multi-file comparison (e.g. "previous" vs "current")
- AI-generated narrative insight, always checked against the computed data

What separates it from a typical "LLM reads your spreadsheet" tool is a
strict division of labor:

> **The LLM interprets. Python resolves, calculates, validates, and
> grounds.**

The LLM never performs arithmetic and never produces a number that
appears in an answer. It classifies intent, proposes semantic labels, and
writes narrative explanations — always over data Python already computed,
and always checked against that data before being shown.

## Why This Project?

LLM-based spreadsheet tools can produce answers that read as confident and
correct but are numerically wrong — the model effectively "eyeballs" the
data. This project exists to explore and demonstrate an alternative
architecture where an LLM adds real value (language understanding,
narrative synthesis) without ever being trusted for arithmetic.

Responsibilities are split explicitly:

| Layer | Responsibility |
|---|---|
| **Python (deterministic engines)** | Data loading, normalization, statistics, anomaly detection, forecasting, resolution of natural-language hints |
| **LLM** | Intent classification, semantic labeling proposals, narrative interpretation |
| **Evidence layer** | Structured, provenance-tracked facts the AI is allowed to reference |
| **Evaluation layer** | Offline, repeatable checks that the above actually holds |

Every LLM call uses a closed output schema (fixed keys, enumerated
values) — there is no field where the model can emit a free-form number
that becomes the answer.

## Key Features

- **Deterministic analytics** — profiling, normalization, KPI/statistics, trend detection, period-over-period comparison
- **Grounded Q&A** — natural-language questions answered from already-computed results, never invented
- **Natural-language charts** — chart requests mapped to a deterministic chart specification, not AI-generated plotting code
- **Multi-file comparison** — role-based (`previous`/`current`) comparison across files, with Q&A, charts, and AI insight
- **Semantic schema detection** — deterministic column-role/type classification with an optional, Python-validated LLM proposal
- **Evidence-based AI interpretation** — AI narrative is generated only from structured, verifiable evidence
- **Forecasting baselines** — naive and seasonal-naive forecasts, explicitly labeled as predictions
- **Anomaly investigation** — deterministic IQR-based detection plus coincident-change context (never a causal claim)
- **Offline evaluation framework** — 90 benchmark cases validating the system end to end without live API calls

## Architecture

```
User
 ↓
Streamlit
 ↓
Intent / Schema Understanding   (LLM — one closed-schema call)
 ↓
Deterministic Analytics         (Python — the numerical authority)
 ↓
Evidence                        (structured, provenance-tracked facts)
 ↓
AI Interpretation               (explains evidence — never invents it)
 ↓
Grounded Answer                 (validated before being shown)
 ↓
Evaluation                      (offline benchmark, 90/90 passing)
```

The LLM is never the numerical authority: it either (a) classifies a
question/chart request into a structured intent, or (b) writes narrative
text that is checked, after generation, against the deterministic data it
was given. A response citing a number, evidence ID, or causal claim not
backed by that data is rejected.

This flow specializes into a few concrete pipelines, all sharing the same
principle:

| Pipeline | What the LLM does | What Python does |
|---|---|---|
| Q&A (`qa_engine.py`) | Classifies question into an intent + hints | Resolves hints, looks up/computes the value, phrases the answer |
| Charts (`qa_chart_engine.py`) | Classifies chart request | Builds the chart spec and renders it (Plotly) |
| Multi-file comparison (`multi_file_comparison.py`) | Classifies comparison request | Resolves files by role/name (exact match only), computes the comparison |
| Semantic schema (`src/schema/`) | Proposes column role/type | Builds the deterministic candidate first; validates or rejects the proposal |
| Evidence-aware insight (`src/evidence/`, `evidence_ai_interpreter.py`) | Writes narrative from supplied evidence | Builds the evidence, validates every cited number/ID/claim |

## Trust & Grounding Model

This is the core engineering property of the project, enforced the same
way at every layer:

- **Closed output schemas** — every LLM call has a fixed set of allowed
  keys and enumerated values; there is no free-form numeric field.
- **Resolve, don't guess** — column and period resolution is exact/
  conservative-fuzzy; an ambiguous match is reported as ambiguous, never
  silently picked.
- **Evidence, not raw data** — AI narrative generation only ever sees a
  small, structured Evidence payload (one deterministic fact per item),
  never a raw DataFrame and never an internal `file_id`.
- **Numeric grounding** — every number in an AI response is checked
  against the payload it was given; an unsupported number invalidates the
  response.
- **Causal-claim guarding** — a coincident change may be reported as a
  coincidence ("X also changed during the same period"); asserting that
  one caused the other is detected and rejected.

These properties are exercised directly by the grounding and safety
categories of the evaluation framework below, not just described.

## V0.8 — Trusted & Intelligent Analytics

V0.8 is the current milestone and the project's strongest engineering
story: five additive layers on top of the V0.1–V0.7 pipeline, none of
which required rewriting existing code.

**Semantic understanding**
- Detects column roles (`dimension`, `measure`, `date`, `identifier`, `unknown`) and semantic types (`numeric`, `currency`, `percentage`, `date`, `text`, `identifier`, `unknown`)
- Deterministic profiling produces a candidate classification first; an optional LLM proposal may only refine it
- Every proposed value is validated in Python; a proposal contradicting strong (dtype-certain) evidence is rejected
- Business-term column resolution (e.g. "total sales") falls back to conservative lexical scoring — never embeddings — and only resolves when exactly one candidate is clearly confident

**Evidence and grounding**
- Every analytical fact shown to the AI is represented as a small, typed Evidence record with explicit provenance
- AI interpretation operates only on that evidence, never on raw data
- Evidence IDs a response claims to have used are checked against what was actually supplied
- Causal language between two evidenced facts is explicitly detected and rejected

**Forecasting**
- Two baselines: naive (flat carry-forward) and seasonal-naive
- Every forecast is explicitly labeled a prediction, with confidence bounds reported only when defensible from history — never fabricated
- Backtested with deterministic metrics (MAE, RMSE, MAPE); insufficient history is reported rather than forced

**Anomaly investigation**
- Extends existing IQR-based anomaly detection with the period's own change and any coincident change in other measures
- Deliberately reports coincidence, never a root cause

**Evaluation**
- 90 offline benchmark cases exercise intent classification, resolution, grounding, and safety behavior against the real deterministic/validation code — no live API calls
- All 90 currently pass (see [Evaluation](#evaluation))

## Evaluation

An offline benchmark validates the system's behavior deterministically,
without any live LLM call — each case supplies a fixed "model response"
and checks that the real interpreter/resolver/grounding code handles it
correctly.

| Category | Cases | Result |
|---|---|---|
| QA | 20 | ✅ |
| Charts | 15 | ✅ |
| Semantic resolution | 20 | ✅ |
| Grounding | 20 | ✅ |
| Safety | 15 | ✅ |
| **Total** | **90** | **90/90 passed** |

Run it with:

```bash
python -m pytest tests/test_evaluation_metrics.py -q
```

## Technology Stack

**Implemented:**

- Python
- Pandas, NumPy
- OpenPyXL / xlrd (Excel I/O)
- Plotly (charts)
- Streamlit (UI)
- Google Gemini API (`google-genai`) via a swappable `AIProvider` interface
- Pytest (866 tests)

**Planned (not yet implemented):**

- python-pptx (PowerPoint export — V0.9)

## Example Usage

```bash
pip install -r requirements.txt
cp .env.example .env        # add your GEMINI_API_KEY
streamlit run app.py
```

Upload one `.xlsx`/`.xls` file to explore deterministic analytics, ask
grounded questions, and request charts. Upload two or more files to
assign `previous`/`current` roles and compare them via Q&A, charts, and
AI insight.

## Testing

Full test suite: **866 passed, 2 deselected, 0 failed** (a `pytest.ini`
marker excludes network-dependent Gemini integration tests by default).

| Area | Tests |
|---|---|
| V0.1–V0.4 foundations (loading, profiling, normalization, analytics, anomaly detection, chart building, AI provider) | 214 |
| V0.5 Q&A | 120 |
| V0.6 Charts | 143 |
| V0.7 Multi-file comparison | 188 |
| V0.8 Trusted & Intelligent Analytics | 201 |

Coverage spans unit tests, deterministic-engine tests, interpreter/
grounding tests, end-to-end pipeline tests, and Streamlit `AppTest` UI
tests. Run the full suite from the repo root with `pytest`.

## Limitations

These are explicit engineering scope boundaries, not defects:

- **Multi-file comparison is pairwise only** — exactly two files per
  comparison, resolved by exact role/display-name match (no fuzzy
  matching, no inference from upload order).
- **Multi-file charting supports one chart type** — a previous-vs-current
  scalar comparison; no multi-file trend/anomaly charts yet.
- **Semantic resolution is lexical, not semantic** — a small explicit
  synonym table and keyword-overlap scoring, no embeddings/ML.
- **Semantic vocabulary is intentionally small** — 5 roles, 7 types, no
  general-purpose ontology, no unit conversion.
- **Forecasting is two baselines** — naive and seasonal-naive; no
  ARIMA/Prophet/ML forecasting.
- **Anomaly investigation reports coincidence, not root cause** — it does
  not rank or explain *why* an anomaly occurred.
- **The causal-language guard is phrase-based**, not a semantic
  understanding of causality — it can miss unusually-phrased claims,
  and deliberately errs toward over-flagging.
- **The evaluation framework runs offline** against fixed, hand-authored
  model responses — it validates the deterministic/validation code paths,
  not a live model's judgment.
- **Evidence-aware Q&A/AI insight** are fully implemented and tested as
  modules, but surfaced in the UI only via the existing "Why this
  answer?" / "Explain with AI" additions — not yet a dedicated workflow.
- **No production infrastructure** — no database, authentication, or
  cloud deployment; this is a local/portfolio application.

## Roadmap

| Version | Milestone | Status |
|---|---|---|
| V0.1 | Excel Loading | COMPLETE |
| V0.2 | Data Understanding / Profiling | COMPLETE |
| V0.3 | Automatic Visualization Engine | COMPLETE |
| V0.4 | Anomaly Detection | COMPLETE |
| V0.5 | Q&A | COMPLETE |
| V0.6 | Q&A-driven Dynamic Charts | COMPLETE |
| V0.7 | Multi-file Comparison | COMPLETE |
| V0.8 | Trusted & Intelligent Analytics | COMPLETE |
| V0.9 | PowerPoint Generator | PLANNED |
| V1.0 | Integration / Polish | PLANNED |

**Deferred ideas** (not implemented, not scheduled): RAG, agentic/
autonomous workflows, database-backed persistence, authentication, cloud
deployment, persistent conversation memory, row-level semantic diffing, a
general-purpose semantic ontology, automatic unit conversion, a larger
forecasting model zoo, richer cross-file anomaly visualization, and
Microsoft 365/SharePoint/Outlook integration.

## Project Structure

```
app.py                          # Streamlit UI
src/
  excel_loader.py, data_profiler.py, data_normalizer.py,
  analytics_engine.py, anomaly_detector.py, chart_builder.py    # V0.1-V0.4 core
  ai_provider.py, ai_prompt_builder.py, ai_interpreter.py,
  gemini_provider.py                                            # AI provider + grounding
  qa_*.py                                                       # V0.5 Q&A pipeline
  chart_intent_*.py                                             # V0.6 chart pipeline
  file_analysis.py, multi_file_*.py, comparison_*.py            # V0.7 multi-file
  schema/                                                       # V0.8 semantic schema
  evidence/, evidence_*.py                                      # V0.8 evidence + grounded AI
  advanced_analytics/, forecasting/                             # V0.8 analytics + forecasting
  anomaly_investigation.py                                      # V0.8 anomaly context
  evaluation/                                                   # V0.8 evaluation framework
tests/                          # 866 tests (unit, engine, pipeline, AppTest)
data/evaluation/                # 90-case offline benchmark dataset
```
