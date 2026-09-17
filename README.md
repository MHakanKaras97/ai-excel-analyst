# AI Excel Analyst

An AI-powered Excel analytics and executive reporting application.

## Project Status

🚧 In development — V0.1 through V0.5 (Q&A) are complete.

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
Excel
 ↓
Data Understanding + Normalization
 ↓
Deterministic Analytics + Anomaly Detection
 ↓
AI Analyst Core
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

## Testing

Full test suite: **340 passed, 1 deselected** (a `pytest.ini` marker excludes
the network-dependent Gemini integration test by default).

Focused Q&A test suites:

| Module | Tests |
|---|---|
| V0.5.1 `qa_prompt_builder` | 13 passed |
| V0.5.2 `qa_interpreter` | 16 passed |
| V0.5.3a + V0.5.3b `qa_engine` | 47 passed |
| V0.5.4 `qa_answer` | 22 passed |
| V0.5.5 end-to-end pipeline | 8 passed |
| V0.5.6 Streamlit Q&A (AppTest) | 6 passed |

Run the full suite from the repo root with `pytest`.

## Principles

- Deterministic calculations are performed by Python, not the LLM
- The LLM is used for interpretation, not calculation
- Ambiguity is surfaced rather than silently guessed
- Existing working modules are reused rather than unnecessarily refactored
- Provider abstraction (`AIProvider`) keeps the AI layer replaceable
- Tests are provider/network independent wherever possible

## Limitations

Coming soon.

## Development Roadmap

| Version | Milestone | Status |
|---|---|---|
| V0.1 | Excel Loading | COMPLETE |
| V0.2 | Data Understanding / Profiling | COMPLETE |
| V0.3 | Automatic Visualization Engine | COMPLETE |
| V0.4 | Anomaly Detection | COMPLETE |
| V0.5 | Q&A | COMPLETE |
| V0.6 | Q&A-driven Dynamic Charts | PLANNED |
| V0.7 | Multi-file Comparison | PLANNED |
| V0.8 | PowerPoint Generator | PLANNED |
| V1.0 | Integration / Polish | PLANNED |
