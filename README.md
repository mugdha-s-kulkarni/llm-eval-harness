# LLM Eval Harness

A lightweight framework for evaluating LLM outputs against a golden dataset, with support for rubric-based scoring, automated judging, and drift detection across model versions or prompt changes.

## Why I built this

I wanted hands-on experience with LLM validation outside a production setting. Most of my QA background has been in deterministic systems where pass/fail is clear cut. LLM outputs are probabilistic and harder to validate consistently, so I built this to explore how a validation framework holds up for non-deterministic systems: what a golden dataset looks like, how to score subjective quality, and how to catch regressions when a model or prompt changes.

## How it works

1. **Dataset** (`llm_eval/dataset.py`) - loads and validates the golden dataset used as the evaluation baseline.
2. **Rubric** (`llm_eval/rubric.py`) - defines the scoring criteria applied to each output.
3. **Scorer** (`llm_eval/scorer.py`) - applies the rubric to generate per-example and aggregate scores.
4. **Judge** (`llm_eval/judge.py`) - uses a secondary LLM call to assess outputs against criteria that are hard to score programmatically, tone, coherence, adherence to instructions.
5. **Drift detection** (`llm_eval/drift.py`) - compares current run results against baseline scores to flag regressions across model or prompt versions.
6. **Report** (`llm_eval/report.py`) - generates a summary report from a completed eval run.

## Usage

```bash
pip install -r requirements.txt
python scripts/run_eval.py
```

## Design decisions

- **Why separate scorer and judge**: programmatic scoring handles objective, rule based criteria. The judge handles nuanced criteria that need reasoning rather than pattern matching, which mirrors the kind of split I'd want in a real validation pipeline.
- **Why drift detection**: LLM behavior can shift silently when a prompt or model version changes. Catching that early is the same principle behind regression testing in traditional QA, just applied to a probabilistic system.
- **Testing approach**: each core module has unit test coverage (`tests/`) to keep scoring logic reliable as the rubric evolves.

## Project structure

```
llm_eval/       # Core library: dataset, scoring, rubric, judge, drift, reporting
scripts/        # Entry point for running evals
tests/          # Unit tests for core modules
data/           # Golden dataset, baseline scores, run history, sample outputs
reports/        # Generated eval reports
```