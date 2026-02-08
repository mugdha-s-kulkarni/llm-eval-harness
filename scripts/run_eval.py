#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_eval.config import EvalConfig
from llm_eval.dataset import (
    load_candidate_outputs,
    load_golden_dataset,
    pair_examples_with_outputs,
)
from llm_eval.drift import (
    append_history,
    compute_drift_signal,
    detect_regressions,
    load_baseline,
    save_baseline,
    summarize_run,
)
from llm_eval.report import write_reports
from llm_eval.scorer import run_scoring


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LLM-as-judge scoring, regression detection, and drift analysis."
    )
    parser.add_argument("--dataset", type=Path, default=None, help="Path to golden_dataset.json")
    parser.add_argument("--outputs", type=Path, default=None, help="Path to sample_outputs.json")
    parser.add_argument("--model", type=str, default=None, help="Override the judge model ID")
    parser.add_argument(
        "--effort",
        type=str,
        default=None,
        choices=["low", "medium", "high", "xhigh", "max"],
        help="Override the judge's thinking effort level",
    )
    parser.add_argument(
        "--set-baseline",
        action="store_true",
        help="Save this run's per-example scores as the new regression baseline.",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Don't append this run to run_history.json (useful for one-off dry runs).",
    )
    args = parser.parse_args()

    config = EvalConfig()
    if args.model:
        config.model = args.model
    if args.effort:
        config.effort = args.effort

    dataset_path = args.dataset or config.dataset_path
    outputs_path = args.outputs or config.outputs_path

    examples = load_golden_dataset(dataset_path)
    outputs = load_candidate_outputs(outputs_path)
    pairs = pair_examples_with_outputs(examples, outputs)

    print(f"Scoring {len(pairs)} examples with {config.model} (effort={config.effort})...")
    results = run_scoring(pairs, config)

    baseline = load_baseline(config.baseline_path)
    regressions = detect_regressions(results, baseline, config.regression_threshold)

    summary = summarize_run(results)
    history = [] if args.no_history else append_history(config.history_path, summary)
    drift_signal = compute_drift_signal(history) if history else None

    md_path, json_path = write_reports(
        config.reports_dir, results, summary, regressions, drift_signal, config
    )

    if args.set_baseline:
        save_baseline(config.baseline_path, results)
        print(f"Baseline updated: {config.baseline_path}")

    print(f"\nReport written to {md_path}")
    print(f"Machine-readable report: {json_path}")
    print(
        f"Mean overall score: {summary['mean_overall_score']} / 5 | "
        f"Pass rate: {summary['pass_rate'] * 100:.1f}%"
    )
    if regressions:
        print(f"⚠️  {len(regressions)} regression(s) detected vs baseline")
    if drift_signal and drift_signal["drift_detected"]:
        print("⚠️  Drift detected vs run history")

    sys.exit(1 if regressions else 0)


if __name__ == "__main__":
    main()
