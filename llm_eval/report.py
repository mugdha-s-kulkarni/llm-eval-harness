import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from llm_eval.config import EvalConfig
from llm_eval.drift import Regression
from llm_eval.scorer import ExampleResult


def render_markdown_report(
    results: list[ExampleResult],
    summary: dict,
    regressions: list[Regression],
    drift_signal: Optional[dict],
    config: EvalConfig,
) -> str:
    lines = []
    lines.append("# LLM Regression & Drift Report")
    lines.append(
        f"\n_Generated: {summary['timestamp']}_  \n_Model: `{config.model}` · Effort: `{config.effort}`_\n"
    )

    lines.append("## Summary")
    lines.append(f"- Examples evaluated: **{summary['n_examples']}**")
    lines.append(f"- Mean overall score: **{summary['mean_overall_score']} / 5**")
    lines.append(f"- Pass rate: **{summary['pass_rate'] * 100:.1f}%**")
    lines.append(
        f"- Hallucination rate: **{summary['hallucination_rate'] * 100:.1f}%** "
        "of examples had at least one flagged claim"
    )

    lines.append("\n### Per-dimension mean scores")
    lines.append("| Dimension | Mean |")
    lines.append("|---|---|")
    for dimension, value in summary["per_dimension_mean"].items():
        lines.append(f"| {dimension} | {value} |")

    lines.append("\n## Regression Detection (vs. baseline)")
    if regressions:
        lines.append(
            f"⚠️ **{len(regressions)} regression(s) found** "
            f"(score dropped ≥ {config.regression_threshold} pts vs baseline)\n"
        )
        lines.append("| ID | Baseline | Current | Delta |")
        lines.append("|---|---|---|---|")
        for r in regressions:
            lines.append(f"| {r.id} | {r.baseline_score} | {r.current_score} | {r.delta} |")
    else:
        lines.append("✅ No regressions vs baseline.")

    lines.append("\n## Drift Signal (vs. run history)")
    if drift_signal is None:
        lines.append("_Not enough run history yet (need ≥ 2 runs) to compute drift._")
    else:
        flag = "⚠️ **DRIFT DETECTED**" if drift_signal["drift_detected"] else "✅ stable"
        lines.append(
            f"{flag} — mean score moved {drift_signal['delta']:+.3f} vs the trailing "
            f"average of the last {drift_signal['prior_run_count']} run(s) "
            f"({drift_signal['prior_avg_mean_score']} → {drift_signal['current_mean_score']})."
        )
        lines.append(
            f"\nHallucination rate change vs trailing average: "
            f"{drift_signal['hallucination_rate_delta']:+.3f}"
        )

    lines.append("\n## Hallucinations Flagged")
    flagged = [r for r in results if r.hallucinations]
    if not flagged:
        lines.append("✅ None flagged this run.")
    else:
        for r in flagged:
            lines.append(f"\n### `{r.id}` ({r.category})")
            for h in r.hallucinations:
                lines.append(f"- **Claim:** {h['claim']}\n  **Why:** {h['explanation']}")

    lines.append("\n## Per-example Results")
    dim_headers = " | ".join(summary["per_dimension_mean"].keys())
    lines.append(f"| ID | Category | Verdict | Overall | {dim_headers} |")
    lines.append("|---|---|---|---|" + "---|" * len(summary["per_dimension_mean"]))
    for r in results:
        dim_values = " | ".join(str(r.scores[d]) for d in summary["per_dimension_mean"].keys())
        lines.append(f"| {r.id} | {r.category} | {r.verdict} | {r.overall_score} | {dim_values} |")

    return "\n".join(lines) + "\n"


def write_reports(
    reports_dir: Path,
    results: list[ExampleResult],
    summary: dict,
    regressions: list[Regression],
    drift_signal: Optional[dict],
    config: EvalConfig,
) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown_report(results, summary, regressions, drift_signal, config)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    timestamped_md_path = reports_dir / f"report_{ts}.md"
    timestamped_md_path.write_text(markdown)

    (reports_dir / "latest.md").write_text(markdown)

    json_payload = {
        "summary": summary,
        "regressions": [r.__dict__ for r in regressions],
        "drift_signal": drift_signal,
        "results": [
            {
                "id": r.id,
                "category": r.category,
                "verdict": r.verdict,
                "overall_score": r.overall_score,
                "scores": r.scores,
                "hallucinations": r.hallucinations,
                "rationale": r.rationale,
            }
            for r in results
        ],
    }
    latest_json_path = reports_dir / "latest.json"
    latest_json_path.write_text(json.dumps(json_payload, indent=2) + "\n")

    return timestamped_md_path, latest_json_path
