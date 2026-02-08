import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from llm_eval.scorer import ExampleResult


@dataclass
class Regression:
    id: str
    baseline_score: float
    current_score: float
    delta: float


def load_baseline(path: Path) -> dict:
    if not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text())


def save_baseline(path: Path, results: list[ExampleResult]) -> None:
    data = {r.id: r.overall_score for r in results}
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def detect_regressions(
    results: list[ExampleResult], baseline: dict, threshold: float
) -> list[Regression]:
    regressions = []
    for r in results:
        if r.id not in baseline:
            continue
        delta = round(r.overall_score - baseline[r.id], 3)
        if delta <= -threshold:
            regressions.append(
                Regression(
                    id=r.id,
                    baseline_score=baseline[r.id],
                    current_score=r.overall_score,
                    delta=delta,
                )
            )
    return regressions


def summarize_run(results: list[ExampleResult]) -> dict:
    if not results:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_examples": 0,
            "mean_overall_score": 0.0,
            "stdev_overall_score": 0.0,
            "per_dimension_mean": {},
            "pass_rate": 0.0,
            "hallucination_rate": 0.0,
        }

    dimensions = results[0].scores.keys()
    per_dimension_mean = {
        dimension: round(statistics.mean(r.scores[dimension] for r in results), 3)
        for dimension in dimensions
    }
    overall_scores = [r.overall_score for r in results]

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_examples": len(results),
        "mean_overall_score": round(statistics.mean(overall_scores), 3),
        "stdev_overall_score": round(statistics.pstdev(overall_scores), 3)
        if len(overall_scores) > 1
        else 0.0,
        "per_dimension_mean": per_dimension_mean,
        "pass_rate": round(sum(1 for r in results if r.verdict == "pass") / len(results), 3),
        "hallucination_rate": round(
            sum(1 for r in results if r.hallucinations) / len(results), 3
        ),
    }


def load_history(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    return json.loads(Path(path).read_text())


def append_history(path: Path, summary: dict, keep_last: int = 50) -> list[dict]:
    history = load_history(path)
    history.append(summary)
    history = history[-keep_last:]
    Path(path).write_text(json.dumps(history, indent=2) + "\n")
    return history


def compute_drift_signal(history: list[dict]) -> Optional[dict]:
    """Compare the latest run's mean score against the trailing average of prior runs.

    This is a lightweight drift heuristic (no external stats dependency): a run is
    flagged as drifting when its mean score falls more than one standard deviation
    below the trailing average of prior runs (or 0.3 points, whichever is larger,
    so two very consistent prior runs don't produce a near-zero threshold).
    """
    if len(history) < 2:
        return None

    current = history[-1]
    prior = history[:-1]
    prior_means = [h["mean_overall_score"] for h in prior]
    prior_avg = statistics.mean(prior_means)
    prior_stdev = statistics.pstdev(prior_means) if len(prior_means) > 1 else 0.0
    delta = round(current["mean_overall_score"] - prior_avg, 3)
    drift_threshold = max(prior_stdev, 0.3)

    prior_hallucination_rates = [h["hallucination_rate"] for h in prior]

    return {
        "prior_run_count": len(prior),
        "prior_avg_mean_score": round(prior_avg, 3),
        "current_mean_score": current["mean_overall_score"],
        "delta": delta,
        "drift_detected": delta <= -drift_threshold,
        "hallucination_rate_delta": round(
            current["hallucination_rate"] - statistics.mean(prior_hallucination_rates), 3
        ),
    }
