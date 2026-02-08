from llm_eval.drift import (
    compute_drift_signal,
    detect_regressions,
    summarize_run,
)
from llm_eval.scorer import ExampleResult


def make_result(id_, overall_score, verdict="pass", hallucinations=None, scores=None):
    return ExampleResult(
        id=id_,
        category="factual",
        prompt="prompt",
        output="output",
        scores=scores or {"correctness": overall_score, "groundedness": overall_score},
        overall_score=overall_score,
        verdict=verdict,
        hallucinations=hallucinations or [],
        rationale="rationale",
    )


def test_detect_regressions_flags_drop_at_or_beyond_threshold():
    results = [make_result("q1", 4.0), make_result("q2", 4.5)]
    baseline = {"q1": 4.6, "q2": 4.6}  # q1 dropped 0.6, q2 dropped 0.1

    regressions = detect_regressions(results, baseline, threshold=0.5)

    assert len(regressions) == 1
    assert regressions[0].id == "q1"
    assert regressions[0].delta == -0.6


def test_detect_regressions_ignores_ids_not_in_baseline():
    results = [make_result("new-id", 1.0)]
    baseline = {"other-id": 5.0}

    regressions = detect_regressions(results, baseline, threshold=0.5)

    assert regressions == []


def test_summarize_run_computes_means_and_rates():
    results = [
        make_result("q1", 5.0, verdict="pass", hallucinations=[]),
        make_result("q2", 2.0, verdict="fail", hallucinations=[{"claim": "x", "explanation": "y"}]),
    ]

    summary = summarize_run(results)

    assert summary["n_examples"] == 2
    assert summary["mean_overall_score"] == 3.5
    assert summary["pass_rate"] == 0.5
    assert summary["hallucination_rate"] == 0.5


def test_summarize_run_handles_empty_results():
    summary = summarize_run([])

    assert summary["n_examples"] == 0
    assert summary["mean_overall_score"] == 0.0
    assert summary["pass_rate"] == 0.0


def test_compute_drift_signal_requires_at_least_two_runs():
    history = [{"mean_overall_score": 4.0, "hallucination_rate": 0.0}]

    assert compute_drift_signal(history) is None


def test_compute_drift_signal_flags_large_drop():
    history = [
        {"mean_overall_score": 4.5, "hallucination_rate": 0.0},
        {"mean_overall_score": 4.4, "hallucination_rate": 0.0},
        {"mean_overall_score": 3.0, "hallucination_rate": 0.4},  # current run: sharp drop
    ]

    signal = compute_drift_signal(history)

    assert signal is not None
    assert signal["drift_detected"] is True
    assert signal["current_mean_score"] == 3.0
    assert signal["hallucination_rate_delta"] == 0.4


def test_compute_drift_signal_stable_when_scores_hold():
    history = [
        {"mean_overall_score": 4.5, "hallucination_rate": 0.0},
        {"mean_overall_score": 4.4, "hallucination_rate": 0.0},
        {"mean_overall_score": 4.45, "hallucination_rate": 0.0},
    ]

    signal = compute_drift_signal(history)

    assert signal["drift_detected"] is False
