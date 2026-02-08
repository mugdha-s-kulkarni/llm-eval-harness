from llm_eval.config import EvalConfig
from llm_eval.dataset import GoldenExample
from llm_eval.scorer import run_scoring, weighted_overall


def test_weighted_overall_computes_weighted_average():
    scores = {
        "correctness": 5,
        "groundedness": 5,
        "completeness": 3,
        "relevance": 5,
        "clarity": 4,
    }
    weights = {
        "correctness": 0.30,
        "groundedness": 0.30,
        "completeness": 0.15,
        "relevance": 0.15,
        "clarity": 0.10,
    }

    result = weighted_overall(scores, weights)

    expected = 5 * 0.30 + 5 * 0.30 + 3 * 0.15 + 5 * 0.15 + 4 * 0.10
    assert result == round(expected, 3)


class FakeJudge:
    """Stands in for llm_eval.judge.Judge without making an API call."""

    def __init__(self, canned_response: dict):
        self.canned_response = canned_response
        self.calls = []

    def score(self, example, output):
        self.calls.append((example.id, output))
        return self.canned_response


def test_run_scoring_builds_example_results_from_judge_output():
    config = EvalConfig()
    example = GoldenExample(
        id="q1", category="factual", prompt="What is 2+2?", reference_answer="4"
    )
    canned = {
        "scores": {
            "correctness": 5,
            "groundedness": 5,
            "completeness": 5,
            "relevance": 5,
            "clarity": 5,
        },
        "hallucinations": [],
        "verdict": "pass",
        "rationale": "Correct and complete.",
    }
    judge = FakeJudge(canned)

    results = run_scoring([(example, "4")], config, judge=judge)

    assert len(results) == 1
    result = results[0]
    assert result.id == "q1"
    assert result.overall_score == 5.0
    assert result.verdict == "pass"
    assert result.hallucinations == []
    assert judge.calls == [("q1", "4")]


def test_run_scoring_carries_hallucinations_through():
    config = EvalConfig()
    example = GoldenExample(
        id="q2", category="summarization", prompt="Summarize.", reference_answer="Ref."
    )
    canned = {
        "scores": {
            "correctness": 2,
            "groundedness": 1,
            "completeness": 3,
            "relevance": 4,
            "clarity": 4,
        },
        "hallucinations": [
            {"claim": "Fabricated detail.", "explanation": "Not present in the source."}
        ],
        "verdict": "fail",
        "rationale": "Contains a fabricated claim.",
    }
    judge = FakeJudge(canned)

    results = run_scoring([(example, "some output")], config, judge=judge)

    assert results[0].hallucinations == canned["hallucinations"]
    assert results[0].verdict == "fail"
