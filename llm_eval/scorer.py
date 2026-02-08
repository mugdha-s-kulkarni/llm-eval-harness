from dataclasses import dataclass
from typing import Optional

from llm_eval.config import EvalConfig
from llm_eval.dataset import GoldenExample
from llm_eval.judge import Judge


@dataclass
class ExampleResult:
    id: str
    category: str
    prompt: str
    output: str
    scores: dict
    overall_score: float
    verdict: str
    hallucinations: list
    rationale: str


def weighted_overall(scores: dict, weights: dict) -> float:
    total_weight = sum(weights.values())
    weighted_sum = sum(scores[dimension] * weight for dimension, weight in weights.items())
    return round(weighted_sum / total_weight, 3)


def run_scoring(
    pairs: list[tuple[GoldenExample, str]],
    config: EvalConfig,
    judge: Optional[Judge] = None,
) -> list[ExampleResult]:
    judge = judge or Judge(config)
    results = []
    for example, output in pairs:
        raw = judge.score(example, output)
        overall = weighted_overall(raw["scores"], config.weights)
        results.append(
            ExampleResult(
                id=example.id,
                category=example.category,
                prompt=example.prompt,
                output=output,
                scores=raw["scores"],
                overall_score=overall,
                verdict=raw["verdict"],
                hallucinations=raw["hallucinations"],
                rationale=raw["rationale"],
            )
        )
    return results
