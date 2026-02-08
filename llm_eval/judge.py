import json
from typing import Optional

import anthropic

from llm_eval.config import EvalConfig
from llm_eval.dataset import GoldenExample
from llm_eval.rubric import build_judge_prompt

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "object",
            "properties": {
                "correctness": {"type": "integer"},
                "groundedness": {"type": "integer"},
                "completeness": {"type": "integer"},
                "relevance": {"type": "integer"},
                "clarity": {"type": "integer"},
            },
            "required": [
                "correctness",
                "groundedness",
                "completeness",
                "relevance",
                "clarity",
            ],
            "additionalProperties": False,
        },
        "hallucinations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "explanation": {"type": "string"},
                },
                "required": ["claim", "explanation"],
                "additionalProperties": False,
            },
        },
        "verdict": {"type": "string", "enum": ["pass", "fail", "borderline"]},
        "rationale": {"type": "string"},
    },
    "required": ["scores", "hallucinations", "verdict", "rationale"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You are a meticulous, impartial evaluator (LLM-as-judge) for an automated regression "
    "test suite. You compare candidate model outputs against golden reference answers and "
    "score them on a fixed rubric. Be strict about factual accuracy and grounding — this "
    "output feeds a hallucination-detection report, so err on the side of flagging "
    "unsupported or fabricated claims rather than letting them slide."
)

# Scores must land in this range; a judge response outside it indicates the model
# didn't follow the rubric and should be treated as a bug, not silently clamped.
SCORE_RANGE = range(1, 6)


class Judge:
    """Wraps a Claude call that grades one candidate output against a golden example."""

    def __init__(self, config: EvalConfig, client: Optional[anthropic.Anthropic] = None):
        self.config = config
        self.client = client or anthropic.Anthropic()

    def score(self, example: GoldenExample, output: str) -> dict:
        prompt = build_judge_prompt(example, output)
        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=SYSTEM_PROMPT,
            thinking={"type": "adaptive"},
            output_config={
                "effort": self.config.effort,
                "format": {"type": "json_schema", "schema": JUDGE_SCHEMA},
            },
            messages=[{"role": "user", "content": prompt}],
        )

        if response.stop_reason == "refusal":
            raise RuntimeError(
                f"Judge refused to score example '{example.id}': {response.stop_details}"
            )

        text_block = next(block for block in response.content if block.type == "text")
        parsed = json.loads(text_block.text)

        for dimension, value in parsed["scores"].items():
            if value not in SCORE_RANGE:
                raise ValueError(
                    f"judge returned out-of-range score for '{dimension}' on example "
                    f"'{example.id}': {value}"
                )

        return parsed
