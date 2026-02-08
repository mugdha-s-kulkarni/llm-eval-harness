from llm_eval.dataset import GoldenExample
from llm_eval.rubric import RUBRIC_DIMENSIONS, build_judge_prompt


def test_build_judge_prompt_includes_prompt_reference_and_output():
    example = GoldenExample(
        id="q1",
        category="factual",
        prompt="What is the capital of France?",
        reference_answer="Paris.",
    )

    prompt = build_judge_prompt(example, "The capital of France is Paris.")

    assert "What is the capital of France?" in prompt
    assert "Paris." in prompt
    assert "The capital of France is Paris." in prompt
    for dimension in RUBRIC_DIMENSIONS:
        assert dimension in prompt


def test_build_judge_prompt_includes_context_when_present():
    example = GoldenExample(
        id="q2",
        category="summarization",
        prompt="Summarize this.",
        reference_answer="A short summary.",
        context="A long passage of source material.",
    )

    prompt = build_judge_prompt(example, "A summary.")

    assert "A long passage of source material." in prompt
    assert "SOURCE CONTEXT" in prompt


def test_build_judge_prompt_omits_context_block_when_absent():
    example = GoldenExample(
        id="q3",
        category="factual",
        prompt="What is 2+2?",
        reference_answer="4",
    )

    prompt = build_judge_prompt(example, "4")

    assert "SOURCE CONTEXT" not in prompt
