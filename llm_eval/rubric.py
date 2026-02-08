from llm_eval.dataset import GoldenExample

# Each dimension is scored 1-5 by the judge model. Keys here must match
# llm_eval.config.EvalConfig.weights and the judge's structured-output schema.
RUBRIC_DIMENSIONS = {
    "correctness": (
        "Is the output factually correct and consistent with the reference answer? "
        "5 = fully correct, 1 = fundamentally wrong."
    ),
    "groundedness": (
        "Is every claim in the output supported by the reference answer and/or the "
        "provided source context, with no fabricated details? "
        "5 = fully grounded, 1 = contains fabricated or unsupported claims."
    ),
    "completeness": (
        "Does the output cover the key points expected from the reference answer? "
        "5 = fully complete, 1 = major omissions."
    ),
    "relevance": (
        "Does the output directly address the prompt without irrelevant tangents? "
        "5 = fully relevant, 1 = off-topic."
    ),
    "clarity": (
        "Is the output clear, well-organized, and unambiguous? "
        "5 = very clear, 1 = confusing or poorly structured."
    ),
}


def build_judge_prompt(example: GoldenExample, output: str) -> str:
    context_block = (
        f"\n\nSOURCE CONTEXT (ground-truth material the answer must stay consistent with):\n{example.context}"
        if example.context
        else ""
    )
    dims = "\n".join(f"- {name} (1-5): {desc}" for name, desc in RUBRIC_DIMENSIONS.items())
    return f"""You are grading a candidate answer against a golden reference answer.

PROMPT GIVEN TO THE SYSTEM UNDER TEST:
{example.prompt}
{context_block}

REFERENCE ANSWER (ground truth):
{example.reference_answer}

CANDIDATE OUTPUT TO GRADE:
{output}

Score the candidate output on each dimension below, using an integer from 1 to 5:
{dims}

Also identify any hallucinations: specific claims in the candidate output that are not \
supported by the reference answer or source context. Do not flag reasonable paraphrases, \
correct additional general knowledge, or stylistic differences as hallucinations — only \
flag claims that are false or unsupported and material to the answer.

Give an overall verdict: "pass" if the candidate is an acceptable answer, "fail" if it has \
a critical factual error or fabrication, "borderline" for a partially correct or incomplete \
answer that isn't outright wrong. Provide a brief rationale (2-3 sentences) explaining your \
scores."""
