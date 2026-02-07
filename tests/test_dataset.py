import json

import pytest

from llm_eval.dataset import (
    load_candidate_outputs,
    load_golden_dataset,
    pair_examples_with_outputs,
)


def write_json(path, data):
    path.write_text(json.dumps(data))


def test_load_golden_dataset_parses_required_and_optional_fields(tmp_path):
    path = tmp_path / "golden.json"
    write_json(
        path,
        [
            {
                "id": "q1",
                "category": "factual",
                "prompt": "What is 2+2?",
                "reference_answer": "4",
            },
            {
                "id": "q2",
                "prompt": "Summarize this.",
                "reference_answer": "A summary.",
                "context": "Some source text.",
            },
        ],
    )

    examples = load_golden_dataset(path)

    assert len(examples) == 2
    assert examples[0].id == "q1"
    assert examples[0].category == "factual"
    assert examples[0].context is None
    # category defaults when omitted
    assert examples[1].category == "uncategorized"
    assert examples[1].context == "Some source text."


def test_load_golden_dataset_rejects_missing_required_field(tmp_path):
    path = tmp_path / "golden.json"
    write_json(path, [{"id": "q1", "prompt": "What is 2+2?"}])  # missing reference_answer

    with pytest.raises(ValueError, match="reference_answer"):
        load_golden_dataset(path)


def test_load_golden_dataset_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "golden.json"
    write_json(
        path,
        [
            {"id": "q1", "prompt": "A", "reference_answer": "a"},
            {"id": "q1", "prompt": "B", "reference_answer": "b"},
        ],
    )

    with pytest.raises(ValueError, match="duplicate golden dataset id"):
        load_golden_dataset(path)


def test_load_candidate_outputs(tmp_path):
    path = tmp_path / "outputs.json"
    write_json(path, [{"id": "q1", "output": "4"}, {"id": "q2", "output": "42"}])

    outputs = load_candidate_outputs(path)

    assert outputs == {"q1": "4", "q2": "42"}


def test_pair_examples_with_outputs_raises_on_missing_output(tmp_path):
    golden_path = tmp_path / "golden.json"
    outputs_path = tmp_path / "outputs.json"
    write_json(
        golden_path,
        [
            {"id": "q1", "prompt": "A", "reference_answer": "a"},
            {"id": "q2", "prompt": "B", "reference_answer": "b"},
        ],
    )
    write_json(outputs_path, [{"id": "q1", "output": "a"}])

    examples = load_golden_dataset(golden_path)
    outputs = load_candidate_outputs(outputs_path)

    with pytest.raises(ValueError, match="q2"):
        pair_examples_with_outputs(examples, outputs)


def test_pair_examples_with_outputs_pairs_correctly(tmp_path):
    golden_path = tmp_path / "golden.json"
    outputs_path = tmp_path / "outputs.json"
    write_json(golden_path, [{"id": "q1", "prompt": "A", "reference_answer": "a"}])
    write_json(outputs_path, [{"id": "q1", "output": "candidate answer"}])

    examples = load_golden_dataset(golden_path)
    outputs = load_candidate_outputs(outputs_path)
    pairs = pair_examples_with_outputs(examples, outputs)

    assert len(pairs) == 1
    example, output = pairs[0]
    assert example.id == "q1"
    assert output == "candidate answer"
