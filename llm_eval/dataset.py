import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class GoldenExample:
    id: str
    category: str
    prompt: str
    reference_answer: str
    context: Optional[str] = None


def load_golden_dataset(path: Path) -> list[GoldenExample]:
    raw = json.loads(Path(path).read_text())
    examples = []
    seen_ids = set()
    for item in raw:
        for required_field in ("id", "prompt", "reference_answer"):
            if required_field not in item:
                raise ValueError(
                    f"golden dataset item missing required field '{required_field}': {item}"
                )
        if item["id"] in seen_ids:
            raise ValueError(f"duplicate golden dataset id: {item['id']}")
        seen_ids.add(item["id"])
        examples.append(
            GoldenExample(
                id=item["id"],
                category=item.get("category", "uncategorized"),
                prompt=item["prompt"],
                reference_answer=item["reference_answer"],
                context=item.get("context"),
            )
        )
    return examples


def load_candidate_outputs(path: Path) -> dict[str, str]:
    raw = json.loads(Path(path).read_text())
    outputs = {}
    for item in raw:
        if "id" not in item or "output" not in item:
            raise ValueError(f"candidate output item missing 'id' or 'output': {item}")
        outputs[item["id"]] = item["output"]
    return outputs


def pair_examples_with_outputs(
    examples: list[GoldenExample], outputs: dict[str, str]
) -> list[tuple[GoldenExample, str]]:
    pairs = []
    missing = []
    for example in examples:
        if example.id not in outputs:
            missing.append(example.id)
            continue
        pairs.append((example, outputs[example.id]))
    if missing:
        raise ValueError(f"no candidate output found for golden dataset ids: {missing}")
    return pairs
