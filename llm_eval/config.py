import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class EvalConfig:
    model: str = os.environ.get("EVAL_MODEL", "claude-opus-4-8")
    effort: str = os.environ.get("EVAL_EFFORT", "medium")
    max_tokens: int = 2048

    # Rubric dimension weights — must match the keys in llm_eval.rubric.RUBRIC_DIMENSIONS.
    weights: dict = field(
        default_factory=lambda: {
            "correctness": 0.30,
            "groundedness": 0.30,
            "completeness": 0.15,
            "relevance": 0.15,
            "clarity": 0.10,
        }
    )

    # A per-example score drop (on the 1-5 scale) at or beyond this is flagged as a regression.
    regression_threshold: float = float(os.environ.get("EVAL_REGRESSION_THRESHOLD", "0.5"))
    # Overall scores at or above this are considered a "pass" for summary reporting.
    pass_threshold: float = float(os.environ.get("EVAL_PASS_THRESHOLD", "3.5"))

    dataset_path: Path = ROOT / "data" / "golden_dataset.json"
    outputs_path: Path = ROOT / "data" / "sample_outputs.json"
    baseline_path: Path = ROOT / "data" / "baseline_scores.json"
    history_path: Path = ROOT / "data" / "run_history.json"
    reports_dir: Path = ROOT / "reports"
