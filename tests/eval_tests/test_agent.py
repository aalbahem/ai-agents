"""Level 3: Evaluation tests — Real LLM + Real MongoDB via LangSmith.

Test cases are driven by tests/evaluation.json (shared with integration tests).
Uses LangSmith's evaluate() to run the agent against a dataset,
score each response with custom evaluators, and upload traces + scores
to the LangSmith UI.

Marked with @pytest.mark.eval and excluded from default runs.
Run with: pytest -m eval
"""

import json
import pathlib

import pytest
from langsmith import Client

from agents.procurement import run_agent

EVAL_FILE = pathlib.Path(__file__).resolve().parent.parent / "evaluation.json"
EVAL_CASES = json.loads(EVAL_FILE.read_text())

DATASET_NAME = "procurement-assistant-eval"


def _build_langsmith_examples(tags=None):
    """Convert evaluation.json into LangSmith dataset format.

    Args:
        tags: Optional list of tags to filter cases by. A case is included
              if it has at least one matching tag. None means include all.
    """
    cases = EVAL_CASES
    if tags:
        cases = [c for c in cases if set(tags) & set(c.get("tags", []))]
    return [
        {
            "inputs": {"question": case["question"]},
            "outputs": {"expected_values": case["expected_values"]},
            "metadata": {"id": case["id"], "tags": case["tags"]},
        }
        for case in cases
        if case.get("expected_values")  # skip cases with no expected values
    ]


# ── Target function ──────────────────────────────────────────────────
def procurement_agent(inputs: dict) -> dict:
    """Wrap run_agent() into the signature LangSmith expects."""
    result = run_agent(inputs["question"], [])
    return {"answer": result}


# ── Evaluators ───────────────────────────────────────────────────────
def contains_expected(outputs: dict, reference_outputs: dict) -> dict:
    """Check that every expected value appears in the agent's answer."""
    answer = (outputs.get("answer") or "").replace(",", "")
    expected_values = reference_outputs.get("expected_values", [])
    score = all(val in answer for val in expected_values)
    return {"key": "contains_expected", "score": score}


def non_empty(outputs: dict) -> dict:
    """Check that the agent returned a non-empty answer."""
    answer = (outputs.get("answer") or "").strip()
    return {"key": "non_empty", "score": bool(answer)}


# ── Summary evaluator ────────────────────────────────────────────────
def pass_rate(outputs: list, reference_outputs: list) -> dict:
    """Fraction of examples where all expected values were found."""
    passed = 0
    for out, ref in zip(outputs, reference_outputs):
        answer = (out.get("answer") or "").replace(",", "")
        if all(val in answer for val in ref.get("expected_values", [])):
            passed += 1
    return {"key": "pass_rate", "score": passed / len(outputs) if outputs else 0}


# ── Helpers ──────────────────────────────────────────────────────────
def _ensure_dataset(client: Client) -> None:
    """Create (or recreate) the eval dataset in LangSmith."""
    examples = _build_langsmith_examples()

    for ds in client.list_datasets(dataset_name=DATASET_NAME):
        client.delete_dataset(dataset_id=ds.id)

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Procurement assistant end-to-end evaluation",
    )
    client.create_examples(dataset_id=dataset.id, examples=examples)


# ── Test ─────────────────────────────────────────────────────────────
@pytest.mark.eval
def test_langsmith_eval(mongo_collection):
    """Run the full evaluation via LangSmith and assert pass rate."""
    client = Client()
    _ensure_dataset(client)

    results = client.evaluate(
        procurement_agent,
        data=DATASET_NAME,
        evaluators=[contains_expected, non_empty],
        summary_evaluators=[pass_rate],
        experiment_prefix="procurement-eval",
        description="End-to-end procurement assistant evaluation",
        max_concurrency=0,  # sequential to respect Gemini rate limits
    )

    # Verify locally too: every example should have contains_expected=True
    for result in results:
        scores = result["evaluation_results"]
        eval_results = scores.get("results", []) if isinstance(scores, dict) else scores
        contains_scores = [
            r for r in eval_results
            if getattr(r, "key", None) == "contains_expected"
        ]
        assert contains_scores, f"No contains_expected score for {result['example'].inputs}"
        assert contains_scores[0].score, (
            f"Expected values not found in answer for: {result['example'].inputs}\n"
            f"Got: {result['run'].outputs}"
        )
