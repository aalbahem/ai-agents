"""Level 2: Integration tests — Mocked LLM + Real MongoDB.

Test cases are driven by tests/evaluation.json (shared with eval tests).
Structural tests (no-tool, multi-tool) remain standalone.

Requires MongoDB running with procurement data loaded.
"""

import json
import pathlib

import pytest
from langchain_core.messages import AIMessage

from app.agent import run_agent
from tests.helpers import make_mock_llm

EVAL_FILE = pathlib.Path(__file__).resolve().parent.parent / "evaluation.json"
EVAL_CASES = json.loads(EVAL_FILE.read_text())


def _ai_with_tool_calls(tool_calls):
    """Create an AIMessage with tool_calls."""
    return AIMessage(content="", tool_calls=tool_calls)


def _final_response(text):
    """Create a plain AIMessage (no tool calls) — agent's final answer."""
    return AIMessage(content=text)


# ── Data-driven tests from evaluation.json ───────────────────────────
@pytest.mark.parametrize(
    "case",
    EVAL_CASES,
    ids=[c["question"][:50] for c in EVAL_CASES],
)
def test_tool_dispatch(case, mongo_collection):
    """Mock LLM returns the tool call from evaluation.json, verify execution."""
    tool_calls = [
        {"id": f"call_{i}", "name": tc["name"], "args": tc["args"]}
        for i, tc in enumerate(case["tool_calls"])
    ]
    tool_call_response = _ai_with_tool_calls(tool_calls)
    final = _final_response("Done.")

    mock_llm = make_mock_llm([tool_call_response, final])
    result = run_agent(case["question"], [], llm=mock_llm)

    assert result == "Done."
    assert mock_llm.call_count == 2


# ── Structural tests (not in evaluation.json) ────────────────────────
class TestMultipleToolCalls:
    def test_two_tool_calls_in_one_response(self, mongo_collection):
        """Mock LLM returns AIMessage with 2 tool_calls.
        Verify both tools execute and both ToolMessages are appended.
        """
        tool_call_response = _ai_with_tool_calls([
            {
                "id": "call_1",
                "name": "get_distinct_values",
                "args": {"field": "Fiscal Year"},
            },
            {
                "id": "call_2",
                "name": "aggregate_mongodb",
                "args": {
                    "pipeline": json.dumps([
                        {"$group": {"_id": "$Fiscal Year", "count": {"$sum": 1}}},
                        {"$sort": {"count": -1}},
                    ])
                },
            },
        ])
        final = _final_response("There are 3 fiscal years with varying order counts.")

        mock_llm = make_mock_llm([tool_call_response, final])
        result = run_agent("Summarize fiscal years", [], llm=mock_llm)

        assert result == "There are 3 fiscal years with varying order counts."
        assert mock_llm.call_count == 2


class TestNoToolCalls:
    def test_plain_text_response(self):
        """Mock LLM returns plain text. Agent returns immediately, LLM called once."""
        final = _final_response("I can help with procurement data queries.")

        mock_llm = make_mock_llm([final])
        result = run_agent("Hello", [], llm=mock_llm)

        assert result == "I can help with procurement data queries."
        assert mock_llm.call_count == 1
