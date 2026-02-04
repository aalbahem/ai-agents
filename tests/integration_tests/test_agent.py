"""Level 2: Integration tests — Mocked LLM + Real MongoDB.

Test cases are driven by tests/evaluation.json (shared with eval tests).
Structural tests (no-tool, multi-tool) remain standalone.

Requires MongoDB running with procurement data loaded.
"""

import json
import pathlib
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage

from agents.procurement import run_agent
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

# Cases that hit Typesense need the search backend mocked.
_SEMANTIC_MOCK_RETURN = [
    {"value": "Health Care Services, Department of", "score": 0.12},
]

_FUZZY_MOCK_RETURN = [
    {"value": "Pitney Bowes", "score": 1234},
]


def _tool_names(case):
    return {tc["name"] for tc in case["tool_calls"]}


@pytest.mark.parametrize(
    "case",
    EVAL_CASES,
    ids=[c["id"] for c in EVAL_CASES],
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
    names = _tool_names(case)

    patches = {}
    if "find_similar_values" in names:
        patches["tools.procurement.search_similar_values"] = _SEMANTIC_MOCK_RETURN
    if "find_supplier" in names:
        patches["tools.procurement.search_fuzzy"] = _FUZZY_MOCK_RETURN

    if patches:
        from contextlib import ExitStack

        with ExitStack() as stack:
            for target, rv in patches.items():
                stack.enter_context(patch(target, return_value=rv))
            result = run_agent(case["question"], [], llm=mock_llm)
    else:
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


class TestFindSimilarValues:
    """Integration tests for the find_similar_values tool.

    These test semantic search resolution — the user says an informal name
    like "Health Department" and the tool resolves it to the exact DB value
    "Health Care Services, Department of".  The wording gap (different words,
    different order) is what makes this a semantic match, not an exact one.

    Mocks the search backend (Typesense + embeddings) since it may not be
    running, but exercises the full agent tool-dispatch path.
    """

    @patch("tools.procurement.search_similar_values")
    def test_tool_dispatches_and_returns_results(self, mock_search):
        """Verify the agent dispatches find_similar_values with the right args.

        User says "Health Department" — not an exact match for any DB value.
        The tool should resolve it to "Health Care Services, Department of".
        """
        mock_search.return_value = [
            {"value": "Health Care Services, Department of", "score": 0.12},
            {"value": "Public Health, Department of", "score": 0.25},
        ]

        tool_call_response = _ai_with_tool_calls([
            {
                "id": "call_0",
                "name": "find_similar_values",
                "args": {
                    "query": "Health Department",
                    "field_name": "Department Name",
                },
            },
        ])
        final = _final_response("The closest match is Health Care Services, Department of.")

        mock_llm = make_mock_llm([tool_call_response, final])
        result = run_agent("orders from Health Department", [], llm=mock_llm)

        assert result == "The closest match is Health Care Services, Department of."
        mock_search.assert_called_once_with(
            "Health Department", "Department Name", 5
        )

    @patch("tools.procurement.search_similar_values")
    def test_chained_with_mongodb_query(self, mock_search, mongo_collection):
        """Full two-step flow: resolve informal name, then query MongoDB.

        Turn 1 — LLM calls find_similar_values("Health Department")
                  → returns "Health Care Services, Department of"
        Turn 2 — LLM uses that exact value to query MongoDB
        Turn 3 — LLM returns final answer
        """
        mock_search.return_value = [
            {"value": "Health Care Services, Department of", "score": 0.1},
        ]

        search_call = _ai_with_tool_calls([
            {
                "id": "call_0",
                "name": "find_similar_values",
                "args": {
                    "query": "Health Department",
                    "field_name": "Department Name",
                },
            },
        ])
        mongo_call = _ai_with_tool_calls([
            {
                "id": "call_1",
                "name": "aggregate_mongodb",
                "args": {
                    "pipeline": json.dumps([
                        {"$match": {"Department Name": "Health Care Services, Department of"}},
                        {"$count": "total"},
                    ])
                },
            },
        ])
        final = _final_response("Found 42 orders.")

        mock_llm = make_mock_llm([search_call, mongo_call, final])
        result = run_agent("How many orders from Health Department?", [], llm=mock_llm)

        assert result == "Found 42 orders."
        assert mock_llm.call_count == 3


class TestFindSupplier:
    """Integration tests for the find_supplier tool.

    These test fuzzy text search for supplier fields — partial names,
    typos, and zip codes resolved via Typesense typo-tolerant matching.

    Mocks the search backend but exercises the full agent tool-dispatch path.
    """

    @patch("tools.procurement.search_fuzzy")
    def test_partial_supplier_name(self, mock_fuzzy):
        """Partial name 'Pitney' resolves to 'Pitney Bowes'."""
        mock_fuzzy.return_value = [
            {"value": "Pitney Bowes", "score": 1234},
        ]

        tool_call_response = _ai_with_tool_calls([
            {
                "id": "call_0",
                "name": "find_supplier",
                "args": {
                    "query": "Pitney",
                    "field_name": "Supplier Name",
                },
            },
        ])
        final = _final_response("The closest match is Pitney Bowes.")

        mock_llm = make_mock_llm([tool_call_response, final])
        result = run_agent("How much did Pitney spend?", [], llm=mock_llm)

        assert result == "The closest match is Pitney Bowes."
        mock_fuzzy.assert_called_once_with("Pitney", "Supplier Name", 5)

    @patch("tools.procurement.search_fuzzy")
    def test_supplier_then_mongodb_query(self, mock_fuzzy, mongo_collection):
        """Two-step: resolve supplier name, then query MongoDB."""
        mock_fuzzy.return_value = [
            {"value": "Pitney Bowes", "score": 1234},
        ]

        search_call = _ai_with_tool_calls([
            {
                "id": "call_0",
                "name": "find_supplier",
                "args": {
                    "query": "Pitney",
                    "field_name": "Supplier Name",
                },
            },
        ])
        mongo_call = _ai_with_tool_calls([
            {
                "id": "call_1",
                "name": "aggregate_mongodb",
                "args": {
                    "pipeline": json.dumps([
                        {"$match": {"Supplier Name": "Pitney Bowes"}},
                        {"$group": {"_id": None, "total": {"$sum": "$Total Price"}}},
                    ])
                },
            },
        ])
        final = _final_response("Pitney Bowes spent $123,456.")

        mock_llm = make_mock_llm([search_call, mongo_call, final])
        result = run_agent("How much did Pitney spend?", [], llm=mock_llm)

        assert result == "Pitney Bowes spent $123,456."
        assert mock_llm.call_count == 3

    @patch("tools.procurement.search_fuzzy")
    def test_supplier_zip_code(self, mock_fuzzy):
        """Zip code lookup dispatches find_supplier with Supplier Zip Code."""
        mock_fuzzy.return_value = [
            {"value": "95841", "score": 999},
        ]

        tool_call_response = _ai_with_tool_calls([
            {
                "id": "call_0",
                "name": "find_supplier",
                "args": {
                    "query": "95841",
                    "field_name": "Supplier Zip Code",
                },
            },
        ])
        final = _final_response("Found suppliers in 95841.")

        mock_llm = make_mock_llm([tool_call_response, final])
        result = run_agent("Which suppliers are in zip 95841?", [], llm=mock_llm)

        assert result == "Found suppliers in 95841."
        mock_fuzzy.assert_called_once_with("95841", "Supplier Zip Code", 5)
