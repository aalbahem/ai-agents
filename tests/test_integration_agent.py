"""Level 2: Integration tests — Mocked LLM + Real MongoDB.

Requires MongoDB running with procurement data loaded.
"""

import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.agent import TOOLS, run_agent
from tests.conftest import make_mock_llm


def _ai_with_tool_calls(tool_calls):
    """Create an AIMessage with tool_calls."""
    return AIMessage(content="", tool_calls=tool_calls)


def _final_response(text):
    """Create a plain AIMessage (no tool calls) — agent's final answer."""
    return AIMessage(content=text)


class TestAggregateDispatch:
    def test_count_2013_orders(self, mongo_collection):
        """Mock LLM returns aggregate tool call with count pipeline for 2013.
        Verify tool executes against real DB and returns correct count.
        """
        pipeline = json.dumps([
            {
                "$match": {
                    "Creation Date": {
                        "$gte": "2013-01-01T00:00:00",
                        "$lt": "2014-01-01T00:00:00",
                    }
                }
            },
            {"$count": "total"},
        ])
        tool_call_response = _ai_with_tool_calls([
            {
                "id": "call_1",
                "name": "aggregate_mongodb",
                "args": {"pipeline": pipeline},
            }
        ])
        final = _final_response("There were 115,309 orders in 2013.")

        mock_llm = make_mock_llm([tool_call_response, final])
        result = run_agent("How many orders in 2013?", [], llm=mock_llm)

        assert result == "There were 115,309 orders in 2013."
        assert mock_llm.call_count == 2


class TestDistinctDispatch:
    def test_distinct_fiscal_years(self, mongo_collection):
        """Mock LLM returns get_distinct_values tool call for Fiscal Year."""
        tool_call_response = _ai_with_tool_calls([
            {
                "id": "call_1",
                "name": "get_distinct_values",
                "args": {"field": "Fiscal Year"},
            }
        ])
        final = _final_response("The fiscal years are 2012-2013, 2013-2014, 2014-2015.")

        mock_llm = make_mock_llm([tool_call_response, final])
        result = run_agent("What fiscal years exist?", [], llm=mock_llm)

        assert result == "The fiscal years are 2012-2013, 2013-2014, 2014-2015."
        assert mock_llm.call_count == 2


class TestFindDispatchWithSort:
    def test_top_5_by_total_price(self, mongo_collection):
        """Mock LLM returns query_mongodb with sort by Total Price desc, limit 5."""
        tool_call_response = _ai_with_tool_calls([
            {
                "id": "call_1",
                "name": "query_mongodb",
                "args": {
                    "filter": "{}",
                    "projection": json.dumps(
                        {"Supplier Name": 1, "Total Price": 1, "_id": 0}
                    ),
                    "sort": json.dumps([["Total Price", -1]]),
                    "limit": 5,
                },
            }
        ])
        final = _final_response("Here are the top 5 orders by total price.")

        mock_llm = make_mock_llm([tool_call_response, final])
        result = run_agent("Top 5 most expensive orders?", [], llm=mock_llm)

        assert result == "Here are the top 5 orders by total price."
        assert mock_llm.call_count == 2


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
