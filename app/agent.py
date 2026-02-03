"""LangChain agent with MongoDB tools powered by Google Gemini."""

import json
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import (
    GEMINI_FALLBACK_MODELS,
    GEMINI_MODEL,
    GEMINI_TEMPERATURE,
    GOOGLE_API_KEY,
)
from app.db import run_aggregate, run_distinct, run_find
from app.search import search_fuzzy, search_similar_values

SYSTEM_PROMPT = """You are a procurement data analyst assistant. You answer questions about California state government purchase order data (2012–2015) stored in a MongoDB collection called `procurement.purchases`.

## Schema

Each document has these fields:

| Field | Type | Example |
|---|---|---|
| Creation Date | datetime | 2013-08-27T00:00:00 |
| Purchase Date | datetime | 2013-09-15T00:00:00 (often null) |
| Fiscal Year | string | "2013-2014" |
| LPA Number | string | "7-12-70-26" |
| Purchase Order Number | string | "REQ0011118" |
| Requisition Number | string | "REQ0011118" |
| Acquisition Type | string | "IT Goods", "NON-IT Goods", "IT Services", "NON-IT Services" |
| Sub-Acquisition Type | string | |
| Acquisition Method | string | "WSCA/Coop", "Informal Competitive", etc. |
| Sub-Acquisition Method | string | |
| Department Name | string | "Consumer Affairs, Department of" |
| Supplier Code | string | "1740272" |
| Supplier Name | string | "Pitney Bowes" |
| Supplier Qualifications | string | |
| Supplier Zip Code | string | "95841" |
| CalCard | string | "YES" or "NO" |
| Item Name | string | "USB", "Tire Disposal", "Labor" |
| Item Description | string | |
| Quantity | float | 4.5 |
| Unit Price | float | 150.00 |
| Total Price | float | 675.00 |
| Classification Codes | string | "76121504" |
| Normalized UNSPSC | string | "76121504" |
| Commodity Title | string | |
| Class | string | |
| Class Title | string | |
| Family | string | |
| Family Title | string | |
| Segment | string | |
| Segment Title | string | |
| Location | string | |

## Query Guidelines

- Dates are stored as ISODate. To query by year, use: `{"Creation Date": {"$gte": "2013-01-01T00:00:00", "$lt": "2014-01-01T00:00:00"}}`
  IMPORTANT: Always pass dates as ISO 8601 strings like "2013-01-01T00:00:00". They will be converted to datetime objects automatically.
- Prices (Unit Price, Total Price) are floats, NOT strings. Query them numerically.
- Fiscal Year is a string like "2013-2014".
- For text matching, use exact match or `{"$regex": "pattern", "$options": "i"}` for case-insensitive.
- When a user refers to a department or item by name, first try querying MongoDB directly. If the query returns zero results, the name was probably informal or abbreviated — use `find_similar_values` to resolve it to the exact database value, then re-query.
- When a user refers to a supplier by name, code, or zip code, use `find_supplier` to resolve it. This uses typo-tolerant text matching suited for proper nouns (e.g. "Pitney" → "Pitney Bowes").
- When `find_similar_values` returns multiple matches, present them as a **numbered list** to the user and ask which one they meant. Do NOT dump raw JSON. Format like:
  "I found several possible matches for 'Health Department':
   1. Health Care Services, Department of
   2. Public Health, Department of
   3. Mental Health, Department of
   Which one did you mean?"
  Once the user picks one, use that exact value in your MongoDB query.
- If `find_similar_values` returns exactly one match with a very low score (< 0.3), you may use it directly without asking.
- For "top N" or "highest/lowest" questions, use aggregation with $group, $sort, $limit.
- For counting, use aggregation with $group and $count, or $match followed by $count.
- For sums/averages, use $group with $sum/$avg on numeric fields.

## Example Aggregation Pipelines

### Count orders by fiscal year:
```json
[{"$group": {"_id": "$Fiscal Year", "count": {"$sum": 1}}}, {"$sort": {"count": -1}}]
```

### Total spending by department for a fiscal year:
```json
[{"$match": {"Fiscal Year": "2014-2015"}}, {"$group": {"_id": "$Department Name", "total": {"$sum": "$Total Price"}}}, {"$sort": {"total": -1}}]
```

### Top 10 most ordered items:
```json
[{"$group": {"_id": "$Item Name", "count": {"$sum": 1}}}, {"$sort": {"count": -1}}, {"$limit": 10}]
```

Always use the tools to query the database. Do not make up data. Present results clearly, with formatting (tables, lists) where appropriate."""


def _extract_text(content: Any) -> str:
    """Extract plain text from an LLM response content field.

    LangChain model responses may return content as a plain string or as a list
    of content blocks (e.g. [{"type": "text", "text": "...", "extras": {...}}]).
    This normalises both forms to a plain string suitable for display and for
    feeding back into conversation history.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def _parse_dates(obj: Any) -> Any:
    """Recursively convert ISO date strings in query dicts to datetime objects."""
    if isinstance(obj, str):
        # Try parsing ISO date strings
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(obj, fmt)
            except ValueError:
                continue
        return obj
    elif isinstance(obj, dict):
        return {k: _parse_dates(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_parse_dates(item) for item in obj]
    return obj


@tool
def query_mongodb(
    filter: str,
    projection: str = "null",
    sort: str = "null",
    limit: int = 20,
) -> str:
    """Run a MongoDB find() query on the procurement purchases collection.

    Args:
        filter: JSON string of the MongoDB query filter, e.g. '{"Fiscal Year": "2013-2014"}'.
                For date queries, use ISO strings like "2013-01-01T00:00:00".
        projection: JSON string of fields to include/exclude, e.g. '{"Item Name": 1, "Total Price": 1}'.
                    Pass "null" to return all fields.
        sort: JSON string of sort specification as a list of [field, direction] pairs,
              e.g. '[["Total Price", -1]]'. Pass "null" for no sort.
        limit: Maximum number of documents to return (default 20).

    Returns:
        JSON string of matching documents.
    """
    filter_doc = _parse_dates(json.loads(filter))
    proj = json.loads(projection) if projection != "null" else None
    sort_spec = json.loads(sort) if sort != "null" else None
    return run_find(filter_doc, proj, sort_spec, limit)


@tool
def aggregate_mongodb(pipeline: str) -> str:
    """Run a MongoDB aggregation pipeline on the procurement purchases collection.

    Use this for analytical queries: counting, grouping, summing, averaging, top-N, etc.

    Args:
        pipeline: JSON string of the aggregation pipeline (a list of stage objects).
                  For date queries within $match stages, use ISO strings like "2013-01-01T00:00:00".
                  Example: '[{"$group": {"_id": "$Fiscal Year", "count": {"$sum": 1}}}, {"$sort": {"count": -1}}]'

    Returns:
        JSON string of aggregation results.
    """
    parsed = _parse_dates(json.loads(pipeline))
    return run_aggregate(parsed)


@tool
def get_distinct_values(field: str) -> str:
    """Get all distinct values for a given field in the procurement collection.

    Useful for discovering what values exist (e.g., department names, fiscal years, acquisition types).

    Args:
        field: The field name to get distinct values for, e.g. "Fiscal Year" or "Department Name".

    Returns:
        JSON string containing a list of distinct values.
    """
    return run_distinct(field)


@tool
def find_similar_values(query: str, field_name: str, limit: int = 5) -> str:
    """Find database values that semantically match a user's natural-language term.

    Use this when the user refers to a department, supplier, item, or other
    categorical field by an informal or abbreviated name. It returns the closest
    exact values stored in the database so you can build accurate MongoDB queries.

    Args:
        query: The user's term, e.g. "Health Department" or "Dell".
        field_name: The collection field to search within, e.g. "Department Name",
                    "Supplier Name", "Item Name", "Acquisition Type",
                    "Acquisition Method".
        limit: Maximum number of matches to return (default 5).

    Returns:
        JSON string of matches, each with "value" and "score" keys.
        Lower score means a closer match.
        Present the matches as a numbered list to the user and ask them
        to pick one — do NOT show raw JSON to the user.
    """
    results = search_similar_values(query, field_name, limit)
    return json.dumps(results)


@tool
def find_supplier(query: str, field_name: str = "Supplier Name", limit: int = 5) -> str:
    """Find suppliers by name, code, or zip code using typo-tolerant text search.

    Use this when the user refers to a supplier by a partial, abbreviated, or
    misspelled name. Unlike find_similar_values (which uses meaning-based
    search), this uses fuzzy text matching suited for proper nouns.

    Args:
        query: The user's term, e.g. "Pitney", "Delta Dental", "95841".
        field_name: The supplier field to search: "Supplier Name",
                    "Supplier Code", or "Supplier Zip Code".
        limit: Maximum number of matches to return (default 5).

    Returns:
        JSON string of matches, each with "value" and "score" keys.
        Higher score means a closer match.
        Present the matches as a numbered list to the user and ask them
        to pick one — do NOT show raw JSON to the user.
    """
    results = search_fuzzy(query, field_name, limit)
    return json.dumps(results)


TOOLS = [query_mongodb, aggregate_mongodb, get_distinct_values, find_similar_values, find_supplier]


def build_agent():
    """Build and return a LangChain tool-calling agent."""
    primary = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=GEMINI_TEMPERATURE,
        google_api_key=GOOGLE_API_KEY,
    )
    fallbacks = [
        ChatGoogleGenerativeAI(
            model=model,
            temperature=GEMINI_TEMPERATURE,
            google_api_key=GOOGLE_API_KEY,
        )
        for model in GEMINI_FALLBACK_MODELS
        if model != GEMINI_MODEL
    ]
    llm = primary.with_fallbacks(fallbacks) if fallbacks else primary
    return llm.bind_tools(TOOLS)


def run_agent(user_input: str, history: list[dict], llm=None) -> str:
    """Run the agent on a user query, maintaining conversation history.

    Args:
        user_input: The user's question.
        history: List of {"role": "user"|"assistant", "content": "..."} dicts.
        llm: Optional pre-built LLM instance. If None, builds a new one.

    Returns:
        The agent's final text response.
    """
    if llm is None:
        llm = build_agent()
    tools_by_name = {t.name: t for t in TOOLS}

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=user_input))

    # Agentic loop: call LLM, invoke tools, repeat until no more tool calls
    for _ in range(10):  # max iterations to prevent infinite loops
        response = llm.invoke(messages)

        if not response.tool_calls:
            return _extract_text(response.content)

        # Keep the full response object in messages for LangChain's
        # tool-call tracking, but replace content with plain text so
        # later turns don't re-send model metadata (signatures, etc.).
        response.content = _extract_text(response.content)
        messages.append(response)

        # Execute each tool call and append results
        from langchain_core.messages import ToolMessage

        for tc in response.tool_calls:
            tool_fn = tools_by_name[tc["name"]]
            result = tool_fn.invoke(tc["args"])
            messages.append(
                ToolMessage(content=str(result), tool_call_id=tc["id"])
            )

    # If we exhausted iterations, return whatever we have
    return _extract_text(response.content)
