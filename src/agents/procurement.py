# pylint: disable=line-too-long
"""LangChain agent with MongoDB tools powered by Google Gemini."""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from common.config import (
    GEMINI_FALLBACK_MODELS,
    GEMINI_MODEL,
    GEMINI_TEMPERATURE,
    GOOGLE_API_KEY,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)
from tools import TOOLS

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


def _build_gemini():
    from langchain_google_genai import ChatGoogleGenerativeAI

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
    return primary.with_fallbacks(fallbacks) if fallbacks else primary


def _build_openai():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=GEMINI_TEMPERATURE,
        api_key=OPENAI_API_KEY,
    )


def _build_ollama():
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=GEMINI_TEMPERATURE,
    )


_BUILDERS = {
    "openai": _build_openai,
    "gemini": _build_gemini,
    "ollama": _build_ollama,
}


def build_agent():
    """Build and return a LangChain tool-calling agent."""
    builder = _BUILDERS.get(LLM_PROVIDER)
    if builder is None:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER!r}. Choose from: {list(_BUILDERS)}")
    return builder().bind_tools(TOOLS)


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
