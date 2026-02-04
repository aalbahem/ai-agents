"""LangChain @tool functions for procurement data queries."""

import json
from datetime import datetime
from typing import Any

from langchain_core.tools import tool

from clients.mongodb import run_aggregate, run_distinct, run_find
from clients.typesense import search_fuzzy, search_similar_values


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
