"""MongoDB connection and query helpers."""

import json
from datetime import datetime
from typing import Any

from bson import ObjectId
from pymongo import MongoClient

from common.config import COLLECTION_NAME, DB_NAME, MONGO_URI

_client: MongoClient | None = None
_override_collection = None


def set_collection(collection):
    global _override_collection
    _override_collection = collection


def _get_collection():
    if _override_collection is not None:
        return _override_collection
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URI)
    return _client[DB_NAME][COLLECTION_NAME]


def _serialize(doc: dict) -> dict:
    """Convert MongoDB document to JSON-serializable dict."""
    out = {}
    for k, v in doc.items():
        if isinstance(v, ObjectId):
            out[k] = str(v)
        elif isinstance(v, datetime):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out


def run_find(
    filter_doc: dict[str, Any],
    projection: dict[str, Any] | None = None,
    sort: list[list] | None = None,
    limit: int = 20,
) -> str:
    """Execute a find() query and return results as JSON string."""
    col = _get_collection()
    cursor = col.find(filter_doc, projection)
    if sort:
        cursor = cursor.sort(sort)
    cursor = cursor.limit(limit)
    results = [_serialize(doc) for doc in cursor]
    return json.dumps(results, default=str)


def run_aggregate(pipeline: list[dict[str, Any]]) -> str:
    """Execute an aggregation pipeline and return results as JSON string."""
    col = _get_collection()
    results = [_serialize(doc) for doc in col.aggregate(pipeline)]
    return json.dumps(results, default=str)


def run_distinct(field: str) -> str:
    """Return distinct values for a field as JSON string."""
    col = _get_collection()
    values = col.distinct(field)
    serialized = []
    for v in values:
        if isinstance(v, datetime):
            serialized.append(v.isoformat())
        elif isinstance(v, ObjectId):
            serialized.append(str(v))
        else:
            serialized.append(v)
    return json.dumps(serialized, default=str)
