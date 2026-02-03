"""Index distinct MongoDB field values into Typesense for semantic search.

Idempotent: drops and recreates the collection on every run.

Usage:
    python scripts/index_typesense.py
"""

import hashlib

from pymongo import MongoClient

from app.config import (
    COLLECTION_NAME,
    DB_NAME,
    MONGO_URI,
    TYPESENSE_COLLECTION,
)
from app.search import embed_text, get_typesense_client

# Fields whose distinct values we want to make searchable.
TARGET_FIELDS = [
    "Department Name",
    "Supplier Name",
    "Item Name",
    "Acquisition Type",
    "Acquisition Method",
]

COLLECTION_SCHEMA = {
    "name": TYPESENSE_COLLECTION,
    "fields": [
        {"name": "field_name", "type": "string", "facet": True},
        {"name": "value", "type": "string"},
        {"name": "embedding", "type": "float[]", "num_dim": 384},
    ],
}


def _stable_id(field_name: str, value: str) -> str:
    """Deterministic document id so upserts are safe."""
    return hashlib.sha256(f"{field_name}:{value}".encode()).hexdigest()[:16]


def main() -> None:
    client = get_typesense_client()

    # Drop existing collection (idempotent).
    try:
        client.collections[TYPESENSE_COLLECTION].delete()
        print(f"Dropped existing collection '{TYPESENSE_COLLECTION}'")
    except Exception:
        pass

    client.collections.create(COLLECTION_SCHEMA)
    print(f"Created collection '{TYPESENSE_COLLECTION}'")

    # Connect to MongoDB.
    mongo = MongoClient(MONGO_URI)
    collection = mongo[DB_NAME][COLLECTION_NAME]

    total = 0
    for field in TARGET_FIELDS:
        values = collection.distinct(field)
        values = [v for v in values if v is not None and str(v).strip()]
        print(f"  {field}: {len(values)} distinct values")

        for value in values:
            str_value = str(value)
            doc = {
                "id": _stable_id(field, str_value),
                "field_name": field,
                "value": str_value,
                "embedding": embed_text(str_value),
            }
            client.collections[TYPESENSE_COLLECTION].documents.upsert(doc)
            total += 1

    print(f"Indexed {total} documents into '{TYPESENSE_COLLECTION}'")
    mongo.close()


if __name__ == "__main__":
    main()
