"""Index distinct MongoDB field values into Typesense.

Semantic fields get vector embeddings for meaning-based search.
Fuzzy fields get text-only indexing for typo-tolerant matching.

Idempotent: drops and recreates the collection on every run.

Usage:
    python scripts/index_typesense.py
"""

import hashlib

from pymongo import MongoClient

from common.config import (
    COLLECTION_NAME,
    DB_NAME,
    MONGO_URI,
    TYPESENSE_COLLECTION,
)
from clients.typesense import embed_text, get_typesense_client

# Semantic fields — meaning-based vector search (user paraphrases concepts).
SEMANTIC_FIELDS = [
    "Department Name",
    "Item Name",
    "Acquisition Type",
    "Acquisition Method",
]

# Fuzzy fields — typo-tolerant text search (proper nouns, codes).
FUZZY_FIELDS = [
    "Supplier Name",
    "Supplier Code",
    "Supplier Zip Code",
]

COLLECTION_SCHEMA = {
    "name": TYPESENSE_COLLECTION,
    "fields": [
        {"name": "field_name", "type": "string", "facet": True},
        {"name": "search_type", "type": "string", "facet": True},
        {"name": "value", "type": "string"},
        {"name": "embedding", "type": "float[]", "num_dim": 384, "optional": True},
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

    # Index semantic fields with embeddings.
    for field in SEMANTIC_FIELDS:
        values = collection.distinct(field)
        values = [v for v in values if v is not None and str(v).strip()]
        print(f"  {field}: {len(values)} distinct values (semantic)")

        for value in values:
            str_value = str(value)
            doc = {
                "id": _stable_id(field, str_value),
                "field_name": field,
                "search_type": "semantic",
                "value": str_value,
                "embedding": embed_text(str_value),
            }
            client.collections[TYPESENSE_COLLECTION].documents.upsert(doc)
            total += 1

    # Index fuzzy fields without embeddings.
    for field in FUZZY_FIELDS:
        values = collection.distinct(field)
        values = [v for v in values if v is not None and str(v).strip()]
        print(f"  {field}: {len(values)} distinct values (fuzzy)")

        for value in values:
            str_value = str(value)
            doc = {
                "id": _stable_id(field, str_value),
                "field_name": field,
                "search_type": "fuzzy",
                "value": str_value,
            }
            client.collections[TYPESENSE_COLLECTION].documents.upsert(doc)
            total += 1

    print(f"Indexed {total} documents into '{TYPESENSE_COLLECTION}'")
    mongo.close()


if __name__ == "__main__":
    main()
