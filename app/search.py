
"""Typesense vector search for resolving user intent to exact field values."""

from app.config import (
    TYPESENSE_API_KEY,
    TYPESENSE_COLLECTION,
    TYPESENSE_HOST,
    TYPESENSE_PORT,
)

_client = None
_embedding_model = None


def get_typesense_client():
    """Return a lazily-initialised Typesense client."""
    import typesense

    global _client
    if _client is None:
        _client = typesense.Client(
            {
                "nodes": [
                    {
                        "host": TYPESENSE_HOST,
                        "port": TYPESENSE_PORT,
                        "protocol": "http",
                    }
                ],
                "api_key": TYPESENSE_API_KEY,
                "connection_timeout_seconds": 5,
            }
        )
    return _client


def _get_embedding_model():
    """Return a lazily-loaded sentence-transformer model."""
    from sentence_transformers import SentenceTransformer

    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def embed_text(text: str) -> list[float]:
    """Embed a single string. Swap this function to change the embedding model."""
    model = _get_embedding_model()
    return model.encode(text).tolist()


def search_similar_values(
    query: str,
    field_name: str,
    limit: int = 5,
) -> list[dict]:
    """Embed *query* and vector-search Typesense for the closest field values.

    Args:
        query: The user's natural-language term (e.g. "Health Department").
        field_name: Which field to search within (e.g. "Department Name").
        limit: Maximum number of results to return.

    Returns:
        List of dicts with ``value`` and ``score`` keys.
    """
    client = get_typesense_client()
    query_embedding = embed_text(query)

    # Use multi_search (POST) to avoid URL length limits with 384-dim vectors.
    results = client.multi_search.perform(
        {
            "searches": [
                {
                    "collection": TYPESENSE_COLLECTION,
                    "q": "*",
                    "filter_by": f"field_name:={field_name}",
                    "vector_query": f"embedding:([{','.join(str(v) for v in query_embedding)}], k:{limit})",
                }
            ]
        },
        {},
    )

    return [
        {
            "value": hit["document"]["value"],
            "score": hit["vector_distance"],
        }
        for hit in results["results"][0]["hits"]
    ]
