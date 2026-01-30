"""Shared fixtures for procurement assistant tests."""

import os

import pytest
from pymongo import MongoClient

from app.config import COLLECTION_NAME, DB_NAME, MONGO_URI
from app.db import set_collection


@pytest.fixture(scope="session")
def mongo_collection():
    """Connect to the real MongoDB collection for integration/eval tests."""
    client = MongoClient(MONGO_URI)
    col = client[DB_NAME][COLLECTION_NAME]
    yield col
    client.close()


@pytest.fixture(autouse=True)
def _use_real_db(request, mongo_collection):
    """Automatically inject the real MongoDB collection for integration and eval tests.

    Only applies to tests in test_integration_* and test_eval_* modules.
    """
    module_name = request.module.__name__
    if "integration" in module_name or "eval" in module_name:
        set_collection(mongo_collection)
        yield
        set_collection(None)
    else:
        yield


def make_mock_llm(responses):
    """Create a mock LLM that returns predetermined responses in sequence.

    Args:
        responses: List of AIMessage objects to return on successive invoke() calls.

    Returns:
        A mock object with an invoke() method.
    """

    class _MockLLM:
        def __init__(self, responses):
            self._responses = list(responses)
            self._call_count = 0

        def invoke(self, messages):
            idx = min(self._call_count, len(self._responses) - 1)
            self._call_count += 1
            return self._responses[idx]

        @property
        def call_count(self):
            return self._call_count

    return _MockLLM(responses)
