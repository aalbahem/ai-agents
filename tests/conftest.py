"""Shared fixtures for procurement assistant tests."""

import pytest
from pymongo import MongoClient

from common.config import COLLECTION_NAME, DB_NAME, MONGO_URI
from clients.mongodb import set_collection


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
    module_path = str(request.fspath)
    if "integration_tests" in module_path or "eval_tests" in module_path:
        set_collection(mongo_collection)
        yield
        set_collection(None)
    else:
        yield


