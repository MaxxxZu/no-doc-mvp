import os

import pytest
from fastapi.testclient import TestClient

# Ensure tests don't write to your real HF cache
os.environ.setdefault("HF_HOME", "/tmp/.hf-tests")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/.hf-tests/transformers")


@pytest.fixture(scope="session")
def client():
    # Import here to respect env overrides above
    from app import app
    return TestClient(app)
