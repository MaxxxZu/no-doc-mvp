import pytest
from fastapi.testclient import TestClient


@pytest.mark.parametrize("origins,origin,allowed", [
    (["https://allowed.example"], "https://allowed.example", True),
    (["https://allowed.example"], "https://denied.example", False),
    ([], "https://allowed.example", False),
])
def test_cors(origins, origin, allowed):
    from app import create_app
    from config import Settings

    with TestClient(create_app(Settings(cors_origins=origins))) as client:
        response = client.get("/healthz", headers={"Origin": origin})
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == (origin if allowed else None)
        preflight = client.options("/classify", headers={
            "Origin": origin, "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        })
        assert preflight.status_code == (200 if allowed else 400)
        expected_origin = origin if allowed else None
        assert preflight.headers.get("access-control-allow-origin") == expected_origin
        if allowed:
            assert "POST" in preflight.headers["access-control-allow-methods"]
            assert "content-type" in preflight.headers["access-control-allow-headers"].lower()
