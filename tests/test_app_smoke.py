import pytest


def test_root_endpoint_returns_api_message():
    pytest.importorskip("fastapi")
    testclient = pytest.importorskip("fastapi.testclient")
    from main import app

    with testclient.TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Data Processing API"}
