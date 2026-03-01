import pytest


def test_dataset_intake_reference_route_smoke():
    pytest.importorskip("fastapi")
    pytest.importorskip("sqlmodel")
    testclient = pytest.importorskip("fastapi.testclient")
    from main import app

    with testclient.TestClient(app) as client:
        response = client.get("/dataset/intake/reference")

    assert response.status_code == 200
    payload = response.json()
    assert payload.get("success") is True


def test_export_history_route_smoke():
    pytest.importorskip("fastapi")
    pytest.importorskip("sqlmodel")
    testclient = pytest.importorskip("fastapi.testclient")
    from main import app

    with testclient.TestClient(app) as client:
        response = client.get("/dataset/exports/history")

    assert response.status_code == 200
    payload = response.json()
    assert payload.get("success") is True
