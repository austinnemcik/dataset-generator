import pytest
from uuid import uuid4


def test_dataset_intake_reference_route_smoke():
    pytest.importorskip("fastapi")
    pytest.importorskip("sqlmodel")
    testclient = pytest.importorskip("fastapi.testclient")
    from app.main import app

    with testclient.TestClient(app) as client:
        response = client.get("/dataset/intake/reference")

    assert response.status_code == 200
    payload = response.json()
    assert payload.get("success") is True


def test_export_history_route_smoke():
    pytest.importorskip("fastapi")
    pytest.importorskip("sqlmodel")
    testclient = pytest.importorskip("fastapi.testclient")
    from app.main import app

    with testclient.TestClient(app) as client:
        response = client.get("/dataset/exports/history")

    assert response.status_code == 200
    payload = response.json()
    assert payload.get("success") is True


def test_delete_completed_batch_run_route():
    pytest.importorskip("fastapi")
    pytest.importorskip("sqlmodel")
    testclient = pytest.importorskip("fastapi.testclient")
    from sqlmodel import Session, select

    from agent.batch_store import create_batch_run
    from app.core.database import BatchRun, BatchRunItem, engine
    from app.main import app

    run_id = f"test-batch-route-delete-{uuid4().hex}"
    item_run_id = f"test-batch-route-item-{uuid4().hex}"
    create_batch_run(
        request_payload={"topic": "route delete", "max_concurrency": 1},
        run_plan=[
            {
                "item_index": 0,
                "run_id": item_run_id,
                "dataset_key": f"{item_run_id}: route delete",
                "slot_key": "route-delete-slot",
                "requested_topic": "route delete",
                "topic": "route delete",
                "agent": "qa",
                "ex_amt": 3,
                "max_retries": 0,
                "retry_backoff_seconds": 1.0,
                "source_material": None,
                "model": "test-model",
                "seed": None,
            }
        ],
        run_id=run_id,
    )

    with Session(engine) as session:
        batch_run = session.exec(select(BatchRun).where(BatchRun.run_id == run_id)).first()
        item = session.exec(select(BatchRunItem).where(BatchRunItem.batch_run_id == batch_run.id)).first()
        batch_run.status = "completed"
        batch_run.completed_runs = 1
        batch_run.queued_runs = 0
        item.status = "completed"
        session.add(batch_run)
        session.add(item)
        session.commit()

    with testclient.TestClient(app) as client:
        response = client.delete(f"/dataset/batch/{run_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload.get("success") is True


def test_delete_dataset_example_route():
    pytest.importorskip("fastapi")
    pytest.importorskip("sqlmodel")
    testclient = pytest.importorskip("fastapi.testclient")
    from sqlmodel import Session

    from app.core.database import Dataset, TrainingExample, engine
    from app.main import app

    with Session(engine) as session:
        dataset = Dataset(
            name=f"example-delete-{uuid4().hex}",
            description="Delete a single example",
            examples=[
                TrainingExample(prompt="p1", instruction="hello", response="world"),
                TrainingExample(prompt="p2", instruction="foo", response="bar"),
            ],
        )
        session.add(dataset)
        session.commit()
        session.refresh(dataset)
        example_id = dataset.examples[0].id
        dataset_id = dataset.id

    with testclient.TestClient(app) as client:
        response = client.delete(f"/dataset/examples/{example_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload.get("success") is True
    assert payload["data"]["dataset_id"] == dataset_id
    assert payload["data"]["remaining_examples"] == 1


def test_update_dataset_example_route():
    pytest.importorskip("fastapi")
    pytest.importorskip("sqlmodel")
    testclient = pytest.importorskip("fastapi.testclient")
    from sqlmodel import Session

    from app.core.database import Dataset, TrainingExample, engine
    from app.main import app

    with Session(engine) as session:
        dataset = Dataset(
            name=f"example-update-{uuid4().hex}",
            description="Update a single example",
            examples=[
                TrainingExample(
                    prompt="p1",
                    instruction="Original instruction",
                    response="Original response",
                    embedding="[0.1, 0.2]",
                )
            ],
        )
        session.add(dataset)
        session.commit()
        session.refresh(dataset)
        example_id = dataset.examples[0].id

    with testclient.TestClient(app) as client:
        response = client.put(
            f"/dataset/examples/{example_id}",
            json={
                "instruction": "Updated instruction",
                "response": "Updated response",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload.get("success") is True
    assert payload["data"]["example"]["instruction"] == "Updated instruction"
    assert payload["data"]["example"]["response"] == "Updated response"

    with Session(engine) as session:
        updated_example = session.get(TrainingExample, example_id)
        assert updated_example is not None
        assert updated_example.instruction == "Updated instruction"
        assert updated_example.response == "Updated response"
        assert updated_example.embedding is None


def test_delete_terminal_batch_runs_route():
    pytest.importorskip("fastapi")
    pytest.importorskip("sqlmodel")
    testclient = pytest.importorskip("fastapi.testclient")
    from sqlmodel import Session, select

    from agent.batch_store import create_batch_run
    from app.core.database import BatchRun, engine
    from app.main import app

    run_id = f"test-batch-route-clear-{uuid4().hex}"
    create_batch_run(
        request_payload={"topic": "route clear", "max_concurrency": 1},
        run_plan=[
            {
                "item_index": 0,
                "run_id": f"test-batch-route-clear-item-{uuid4().hex}",
                "dataset_key": "test-batch-route-clear-item: route clear",
                "slot_key": "route-clear-slot",
                "requested_topic": "route clear",
                "topic": "route clear",
                "agent": "qa",
                "ex_amt": 3,
                "max_retries": 0,
                "retry_backoff_seconds": 1.0,
                "source_material": None,
                "model": "test-model",
                "seed": None,
            }
        ],
        run_id=run_id,
    )

    with Session(engine) as session:
        batch_run = session.exec(select(BatchRun).where(BatchRun.run_id == run_id)).first()
        batch_run.status = "completed"
        session.add(batch_run)
        session.commit()

    with testclient.TestClient(app) as client:
        response = client.delete("/dataset/batch")

    assert response.status_code == 200
    payload = response.json()
    assert payload.get("success") is True
    assert run_id in payload["data"]["deleted_run_ids"]


def test_batch_generation_model_defaults_topic_variations_off():
    pytest.importorskip("sqlmodel")
    from app.core.enums import AgentType
    from routes.dataset_models import BatchGeneration

    body = BatchGeneration(
        amount=1,
        topics=["Dealing with a rough work/school week + small wins"],
        agent_types=[AgentType.conversation],
        ex_amt=5,
    )

    assert body.allow_topic_variations is False

