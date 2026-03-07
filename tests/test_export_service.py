import pytest


def test_serialize_export_record_sharegpt():
    pytest.importorskip("sqlmodel")
    from app.core.database import TrainingExample
    from services.export_service import _serialize_export_record

    row = _serialize_export_record(
        TrainingExample(prompt="p", instruction="inst", response="resp", dataset_id=1),
        "sharegpt",
    )
    assert "conversations" in row
    assert row["conversations"][-1]["from"] == "gpt"


def test_serialize_export_record_chatml():
    pytest.importorskip("sqlmodel")
    from app.core.database import TrainingExample
    from services.export_service import _serialize_export_record

    row = _serialize_export_record(
        TrainingExample(prompt="p", instruction="inst", response="resp", dataset_id=1),
        "chatml",
    )
    assert "messages" in row
    assert row["messages"][-1]["role"] == "assistant"


def test_serialize_export_record_alpaca():
    pytest.importorskip("sqlmodel")
    from app.core.database import TrainingExample
    from services.export_service import _serialize_export_record

    row = _serialize_export_record(
        TrainingExample(prompt="p", instruction="inst", response="resp", dataset_id=1),
        "alpaca",
    )
    assert row["output"] == "resp"


def test_serialize_export_record_sharegpt_expands_flattened_conversation():
    pytest.importorskip("sqlmodel")
    from app.core.database import TrainingExample
    from services.export_service import _serialize_export_record

    row = _serialize_export_record(
        TrainingExample(
            prompt="p",
            instruction="User: Hey\nAssistant: Hi there\nUser: I had a rough week.",
            response="I'm sorry. Want to talk through it?",
            dataset_id=1,
        ),
        "sharegpt",
    )

    assert row["conversations"] == [
        {"from": "system", "value": "You are a helpful assistant."},
        {"from": "human", "value": "Hey"},
        {"from": "gpt", "value": "Hi there"},
        {"from": "human", "value": "I had a rough week."},
        {"from": "gpt", "value": "I'm sorry. Want to talk through it?"},
    ]


def test_serialize_export_record_chatml_expands_flattened_conversation():
    pytest.importorskip("sqlmodel")
    from app.core.database import TrainingExample
    from services.export_service import _serialize_export_record

    row = _serialize_export_record(
        TrainingExample(
            prompt="p",
            instruction="User: Hey\nAssistant: Hi there\nUser: I had a rough week.",
            response="I'm sorry. Want to talk through it?",
            dataset_id=1,
        ),
        "chatml",
    )

    assert row["messages"] == [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hey"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "I had a rough week."},
        {"role": "assistant", "content": "I'm sorry. Want to talk through it?"},
    ]

