import pytest


def test_serialize_export_record_sharegpt():
    pytest.importorskip("sqlmodel")
    from database import TrainingExample
    from services.export_service import _serialize_export_record

    row = _serialize_export_record(
        TrainingExample(prompt="p", instruction="inst", response="resp", dataset_id=1),
        "sharegpt",
    )
    assert "conversations" in row


def test_serialize_export_record_chatml():
    pytest.importorskip("sqlmodel")
    from database import TrainingExample
    from services.export_service import _serialize_export_record

    row = _serialize_export_record(
        TrainingExample(prompt="p", instruction="inst", response="resp", dataset_id=1),
        "chatml",
    )
    assert "messages" in row


def test_serialize_export_record_alpaca():
    pytest.importorskip("sqlmodel")
    from database import TrainingExample
    from services.export_service import _serialize_export_record

    row = _serialize_export_record(
        TrainingExample(prompt="p", instruction="inst", response="resp", dataset_id=1),
        "alpaca",
    )
    assert row["output"] == "resp"
