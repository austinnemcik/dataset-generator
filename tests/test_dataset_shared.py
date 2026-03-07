import re
from uuid import uuid4

import pytest


def _build_dataset_with_examples(example_count: int) -> int:
    from sqlmodel import Session

    from app.core.database import Dataset, TrainingExample, engine

    with Session(engine) as session:
        dataset = Dataset(
            name=f"source-material-sampling-{uuid4().hex}",
            description="Source material sampling test",
            examples=[
                TrainingExample(
                    prompt=f"prompt-{index}",
                    instruction=f"instruction-{index}",
                    response=f"response-{index}",
                )
                for index in range(1, example_count + 1)
            ],
        )
        session.add(dataset)
        session.commit()
        session.refresh(dataset)
        return dataset.id


def test_resolve_source_material_first_limit_applies_per_dataset():
    pytest.importorskip("sqlmodel")
    from sqlmodel import Session

    from app.core.database import engine
    from routes.dataset_shared import resolve_source_material

    dataset_id = _build_dataset_with_examples(12)

    with Session(engine) as session:
        material, used_dataset_ids, _ = resolve_source_material(
            [dataset_id],
            session,
            dataset_example_limit=5,
            dataset_example_selection="first",
        )

    assert used_dataset_ids == [dataset_id]
    assert material is not None
    instruction_lines = re.findall(r"EXAMPLE \d+ INSTRUCTION: instruction-\d+", material)
    assert len(instruction_lines) == 5
    assert "instruction-1" in material
    assert "instruction-5" in material
    assert "instruction-6" not in material


def test_resolve_source_material_random_limit_is_seeded_and_capped():
    pytest.importorskip("sqlmodel")
    from sqlmodel import Session

    from app.core.database import engine
    from routes.dataset_shared import resolve_source_material

    dataset_id = _build_dataset_with_examples(20)

    with Session(engine) as session:
        material_a, _, _ = resolve_source_material(
            [dataset_id],
            session,
            dataset_example_limit=6,
            dataset_example_selection="random",
            seed=42,
        )
        material_b, _, _ = resolve_source_material(
            [dataset_id],
            session,
            dataset_example_limit=6,
            dataset_example_selection="random",
            seed=42,
        )
        material_c, _, _ = resolve_source_material(
            [dataset_id],
            session,
            dataset_example_limit=6,
            dataset_example_selection="random",
            seed=7,
        )

    assert material_a == material_b
    assert material_a is not None
    instruction_lines = re.findall(r"EXAMPLE \d+ INSTRUCTION: instruction-\d+", material_a)
    assert len(instruction_lines) == 6
    assert material_c is not None
    assert material_c != material_a
