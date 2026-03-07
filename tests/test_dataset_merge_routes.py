import pytest


def test_discover_merge_pools_can_be_scoped_to_candidate_ids():
    pytest.importorskip("sqlmodel")

    from sqlmodel import Session, SQLModel, create_engine

    from app.core.database import Dataset, TrainingExample
    from routes.dataset_merge_routes import discover_merge_pools

    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        first = Dataset(name="alpha", description="a")
        second = Dataset(name="beta", description="b")
        third = Dataset(name="gamma", description="c")
        session.add(first)
        session.add(second)
        session.add(third)
        session.flush()

        session.add(
            TrainingExample(
                prompt="p1",
                instruction="i1",
                response="r1",
                dataset_id=first.id,
                embedding="[1.0, 0.0]",
            )
        )
        session.add(
            TrainingExample(
                prompt="p2",
                instruction="i2",
                response="r2",
                dataset_id=second.id,
                embedding="[0.99, 0.01]",
            )
        )
        session.add(
            TrainingExample(
                prompt="p3",
                instruction="i3",
                response="r3",
                dataset_id=third.id,
                embedding="[0.98, 0.02]",
            )
        )
        session.commit()

        all_pools, skipped, skipped_dim = discover_merge_pools(session, 0.95)
        scoped_pools, scoped_skipped, scoped_skipped_dim = discover_merge_pools(
            session, 0.95, candidate_dataset_ids=[first.id, second.id]
        )

    assert skipped == 0
    assert skipped_dim == 0
    assert scoped_skipped == 0
    assert scoped_skipped_dim == 0
    assert all_pools == [[first.id, second.id, third.id]]
    assert scoped_pools == [[first.id, second.id]]


def test_discover_merge_pools_skips_dimension_mismatch_instead_of_failing():
    pytest.importorskip("sqlmodel")

    from sqlmodel import Session, SQLModel, create_engine

    from app.core.database import Dataset, TrainingExample
    from routes.dataset_merge_routes import discover_merge_pools

    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        first = Dataset(name="d1", description="a")
        second = Dataset(name="d2", description="b")
        session.add(first)
        session.add(second)
        session.flush()

        session.add(
            TrainingExample(
                prompt="p1",
                instruction="i1",
                response="r1",
                dataset_id=first.id,
                embedding="[1.0, 0.0, 0.0]",
            )
        )
        session.add(
            TrainingExample(
                prompt="p2",
                instruction="i2",
                response="r2",
                dataset_id=second.id,
                embedding="[0.9, 0.1]",
            )
        )
        session.commit()

        pools, skipped_without_embeddings, skipped_dimension_mismatch = discover_merge_pools(session, 0.8)

    assert pools == []
    assert skipped_without_embeddings == 0
    assert skipped_dimension_mismatch == 1
