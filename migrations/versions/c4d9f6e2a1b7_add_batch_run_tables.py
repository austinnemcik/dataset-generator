"""add batch run tables

Revision ID: c4d9f6e2a1b7
Revises: b91d7f6a4c2e
Create Date: 2026-02-28 09:30:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = "c4d9f6e2a1b7"
down_revision: Union[str, Sequence[str], None] = "b91d7f6a4c2e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "dataset",
        sa.Column("source_run_id", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    )
    op.create_index("ix_dataset_source_run_id", "dataset", ["source_run_id"], unique=True)

    op.create_table(
        "batchrun",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("status", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("request_json", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("summary_json", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("total_runs", sa.Integer(), nullable=False),
        sa.Column("queued_runs", sa.Integer(), nullable=False),
        sa.Column("running_runs", sa.Integer(), nullable=False),
        sa.Column("completed_runs", sa.Integer(), nullable=False),
        sa.Column("failed_runs", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_batchrun_run_id"), "batchrun", ["run_id"], unique=True)
    op.create_index(op.f("ix_batchrun_status"), "batchrun", ["status"], unique=False)

    op.create_table(
        "batchrunitem",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("batch_run_id", sa.Integer(), nullable=False),
        sa.Column("item_index", sa.Integer(), nullable=False),
        sa.Column("run_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("dataset_key", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("slot_key", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("requested_topic", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("topic", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("agent", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("ex_amt", sa.Integer(), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=True),
        sa.Column("status", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("max_retries", sa.Integer(), nullable=False),
        sa.Column("retry_backoff_seconds", sa.Float(), nullable=False),
        sa.Column("source_material", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("model", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("error_type", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("error", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("created_dataset_id", sa.Integer(), nullable=True),
        sa.Column("result_json", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["batch_run_id"], ["batchrun.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_batchrunitem_batch_run_id"), "batchrunitem", ["batch_run_id"], unique=False)
    op.create_index(op.f("ix_batchrunitem_item_index"), "batchrunitem", ["item_index"], unique=False)
    op.create_index(op.f("ix_batchrunitem_run_id"), "batchrunitem", ["run_id"], unique=True)
    op.create_index(op.f("ix_batchrunitem_status"), "batchrunitem", ["status"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_batchrunitem_status"), table_name="batchrunitem")
    op.drop_index(op.f("ix_batchrunitem_run_id"), table_name="batchrunitem")
    op.drop_index(op.f("ix_batchrunitem_item_index"), table_name="batchrunitem")
    op.drop_index(op.f("ix_batchrunitem_batch_run_id"), table_name="batchrunitem")
    op.drop_table("batchrunitem")

    op.drop_index(op.f("ix_batchrun_status"), table_name="batchrun")
    op.drop_index(op.f("ix_batchrun_run_id"), table_name="batchrun")
    op.drop_table("batchrun")

    op.drop_index("ix_dataset_source_run_id", table_name="dataset")
    op.drop_column("dataset", "source_run_id")
