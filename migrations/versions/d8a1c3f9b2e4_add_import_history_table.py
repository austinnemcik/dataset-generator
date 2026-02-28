"""add import history table

Revision ID: d8a1c3f9b2e4
Revises: c4d9f6e2a1b7
Create Date: 2026-02-28 10:45:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = "d8a1c3f9b2e4"
down_revision: Union[str, Sequence[str], None] = "c4d9f6e2a1b7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "importhistory",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("status", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("source_url", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("method", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("detected_format", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("dataset_id", sa.Integer(), nullable=True),
        sa.Column("dataset_name", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("request_headers_json", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("request_body_json", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("field_mapper_json", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("prompt", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("source_label", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("fetched_records", sa.Integer(), nullable=False),
        sa.Column("normalized_records", sa.Integer(), nullable=False),
        sa.Column("imported_records", sa.Integer(), nullable=False),
        sa.Column("duplicate_records", sa.Integer(), nullable=False),
        sa.Column("invalid_records", sa.Integer(), nullable=False),
        sa.Column("error", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["dataset_id"], ["dataset.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_importhistory_status"), "importhistory", ["status"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_importhistory_status"), table_name="importhistory")
    op.drop_table("importhistory")
