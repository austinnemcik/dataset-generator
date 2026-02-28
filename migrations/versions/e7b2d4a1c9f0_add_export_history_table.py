"""add export history table

Revision ID: e7b2d4a1c9f0
Revises: d8a1c3f9b2e4
Create Date: 2026-02-28 11:20:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = "e7b2d4a1c9f0"
down_revision: Union[str, Sequence[str], None] = "d8a1c3f9b2e4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "exporthistory",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("status", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("export_format", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("dataset_ids_json", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("options_json", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("output_filename", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("total_examples", sa.Integer(), nullable=False),
        sa.Column("train_examples", sa.Integer(), nullable=False),
        sa.Column("val_examples", sa.Integer(), nullable=False),
        sa.Column("error", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_exporthistory_status"), "exporthistory", ["status"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_exporthistory_status"), table_name="exporthistory")
    op.drop_table("exporthistory")
