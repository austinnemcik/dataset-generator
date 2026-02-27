"""add dataset cost columns

Revision ID: a3f5c1d2b8e9
Revises: 91888025f7ae
Create Date: 2026-02-27 07:05:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a3f5c1d2b8e9"
down_revision: Union[str, Sequence[str], None] = "91888025f7ae"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "dataset",
        sa.Column("generation_cost", sa.Float(), nullable=False, server_default="0"),
    )
    op.add_column(
        "dataset",
        sa.Column("grading_cost", sa.Float(), nullable=False, server_default="0"),
    )
    op.add_column(
        "dataset",
        sa.Column("total_cost", sa.Float(), nullable=False, server_default="0"),
    )
    op.alter_column("dataset", "generation_cost", server_default=None)
    op.alter_column("dataset", "grading_cost", server_default=None)
    op.alter_column("dataset", "total_cost", server_default=None)


def downgrade() -> None:
    op.drop_column("dataset", "total_cost")
    op.drop_column("dataset", "grading_cost")
    op.drop_column("dataset", "generation_cost")

