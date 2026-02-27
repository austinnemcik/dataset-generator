"""add model to dataset

Revision ID: b91d7f6a4c2e
Revises: a3f5c1d2b8e9
Create Date: 2026-02-27 07:40:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = "b91d7f6a4c2e"
down_revision: Union[str, Sequence[str], None] = "a3f5c1d2b8e9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "dataset",
        sa.Column("model", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("dataset", "model")

