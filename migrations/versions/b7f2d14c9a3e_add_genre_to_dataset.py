"""add genre to dataset

Revision ID: b7f2d14c9a3e
Revises: a2b6c8d0e4f1
Create Date: 2026-03-01 06:10:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = "b7f2d14c9a3e"
down_revision: Union[str, Sequence[str], None] = "a2b6c8d0e4f1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("dataset", sa.Column("genre", sqlmodel.sql.sqltypes.AutoString(), nullable=True))
    op.create_index(op.f("ix_dataset_genre"), "dataset", ["genre"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_dataset_genre"), table_name="dataset")
    op.drop_column("dataset", "genre")
