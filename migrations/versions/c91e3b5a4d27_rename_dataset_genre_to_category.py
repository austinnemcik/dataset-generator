"""rename dataset genre to category

Revision ID: c91e3b5a4d27
Revises: b7f2d14c9a3e
Create Date: 2026-03-01 06:30:00.000000

"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "c91e3b5a4d27"
down_revision: Union[str, Sequence[str], None] = "b7f2d14c9a3e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column("dataset", "genre", new_column_name="category")
    op.execute("ALTER INDEX IF EXISTS ix_dataset_genre RENAME TO ix_dataset_category")


def downgrade() -> None:
    op.alter_column("dataset", "category", new_column_name="genre")
    op.execute("ALTER INDEX IF EXISTS ix_dataset_category RENAME TO ix_dataset_genre")
