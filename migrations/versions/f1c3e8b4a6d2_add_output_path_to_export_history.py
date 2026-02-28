"""add output path to export history

Revision ID: f1c3e8b4a6d2
Revises: e7b2d4a1c9f0
Create Date: 2026-02-28 11:45:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = "f1c3e8b4a6d2"
down_revision: Union[str, Sequence[str], None] = "e7b2d4a1c9f0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "exporthistory",
        sa.Column("output_path", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("exporthistory", "output_path")
