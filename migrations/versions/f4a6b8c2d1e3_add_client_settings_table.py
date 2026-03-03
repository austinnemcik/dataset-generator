"""add client settings table

Revision ID: f4a6b8c2d1e3
Revises: c91e3b5a4d27
Create Date: 2026-03-03 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "f4a6b8c2d1e3"
down_revision: Union[str, None] = "c91e3b5a4d27"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table("clientsettings"):
        return

    op.create_table(
        "clientsettings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("default_model", sa.String(), nullable=False, server_default="z-ai/glm-5"),
        sa.Column("grading_model", sa.String(), nullable=False, server_default="google/gemini-2.5-flash"),
        sa.Column("naming_model", sa.String(), nullable=False, server_default="google/gemini-2.5-flash"),
        sa.Column("threshold", sa.Float(), nullable=False, server_default="0.8"),
        sa.Column("min_grading_score", sa.Float(), nullable=False, server_default="8.0"),
        sa.Column("min_response_char_length", sa.Integer(), nullable=False, server_default="40"),
        sa.Column("max_grading_json_retries", sa.Integer(), nullable=False, server_default="2"),
        sa.Column("max_naming_json_retries", sa.Integer(), nullable=False, server_default="2"),
        sa.Column("max_low_quality_retries", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("max_generation_retries", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("min_save_ratio", sa.Float(), nullable=False, server_default="0.8"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table("clientsettings"):
        op.drop_table("clientsettings")
