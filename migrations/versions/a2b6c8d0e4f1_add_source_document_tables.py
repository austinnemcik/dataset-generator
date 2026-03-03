"""add source document tables

Revision ID: a2b6c8d0e4f1
Revises: f1c3e8b4a6d2
Create Date: 2026-03-01 05:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = "a2b6c8d0e4f1"
down_revision: Union[str, Sequence[str], None] = "f1c3e8b4a6d2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "sourcedocument",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("file_type", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("char_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "sourcechunk",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("char_count", sa.Integer(), nullable=False, server_default="0"),
        sa.ForeignKeyConstraint(["document_id"], ["sourcedocument.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_sourcechunk_chunk_index"), "sourcechunk", ["chunk_index"], unique=False)
    op.create_index(op.f("ix_sourcechunk_document_id"), "sourcechunk", ["document_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_sourcechunk_document_id"), table_name="sourcechunk")
    op.drop_index(op.f("ix_sourcechunk_chunk_index"), table_name="sourcechunk")
    op.drop_table("sourcechunk")
    op.drop_table("sourcedocument")
