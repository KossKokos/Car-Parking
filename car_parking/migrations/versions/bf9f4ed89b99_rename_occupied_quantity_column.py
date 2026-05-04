"""rename occupied quantity column

Revision ID: bf9f4ed89b99
Revises: bbd1b831e087
Create Date: 2026-04-28 12:46:25.771734

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = 'bf9f4ed89b99'
down_revision = 'bbd1b831e087'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "parking_count_table",
        "ococcupied_quantity",
        new_column_name="occupied_quantity",
    )


def downgrade() -> None:
    op.alter_column(
        "parking_count_table",
        "occupied_quantity",
        new_column_name="ococcupied_quantity",
    )