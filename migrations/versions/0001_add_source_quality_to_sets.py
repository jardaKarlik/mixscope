"""add_source_quality_to_sets

Revision ID: 0001
Revises: 
Create Date: 2026-04-16 02:38:03.547510

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0001'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add source_quality column to sets table.

    1 = whitelist channel (high trust, weight 3x)
    2 = search result, tracklist confirmed (medium trust, weight 1x)  [DEFAULT]
    3 = search result, no tracklist (low trust, weight 0.3x)
    """
    op.add_column(
        "sets",
        sa.Column(
            "source_quality",
            sa.SmallInteger(),
            nullable=False,
            server_default="2",
            comment="1=whitelist 2=search+tracklist 3=search+no-tracklist",
        ),
    )


def downgrade() -> None:
    op.drop_column("sets", "source_quality")
