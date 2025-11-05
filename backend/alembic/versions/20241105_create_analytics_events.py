"""Create analytics events table."""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20241105_create_analytics_events"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "analytics_events",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.String(length=64), nullable=True),
        sa.Column("role", sa.String(length=255), nullable=True),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("rating", sa.Integer(), nullable=True),
        sa.Column("relevant", sa.Boolean(), nullable=True),
        sa.Column("context", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("analytics_events")
