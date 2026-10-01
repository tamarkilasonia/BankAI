"""Initial schema: users, accounts, transactions, products, conversation logs, evaluations.

Revision ID: 0001
Revises:
Create Date: 2026-10-01 20:02:15.990162
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "banking_products",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("code", sa.String(length=40), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column(
            "product_type",
            sa.Enum(
                "account",
                "card",
                "loan",
                "mortgage",
                "deposit",
                name="producttype",
                native_enum=False,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column(
            "currency",
            sa.Enum("GEL", "USD", "EUR", name="currency", native_enum=False, length=32),
            nullable=False,
        ),
        sa.Column("interest_rate_pct", sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column("monthly_fee", sa.Numeric(precision=10, scale=2), nullable=True),
        sa.Column("min_amount", sa.Numeric(precision=14, scale=2), nullable=True),
        sa.Column("max_amount", sa.Numeric(precision=14, scale=2), nullable=True),
        sa.Column("min_term_months", sa.Integer(), nullable=True),
        sa.Column("max_term_months", sa.Integer(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("currency IN ('GEL', 'USD', 'EUR')", name=op.f("ck_banking_products_currency")),
        sa.CheckConstraint(
            "product_type IN ('account', 'card', 'loan', 'mortgage', 'deposit')",
            name=op.f("ck_banking_products_producttype"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_banking_products")),
        sa.UniqueConstraint("code", name=op.f("uq_banking_products_code")),
    )
    op.create_table(
        "evaluation_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "running",
                "completed",
                "failed",
                name="evaluationrunstatus",
                native_enum=False,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("llm_provider", sa.String(length=32), nullable=False),
        sa.Column("model_name", sa.String(length=64), nullable=True),
        sa.Column("dataset_name", sa.String(length=64), nullable=False),
        sa.Column("dataset_version", sa.String(length=64), nullable=False),
        sa.Column("total_cases", sa.Integer(), nullable=False),
        sa.Column("passed_cases", sa.Integer(), nullable=False),
        sa.Column(
            "summary",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('running', 'completed', 'failed')",
            name=op.f("ck_evaluation_runs_evaluationrunstatus"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_evaluation_runs")),
    )
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("customer_number", sa.String(length=16), nullable=False),
        sa.Column("full_name", sa.String(length=120), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column(
            "preferred_language",
            sa.Enum("en", "ka", name="language", native_enum=False, length=32),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("preferred_language IN ('en', 'ka')", name=op.f("ck_users_language")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_users")),
        sa.UniqueConstraint("customer_number", name=op.f("uq_users_customer_number")),
        sa.UniqueConstraint("email", name=op.f("uq_users_email")),
    )
    op.create_table(
        "accounts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("account_number", sa.String(length=34), nullable=False),
        sa.Column("name", sa.String(length=80), nullable=False),
        sa.Column(
            "account_type",
            sa.Enum(
                "checking",
                "savings",
                name="accounttype",
                native_enum=False,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column(
            "currency",
            sa.Enum("GEL", "USD", "EUR", name="currency", native_enum=False, length=32),
            nullable=False,
        ),
        sa.Column("balance", sa.Numeric(precision=14, scale=2), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "active",
                "frozen",
                "closed",
                name="accountstatus",
                native_enum=False,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("account_type IN ('checking', 'savings')", name=op.f("ck_accounts_accounttype")),
        sa.CheckConstraint("currency IN ('GEL', 'USD', 'EUR')", name=op.f("ck_accounts_currency")),
        sa.CheckConstraint(
            "status IN ('active', 'frozen', 'closed')", name=op.f("ck_accounts_accountstatus")
        ),
        sa.CheckConstraint("balance >= 0", name=op.f("ck_accounts_balance_non_negative")),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_accounts_user_id_users"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_accounts")),
        sa.UniqueConstraint("account_number", name=op.f("uq_accounts_account_number")),
    )
    op.create_index(op.f("ix_accounts_user_id"), "accounts", ["user_id"], unique=False)
    op.create_table(
        "conversation_logs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=False),
        sa.Column("conversation_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=True),
        sa.Column(
            "channel",
            sa.Enum("text", "voice", name="channel", native_enum=False, length=32),
            nullable=False,
        ),
        sa.Column("user_input", sa.Text(), nullable=False),
        sa.Column("detected_intent", sa.String(length=40), nullable=True),
        sa.Column("selected_tool", sa.String(length=64), nullable=True),
        sa.Column(
            "tool_arguments",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=True,
        ),
        sa.Column("tool_latency_ms", sa.Float(), nullable=True),
        sa.Column("llm_latency_ms", sa.Float(), nullable=True),
        sa.Column("total_latency_ms", sa.Float(), nullable=True),
        sa.Column("response_text", sa.Text(), nullable=True),
        sa.Column(
            "safety_status",
            sa.Enum(
                "allowed",
                "refused",
                "confirmation_required",
                name="safetystatus",
                native_enum=False,
                length=32,
            ),
            nullable=True,
        ),
        sa.Column(
            "safety_flags",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("channel IN ('text', 'voice')", name=op.f("ck_conversation_logs_channel")),
        sa.CheckConstraint(
            "safety_status IN ('allowed', 'refused', 'confirmation_required')",
            name=op.f("ck_conversation_logs_safetystatus"),
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_conversation_logs_user_id_users"), ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_conversation_logs")),
    )
    op.create_index(
        op.f("ix_conversation_logs_conversation_id"), "conversation_logs", ["conversation_id"], unique=False
    )
    op.create_index(
        op.f("ix_conversation_logs_request_id"), "conversation_logs", ["request_id"], unique=False
    )
    op.create_index(op.f("ix_conversation_logs_user_id"), "conversation_logs", ["user_id"], unique=False)
    op.create_table(
        "evaluation_results",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("case_id", sa.String(length=64), nullable=False),
        sa.Column("category", sa.String(length=32), nullable=False),
        sa.Column("input_text", sa.Text(), nullable=False),
        sa.Column("expected_behavior", sa.String(length=64), nullable=False),
        sa.Column("actual_behavior", sa.String(length=64), nullable=True),
        sa.Column("expected_intent", sa.String(length=40), nullable=True),
        sa.Column("actual_intent", sa.String(length=40), nullable=True),
        sa.Column("expected_tool", sa.String(length=64), nullable=True),
        sa.Column("actual_tool", sa.String(length=64), nullable=True),
        sa.Column("passed", sa.Boolean(), nullable=False),
        sa.Column(
            "scores",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.Column(
            "failure_reasons",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.Column("response_text", sa.Text(), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["evaluation_runs.id"],
            name=op.f("fk_evaluation_results_run_id_evaluation_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_evaluation_results")),
        sa.UniqueConstraint("run_id", "case_id", name="uq_evaluation_results_run_id_case_id"),
    )
    op.create_index(op.f("ix_evaluation_results_category"), "evaluation_results", ["category"], unique=False)
    op.create_index(op.f("ix_evaluation_results_run_id"), "evaluation_results", ["run_id"], unique=False)
    op.create_table(
        "transactions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("account_id", sa.Uuid(), nullable=False),
        sa.Column(
            "direction",
            sa.Enum(
                "debit",
                "credit",
                name="transactiondirection",
                native_enum=False,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("amount", sa.Numeric(precision=14, scale=2), nullable=False),
        sa.Column(
            "currency",
            sa.Enum("GEL", "USD", "EUR", name="currency", native_enum=False, length=32),
            nullable=False,
        ),
        sa.Column(
            "category",
            sa.Enum(
                "groceries",
                "restaurants",
                "transportation",
                "shopping",
                "utilities",
                "subscriptions",
                "healthcare",
                "entertainment",
                "salary",
                "transfer",
                "deposit",
                name="transactioncategory",
                native_enum=False,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("merchant", sa.String(length=120), nullable=False),
        sa.Column("description", sa.String(length=255), nullable=False),
        sa.Column("booked_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "category IN ('groceries', 'restaurants', 'transportation', 'shopping', 'utilities', 'subscriptions', 'healthcare', 'entertainment', 'salary', 'transfer', 'deposit')",
            name=op.f("ck_transactions_transactioncategory"),
        ),
        sa.CheckConstraint("currency IN ('GEL', 'USD', 'EUR')", name=op.f("ck_transactions_currency")),
        sa.CheckConstraint(
            "direction IN ('debit', 'credit')", name=op.f("ck_transactions_transactiondirection")
        ),
        sa.CheckConstraint("amount > 0", name=op.f("ck_transactions_amount_positive")),
        sa.ForeignKeyConstraint(
            ["account_id"],
            ["accounts.id"],
            name=op.f("fk_transactions_account_id_accounts"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_transactions")),
    )
    op.create_index(
        "ix_transactions_account_id_booked_at", "transactions", ["account_id", "booked_at"], unique=False
    )
    op.create_index(op.f("ix_transactions_category"), "transactions", ["category"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_transactions_category"), table_name="transactions")
    op.drop_index("ix_transactions_account_id_booked_at", table_name="transactions")
    op.drop_table("transactions")
    op.drop_index(op.f("ix_evaluation_results_run_id"), table_name="evaluation_results")
    op.drop_index(op.f("ix_evaluation_results_category"), table_name="evaluation_results")
    op.drop_table("evaluation_results")
    op.drop_index(op.f("ix_conversation_logs_user_id"), table_name="conversation_logs")
    op.drop_index(op.f("ix_conversation_logs_request_id"), table_name="conversation_logs")
    op.drop_index(op.f("ix_conversation_logs_conversation_id"), table_name="conversation_logs")
    op.drop_table("conversation_logs")
    op.drop_index(op.f("ix_accounts_user_id"), table_name="accounts")
    op.drop_table("accounts")
    op.drop_table("users")
    op.drop_table("evaluation_runs")
    op.drop_table("banking_products")
