"""Constants shared by unit and integration tests."""

from datetime import UTC, datetime

from app.database.seed import stable_uuid

# A fixed "today" for the mock history so every test sees identical data.
# 2026-09-28 18:00 UTC = 22:00 bank time (UTC+4) on Monday 28 September.
ANCHOR = datetime(2026, 9, 28, 18, 0, tzinfo=UTC)

# HS256 secret for principal tokens in tests (>= 32 characters, like production).
TEST_MCP_SECRET = "test-only-mcp-secret-0123456789abcdef"

# Mock customers (see USER_PROFILES in app/database/seed.py).
NINO = stable_uuid("user", 1)  # CUST-0001: checking GEL, savings GEL, savings USD
GIORGI = stable_uuid("user", 2)  # CUST-0002: checking GEL, savings GEL
EMMA = stable_uuid("user", 10)  # CUST-0010: checking GEL, FROZEN checking EUR

NINO_CHECKING = stable_uuid("account", 1, 1)
NINO_SAVINGS_GEL = stable_uuid("account", 1, 2)
NINO_SAVINGS_USD = stable_uuid("account", 1, 3)
GIORGI_CHECKING = stable_uuid("account", 2, 1)
EMMA_FROZEN_EUR = stable_uuid("account", 10, 2)
