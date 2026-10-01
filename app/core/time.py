"""Time helpers.

All timestamps are stored in UTC. Calendar questions ("this month", "today")
are answered in the bank's local time: Georgia uses UTC+4 all year with no
daylight saving time, so a fixed offset is exact and needs no tz database.
"""

from datetime import UTC, datetime, timedelta, timezone

BANK_TZ = timezone(timedelta(hours=4), name="Asia/Tbilisi")


def utcnow() -> datetime:
    """Current time, timezone-aware, in UTC. Injected as a ``clock`` so tests can pin it."""
    return datetime.now(UTC)


def as_utc(moment: datetime) -> datetime:
    """Normalize a timestamp read from the database to aware UTC.

    PostgreSQL returns timezone-aware values; SQLite (unit tests) returns naive
    ones that were stored as UTC.
    """
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)
