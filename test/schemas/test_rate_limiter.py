"""Unit tests for RateLimit dataclass."""

from datetime import datetime, date, timedelta
from rag.schemas.rate_limiter import RateLimit


class TestRateLimitInitialization:
    """Test RateLimit initialization."""

    def test_default_initialization(self):
        """Test creating RateLimit with defaults."""
        limit = RateLimit()

        assert limit.requests == 0
        assert limit.tokens == 0
        assert isinstance(limit.last_reset, datetime)
        assert limit.daily_requests == 0
        assert limit.daily_tokens == 0
        assert isinstance(limit.day_start, date)
        assert limit.day_start == datetime.now().date()

    def test_initialization_with_custom_values(self):
        """Test creating RateLimit with custom values."""
        now = datetime(2025, 1, 1, 12, 0, 0)
        today = date(2025, 1, 1)

        limit = RateLimit(
            requests=100,
            tokens=5000,
            last_reset=now,
            daily_requests=500,
            daily_tokens=25000,
            day_start=today,
        )

        assert limit.requests == 100
        assert limit.tokens == 5000
        assert limit.last_reset == now
        assert limit.daily_requests == 500
        assert limit.daily_tokens == 25000
        assert limit.day_start == today


class TestRateLimitFieldTypes:
    """Test RateLimit field types."""

    def test_requests_is_int(self):
        """Test requests field is integer."""
        limit = RateLimit(requests=42)
        assert isinstance(limit.requests, int)
        assert limit.requests == 42

    def test_tokens_is_int(self):
        """Test tokens field is integer."""
        limit = RateLimit(tokens=1000)
        assert isinstance(limit.tokens, int)
        assert limit.tokens == 1000

    def test_last_reset_is_datetime(self):
        """Test last_reset field is datetime."""
        now = datetime(2025, 1, 1, 12, 0, 0)
        limit = RateLimit(last_reset=now)
        assert isinstance(limit.last_reset, datetime)
        assert limit.last_reset == now

    def test_daily_requests_is_int(self):
        """Test daily_requests field is integer."""
        limit = RateLimit(daily_requests=100)
        assert isinstance(limit.daily_requests, int)
        assert limit.daily_requests == 100

    def test_daily_tokens_is_int(self):
        """Test daily_tokens field is integer."""
        limit = RateLimit(daily_tokens=5000)
        assert isinstance(limit.daily_tokens, int)
        assert limit.daily_tokens == 5000

    def test_day_start_is_date(self):
        """Test day_start field is date."""
        today = date(2025, 1, 1)
        limit = RateLimit(day_start=today)
        assert isinstance(limit.day_start, date)
        assert limit.day_start == today


class TestRateLimitDefaultFactories:
    """Test RateLimit default factory functions."""

    def test_last_reset_defaults_to_now(self):
        """Test last_reset defaults to current datetime."""
        before = datetime.now()
        limit = RateLimit()
        after = datetime.now()

        assert before <= limit.last_reset <= after

    def test_day_start_defaults_to_today(self):
        """Test day_start defaults to today's date."""
        limit = RateLimit()

        assert limit.day_start == datetime.now().date()

    def test_multiple_instances_have_different_defaults(self):
        """Test multiple instances get independent default values."""
        limit1 = RateLimit()

        # Wait a tiny bit
        import time

        time.sleep(0.01)

        limit2 = RateLimit()

        # Should have slightly different timestamps
        # (or at least not share the same object)
        assert limit1.last_reset is not limit2.last_reset


class TestRateLimitMutability:
    """Test RateLimit is mutable (not frozen)."""

    def test_can_update_requests(self):
        """Test requests field can be updated."""
        limit = RateLimit()

        limit.requests = 10
        assert limit.requests == 10

        limit.requests += 5
        assert limit.requests == 15

    def test_can_update_tokens(self):
        """Test tokens field can be updated."""
        limit = RateLimit()

        limit.tokens = 1000
        assert limit.tokens == 1000

        limit.tokens += 500
        assert limit.tokens == 1500

    def test_can_update_last_reset(self):
        """Test last_reset field can be updated."""
        limit = RateLimit()
        new_time = datetime(2025, 1, 1, 12, 0, 0)

        limit.last_reset = new_time
        assert limit.last_reset == new_time

    def test_can_update_daily_counters(self):
        """Test daily counter fields can be updated."""
        limit = RateLimit()

        limit.daily_requests = 100
        limit.daily_tokens = 5000

        assert limit.daily_requests == 100
        assert limit.daily_tokens == 5000


class TestRateLimitUsageScenarios:
    """Test common RateLimit usage scenarios."""

    def test_tracking_requests(self):
        """Test tracking request count."""
        limit = RateLimit()

        # Simulate 5 requests
        for _ in range(5):
            limit.requests += 1

        assert limit.requests == 5

    def test_tracking_tokens(self):
        """Test tracking token usage."""
        limit = RateLimit()

        # Simulate using tokens
        limit.tokens += 1000
        limit.tokens += 500
        limit.tokens += 250

        assert limit.tokens == 1750

    def test_daily_reset_scenario(self):
        """Test scenario of resetting daily counters."""
        yesterday = date.today() - timedelta(days=1)
        limit = RateLimit(
            requests=100,
            tokens=5000,
            daily_requests=500,
            daily_tokens=25000,
            day_start=yesterday,
        )

        # Simulate daily reset
        if limit.day_start < date.today():
            limit.requests = 0
            limit.tokens = 0
            limit.day_start = date.today()

        assert limit.requests == 0
        assert limit.tokens == 0
        assert limit.day_start == date.today()
        assert limit.daily_requests == 500  # Historical data preserved

    def test_rate_limit_check_scenario(self):
        """Test scenario of checking rate limits."""
        limit = RateLimit(requests=90, tokens=9000)

        # Define limits
        MAX_REQUESTS = 100
        MAX_TOKENS = 10000

        # Check if we can make another request
        can_make_request = limit.requests < MAX_REQUESTS and limit.tokens < MAX_TOKENS

        assert can_make_request is True

        # Make request
        if can_make_request:
            limit.requests += 1
            limit.tokens += 500

        assert limit.requests == 91
        assert limit.tokens == 9500


class TestRateLimitEdgeCases:
    """Test RateLimit edge cases."""

    def test_zero_values(self):
        """Test RateLimit with all zero values."""
        limit = RateLimit(
            requests=0,
            tokens=0,
            daily_requests=0,
            daily_tokens=0,
        )

        assert limit.requests == 0
        assert limit.tokens == 0
        assert limit.daily_requests == 0
        assert limit.daily_tokens == 0

    def test_large_values(self):
        """Test RateLimit with very large values."""
        limit = RateLimit(
            requests=1_000_000,
            tokens=1_000_000_000,
            daily_requests=10_000_000,
            daily_tokens=10_000_000_000,
        )

        assert limit.requests == 1_000_000
        assert limit.tokens == 1_000_000_000
        assert limit.daily_requests == 10_000_000
        assert limit.daily_tokens == 10_000_000_000

    def test_negative_values_allowed(self):
        """Test that negative values are allowed (no validation)."""
        # Note: The dataclass doesn't prevent negative values
        limit = RateLimit(
            requests=-10,
            tokens=-1000,
        )

        assert limit.requests == -10
        assert limit.tokens == -1000

    def test_future_date(self):
        """Test RateLimit with future date."""
        future_date = date.today() + timedelta(days=10)
        limit = RateLimit(day_start=future_date)

        assert limit.day_start == future_date

    def test_past_datetime(self):
        """Test RateLimit with past datetime."""
        past = datetime(2020, 1, 1, 0, 0, 0)
        limit = RateLimit(last_reset=past)

        assert limit.last_reset == past
        assert limit.last_reset < datetime.now()
