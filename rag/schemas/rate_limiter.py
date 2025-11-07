from dataclasses import dataclass, field
from datetime import datetime, date


@dataclass
class RateLimit:
    requests: int = 0
    tokens: int = 0
    last_reset: datetime = field(default_factory=datetime.now)
    daily_requests: int = 0
    daily_tokens: int = 0
    day_start: date = field(default_factory=lambda: datetime.now().date())
