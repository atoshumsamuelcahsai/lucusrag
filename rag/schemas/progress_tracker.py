from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProgressTracker:
    total_elements: int
    processed_elements: int = 0
    total_expected_tokens: int = 0
    tokens_used: int = 0
    total_expected_cost: float = 0.0
    cost_so_far: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)

    def update(self, tokens_used: int, is_element_complete: bool = True) -> None:
        self.tokens_used += tokens_used
        if is_element_complete:
            self.processed_elements += 1

        # Update every 30 seconds or every 50 elements
        now = datetime.now()
        if (now - self.last_update).seconds >= 30 or self.processed_elements % 50 == 0:
            self._print_progress()
            self.last_update = now

    def _print_progress(self) -> None:
        elapsed_time = datetime.now() - self.start_time
        if self.processed_elements > 0:
            avg_time_per_element = elapsed_time / self.processed_elements
            estimated_remaining_time = avg_time_per_element * (
                self.total_elements - self.processed_elements
            )
        else:
            estimated_remaining_time = timedelta(0)

        # Calculate costs
        input_cost = (self.tokens_used / 1000) * 0.00025  # Claude input cost
        output_cost = (self.tokens_used / 1000) * 0.00075  # Claude output cost
        current_cost = input_cost + output_cost

        logger.info("\n" + "=" * 50)
        logger.info(f"Progress Update ({datetime.now().strftime('%H:%M:%S')})")
        logger.info(
            f"Elements: {self.processed_elements:,}/{self.total_elements:,} ({(self.processed_elements/self.total_elements*100):.1f}%)"
        )

        # Handle division by zero for token percentage
        if self.total_expected_tokens > 0:
            token_pct = self.tokens_used / self.total_expected_tokens * 100
            logger.info(
                f"Tokens Used: {self.tokens_used:,}/{self.total_expected_tokens:,} ({token_pct:.1f}%)"
            )
        else:
            logger.info(f"Tokens Used: {self.tokens_used:,}")

        logger.info(f"Cost: ${current_cost:.2f}/${self.total_expected_cost:.2f}")
        logger.info(f"Time Elapsed: {str(elapsed_time).split('.')[0]}")
        logger.info(
            f"Estimated Remaining: {str(estimated_remaining_time).split('.')[0]}"
        )
        logger.info("=" * 50)
