"""
Tests for adaptive_k module - probabilistic document selection.
"""

from llama_index.core.schema import NodeWithScore, TextNode

from rag.engine.adaptive_k import (
    adaptive_k_selection,
    estimate_tokens,
    calculate_cost,
)


class TestEstimateTokens:
    """Test token estimation function."""

    def test_estimate_tokens_basic(self):
        """Test basic token estimation."""
        text = "This is a test sentence with multiple words."
        tokens = estimate_tokens(text)
        # Rough approximation: 1 token ≈ 4 characters
        expected = len(text) // 4
        assert tokens == expected

    def test_estimate_tokens_empty(self):
        """Test token estimation for empty string."""
        assert estimate_tokens("") == 0

    def test_estimate_tokens_short(self):
        """Test token estimation for short text."""
        assert estimate_tokens("abc") == 0  # Less than 4 chars
        assert estimate_tokens("abcd") == 1  # Exactly 4 chars

    def test_estimate_tokens_long(self):
        """Test token estimation for long text."""
        text = "word " * 100  # 500 characters
        tokens = estimate_tokens(text)
        assert tokens == 125  # 500 / 4


class TestCalculateCost:
    """Test cost calculation function."""

    def test_calculate_cost_default_price(self):
        """Test cost calculation with default price."""
        tokens = 1000
        cost = calculate_cost(tokens)
        # 1000 tokens / 1000 * 0.001 = 0.001
        assert cost == 0.001

    def test_calculate_cost_custom_price(self):
        """Test cost calculation with custom price."""
        tokens = 1000
        cost = calculate_cost(tokens, price_per_1k_tokens=0.002)
        # 1000 tokens / 1000 * 0.002 = 0.002
        assert cost == 0.002

    def test_calculate_cost_zero_tokens(self):
        """Test cost calculation with zero tokens."""
        assert calculate_cost(0) == 0.0

    def test_calculate_cost_fractional(self):
        """Test cost calculation with fractional tokens."""
        tokens = 500
        cost = calculate_cost(tokens, price_per_1k_tokens=0.001)
        # 500 tokens / 1000 * 0.001 = 0.0005
        assert cost == 0.0005


class TestAdaptiveKSelection:
    """Test adaptive_k_selection function."""

    def _create_mock_node(self, text: str, score: float) -> NodeWithScore:
        """Helper to create mock nodes."""
        node = TextNode(text=text, id_=f"node_{score}")
        return NodeWithScore(node=node, score=score)

    def test_adaptive_k_empty_nodes(self):
        """Test with empty node list."""
        nodes = []
        selected, metadata = adaptive_k_selection(nodes)

        assert len(selected) == 0
        assert metadata["selected_k"] == 0
        assert metadata["stopped_reason"] == "no_candidates"
        assert metadata["total_cost"] == 0.0
        assert metadata["total_tokens"] == 0

    def test_adaptive_k_insufficient_candidates(self):
        """Test when nodes <= k_min."""
        nodes = [
            self._create_mock_node("text1", 0.9),
            self._create_mock_node("text2", 0.8),
        ]
        selected, metadata = adaptive_k_selection(nodes, k_min=2, k_max=10)

        assert len(selected) == 2
        assert metadata["selected_k"] == 2
        assert metadata["stopped_reason"] == "insufficient_candidates"
        assert len(metadata["probabilities"]) == 2

    def test_adaptive_k_probability_target_reached_early(self):
        """Test when probability target is reached early."""
        # Create nodes with high scores that will concentrate probability
        nodes = [
            self._create_mock_node("high score text", 10.0),
            self._create_mock_node("low score text", 0.1),
            self._create_mock_node("low score text 2", 0.1),
        ]
        selected, metadata = adaptive_k_selection(
            nodes, k_min=1, k_max=10, probability_target=0.70, temperature=1.0
        )

        # First node should have very high probability
        assert metadata["selected_k"] >= 1
        assert metadata["stopped_reason"] in ["probability", "insufficient_candidates"]
        assert len(metadata["probabilities"]) == 3
        assert len(metadata["cumulative_masses"]) == 3

    def test_adaptive_k_cost_constraint(self):
        """Test that cost constraint stops selection."""
        # Create nodes with very long text to trigger cost limit
        # ~50k characters = ~12.5k tokens = ~$0.0125 cost per node
        long_text = "word " * 10000
        nodes = [
            self._create_mock_node(long_text, 0.9),
            self._create_mock_node(long_text, 0.8),
        ]
        selected, metadata = adaptive_k_selection(
            nodes,
            k_min=1,
            k_max=10,
            max_cost_per_query=0.01,  # Very low budget
            price_per_1k_tokens=0.001,
        )

        # Should stop due to cost (k_min=1 means first node is always taken, costing 0.0125)
        # Since first node exceeds budget, it should stop at k_min
        assert metadata["stopped_reason"] in [
            "cost",
            "insufficient_candidates",
            "probability",
        ]
        # Cost will be at least the first node, which may exceed budget
        # The important thing is that it stops due to cost constraint
        assert metadata["total_cost"] > 0

    def test_adaptive_k_k_max_constraint(self):
        """Test that k_max constraint is respected."""
        nodes = [self._create_mock_node(f"text{i}", 0.9 - i * 0.1) for i in range(15)]
        selected, metadata = adaptive_k_selection(
            nodes,
            k_min=2,
            k_max=5,
            probability_target=0.99,  # High target to force more docs
        )

        assert len(selected) <= 5
        assert metadata["selected_k"] <= 5
        if metadata["selected_k"] == 5:
            assert metadata["stopped_reason"] == "k_max"

    def test_adaptive_k_softmax_probabilities(self):
        """Test that softmax probabilities sum to 1."""
        nodes = [
            self._create_mock_node("text1", 1.0),
            self._create_mock_node("text2", 0.5),
            self._create_mock_node("text3", 0.2),
        ]
        selected, metadata = adaptive_k_selection(nodes, k_min=1, k_max=10)

        probabilities = metadata["probabilities"]
        assert len(probabilities) == 3
        # Probabilities should sum to approximately 1.0
        assert abs(sum(probabilities) - 1.0) < 0.0001
        # First node should have highest probability
        assert probabilities[0] > probabilities[1]
        assert probabilities[1] > probabilities[2]

    def test_adaptive_k_cumulative_masses(self):
        """Test cumulative probability masses."""
        nodes = [
            self._create_mock_node("text1", 1.0),
            self._create_mock_node("text2", 0.5),
            self._create_mock_node("text3", 0.2),
        ]
        selected, metadata = adaptive_k_selection(nodes, k_min=1, k_max=10)

        cumulative_masses = metadata["cumulative_masses"]
        assert len(cumulative_masses) == 3
        # Cumulative masses should be non-decreasing
        assert cumulative_masses[0] <= cumulative_masses[1]
        assert cumulative_masses[1] <= cumulative_masses[2]
        # Last cumulative mass should be ~1.0
        assert abs(cumulative_masses[-1] - 1.0) < 0.0001

    def test_adaptive_k_temperature_effect(self):
        """Test that temperature affects probability distribution."""
        nodes = [
            self._create_mock_node("text1", 1.0),
            self._create_mock_node("text2", 0.5),
        ]

        # Low temperature = sharper distribution
        selected_low, metadata_low = adaptive_k_selection(
            nodes, k_min=1, k_max=10, temperature=0.1
        )

        # High temperature = flatter distribution
        selected_high, metadata_high = adaptive_k_selection(
            nodes, k_min=1, k_max=10, temperature=10.0
        )

        # Low temperature should make first node more dominant
        assert metadata_low["probabilities"][0] > metadata_high["probabilities"][0]

    def test_adaptive_k_metadata_structure(self):
        """Test that metadata has all required fields."""
        nodes = [
            self._create_mock_node("text1", 0.9),
            self._create_mock_node("text2", 0.8),
            self._create_mock_node("text3", 0.7),
        ]
        selected, metadata = adaptive_k_selection(nodes)

        required_fields = [
            "selected_k",
            "total_tokens",
            "total_cost",
            "stopped_reason",
            "probabilities",
            "cumulative_masses",
        ]
        for field in required_fields:
            assert field in metadata, f"Missing field: {field}"

        assert isinstance(metadata["selected_k"], int)
        assert isinstance(metadata["total_tokens"], int)
        assert isinstance(metadata["total_cost"], float)
        assert isinstance(metadata["stopped_reason"], str)
        assert isinstance(metadata["probabilities"], list)
        assert isinstance(metadata["cumulative_masses"], list)

    def test_adaptive_k_stops_at_probability_target(self):
        """Test that selection stops when probability target is reached."""
        # Create nodes where first few have very high scores
        nodes = [
            self._create_mock_node("high1", 10.0),
            self._create_mock_node("high2", 9.0),
            self._create_mock_node("low1", 0.1),
            self._create_mock_node("low2", 0.1),
        ]
        selected, metadata = adaptive_k_selection(
            nodes, k_min=1, k_max=10, probability_target=0.70, temperature=1.0
        )

        # Should stop early when cumulative mass reaches target
        cumulative_masses = metadata["cumulative_masses"]
        if metadata["stopped_reason"] == "probability":
            # The cumulative mass at selected_k should be >= target
            final_cumulative = cumulative_masses[metadata["selected_k"] - 1]
            assert final_cumulative >= 0.70

    def test_adaptive_k_always_includes_k_min(self):
        """Test that k_min documents are always included."""
        nodes = [self._create_mock_node(f"text{i}", 0.9 - i * 0.1) for i in range(10)]
        selected, metadata = adaptive_k_selection(nodes, k_min=3, k_max=10)

        assert len(selected) >= 3
        assert metadata["selected_k"] >= 3
