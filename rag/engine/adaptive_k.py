"""
Adaptive K selection for document retrieval.

Implements early stopping algorithm to select optimal number of documents
based on score distribution and cost constraints.

Uses softmax probability mass approach:
1. Convert scores to probabilities using softmax (temperature τ)
2. Compute cumulative probability mass
3. Stop when cumulative mass >= target (e.g., 0.95)
4. Respect k_min, k_max, and cost constraints
"""

import typing as t
import math
from llama_index.core.schema import NodeWithScore
from rag.logging_config import get_logger

logger = get_logger(__name__)


def estimate_tokens(text: str) -> int:
    """
    Estimate token count from text.
    Uses rough approximation: 1 token ≈ 4 characters for English text.
    Can be improved with actual tokenizer (tiktoken/anthropic) later.
    """
    return len(text) // 4


def calculate_cost(tokens: int, price_per_1k_tokens: float = 0.001) -> float:
    """
    Calculate cost for given token count.

    Args:
        tokens: Number of tokens
        price_per_1k_tokens: Price per 1000 tokens (default: $0.001 = $1 per 1M tokens)

    Returns:
        Cost in dollars
    """
    return (tokens / 1000.0) * price_per_1k_tokens


def adaptive_k_selection(
    nodes: t.List[NodeWithScore],
    k_min: int = 2,
    k_max: int = 10,
    probability_target: float = 0.70,
    temperature: float = 1.0,
    max_cost_per_query: float = 0.01,
    price_per_1k_tokens: float = 0.001,
) -> t.Tuple[t.List[NodeWithScore], dict]:
    """
    Select optimal K documents based on softmax probability mass and cost constraints.

    Algorithm (softmax probability mass approach):
    1. Convert scores to probabilities using softmax with temperature τ
    2. Compute cumulative probability mass P_k = sum(p_1 to p_k)
    3. Start with k_min documents (always take top k_min)
    4. For each next candidate:
       - Check if cumulative mass >= target (e.g., 0.95) → stop
       - Check if cost exceeds budget → stop
       - Otherwise → continue adding documents

    Args:
        nodes: List of NodeWithScore candidates (already ranked/reranked)
        k_min: Minimum number of documents to select (default: 2)
        k_max: Maximum number of documents to select (default: 10)
        probability_target: Target cumulative probability mass (default: 0.95)
            Stop when cumulative mass >= target
        temperature: Temperature for softmax (default: 1.0)
            Lower temperature = sharper distribution, higher = flatter
        max_cost_per_query: Maximum cost per query in dollars (default: $0.01)
        price_per_1k_tokens: Price per 1000 input tokens (default: $0.001)

    Returns:
        Tuple of (selected_nodes, metadata_dict)
        metadata_dict contains:
            - selected_k: Number of documents selected
            - total_tokens: Estimated token count
            - total_cost: Estimated cost
            - stopped_reason: Why selection stopped ("probability", "cost", "k_max", "end")
            - probabilities: List of probabilities for each document
            - cumulative_masses: List of cumulative probability masses
    """
    if len(nodes) == 0:
        return [], {
            "selected_k": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "stopped_reason": "no_candidates",
            "probabilities": [],
            "cumulative_masses": [],
        }

    if len(nodes) <= k_min:
        # Not enough candidates, take all
        selected = nodes
        total_text = " ".join(node.node.get_content() for node in selected)
        tokens = estimate_tokens(total_text)
        cost = calculate_cost(tokens, price_per_1k_tokens)
        return selected, {
            "selected_k": len(selected),
            "total_tokens": tokens,
            "total_cost": cost,
            "stopped_reason": "insufficient_candidates",
            "probabilities": [1.0 / len(nodes)] * len(nodes) if nodes else [],
            "cumulative_masses": [],
        }

    # Extract scores
    scores = [node.score for node in nodes]

    # Log all candidate scores for debugging
    logger.info(f"📊 Adaptive K: {len(nodes)} candidates with scores:")
    for idx, (node, score) in enumerate(zip(nodes, scores), 1):
        logger.info(f"  Rank {idx}: score={score:.4f}")

    # Convert scores to probabilities using softmax
    # p_i = exp(s_i / τ) / sum(exp(s_j / τ))
    exp_scores = [math.exp(score / temperature) for score in scores]
    sum_exp = sum(exp_scores)
    probabilities = [exp_score / sum_exp for exp_score in exp_scores]

    # Compute cumulative probability masses
    cumulative_masses = []
    cumsum = 0.0
    for prob in probabilities:
        cumsum += prob
        cumulative_masses.append(cumsum)

    # Log probabilities and cumulative masses
    logger.info(f"📊 Softmax probabilities (τ={temperature:.2f}):")
    for idx, (prob, cum_mass) in enumerate(zip(probabilities, cumulative_masses), 1):
        logger.info(
            f"  Rank {idx}: p={prob:.4f}, cumulative={cum_mass:.4f} "
            f"{'✅' if cum_mass >= probability_target else ''}"
        )

    # Start with minimum K
    selected = nodes[:k_min]
    selected_k = k_min
    stopped_reason = None

    # Calculate initial cost
    total_text = " ".join(node.node.get_content() for node in selected)
    total_tokens = estimate_tokens(total_text)
    total_cost = calculate_cost(total_tokens, price_per_1k_tokens)

    # Check if k_min already meets probability target
    if cumulative_masses[k_min - 1] >= probability_target:
        logger.info(
            f"✅ Already at probability target with k_min={k_min} "
            f"(cumulative={cumulative_masses[k_min-1]:.4f} >= {probability_target:.2f})"
        )
        stopped_reason = "probability"
    else:
        # Iterate through remaining candidates
        for i in range(k_min, min(k_max, len(nodes))):
            # Check if we can add this candidate
            candidate = nodes[i]
            candidate_text = candidate.node.get_content()
            candidate_tokens = estimate_tokens(candidate_text)
            candidate_cost = calculate_cost(candidate_tokens, price_per_1k_tokens)

            # Check cost constraint
            if total_cost + candidate_cost > max_cost_per_query:
                stopped_reason = "cost"
                logger.info(
                    f"💰 Adaptive K: Stopped at {selected_k} due to cost limit "
                    f"(${total_cost:.4f} + ${candidate_cost:.4f} > ${max_cost_per_query:.4f})"
                )
                break

            # Check probability target (cumulative mass)
            current_cumulative = cumulative_masses[i]
            logger.info(
                f"📈 Adaptive K: Checking position {i+1} "
                f"(cumulative mass: {current_cumulative:.4f}, target: {probability_target:.2f})"
            )

            if current_cumulative >= probability_target:
                stopped_reason = "probability"
                logger.info(
                    f"✅ Adaptive K: STOPPED at {selected_k+1} due to probability target "
                    f"(cumulative={current_cumulative:.4f} >= {probability_target:.2f})"
                )
                # Add this candidate before stopping (it reaches the target)
                selected.append(candidate)
                selected_k += 1
                total_text += " " + candidate_text
                total_tokens = estimate_tokens(total_text)
                total_cost = calculate_cost(total_tokens, price_per_1k_tokens)
                break

            # Add this candidate
            selected.append(candidate)
            selected_k += 1
            total_text += " " + candidate_text
            total_tokens = estimate_tokens(total_text)
            total_cost = calculate_cost(total_tokens, price_per_1k_tokens)

    # Determine final stopped reason
    if stopped_reason is None:
        if selected_k >= k_max:
            stopped_reason = "k_max"
        elif selected_k >= len(nodes):
            stopped_reason = "end"
        else:
            stopped_reason = "unknown"

    metadata = {
        "selected_k": selected_k,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "stopped_reason": stopped_reason,
        "probabilities": probabilities,
        "cumulative_masses": cumulative_masses,
    }

    # Log summary
    final_cumulative = cumulative_masses[selected_k - 1] if selected_k > 0 else 0.0
    logger.info(
        f"🎯 Adaptive K selection: {len(nodes)} candidates → {selected_k} selected "
        f"(stopped: {stopped_reason}, cumulative mass: {final_cumulative:.4f}, "
        f"cost: ${total_cost:.4f}, tokens: {total_tokens})"
    )

    return selected, metadata
