"""Input validation utilities for MS_RAG.

All numeric user inputs must pass through validate_numeric() before being
stored in PipelineConfig. This centralises validation and eliminates
duplicated range-check logic across workflow steps.

Validation rules (from design.md error-handling table):
    - Chunk size:                   > 0 (positive integer)
    - Chunk overlap:                0 <= overlap < chunk_size  (checked separately)
    - Retrieval top_k:              1 <= k <= 1000
    - Reranking top_k:              1 <= k <= retrieval_top_k  (max_val passed by caller)
    - Hybrid alpha:                 0.0 <= alpha <= 1.0
    - MMR lambda:                   0.0 <= lambda <= 1.0
    - Embeddings filter threshold:  0.0 <= t <= 1.0
    - CI/CD threshold:              0.0 <= t <= 1.0
    - Query length:                 1 <= len <= 4096
    - System prompt length:         <= 10000 chars
    - Compression technique count:  1 <= count <= 6
    - Ensemble weights sum:         abs(sum - 1.0) <= 0.01
"""

from __future__ import annotations

from ms_rag.utils.exceptions import ValidationError


def validate_numeric(
    value: int | float,
    min_val: int | float,
    max_val: int | float,
    field_name: str,
) -> None:
    """Validate that *value* is within [min_val, max_val] (inclusive).

    Args:
        value:      The numeric value to validate.
        min_val:    The minimum allowed value (inclusive).
        max_val:    The maximum allowed value (inclusive).
        field_name: Human-readable field name used in the error message.

    Raises:
        ValidationError: If ``value < min_val`` or ``value > max_val``.

    Example::

        validate_numeric(512, 1, 8192, "chunk_size")          # OK
        validate_numeric(0, 1, 8192, "chunk_size")             # raises ValidationError
        validate_numeric(0.6, 0.0, 1.0, "hybrid_alpha")       # OK
        validate_numeric(1.5, 0.0, 1.0, "hybrid_alpha")       # raises ValidationError
    """
    if value < min_val or value > max_val:
        raise ValidationError(
            f"{field_name} must be between {min_val} and {max_val} (got {value}).",
            field_name=field_name,
            value=value,
            min_val=min_val,
            max_val=max_val,
        )


def validate_ensemble_weights(weights: list[float], field_name: str = "ensemble_weights") -> None:
    """Validate that ensemble retriever weights sum to 1.0 (±0.01 tolerance).

    Args:
        weights:    List of float weights, one per sub-retriever.
        field_name: Field name for the error message.

    Raises:
        ValidationError: If ``abs(sum(weights) - 1.0) > 0.01``.
    """
    if not weights:
        raise ValidationError(
            f"{field_name}: at least one weight is required.",
            field_name=field_name,
            value=0,
            min_val=1,
            max_val=None,
        )
    total = sum(weights)
    if abs(total - 1.0) > 0.01:
        raise ValidationError(
            f"{field_name} must sum to 1.0 (±0.01), got {total:.4f}.",
            field_name=field_name,
            value=total,
            min_val=0.99,
            max_val=1.01,
        )


def validate_chunk_overlap(chunk_size: int, chunk_overlap: int) -> None:
    """Validate that chunk_overlap is strictly less than chunk_size.

    Args:
        chunk_size:    The configured chunk size (positive integer).
        chunk_overlap: The configured chunk overlap (non-negative integer).

    Raises:
        ValidationError: If ``chunk_overlap >= chunk_size``.
    """
    if chunk_overlap >= chunk_size:
        raise ValidationError(
            f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size}).",
            field_name="chunk_overlap",
            value=chunk_overlap,
            min_val=0,
            max_val=chunk_size - 1,
        )
