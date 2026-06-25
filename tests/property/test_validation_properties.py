"""Property-based tests for validate_numeric and related validators.

Property 27: Numeric Input Range Validation
    For any numeric configuration field and any input value outside the
    field's allowed range, validate_numeric() must return a ValidationError.
    For any input value within the allowed range, no error should be raised.

Validates: Requirements 19.4
"""

from __future__ import annotations

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from ms_rag.utils.exceptions import ValidationError
from ms_rag.utils.validation import (
    validate_chunk_overlap,
    validate_ensemble_weights,
    validate_numeric,
)


# ---------------------------------------------------------------------------
# Property 27: Numeric Input Range Validation (integers)
# ---------------------------------------------------------------------------


@given(
    value=st.integers(min_value=-10_000, max_value=10_000),
    min_val=st.integers(min_value=-5_000, max_value=0),
    max_val=st.integers(min_value=1, max_value=5_000),
)
@settings(max_examples=200)
def test_validate_numeric_integers_outside_range_raises(
    value: int, min_val: int, max_val: int
) -> None:
    """Feature: ms-rag, Property 27: Numeric Input Range Validation (integers).

    validate_numeric() must raise ValidationError iff value is outside [min_val, max_val].
    """
    assume(min_val < max_val)

    if value < min_val or value > max_val:
        with pytest.raises(ValidationError) as exc_info:
            validate_numeric(value, min_val, max_val, "test_field")
        err = exc_info.value
        assert err.field_name == "test_field"
        assert err.value == value
        assert err.min_val == min_val
        assert err.max_val == max_val
    else:
        # Must NOT raise
        validate_numeric(value, min_val, max_val, "test_field")


# ---------------------------------------------------------------------------
# Property 27: Numeric Input Range Validation (floats)
# ---------------------------------------------------------------------------


@given(
    value=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200)
def test_validate_numeric_floats_unit_interval(value: float) -> None:
    """Float values outside [0.0, 1.0] must raise; within range must not raise."""
    if value < 0.0 or value > 1.0:
        with pytest.raises(ValidationError):
            validate_numeric(value, 0.0, 1.0, "alpha")
    else:
        validate_numeric(value, 0.0, 1.0, "alpha")


# ---------------------------------------------------------------------------
# Specific field validation — exact rules from design.md table
# ---------------------------------------------------------------------------


class TestChunkSizeValidation:
    def test_chunk_size_zero_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            validate_numeric(0, 1, 8192, "chunk_size")
        assert "chunk_size" in str(exc_info.value)

    def test_chunk_size_negative_raises(self) -> None:
        with pytest.raises(ValidationError):
            validate_numeric(-1, 1, 8192, "chunk_size")

    def test_chunk_size_valid_passes(self) -> None:
        validate_numeric(512, 1, 8192, "chunk_size")
        validate_numeric(1, 1, 8192, "chunk_size")
        validate_numeric(8192, 1, 8192, "chunk_size")


class TestRetrievalTopKValidation:
    def test_top_k_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            validate_numeric(0, 1, 1000, "retrieval_top_k")

    def test_top_k_above_max_raises(self) -> None:
        with pytest.raises(ValidationError):
            validate_numeric(1001, 1, 1000, "retrieval_top_k")

    def test_top_k_boundary_values_pass(self) -> None:
        validate_numeric(1, 1, 1000, "retrieval_top_k")
        validate_numeric(1000, 1, 1000, "retrieval_top_k")
        validate_numeric(5, 1, 1000, "retrieval_top_k")


class TestAlphaAndLambdaValidation:
    @pytest.mark.parametrize("val", [-0.01, -1.0, 1.01, 2.0])
    def test_out_of_range_raises(self, val: float) -> None:
        with pytest.raises(ValidationError):
            validate_numeric(val, 0.0, 1.0, "hybrid_alpha")

    @pytest.mark.parametrize("val", [0.0, 0.5, 1.0])
    def test_boundary_and_midpoint_pass(self, val: float) -> None:
        validate_numeric(val, 0.0, 1.0, "hybrid_alpha")
        validate_numeric(val, 0.0, 1.0, "mmr_lambda")


class TestThresholdValidation:
    @pytest.mark.parametrize("val", [-0.1, 1.1])
    def test_out_of_range_raises(self, val: float) -> None:
        with pytest.raises(ValidationError):
            validate_numeric(val, 0.0, 1.0, "embeddings_threshold")

    @pytest.mark.parametrize("val", [0.0, 0.75, 1.0])
    def test_valid_passes(self, val: float) -> None:
        validate_numeric(val, 0.0, 1.0, "embeddings_threshold")


# ---------------------------------------------------------------------------
# validate_chunk_overlap
# ---------------------------------------------------------------------------


@given(
    chunk_size=st.integers(min_value=1, max_value=4096),
    overlap=st.integers(min_value=0, max_value=4096),
)
@settings(max_examples=200)
def test_chunk_overlap_property(chunk_size: int, overlap: int) -> None:
    """chunk_overlap >= chunk_size must raise; overlap < chunk_size must pass."""
    if overlap >= chunk_size:
        with pytest.raises(ValidationError) as exc_info:
            validate_chunk_overlap(chunk_size, overlap)
        assert exc_info.value.field_name == "chunk_overlap"
    else:
        validate_chunk_overlap(chunk_size, overlap)


# ---------------------------------------------------------------------------
# validate_ensemble_weights
# ---------------------------------------------------------------------------


class TestEnsembleWeights:
    def test_equal_weights_pass(self) -> None:
        validate_ensemble_weights([0.5, 0.5])
        validate_ensemble_weights([0.33, 0.33, 0.34])

    def test_weights_not_summing_to_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            validate_ensemble_weights([0.4, 0.4])  # sum = 0.8

    def test_weights_slightly_over_tolerance_raises(self) -> None:
        with pytest.raises(ValidationError):
            validate_ensemble_weights([0.5, 0.52])  # sum = 1.02 (> 0.01 tolerance)

    def test_weights_within_tolerance_pass(self) -> None:
        # sum = 1.005 — within ±0.01
        validate_ensemble_weights([0.5, 0.505])

    def test_empty_list_raises(self) -> None:
        with pytest.raises(ValidationError):
            validate_ensemble_weights([])

    @given(
        weights=st.lists(
            st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=2, max_size=5,
        )
    )
    @settings(max_examples=100)
    def test_property_ensemble_weights(self, weights: list[float]) -> None:
        """Weights summing outside 1.0±0.01 must raise; within range must pass."""
        total = sum(weights)
        if abs(total - 1.0) > 0.01:
            with pytest.raises(ValidationError):
                validate_ensemble_weights(weights)
        else:
            validate_ensemble_weights(weights)
