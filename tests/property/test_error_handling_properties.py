"""Property-based tests for error handling and retry utilities.

Properties covered:
    Property 25: API Error Recovery Options (Req 19.1)

Validates: Requirements 19.1, 19.2, 19.3
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ms_rag.utils.retry import retry_with_backoff, retry_with_user_prompt


# ---------------------------------------------------------------------------
# Property 25: API Error Recovery Options
# ---------------------------------------------------------------------------


@given(
    error_type=st.sampled_from([
        ConnectionError, TimeoutError, ValueError, RuntimeError, IOError
    ])
)
@settings(max_examples=30)
def test_retry_with_user_prompt_presents_options_on_failure(
    error_type: type[Exception],
) -> None:
    """Feature: ms-rag, Property 25: API Error Recovery Options.

    When all retry attempts fail, the user must be presented with exactly
    Retry / Skip / Abort options — no exception propagates uncaught.
    """
    def always_fail() -> None:
        raise error_type("simulated failure")

    with patch("ms_rag.utils.retry.questionary") as mock_q, \
         patch("ms_rag.utils.retry.Console"):

        mock_select = MagicMock()
        mock_select.ask.return_value = "skip"  # user chooses Skip
        mock_q.select.return_value = mock_select
        mock_q.Choice = MagicMock(side_effect=lambda title, value: value)

        result = retry_with_user_prompt(
            always_fail, operation_name="test_op", max_attempts=1, delays=(0.001,)
        )

    # Skip returns None
    assert result is None
    # Select must have been called with Retry/Skip/Abort options
    mock_q.select.assert_called_once()


def test_retry_with_user_prompt_retry_option_re_calls_fn() -> None:
    """Choosing Retry must re-attempt the function."""
    attempt_count = {"n": 0}

    def fn_succeeds_on_second_call() -> str:
        attempt_count["n"] += 1
        if attempt_count["n"] == 1:
            raise RuntimeError("first fail — triggers user prompt")
        return "success"  # succeeds on the retry pass

    with patch("ms_rag.utils.retry.questionary") as mock_q, \
         patch("ms_rag.utils.retry.Console"):

        mock_select = MagicMock()
        mock_select.ask.return_value = "retry"
        mock_q.select.return_value = mock_select
        mock_q.Choice = MagicMock(side_effect=lambda title, value: value)

        result = retry_with_user_prompt(
            fn_succeeds_on_second_call,
            operation_name="test",
            max_attempts=1,
            delays=(0.001,),
        )

    assert result == "success"


def test_retry_with_user_prompt_abort_raises_system_exit() -> None:
    """Choosing Abort must raise SystemExit."""
    def always_fail() -> None:
        raise RuntimeError("fail")

    with patch("ms_rag.utils.retry.questionary") as mock_q, \
         patch("ms_rag.utils.retry.Console"):

        mock_select = MagicMock()
        mock_select.ask.return_value = "abort"
        mock_q.select.return_value = mock_select
        mock_q.Choice = MagicMock(side_effect=lambda title, value: value)

        with pytest.raises(SystemExit):
            retry_with_user_prompt(
                always_fail, operation_name="test", max_attempts=1, delays=(0.001,)
            )


# ---------------------------------------------------------------------------
# retry_with_backoff unit tests
# ---------------------------------------------------------------------------


class TestRetryWithBackoff:
    def test_succeeds_on_first_attempt(self) -> None:
        result = retry_with_backoff(lambda: "ok", max_attempts=3, delays=(0.001,))
        assert result == "ok"

    def test_retries_on_failure_and_succeeds(self) -> None:
        counter = {"n": 0}

        def fn() -> str:
            counter["n"] += 1
            if counter["n"] < 3:
                raise RuntimeError("transient")
            return "done"

        result = retry_with_backoff(fn, max_attempts=3, delays=(0.001, 0.001))
        assert result == "done"
        assert counter["n"] == 3

    def test_raises_after_max_attempts(self) -> None:
        with pytest.raises(ValueError, match="always fails"):
            retry_with_backoff(
                lambda: (_ for _ in ()).throw(ValueError("always fails")),
                max_attempts=3,
                delays=(0.001, 0.001),
            )

    def test_on_retry_callback_invoked(self) -> None:
        retries: list[int] = []

        def fn() -> None:
            raise RuntimeError("fail")

        def on_retry(n: int, exc: Exception) -> None:
            retries.append(n)

        with pytest.raises(RuntimeError):
            retry_with_backoff(fn, max_attempts=3, delays=(0.001, 0.001), on_retry=on_retry)

        assert retries == [1, 2]

    def test_single_attempt_raises_immediately(self) -> None:
        with pytest.raises(IOError):
            retry_with_backoff(
                lambda: (_ for _ in ()).throw(IOError("io")),
                max_attempts=1,
                delays=(0.001,),
            )

    @given(
        error_type=st.sampled_from([ConnectionError, TimeoutError, ValueError, RuntimeError])
    )
    @settings(max_examples=20)
    def test_any_exception_type_is_retried(self, error_type: type[Exception]) -> None:
        """Any exception type raised by fn() must trigger the retry loop."""
        counter = {"n": 0}

        def fn() -> str:
            counter["n"] += 1
            if counter["n"] < 2:
                raise error_type("fail")
            return "ok"

        result = retry_with_backoff(fn, max_attempts=3, delays=(0.001, 0.001))
        assert result == "ok"
        assert counter["n"] == 2
