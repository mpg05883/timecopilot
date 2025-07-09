"""
Test that the agent works with a live LLM.
Keeping it separate from the other tests because costs and requires a live LLM.
"""

import logfire
import pytest
from dotenv import load_dotenv
from utilsforecast.data import generate_series

from timecopilot import TimeCopilot

load_dotenv()
logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_pydantic_ai()


@pytest.mark.live
@pytest.mark.parametrize("n_series", [1, 2])
def test_forecast_returns_expected_output(n_series):
    h = 2
    df = generate_series(
        n_series=n_series,
        freq="D",
        min_length=30,
        static_as_categorical=False,
    )
    tc = TimeCopilot(
        llm="openai:gpt-4o-mini",
        retries=3,
    )
    result = tc.forecast(
        df=df,
        query=f"Please forecast the series with a horizon of {h} and frequency D.",
    )
    assert len(result.fcst_df) == n_series * h
    assert result.output.is_better_than_seasonal_naive
    assert result.output.forecast_analysis is not None
    assert result.output.reason_for_selection is not None
