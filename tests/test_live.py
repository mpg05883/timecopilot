"""
Test that the agent works with a live LLM.
Keeping it separate from the other tests because costs and requires a live LLM.
"""

import pytest
from utilsforecast.data import generate_series

from timecopilot import TimeCopilot


@pytest.mark.live
def test_forecast_returns_expected_output():
    df = generate_series(
        n_series=1,
        freq="D",
        min_length=30,
        static_as_categorical=False,
    )
    forecasting_agent = TimeCopilot(
        llm="openai:gpt-4o-mini",
        retries=3,
    )
    result = forecasting_agent.forecast(
        df=df,
        query="Please forecast the series with a horizon of 2 and frequency D.",
    )
    assert len(result.output.forecast) == 2
    assert result.output.is_better_than_seasonal_naive
    assert result.output.forecast_analysis is not None
    assert result.output.reason_for_selection is not None
