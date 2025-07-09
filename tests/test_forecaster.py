import pytest
from utilsforecast.data import generate_series

from timecopilot.forecaster import TimeCopilotForecaster
from timecopilot.models import SeasonalNaive, ZeroModel


@pytest.fixture
def models():
    return [SeasonalNaive(), ZeroModel()]


@pytest.mark.parametrize(
    "freq,h",
    [
        ("D", 2),
        ("W-MON", 3),
    ],
)
def test_forecaster_forecast(models, freq, h):
    n_uids = 3
    df = generate_series(n_series=n_uids, freq=freq, min_length=30)
    forecaster = TimeCopilotForecaster(models=models)
    fcst_df = forecaster.forecast(df=df, h=h, freq=freq)
    assert len(fcst_df.columns) == 2 + len(models)
    assert len(fcst_df) == h * n_uids
    for model in models:
        assert model.alias in fcst_df.columns


@pytest.mark.parametrize(
    "freq,h,n_windows,step_size",
    [
        ("D", 2, 2, 1),
        ("W-MON", 3, 2, 2),
    ],
)
def test_forecaster_cross_validation(models, freq, h, n_windows, step_size):
    n_uids = 3
    df = generate_series(n_series=n_uids, freq=freq, min_length=30)
    forecaster = TimeCopilotForecaster(models=models)
    fcst_df = forecaster.cross_validation(
        df=df,
        h=h,
        freq=freq,
        n_windows=n_windows,
        step_size=step_size,
    )
    assert len(fcst_df.columns) == 4 + len(models)
    uids = df["unique_id"].unique()
    for uid in uids:  # noqa: B007
        fcst_df_uid = fcst_df.query("unique_id == @uid")
        assert fcst_df_uid["cutoff"].nunique() == n_windows
        assert len(fcst_df_uid) == n_windows * h
    for model in models:
        assert model.alias in fcst_df.columns


def test_forecaster_forecast_with_level(models):
    n_uids = 3
    level = [80, 90]
    df = generate_series(n_series=n_uids, freq="D", min_length=30)
    forecaster = TimeCopilotForecaster(models=models)
    fcst_df = forecaster.forecast(df=df, h=2, freq="D", level=level)
    assert len(fcst_df) == 2 * n_uids
    assert len(fcst_df.columns) == 2 + len(models) * (1 + 2 * len(level))
    for model in models:
        assert model.alias in fcst_df.columns
        for lv in level:
            assert f"{model.alias}-lo-{lv}" in fcst_df.columns
            assert f"{model.alias}-hi-{lv}" in fcst_df.columns


def test_forecaster_forecast_with_quantiles(models):
    n_uids = 3
    quantiles = [0.1, 0.9]
    df = generate_series(n_series=n_uids, freq="D", min_length=30)
    forecaster = TimeCopilotForecaster(models=models)
    fcst_df = forecaster.forecast(df=df, h=2, freq="D", quantiles=quantiles)
    assert len(fcst_df) == 2 * n_uids
    assert len(fcst_df.columns) == 2 + len(models) * (1 + len(quantiles))
    for model in models:
        assert model.alias in fcst_df.columns
        for q in quantiles:
            assert f"{model.alias}-q-{int(100 * q)}" in fcst_df.columns
