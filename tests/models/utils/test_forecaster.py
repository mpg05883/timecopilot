import pandas as pd
import pytest
from utilsforecast.data import generate_series

from timecopilot.models.utils.forecaster import QuantileConverter


def test_prepare_level_and_quantiles_with_levels():
    qc = QuantileConverter(level=[80, 95])
    assert qc.level == [80, 95]
    assert qc.level_was_provided


@pytest.mark.parametrize(
    "quantiles,expected_level",
    [
        ([0.1, 0.5, 0.9], [0, 80]),
        ([0.1, 0.5, 0.2, 0.9], [0, 60, 80]),
        ([0.5], [0]),
    ],
)
def test_prepare_level_and_quantiles_with_quantiles(quantiles, expected_level):
    qc = QuantileConverter(level=None, quantiles=quantiles)
    assert qc.quantiles == quantiles
    assert qc.level == expected_level
    assert not qc.level_was_provided


def test_prepare_level_and_quantiles_error_both():
    with pytest.raises(ValueError):
        QuantileConverter(level=[90], quantiles=[0.9])


@pytest.mark.parametrize(
    "n_models,quantiles",
    [
        (1, [0.1]),
        (2, [0.1, 0.5, 0.9]),
        (2, [0.1, 0.5, 0.2, 0.9]),
    ],
)
def test_maybe_convert_level_to_quantiles(n_models, quantiles):
    models = [f"model{i}" for i in range(n_models)]
    qc = QuantileConverter(quantiles=quantiles)
    assert not qc.level_was_provided
    df = generate_series(
        n_series=2,
        freq="D",
        min_length=10,
        n_models=n_models,
        level=qc.level,
    )
    result_df = qc.maybe_convert_level_to_quantiles(
        df,
        models=models,
    )
    exp_n_cols = 3 + (1 + len(quantiles)) * n_models
    assert result_df.shape[1] == exp_n_cols
    for model in models:
        assert qc.quantiles is not None
        for q in qc.quantiles:
            assert f"{model}-q-{int(q * 100)}" in result_df.columns
        if 0.5 in qc.quantiles:
            pd.testing.assert_series_equal(
                result_df[f"{model}-q-50"],
                result_df[f"{model}"],
                check_names=False,
            )
    # check that maybe convert quantiles to level returns the same result
    pd.testing.assert_frame_equal(
        df,
        qc.maybe_convert_quantiles_to_level(df, models=models),
    )


@pytest.mark.parametrize(
    "n_models,level",
    [
        (1, [80]),
        (2, [0, 80]),
        (2, [60, 80]),
    ],
)
def test_maybe_convert_quantiles_to_level(n_models, level):
    models = [f"model{i}" for i in range(n_models)]
    qc = QuantileConverter(level=level)
    assert qc.level_was_provided
    df = generate_series(
        n_series=2,
        freq="D",
        min_length=10,
        n_models=n_models,
    )
    for model in models:
        for q in qc.quantiles:  # type: ignore
            df[f"{model}-q-{int(q * 100)}"] = q
    result_df = qc.maybe_convert_quantiles_to_level(
        df,
        models=models,
    )
    exp_n_cols = 3 + (1 + len(level) * 2) * n_models
    if 0 in level:
        exp_n_cols -= n_models * 2
    assert result_df.shape[1] == exp_n_cols
    for model in models:
        for lv in level:
            if lv == 0:
                pd.testing.assert_series_equal(
                    result_df[model],
                    df[f"{model}-q-50"],
                    check_names=False,
                )
            else:
                alpha = 1 - lv / 100
                q_lo = int((alpha / 2) * 100)
                q_hi = int((1 - alpha / 2) * 100)
                pd.testing.assert_series_equal(
                    result_df[f"{model}-lo-{lv}"],
                    df[f"{model}-q-{q_lo}"],
                    check_names=False,
                )
                pd.testing.assert_series_equal(
                    result_df[f"{model}-hi-{lv}"],
                    df[f"{model}-q-{q_hi}"],
                    check_names=False,
                )
    # check that maybe convert level to quantiles returns the same result
    pd.testing.assert_frame_equal(
        df,
        qc.maybe_convert_level_to_quantiles(df, models=models),
    )
