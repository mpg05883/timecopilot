import pytest
from utilsforecast.data import generate_series

from timecopilot.models.utils.forecaster import QuantileConverter


def test_prepare_level_and_quantiles_with_levels():
    qc = QuantileConverter(level=[80, 95])
    assert qc.level == [80, 95]
    assert qc.quantiles is None


def test_prepare_level_and_quantiles_with_quantiles():
    quantiles = [0.1, 0.5, 0.9]
    qc = QuantileConverter(level=None, quantiles=quantiles)
    expected_level = [0, 80]
    assert qc.quantiles == quantiles
    assert qc.level == expected_level


def test_prepare_level_and_quantiles_error_both():
    with pytest.raises(ValueError):
        QuantileConverter(level=[90], quantiles=[0.9])


@pytest.mark.parametrize(
    "n_models,quantiles",
    [
        (1, [0.1]),
        (2, [0.1, 0.5, 0.9]),
    ],
)
def test_maybe_convert_level_to_quantiles(n_models, quantiles):
    models = [f"model{i}" for i in range(n_models)]
    qc = QuantileConverter(quantiles=quantiles)
    df = generate_series(
        n_series=1,
        freq="D",
        min_length=10,
        n_models=n_models,
        level=qc.level,
    )
    result_df = qc.maybe_convert_level_to_quantiles(
        df,
        models=models,
    )
    for model in models:
        assert qc.quantiles is not None
        for q in qc.quantiles:
            assert f"{model}-q-{int(q * 100)}" in result_df.columns
        if 0.5 in qc.quantiles:
            assert result_df.loc[0, f"{model}-q-50"] == result_df.loc[0, f"{model}"]
