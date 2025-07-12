import pandas as pd
import timesfm
import torch
import utilsforecast.processing as ufp
from timesfm.timesfm_base import DEFAULT_QUANTILES as DEFAULT_QUANTILES_TFM

from ..utils.forecaster import Forecaster, QuantileConverter


class TimesFM(Forecaster):
    def __init__(
        self,
        repo_id: str = "google/timesfm-1.0-200m-pytorch",
        context_length: int = 512,
        per_core_batch_size: int = 64,
        num_layers: int = 20,
        model_dims: int = 1280,
        alias: str = "TimesFM",
    ):
        if "pytorch" not in repo_id:
            raise ValueError(
                "TimesFM only supports pytorch models, "
                "if you'd like to use jax, please open an issue"
            )

        if "2.0" in repo_id:
            raise ValueError(
                "TimesFM 2.0 is not supported yet, "
                "see https://github.com/google-research/timesfm/issues/269"
                "please use TimesFM 1.0"
            )

        self.repo_id = repo_id
        self.context_length = context_length
        self.per_core_batch_size = per_core_batch_size
        self.num_layers = num_layers
        self.model_dims = model_dims
        self.alias = alias

    def get_predictor(
        self,
        prediction_length: int,
        quantiles: list[float] | None = None,
    ) -> timesfm.TimesFm:
        backend = "gpu" if torch.cuda.is_available() else "cpu"
        tfm_hparams = timesfm.TimesFmHparams(
            backend=backend,
            horizon_len=prediction_length,
            quantiles=quantiles,
            context_len=self.context_length,
            num_layers=self.num_layers,
            model_dims=self.model_dims,
            per_core_batch_size=self.per_core_batch_size,
        )
        tfm_checkpoint = timesfm.TimesFmCheckpoint(huggingface_repo_id=self.repo_id)
        tfm = timesfm.TimesFm(
            hparams=tfm_hparams,
            checkpoint=tfm_checkpoint,
        )
        return tfm

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        """Generate forecasts for time series data using the model.

        This method produces point forecasts and, optionally, prediction
        intervals or quantile forecasts. The input DataFrame can contain one
        or multiple time series in stacked (long) format.

        Args:
            df (pd.DataFrame):
                DataFrame containing the time series to forecast. It must
                include as columns:

                    - "unique_id": an ID column to distinguish multiple series.
                    - "ds": a time column indicating timestamps or periods.
                    - "y": a target column with the observed values.

            h (int):
                Forecast horizon specifying how many future steps to predict.
            freq (str):
                Frequency of the time series (e.g. "D" for daily, "M" for
                monthly). See [Pandas frequency aliases](https://pandas.pydata.org/
                pandas-docs/stable/user_guide/timeseries.html#offset-aliases) for
                valid values.
            level (list[int | float], optional):
                Confidence levels for prediction intervals, expressed as
                percentages (e.g. [80, 95]). If provided, the returned
                DataFrame will include lower and upper interval columns for
                each specified level.
            quantiles (list[float], optional):
                List of quantiles to forecast, expressed as floats between 0
                and 1. Should not be used simultaneously with `level`. When
                provided, the output DataFrame will contain additional columns
                named in the format "model-q-{percentile}", where {percentile}
                = 100 × quantile value.

        Returns:
            pd.DataFrame:
                DataFrame containing forecast results. Includes:

                    - point forecasts for each timestamp and series.
                    - prediction intervals if `level` is specified.
                    - quantile forecasts if `quantiles` is specified.

                For multi-series data, the output retains the same unique
                identifiers as the input DataFrame.
        """
        qc = QuantileConverter(level=level, quantiles=quantiles)
        if qc.quantiles is not None and len(qc.quantiles) != len(DEFAULT_QUANTILES_TFM):
            raise ValueError(
                "TimesFM only supports the default quantiles, "
                "please use the default quantiles or default level, "
                "see https://github.com/google-research/timesfm/issues/286"
            )
        predictor = self.get_predictor(
            prediction_length=h,
            quantiles=qc.quantiles or DEFAULT_QUANTILES_TFM,
        )
        fcst_df = predictor.forecast_on_df(
            inputs=df,
            freq=freq,
            value_name="y",
            model_name=self.alias,
            num_jobs=1,
        )
        if qc.quantiles is not None:
            for q in qc.quantiles:
                fcst_df = ufp.assign_columns(
                    fcst_df,
                    f"{self.alias}-q-{int(q * 100)}",
                    fcst_df[f"{self.alias}-q-{q}"],
                )
            fcst_df = qc.maybe_convert_quantiles_to_level(
                fcst_df,
                models=[self.alias],
            )
        else:
            fcst_df = fcst_df[["unique_id", "ds", self.alias]]
        return fcst_df
