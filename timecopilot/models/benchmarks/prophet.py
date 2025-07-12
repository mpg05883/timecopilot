from copy import deepcopy

import pandas as pd
from prophet import Prophet as ProphetBase
from threadpoolctl import threadpool_limits

from ..utils.forecaster import Forecaster, QuantileConverter
from ..utils.parallel_forecaster import ParallelForecaster


class Prophet(ProphetBase, ParallelForecaster, Forecaster):
    def __init__(
        self,
        alias: str = "Prophet",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.alias = alias
        if "interval_width" in kwargs:
            raise ValueError(
                "interval_width is not supported, "
                "use `level` or `quantiles` instead when using "
                "`forecast` or `cross_validation`"
            )

    def predict_uncertainty(
        self,
        df: pd.DataFrame,
        vectorized: bool,
        quantiles: list[float],
    ) -> pd.DataFrame:
        # adapted from https://github.com/facebook/prophet/blob/e64606036325bfb225333ef0991e41bdfb66f7c1/python/prophet/forecaster.py#L1431-L1455
        sim_values = self.sample_posterior_predictive(df, vectorized)
        series = {}
        for q in quantiles:
            series[f"yhat-q-{int(q * 100)}"] = self.percentile(
                sim_values["yhat"], 100 * q, axis=1
            )
        return pd.DataFrame(series)

    def predict(
        self,
        df: pd.DataFrame = None,
        vectorized: bool = True,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        # Predict using the prophet model.
        # adapted from https://github.com/facebook/prophet/blob/e64606036325bfb225333ef0991e41bdfb66f7c1/python/prophet/forecaster.py#L1249C2-L1295C1
        # to allow for quantiles
        if self.history is None:
            raise Exception("Model has not been fit.")
        if df is None:
            df = self.history.copy()
        else:
            if df.shape[0] == 0:
                raise ValueError("Dataframe has no rows.")
            df = self.setup_dataframe(df.copy())
        df["trend"] = self.predict_trend(df)
        seasonal_components = self.predict_seasonal_components(df)
        if self.uncertainty_samples and quantiles is not None:
            intervals = self.predict_uncertainty(df, vectorized, quantiles)
        else:
            intervals = None
        # Drop columns except ds, cap, floor, and trend
        cols = ["ds", "trend"]
        if "cap" in df:
            cols.append("cap")
        if self.logistic_floor:
            cols.append("floor")
        # Add in forecast components
        df2 = pd.concat((df[cols], intervals, seasonal_components), axis=1)
        df2["yhat"] = (
            df2["trend"] * (1 + df2["multiplicative_terms"]) + df2["additive_terms"]
        )
        df2.columns = [col.replace("yhat", self.alias) for col in df2.columns]
        cols = [col for col in df2.columns if self.alias in col or "ds" in col]
        return df2[cols]

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
        return self._local_forecast(
            df=df,
            h=h,
            freq=freq,
            level=level,
            quantiles=quantiles,
        )

    def _local_forecast_impl(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        qc = QuantileConverter(level=level, quantiles=quantiles)
        model = deepcopy(self)
        model.fit(df=df)
        future_df = model.make_future_dataframe(
            periods=h,
            include_history=False,
            freq=freq,
        )
        fcst_df = model.predict(future_df, quantiles=qc.quantiles)
        fcst_df = qc.maybe_convert_quantiles_to_level(fcst_df, models=[self.alias])
        return fcst_df

    def _local_forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        with threadpool_limits(limits=1):
            return self._local_forecast_impl(
                df=df,
                h=h,
                freq=freq,
                level=level,
                quantiles=quantiles,
            )
