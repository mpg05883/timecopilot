import pandas as pd

from .models.utils.forecaster import Forecaster


class TimeCopilotForecaster:
    def __init__(self, models: list[Forecaster]):
        self.models = models

    def _call_models(
        self,
        attr: str,
        merge_on: list[str],
        df: pd.DataFrame,
        h: int,
        freq: str,
        level: list[int] | None = None,
        quantiles: list[float] | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        res_df: pd.DataFrame | None = None
        for model in self.models:
            res_df_model = getattr(model, attr)(
                df=df,
                h=h,
                freq=freq,
                level=level,
                quantiles=quantiles,
                **kwargs,
            )
            if res_df is None:
                res_df = res_df_model
            else:
                if "y" in res_df_model:
                    # drop y to avoid duplicate columns
                    # y was added by the previous condition
                    # to cross validation
                    # (the initial model)
                    res_df_model = res_df_model.drop(columns=["y"])
                res_df = res_df.merge(
                    res_df_model,
                    on=merge_on,
                )
        return res_df

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
        level: list[int] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        return self._call_models(
            "forecast",
            merge_on=["unique_id", "ds"],
            df=df,
            h=h,
            freq=freq,
            level=level,
            quantiles=quantiles,
        )

    def cross_validation(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
        n_windows: int = 1,
        step_size: int | None = None,
        level: list[int] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        return self._call_models(
            "cross_validation",
            merge_on=["unique_id", "ds", "cutoff"],
            df=df,
            h=h,
            freq=freq,
            n_windows=n_windows,
            step_size=step_size,
            level=level,
            quantiles=quantiles,
        )
