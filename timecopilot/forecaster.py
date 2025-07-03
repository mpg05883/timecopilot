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
        **kwargs,
    ) -> pd.DataFrame:
        res_df: pd.DataFrame | None = None
        for model in self.models:
            res_df_model = getattr(model, attr)(df=df, h=h, freq=freq, **kwargs)
            if res_df is None:
                res_df = res_df_model
            else:
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
    ) -> pd.DataFrame:
        return self._call_models(
            "forecast",
            merge_on=["unique_id", "ds"],
            df=df,
            h=h,
            freq=freq,
        )

    def cross_validation(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
        n_windows: int = 1,
        step_size: int | None = None,
    ) -> pd.DataFrame:
        return self._call_models(
            "cross_validation",
            merge_on=["unique_id", "ds", "cutoff"],
            df=df,
            h=h,
            freq=freq,
            n_windows=n_windows,
            step_size=step_size,
        )
