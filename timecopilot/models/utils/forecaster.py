import pandas as pd
import utilsforecast.processing as ufp
from gluonts.time_feature.seasonality import (
    DEFAULT_SEASONALITIES,
)
from gluonts.time_feature.seasonality import (
    get_seasonality as _get_seasonality,
)
from tqdm import tqdm
from utilsforecast.processing import (
    backtest_splits,
    drop_index_if_pandas,
    join,
    maybe_compute_sort_indices,
    take_rows,
    vertical_concat,
)


def get_seasonality(freq: str) -> int:
    return _get_seasonality(freq, seasonalities=DEFAULT_SEASONALITIES | {"D": 7})


def maybe_convert_col_to_datetime(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    if not pd.api.types.is_datetime64_any_dtype(df[col_name]):
        df = df.copy()
        df[col_name] = pd.to_datetime(df[col_name])
    return df


class Forecaster:
    alias: str

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
    ) -> pd.DataFrame:
        raise NotImplementedError

    def cross_validation(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
        n_windows: int = 1,
        step_size: int | None = None,
    ) -> pd.DataFrame:
        df = maybe_convert_col_to_datetime(df, "ds")
        # mlforecast cv code
        results = []
        sort_idxs = maybe_compute_sort_indices(df, "unique_id", "ds")
        if sort_idxs is not None:
            df = take_rows(df, sort_idxs)
        splits = backtest_splits(
            df,
            n_windows=n_windows,
            h=h,
            id_col="unique_id",
            time_col="ds",
            freq=pd.tseries.frequencies.to_offset(freq),
            step_size=h if step_size is None else step_size,
        )
        for _, (cutoffs, train, valid) in tqdm(enumerate(splits)):
            if len(valid.columns) > 3:
                raise NotImplementedError(
                    "Cross validation with exogenous variables is not yet supported."
                )
            y_pred = self.forecast(
                df=train,
                h=h,
                freq=freq,
            )
            y_pred = join(y_pred, cutoffs, on="unique_id", how="left")
            result = join(
                valid[["unique_id", "ds", "y"]],
                y_pred,
                on=["unique_id", "ds"],
            )
            if result.shape[0] < valid.shape[0]:
                raise ValueError(
                    "Cross validation result produced less results than expected. "
                    "Please verify that the frequency parameter (freq) "
                    "matches your series' "
                    "and that there aren't any missing periods."
                )
            results.append(result)
        out = vertical_concat(results)
        out = drop_index_if_pandas(out)
        first_out_cols = ["unique_id", "ds", "cutoff", "y"]
        remaining_cols = [c for c in out.columns if c not in first_out_cols]
        fcst_cv_df = out[first_out_cols + remaining_cols]
        return fcst_cv_df


class QuantileConverter:
    """Handles inputs and outputs for probabilistic forecasts."""

    def __init__(
        self,
        level: list[int | float] | None = None,
        quantiles: list[int | float] | None = None,
    ):
        level, quantiles = self._prepare_level_and_quantiles(level, quantiles)
        self.level = level
        self.quantiles = quantiles

    @staticmethod
    def _prepare_level_and_quantiles(
        level: list[int | float] | None,
        quantiles: list[float] | None,
    ) -> tuple[list[int | float] | None, list[float] | None]:
        # based on https://github.com/Nixtla/nixtla/blob/e74d98d9346a055153f84801cac94715c2342946/nixtla/nixtla_client.py#L444
        if level is not None and quantiles is not None:
            raise ValueError("You should provide `level` or `quantiles`, but not both.")
        if quantiles is None:
            return level, quantiles
        # we recover level from quantiles
        if not all(0 < q < 1 for q in quantiles):
            raise ValueError("`quantiles` should be floats between 0 and 1.")
        level = [abs(int(100 - 200 * q)) for q in quantiles]
        return level, quantiles

    def maybe_convert_level_to_quantiles(
        self,
        df: pd.DataFrame,
        models: list[str],
    ) -> pd.DataFrame:
        if self.quantiles is None:
            return df
        out_cols = [c for c in df.columns if "-lo-" not in c and "-hi-" not in c]
        df = ufp.copy_if_pandas(df, deep=False)
        for model in models:
            for q in sorted(self.quantiles):
                if q == 0.5:
                    col = model
                else:
                    lv = int(100 - 200 * q)
                    hi_or_lo = "lo" if lv > 0 else "hi"
                    lv = abs(lv)
                    col = f"{model}-{hi_or_lo}-{lv}"
                q_col = f"{model}-q-{int(q * 100)}"
                df = ufp.assign_columns(df, q_col, df[col])
                out_cols.append(q_col)
        return df[out_cols]
