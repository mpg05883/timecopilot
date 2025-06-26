import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import _zero_to_nan, mae

from ..models.utils.forecaster import get_seasonality

warnings.simplefilter(
    action="ignore",
    category=FutureWarning,
)


def mase(
    df: pd.DataFrame,
    models: list[str],
    seasonality: int,
    train_df: pd.DataFrame,
    id_col: str = "unique_id",
    target_col: str = "y",
) -> pd.DataFrame:
    mean_abs_err = mae(df, models, id_col, target_col)
    mean_abs_err = mean_abs_err.set_index(id_col)
    # assume train_df is sorted
    lagged = train_df.groupby(id_col, observed=True)[target_col].shift(seasonality)
    scale = train_df[target_col].sub(lagged).abs()
    scale = scale.groupby(train_df[id_col], observed=True).mean()
    scale[scale < 1e-2] = 0.0
    res = mean_abs_err.div(_zero_to_nan(scale), axis=0).fillna(0)
    res.index.name = id_col
    res = res.reset_index()
    return res


def generate_train_cv_splits(
    df: pd.DataFrame,
    cutoffs: pd.DataFrame,
) -> pd.DataFrame:
    """
    based on `cutoffs` (columns `unique_id`, `cutoffs`)
    generates train cv splits using `df`
    """
    df = df.merge(cutoffs, on="unique_id", how="outer")
    df = df.query("ds <= cutoff")
    df = df.reset_index(drop=True)
    return df


@dataclass
class DatasetParams:
    pandas_frequency: str
    horizon: int
    seasonality: int

    @staticmethod
    def _get_value_from_df_col(
        df: pd.DataFrame,
        col: str,
        dtype: Callable | None = None,
    ) -> Any:
        col_values = df[col].unique()
        if len(col_values) > 1:
            raise ValueError(f"{col} is not unique: {col_values}")
        value = col_values[0]
        if dtype is not None:
            value = dtype(value)
        return value

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> "DatasetParams":
        if "unique_id" not in df.columns:
            df["unique_id"] = "uid_0"
        dataset_params = {}
        dataset_params_cols = [
            "pandas_frequency",
            "seasonality",
            "horizon",
        ]
        dataset_params_cols_dtypes = [str, int, int]
        for col, dtype in zip(
            dataset_params_cols,
            dataset_params_cols_dtypes,
            strict=False,
        ):
            if col == "pandas_frequency" and col not in df.columns:
                dataset_params[col] = pd.infer_freq(df["ds"])
            elif col == "seasonality" and col not in df.columns:
                dataset_params[col] = get_seasonality(
                    dataset_params["pandas_frequency"]
                )
            elif col == "horizon" and col not in df.columns:
                dataset_params[col] = 2 * dataset_params["seasonality"]
            else:
                dataset_params[col] = cls._get_value_from_df_col(df, col, dtype=dtype)
        return cls(**dataset_params)


@dataclass
class ExperimentDataset(DatasetParams):
    df: pd.DataFrame

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> "ExperimentDataset":
        """
        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame must contain the following columns:
            - unique_id: A unique identifier for each time series (str).
                If not present, it will be set to "uid_0".
            - ds: The datetime column representing the time index (datetime).
            - y: The target variable to forecast (float).
            - pandas_frequency: The frequency of the data, e.g.,
                'D' for daily, 'M' for monthly (str).
                If not present, it will be inferred from the data.
            - horizon: The number of periods to forecast (int).
                If not present, it will be set to 2 * seasonality.
            - seasonality: The seasonal period of the data,
                typically inferred from the frequency (int).
                If not present, it will be inferred from the frequency.
        """
        ds_params = DatasetParams.from_df(df=df)
        df = df[["unique_id", "ds", "y"]]  # type: ignore
        return cls(
            df=df,
            **asdict(ds_params),
        )

    @classmethod
    def from_parquet(
        cls,
        parquet_path: str | Path,
    ) -> "ExperimentDataset":
        df = pd.read_parquet(parquet_path)
        return cls.from_df(df=df)

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
    ) -> "ExperimentDataset":
        df = pd.read_csv(csv_path)
        return cls.from_df(df=df)

    def evaluate_forecast_df(
        self,
        forecast_df: pd.DataFrame,
        models: list[str],
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        forecast_df : pd.DataFrame
            df should have columns: unique_id, ds, cutoff, y, and models
        """
        for model in models:
            if forecast_df[model].isna().sum() > 0:
                print(forecast_df.loc[forecast_df[model].isna()]["unique_id"].unique())
                raise ValueError(f"model {model} has NaN values")
        cutoffs = forecast_df[["unique_id", "cutoff"]].drop_duplicates()
        train_cv_splits = generate_train_cv_splits(df=self.df, cutoffs=cutoffs)

        def add_id_cutoff(df: pd.DataFrame):
            df["id_cutoff"] = (
                df["unique_id"].astype(str) + "-" + df["cutoff"].astype(str)
            )

        for df in [cutoffs, train_cv_splits, forecast_df]:
            add_id_cutoff(df)
        partial_mase = partial(mase, seasonality=self.seasonality)
        eval_df = evaluate(
            df=forecast_df,
            train_df=train_cv_splits,
            metrics=[partial_mase],
            models=models,
            id_col="id_cutoff",
        )
        eval_df = eval_df.merge(cutoffs, on=["id_cutoff"])
        eval_df = eval_df.drop(columns=["id_cutoff"])
        eval_df = eval_df[["unique_id", "cutoff", "metric"] + models]
        return eval_df


@dataclass
class ForecastDataset:
    forecast_df: pd.DataFrame
    time_df: pd.DataFrame

    @classmethod
    def from_dir(cls, dir: str | Path):
        dir_ = Path(dir)
        forecast_df = pd.read_parquet(dir_ / "forecast_df.parquet")
        time_df = pd.read_parquet(dir_ / "time_df.parquet")
        return cls(forecast_df=forecast_df, time_df=time_df)

    @staticmethod
    def is_forecast_ready(dir: str | Path):
        dir_ = Path(dir)
        forecast_path = dir_ / "forecast_df.parquet"
        time_path = dir_ / "time_df.parquet"
        return forecast_path.exists() and time_path.exists()

    def save_to_dir(self, dir: str | Path):
        dir_ = Path(dir)
        dir_.mkdir(parents=True, exist_ok=True)
        self.forecast_df.to_parquet(dir_ / "forecast_df.parquet")
        self.time_df.to_parquet(dir_ / "time_df.parquet")
