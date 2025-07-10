import os
import sys

if sys.version_info < (3, 11):
    raise ImportError("TiRex requires Python >= 3.11")

import numpy as np
import pandas as pd
import torch
from tirex import load_model
from tqdm import tqdm

from ..utils.forecaster import Forecaster, QuantileConverter
from .utils import TimeSeriesDataset


class TiRex(Forecaster):
    def __init__(
        self,
        repo_id: str = "NX-AI/TiRex",
        batch_size: int = 16,
        alias: str = "TiRex",
    ):
        self.repo_id = repo_id
        self.batch_size = batch_size
        self.alias = alias
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            # see https://github.com/NX-AI/tirex/tree/main?tab=readme-ov-file#cuda-kernels
            os.environ["TIREX_NO_CUDA"] = "1"
        self.model = load_model(repo_id, device=device)

    def _forecast(
        self,
        dataset: TimeSeriesDataset,
        h: int,
        quantiles: list[float] | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """handles distinction between quantiles and no quantiles"""
        if quantiles is not None:
            fcsts = [
                self.model.forecast(
                    batch,
                    prediction_length=h,
                    quantile_levels=quantiles,
                    output_type="numpy",
                )
                for batch in tqdm(dataset)
            ]  # list of tuples
            fcsts_quantiles, fcsts_mean = zip(*fcsts, strict=False)
            fcsts_quantiles_np = np.concatenate(fcsts_quantiles)
            fcsts_mean_np = np.concatenate(fcsts_mean)
        else:
            fcsts = [
                self.model.forecast(
                    batch,
                    prediction_length=h,
                    output_type="numpy",
                )
                for batch in tqdm(dataset)
            ]
            _, fcsts_mean = zip(*fcsts, strict=False)
            fcsts_mean_np = np.concatenate(fcsts_mean)
            fcsts_quantiles_np = None
        return fcsts_mean_np, fcsts_quantiles_np

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        qc = QuantileConverter(level=level, quantiles=quantiles)
        dataset = TimeSeriesDataset.from_df(df, batch_size=self.batch_size)
        fcst_df = dataset.make_future_dataframe(h=h, freq=freq)
        fcsts_mean_np, fcsts_quantiles_np = self._forecast(
            dataset,
            h,
            quantiles=qc.quantiles,
        )
        fcst_df[self.alias] = fcsts_mean_np.reshape(-1, 1)
        if qc.quantiles is not None and fcsts_quantiles_np is not None:
            for i, q in enumerate(qc.quantiles):
                fcst_df[f"{self.alias}-q-{int(q * 100)}"] = fcsts_quantiles_np[
                    ..., i
                ].reshape(-1, 1)
            fcst_df = qc.maybe_convert_quantiles_to_level(
                fcst_df,
                models=[self.alias],
            )
        return fcst_df
