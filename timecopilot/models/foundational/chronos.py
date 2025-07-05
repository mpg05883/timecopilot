from collections.abc import Iterable

import pandas as pd
import torch
from chronos import BaseChronosPipeline
from tqdm import tqdm
from utilsforecast.processing import make_future_dataframe

from ..utils.forecaster import Forecaster, QuantileConverter


class TimeSeriesDataset:
    def __init__(
        self,
        data: torch.Tensor,
        uids: Iterable,
        last_times: Iterable,
        batch_size: int,
    ):
        self.data = data
        self.uids = uids
        self.last_times = last_times
        self.batch_size = batch_size
        self.n_batches = len(data) // self.batch_size + (
            0 if len(data) % self.batch_size == 0 else 1
        )
        self.current_batch = 0

    @classmethod
    def from_df(cls, df: pd.DataFrame, batch_size: int):
        tensors = []
        df_sorted = df.sort_values(by=["unique_id", "ds"])
        for _, group in df_sorted.groupby("unique_id"):
            tensors.append(torch.tensor(group["y"].values, dtype=torch.bfloat16))
        uids = df_sorted["unique_id"].unique()
        last_times = df_sorted.groupby("unique_id")["ds"].tail(1)
        return cls(tensors, uids, last_times, batch_size)

    def __len__(self):
        return self.n_batches

    def make_future_dataframe(self, h: int, freq: str) -> pd.DataFrame:
        return make_future_dataframe(
            uids=self.uids,
            last_times=pd.to_datetime(self.last_times),
            h=h,
            freq=freq,
        )  # type: ignore

    def __iter__(self):
        self.current_batch = 0  # Reset for new iteration
        return self

    def __next__(self):
        if self.current_batch < self.n_batches:
            start_idx = self.current_batch * self.batch_size
            end_idx = start_idx + self.batch_size
            self.current_batch += 1
            return self.data[start_idx:end_idx]
        else:
            raise StopIteration


class Chronos(Forecaster):
    def __init__(
        self,
        repo_id: str = "amazon/chronos-t5-large",
        batch_size: int = 16,
        alias: str = "Chronos",
    ):
        self.repo_id = repo_id
        self.batch_size = batch_size
        self.alias = alias
        self.model = BaseChronosPipeline.from_pretrained(
            repo_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

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
        fcsts = [
            self.model.predict_quantiles(
                batch,
                prediction_length=h,
                quantile_levels=qc.quantiles,
            )
            for batch in tqdm(dataset)
        ]  # list of tuples
        fcsts_quantiles, fcsts_mean = zip(*fcsts, strict=False)
        fcsts_quantiles_np = torch.cat(fcsts_quantiles).numpy()
        fcsts_mean_np = torch.cat(fcsts_mean).numpy()
        fcst_df = dataset.make_future_dataframe(h=h, freq=freq)
        fcst_df[self.alias] = fcsts_mean_np.reshape(-1, 1)
        if qc.quantiles is not None:
            for i, q in enumerate(qc.quantiles):
                fcst_df[f"{self.alias}-q-{int(q * 100)}"] = fcsts_quantiles_np[
                    ..., i
                ].reshape(-1, 1)
            fcst_df = qc.maybe_convert_quantiles_to_level(
                fcst_df,
                models=[self.alias],
            )
        return fcst_df
