import numpy as np
import pandas as pd
import torch
from toto.data.util.dataset import MaskedTimeseries
from toto.inference.forecaster import TotoForecaster
from toto.model.toto import Toto as TotoModel
from tqdm import tqdm

from ..utils.forecaster import Forecaster, QuantileConverter
from .utils import TimeSeriesDataset


class Toto(Forecaster):
    def __init__(
        self,
        repo_id: str = "Datadog/Toto-Open-Base-1.0",
        context_length: int = 4096,
        batch_size: int = 16,
        num_samples: int = 256,
        samples_per_batch: int = 256,
        alias: str = "Toto",
    ):
        self.repo_id = repo_id
        self.context_length = context_length
        self.batch_size = batch_size
        self.num_samples = (
            num_samples  # Number of samples for probabilistic forecasting
        )
        self.samples_per_batch = (
            samples_per_batch  # Control memory usage during inference
        )
        self.alias = alias
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = TotoModel.from_pretrained(self.repo_id).to(self.device)
        self.model = TotoForecaster(model.model)

    def _to_masked_timeseries(self, batch: list[torch.Tensor]) -> MaskedTimeseries:
        batch_size = len(batch)
        # using toch.float as stated in the docs
        # https://github.com/DataDog/toto/blob/main/toto/notebooks/inference_tutorial.ipynb
        padded_tensor = torch.zeros(
            batch_size,
            self.context_length,
            dtype=torch.float,
            device=self.device,
        )
        padding_mask = torch.zeros(
            batch_size,
            self.context_length,
            dtype=torch.float,
            device=self.device,
        )
        for idx, ts in enumerate(batch):
            series_length = len(ts)
            if series_length > self.context_length:
                ts = ts[-self.context_length :]
            padded_tensor[idx, -series_length:] = ts
            padding_mask[idx, -series_length:] = 1.0
        masked_ts = MaskedTimeseries(
            series=padded_tensor,
            padding_mask=padding_mask,
            id_mask=torch.zeros_like(padded_tensor),
            # Prepare timestamp information (optional, but expected by API;
            # not used by the current model release)
            timestamp_seconds=torch.zeros_like(padded_tensor),
            time_interval_seconds=torch.full(
                (batch_size,),
                1,
                device=self.device,
            ),
        )
        return masked_ts

    def _forecast(
        self,
        dataset: TimeSeriesDataset,
        h: int,
        quantiles: list[float] | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """handles distinction between quantiles and no quantiles"""
        fcsts = [
            self.model.forecast(
                self._to_masked_timeseries(batch),
                prediction_length=h,
                num_samples=self.num_samples,
                samples_per_batch=self.samples_per_batch,
                use_kv_cache=True,
            )
            for batch in tqdm(dataset)
        ]  # list of fcsts objects
        fcsts_mean = [fcst.mean.cpu().numpy() for fcst in fcsts]
        fcsts_mean_np = np.concatenate(fcsts_mean)
        if fcsts_mean_np.shape[0] != 1:
            raise ValueError(
                f"fcsts_mean_np.shape[0] != 1: {fcsts_mean_np.shape[0]} != 1, "
                "this is not expected, please open an issue on github"
            )
        fcsts_mean_np = fcsts_mean_np.squeeze(axis=0)
        if quantiles is not None:
            quantiles_torch = torch.tensor(
                quantiles,
                device=self.device,
                dtype=torch.float,
            )
            fcsts_quantiles = [
                fcst.quantile(quantiles_torch).cpu().numpy() for fcst in fcsts
            ]
            fcsts_quantiles_np = np.concatenate(fcsts_quantiles)
            if fcsts_quantiles_np.shape[1] != 1:
                raise ValueError(
                    "fcsts_quantiles_np.shape[1] != 1: "
                    f"{fcsts_quantiles_np.shape[1]} != 1, "
                    "this is not expected, please open an issue on github"
                )
            fcsts_quantiles_np = np.moveaxis(fcsts_quantiles_np, 0, -1).squeeze(axis=0)
        else:
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
