from typing import Any

import numpy as np
import pandas as pd
import tqdm
import utilsforecast.processing as ufp
from gluonts.dataset import Dataset
from gluonts.dataset.util import forecast_start
from gluonts.model import Forecast
from gluonts.model.forecast import QuantileForecast
from gluonts.model.predictor import RepresentablePredictor
from gluonts.transform.feature import LastValueImputation, MissingValueImputation

from timecopilot.models.utils.forecaster import Forecaster


class GluonTSPredictor(RepresentablePredictor):
    """
    Adapter to use a TimeCopilot Forecaster as a GluonTS Predictor.

    This class wraps a TimeCopilot Forecaster and exposes the GluonTS Predictor
    interface, allowing it to be used with GluonTS evaluation and processing utilities.
    """

    def __init__(
        self,
        forecaster: Forecaster,
        h: int,
        freq: str,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
        max_length: int | None = None,
        imputation_method: MissingValueImputation | None = None,
        batch_size: int | None = 1024,
    ):
        """
        Initialize a GluonTSPredictor.

        Args:
            forecaster (Forecaster): The TimeCopilot forecaster to wrap.
            h (int): Forecast horizon.
            freq (str): Frequency string (e.g., 'D', 'H').
            level (list[int | float] | None): Not supported; use quantiles instead.
            quantiles (list[float] | None): Quantiles to forecast, if any.
            max_length (int | None): Maximum length of input series.
            imputation_method (MissingValueImputation | None): Imputation method for
                missing values.
            batch_size (int | None): Batch size for prediction.

        Raises:
            NotImplementedError: If level is provided (use quantiles instead).
        """
        self.forecaster = forecaster
        self.h = h
        self.freq = freq
        self.level = level
        self.max_length = max_length
        if level is not None:
            raise NotImplementedError("level is not supported, use quantiles instead")
        self.quantiles = quantiles
        self.imputation_method = imputation_method or LastValueImputation()
        self.batch_size = batch_size
        self.alias = forecaster.alias

    def _gluonts_dataset_to_df(
        self,
        dataset: Dataset,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        dfs: list[pd.DataFrame] = []
        metadata: dict[str, Any] = {}
        for _, entry in enumerate(dataset):
            target = np.asarray(entry["target"], dtype=np.float32)
            if self.max_length is not None and len(target) > self.max_length:
                entry["start"] += len(target[: -self.max_length])
                target = target[-self.max_length :]
            if np.isnan(target).any():
                target = self.imputation_method(target)
            if target.ndim > 1:
                raise ValueError("only for univariate time series")
            fcst_start = forecast_start(entry)
            uid = f"{entry['item_id']}-{fcst_start}"
            ds_start = entry["start"]
            ds = pd.date_range(
                start=ds_start.to_timestamp(),
                freq=ds_start.freq,
                periods=len(target),
            )
            uid_df = pd.DataFrame(
                {
                    "unique_id": uid,
                    "ds": ds,
                    "y": target,
                }
            )
            dfs.append(uid_df)
            metadata[uid] = {
                "item_id": entry["item_id"],
                "fcst_start": fcst_start,
            }
        df = pd.concat(dfs, ignore_index=True)
        return df, metadata

    def _predict_df(self, df: pd.DataFrame, metadata: dict[str, Any]) -> list[Forecast]:
        fcst_df = self.forecaster.forecast(
            df=df,
            h=self.h,
            freq=self.freq,
            level=self.level,
            quantiles=self.quantiles,
        )
        fcst_df = fcst_df.set_index("unique_id")
        fcsts: list[Forecast] = []
        for uid, metadata_uid in metadata.items():
            fcst_df_uid = fcst_df.loc[uid]
            forecast_arrays = ufp.value_cols_to_numpy(
                df=fcst_df_uid,
                id_col="unique_id",
                time_col="ds",
                target_col=None,
            )
            forecast_keys = ["mean"]
            if self.quantiles is not None:
                forecast_keys += [f"{q}" for q in self.quantiles]
            q_fcst = QuantileForecast(
                forecast_arrays=forecast_arrays.T,
                forecast_keys=forecast_keys,
                item_id=metadata_uid["item_id"],
                start_date=metadata_uid["fcst_start"],
            )
            fcsts.append(q_fcst)
        return fcsts

    def _predict_batch(self, batch: list[Dataset]) -> list[Forecast]:
        df, metadata = self._gluonts_dataset_to_df(batch)
        return self._predict_df(df, metadata)

    def predict(self, dataset: Dataset, **kwargs: Any) -> list[Forecast]:
        """
        Predict forecasts for a GluonTS Dataset.

        Args:
            dataset (Dataset): GluonTS Dataset to forecast.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            list[Forecast]: List of GluonTS Forecast objects for the dataset.
        """
        fcsts: list[Forecast] = []
        batch: list[Dataset] = []
        for _, entry in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
            batch.append(entry)
            if len(batch) == self.batch_size:
                fcsts.extend(self._predict_batch(batch))
                batch = []
        if len(batch) > 0:
            fcsts.extend(self._predict_batch(batch))
        return fcsts
