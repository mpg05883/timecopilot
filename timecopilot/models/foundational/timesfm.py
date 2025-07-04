import pandas as pd
import timesfm
import torch

from ..utils.forecaster import Forecaster


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
    ) -> timesfm.TimesFm:
        backend = "gpu" if torch.cuda.is_available() else "cpu"
        tfm_hparams = timesfm.TimesFmHparams(
            backend=backend,
            horizon_len=prediction_length,
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
        if level is not None and quantiles is not None:
            raise NotImplementedError(
                "Level and quantiles are not supported for TimesFM yet"
            )
        predictor = self.get_predictor(prediction_length=h)
        fcst_df = predictor.forecast_on_df(
            inputs=df,
            freq=freq,
            value_name="y",
            model_name=self.alias,
            num_jobs=1,
        )
        fcst_df = fcst_df[["unique_id", "ds", self.alias]]
        return fcst_df
