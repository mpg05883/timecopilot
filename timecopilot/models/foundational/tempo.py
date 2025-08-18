from contextlib import contextmanager
from functools import cached_property
from pathlib import Path

import torch
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import Chain
from omegaconf import DictConfig

from ..utils.gluonts_forecaster import GluonTSForecaster
from .src.models.lightning_tempo import LightningTEMPO
from .src.utils.gluonts import get_input_transform


class TEMPOForecaster(GluonTSForecaster):
    """
    Wrapper for using TEMPO models with TimeCopilot.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        batch_size: int = 256,
        num_samples: int = 100,
        alias: str = "TEMPO",
        verbose: bool = False,
    ):
        """
        Args:
            checkpoint_path (str | Path): Path to the TEMPO model checkpoint
                (.ckpt file). The checkpoint should contain the model's
                configuration.
            batch_size (int, optional): Batch size for inference. Defaults to
                256.
            num_samples (int, optional): Number of samples for probabilistic
                forecasting. Defaults to 100.
            alias (str, optional): Name for the model in output DataFrames.
                Defaults to "TEMPO".
            verbose (bool, optional): Whether to print loading messages when
                instantiating the predictor. Defaults to False.
        """
        super().__init__(
            repo_id="",  # Not used for local checkpoints
            filename="",  # Not used for local checkpoints
            alias=alias,
            num_samples=num_samples,
        )
        self.checkpoint_path = Path(checkpoint_path)
        self.batch_size = batch_size
        self.verbose = verbose
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @cached_property
    def lightning_module(self) -> LightningTEMPO:
        """Load the LightningTEMPO model from its checkpoint"""
        return LightningTEMPO.load_from_checkpoint(
            checkpoint_path=self.checkpoint_path,
            map_location=self.device,
        ).eval()

    @cached_property
    def config(self) -> DictConfig:
        """Get the model's configuration from the checkpoint's hparams"""
        return self.lightning_module.hparams.config

    def get_input_transform(self, prediction_length: int) -> Chain:
        """Create the input transform for the predictor"""
        return get_input_transform(
            context_length=self.config.context_length,
            prediction_length=prediction_length,
            use_time_features=self.config.use_time_features,
        )

    @contextmanager
    def get_predictor(self, prediction_length: int) -> PyTorchPredictor:
        """
        Create a GluonTS predictor from your LightningTEMPO model.
        """
        input_transform = self.get_input_transform(prediction_length)

        predictor = self.lightning_module.get_predictor(
            input_transform=input_transform,
            batch_size=self.batch_size,
            prediction_length=prediction_length,
            verbose=self.verbose,
        )

        try:
            yield predictor
        finally:
            del predictor
            torch.cuda.empty_cache()
