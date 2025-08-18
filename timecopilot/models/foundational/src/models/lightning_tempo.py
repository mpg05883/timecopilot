from typing import Literal

import pytorch_lightning as pl
import torch
from omegaconf.dictconfig import DictConfig
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR

from .tempo import TEMPO


class LightningTEMPO(TEMPO, pl.LightningModule):
    """
    A Lightning wrapper around TEMPO.
    """

    def __init__(
        self,
        config: DictConfig | None = None,
        **kwargs,
    ):
        """
        Initializes a LightningTEMPO instance.

        Args:
            config (DictConfig): Configuration object containing model settings
            and hyperparameters (e.g. loss function, learning rate, etc.).
        """
        # Initialize TEMPO
        super().__init__(config)

        # Save all arguments passed to __init__ as hyperparameters
        self.save_hyperparameters()
        # Restore config from hparams if loading checkpoint
        if config is None:
            config = self.hparams.config

        self.config = config
        self.lr = config.lr
        self.T_max = config.T_max
        self.eta_min = config.eta_min
        self.use_time_features = config.get("use_time_features", False)
        self.is_dec_loss = config.get("is_dec_loss", False)
        self.stl_weight = config.get("stl_weight", 1e-2)

        # Handle any other kwargs that might be passed
        for key, value in kwargs.items():
            setattr(self, key, value)

    # ========================= Public Methods =========================
    def training_step(
        self,
        batch: dict[str, Tensor],
        batch_index: int,
    ) -> float:
        """
        Performs a forward pass on TEMPO using the inputs from batch, computes
        the loss between the forecasted and ground truth values, and prints the
        decomposition loss and the forecast loss.

        Args:
            batch (Dict[str, Tensor]): A dictionary containing the following
            keys-value pairs:
                - "past_target" (Tensor): Ground truth values over the
                context window.
                - "future_target" (Tensor): Ground truth values over the
                prediction window.
                - "past_trend" (Tensor): Trend component over the context
                window.
                - "past_seasonal" (Tensor): Seasonal component over the
                context window.
                - "past_residual" (Tensor): Residual component over the
                context window.
            batch_index (int): Index of the current batch in the validation
            loop (not used).

        Returns:
            forecast_loss (float): The loss between the predicted and ground
            truth values over the prediction window.
        """
        # Get forecasts (or distribution arguments) and decomposition loss
        outputs, stl_loss = self._forward_pass(batch, False)

        # Extract ground truth values from batch
        future_target = batch["future_target"]

        # Compute forecast loss
        forecast_loss = self._compute_forecast_loss(outputs, future_target)

        total_loss = (
            forecast_loss.float() + self.stl_weight * stl_loss.float()
            if self.is_dec_loss
            else forecast_loss.float()
        )

        # Print decomposition loss and forecast loss
        self._print_loss(
            stl_loss=stl_loss,
            forecast_loss=forecast_loss,
            total_loss=total_loss,
            split="train",
            batch_size=future_target.shape[0],
        )

        return total_loss

    def validation_step(
        self,
        batch: dict[str, Tensor],
        batch_index: int,
    ) -> None:
        """
        Performs a forward pass on TEMPO using the inputs from batch, computes
        the loss between the forecasted and ground truth values, and prints the
        decomposition loss and the forecast loss.

        This method is exactly the same as training_step, except it doesn't
        return the forecast loss.

        Args:
            batch (Dict[str, Tensor]): A dictionary containing the
            following keys-value pairs:
                - "past_target" (Tensor): Ground truth values over the
                context window.
                - "future_target" (Tensor): Ground truth values over the
                prediction window.
                - "past_trend" (Tensor): Trend component over the context
                window.
                - "past_seasonal" (Tensor): Seasonal component over the
                context window.
                - "past_residual" (Tensor): Residual component over the
                context window.
            batch_index (int): Index of the current batch in the validation
            loop (not used in this method).
        """
        # Get forecasts (or distribution arguments) and decomposition loss
        outputs, stl_loss = self._forward_pass(batch, False)

        # Extract ground truth values from batch
        future_target = batch["future_target"]

        # future_target_norm, (loc, scale) = self.norm(future_target)

        # Compute forecast loss
        forecast_loss = self._compute_forecast_loss(outputs, future_target)

        total_loss = (
            forecast_loss.float() + self.stl_weight * stl_loss.float()
            if self.is_dec_loss
            else forecast_loss.float()
        )

        # Print decomposition loss and forecast loss
        self._print_loss(
            stl_loss=stl_loss,
            forecast_loss=forecast_loss,
            total_loss=total_loss,
            split="val",
            batch_size=future_target.shape[0],
        )

    def configure_optimizers(self) -> Optimizer:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
        )

        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.T_max,
            eta_min=self.eta_min,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
                "strict": True,
                "name": "cosine_annealing",
            },
        }

    # ========================= Private Methods =========================
    def _forward_pass(
        self,
        batch: dict[str, Tensor],
        test=True,
    ):
        """
        Extracts inputs from batch, performs a forward pass on TEMPO, and
        returns the model outputs. If Test is False, the decomposition loss is
        also returned.

        Args:
            batch (Dict[str, Tensor]): A dictionary containing the
            following keys-value pairs:
                - "past_target" (Tensor): Ground truth values over the
                context window.
                - "future_target" (Tensor): Ground truth values over the
                prediction window (not used here).
                - "past_trend" (Tensor): Trend component over the context
                window.
                - "past_seasonal" (Tensor): Seasonal component over the
                context window.
                - "past_residual" (Tensor): Residual component over the
                context window.
            test (bool): Set to True if the forward pass is being used on the
            test set and you DO NOT want to get the decomposition loss.

        Returns:
            Tuple:
                - outputs: If the model is deterministic, outputs simply
                contains the model's forecasts. Else, outputs contains a tuple
                of the forecasted distribution arguments, location, and scale.
                - stl_loss: The loss between the estimated STL
                components and the true STL components.
        """
        # Extract tensors from batch
        # import pdb, pdb; pdb.set_trace()
        past_target, past_trend, past_seasonal, past_residual = (
            batch["past_target"],
            batch["past_trend"],
            batch["past_seasonal"],
            batch["past_residual"],
        )
        past_time_feat = None
        future_time_feat = None
        if self.use_time_features:
            past_time_feat, future_time_feat = (
                batch["past_time_feat"],
                batch["future_time_feat"],
            )

        if test:
            # Only get forecasts (or distribution arguments)
            outputs = self(
                past_target=past_target,
                past_trend=past_trend,
                past_seasonal=past_seasonal,
                past_residual=past_residual,
                test=test,
                past_time_feat=past_time_feat,
                future_time_feat=future_time_feat,
            )

            return outputs
        else:
            # Get forecasts (or distribution arguments) and decomposition loss
            *outputs, stl_loss = self(
                past_target=past_target,
                past_trend=past_trend,
                past_seasonal=past_seasonal,
                past_residual=past_residual,
                test=test,
                past_time_feat=past_time_feat,
                future_time_feat=future_time_feat,
            )

            # If outputs is a tuple only containing one element, extract the
            # single element from the tuple
            # NOTE: This happens if test is False and the model's
            # deterministic because *outputs converts outputs to a tuple
            outputs = outputs[0] if len(outputs) == 1 else outputs

            return outputs, stl_loss

    def _compute_forecast_loss(self, outputs, targets, target_mask=None) -> float:
        """
        Computes the loss between the forecasted and ground truth values using
        the loss function specified in config if the model is deterministic.
        Else, the loss is computed using the output layer's distribution.

        Args:
            outputs: The forecasted time series values.
            targets: The ground truth time series values.

        Returns:
            float: The loss between predicted and true values.
        """
        if self.deterministic:
            return self.criterion(outputs, targets)
        elif self.probabilistic:
            distr_args, loc, scale = outputs

            # Create a distribution at each timestep of the prediction window
            distr = self.distr_output.distribution(distr_args, loc, scale)

            # Compute NLL loss
            nll_loss = -distr.log_prob(targets)
            loss = nll_loss.mean()
            # self.prob_quantile = False
            if self.is_prob_quant:
                # Generate quantile predictions
                num_quantiles = 9

                quantiles = torch.linspace(0.1, 0.9, 9, device=targets.device).view(
                    -1, 1, 1
                )  # (9,1,1)
                # [0.1, 0.2, ..., 0.9]
                # Add batch dimension to quantiles
                # quantiles = quantiles.reshape(1, -1, 1)  # shape: (1, num_quantiles, 1)
                quantile_preds = distr.icdf(quantiles).permute(
                    1, 0, 2
                )  # shape: (batch_size, num_quantiles, prediction_length)
                # Expand targets to match quantile predictions shape
                targets_expanded = targets.unsqueeze(
                    1
                )  # shape: (batch_size, 1, prediction_length)

                # Compute quantile loss
                quantile_loss = 2 * torch.abs(
                    (targets_expanded - quantile_preds)
                    * (
                        (targets_expanded <= quantile_preds).float()
                        - quantiles.view(1, num_quantiles, 1)
                    )
                )

                # Average over prediction horizon and sum over quantile levels
                quantile_loss = quantile_loss.mean(dim=-1).sum(dim=-1).mean()
                # Combine losses (you can adjust the weighting)
                # loss = nll_loss.mean() + 0.001*quantile_loss
                loss += 0.001 * quantile_loss

            return loss  # loss.mean()
        else:
            # import pdb; pdb.set_trace()
            (quantile_preds,), loc, scale = outputs
            loc_scale = (loc, scale)
            if self.norm_style == "instancenorm":
                target, _ = self.norm(targets)  # , loc_scale
            else:
                targets_scale = self.scaling(targets)
                target = targets / targets_scale
            target = target.unsqueeze(1)  # type: ignore

            assert self.prediction_length >= target.shape[-1]

            target = target.to(quantile_preds.device)
            target_mask = (
                target_mask.unsqueeze(1).to(quantile_preds.device)
                if target_mask is not None
                else ~torch.isnan(target)
            )
            target[~target_mask] = 0.0

            # pad target and target_mask if they are shorter than model's prediction_length
            if self.prediction_length > target.shape[-1]:
                padding_shape = (
                    *target.shape[:-1],
                    self.prediction_length - target.shape[-1],
                )
                target = torch.cat(
                    [target, torch.zeros(padding_shape).to(target)], dim=-1
                )
                target_mask = torch.cat(
                    [target_mask, torch.zeros(padding_shape).to(target_mask)], dim=-1
                )

            quantile_preds_shape = (quantile_preds.shape[0], self.num_quantiles, -1)
            # import pdb; pdb.set_trace()
            quantile_preds = quantile_preds.view(*quantile_preds_shape)
            loss = (
                2
                * torch.abs(
                    (target - quantile_preds)
                    * (
                        (target <= quantile_preds).float()
                        - self.quantiles.to(quantile_preds.device).view(
                            1, self.num_quantiles, 1
                        )  # type: ignore
                    )
                )
                * target_mask.float()
            )
            loss = loss.mean(dim=-2)  # Mean over prediction horizon
            loss = loss.sum(dim=-1)  # Sum over quantile levels
            loss = loss.mean()  # Mean over batch

            # Unscale predictions
            if self.norm_style == "instancenorm":
                quantile_preds = self.norm.inverse(
                    quantile_preds.view(quantile_preds.shape[0], -1),
                    loc_scale,
                ).view(*quantile_preds_shape)
            else:
                quantile_preds = (
                    quantile_preds.view(quantile_preds.shape[0], -1) * targets_scale
                )
                quantile_preds = quantile_preds.view(*quantile_preds_shape)

            return loss

    def _print_loss(
        self,
        batch_size: int,
        stl_loss: float = None,
        forecast_loss: float = None,
        total_loss: float = None,
        split: Literal["train", "val", "test"] = "train",
    ) -> None:
        """
        Prints the decomposition loss and the forecast loss, if they're given.

        Args:
            batch_size (int): The number of samples in the current batch.
            stl_loss (float): The loss between the estimated STL
            components and the true STL components.
            forecast_loss (float): The loss between the predicted and ground
            truth values over the prediction window.
            split (str, optional): Name of the split whose loss is being
            printed.
        """
        if stl_loss:
            self.log(
                f"{split}_stl_loss",
                stl_loss,
                prog_bar=True,
                on_step=True if split == "train" else False,
                on_epoch=True,
                batch_size=batch_size,
            )

        if forecast_loss:
            self.log(
                f"{split}_{self.loss_name}_loss",
                forecast_loss,
                prog_bar=True,
                on_step=True if split == "train" else False,
                on_epoch=True,
                batch_size=batch_size,
            )

        if total_loss:
            self.log(
                f"{split}_loss",
                total_loss,
                prog_bar=True,
                on_step=True if split == "train" else False,
                on_epoch=True,
                batch_size=batch_size,
            )
