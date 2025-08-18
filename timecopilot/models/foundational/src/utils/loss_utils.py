import torch.nn as nn
from gluonts.torch.distributions import NegativeBinomialOutput, StudentTOutput
from gluonts.torch.distributions.distribution_output import DistributionOutput
from torch.nn import Module


def is_deterministic(loss_name: str) -> bool:
    """
    Returns True if the loss function is deterministic, False
    otherwise.

    Returns:
        bool: True if the loss function is deterministic.
    """
    return loss_name in {"mse", "mae", "hubber"}


def is_quantile(loss_name: str) -> bool:
    """
    Returns True if the loss function is quantile_loss, False
    otherwise.

    Returns:
        bool: True if the loss function is quantile.
    """
    return loss_name in {"quantile"}


def get_criterion(loss_name: str) -> Module | None:
    """
    Returns the loss function corresponding to `loss_name`.

    Supported loss names:
        - "mse": Returns `nn.MSELoss()`
        - "mae": Returns `nn.L1Loss()`
        - "huber": Returns `nn.HuberLoss()`
        - Else: Returns None

    Returns:
        Module: An instance of the selected loss function.
    """
    if loss_name == "mse":
        return nn.MSELoss()
    elif loss_name == "mae":
        return nn.L1Loss()
    elif loss_name == "huber":
        return nn.HuberLoss()


def get_distr_output(loss_name: str) -> DistributionOutput:
    """
    Returns the distribution output corresponding to `loss_name`.

    Returns:
        DistributionOutput: _description_
    """
    if loss_name == "student_t":
        return StudentTOutput()
    else:
        return NegativeBinomialOutput()
