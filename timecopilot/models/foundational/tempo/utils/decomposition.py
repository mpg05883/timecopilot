import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module


def ensure_2d(x: Tensor = None) -> Tensor | None:
    """
    Ensures that the input tensor has exactly2 dimensions. If it has more
    dimensions, it squeezes out the extra dimension.

    Args:
        x (Tensor): The input tensor to be checked.

    Returns:
        Union[Tensor, None]: If x is None, then returns None, Else. returns the
        x, possibly squeezed to 2 dimensions.
    """
    if x is None:
        return None
    return torch.squeeze(x, -1) if x.dim() == 3 else x


def ensure_3d(x: Tensor = None) -> Tensor | None:
    """
    Ensures that the input tensor has at least 3 dimensions. If it has fewer
    dimensions, it expands the tensor by adding extra dimensions.

    Args:
        x (Tensor): The input tensor to be checked.

    Returns:
        Union[Tensor, None]: If x is None, then returns None, Else. returns the
        x, possibly expanded to 3 dimensions.
    """
    if x is None:
        return None
    return torch.unsqueeze(x, -1) if x.dim() < 3 else x


def have_all(trend: Tensor, seasonal: Tensor, residual: Tensor) -> bool:
    """
    Returns True if all three STL component (trend, seasonal, residual) are not
    None.

    Args:
        trend (Tensor): The trend component.
        seasonal (Tensor): The seasonal component.
        residual (Tensor): The residual component.

    Returns:
        bool: True if all components are not None, False otherwise.
    """
    return trend is not None and seasonal is not None and residual is not None


def compute_decomposition_loss(
    true_stl_components: tuple[Tensor, Tensor, Tensor],
    pred_stl_components: tuple[Tensor, Tensor, Tensor],
    scale: float,
    criterion: Module = nn.MSELoss(),
) -> float:
    """
    Computes the MSE between the ground truth STL components and the estimated
    STL components

    Args:
        true_stl_components (Tuple[Tensor, Tensor, Tensor]):
            A tuple containing the true trend, seasonal, and residual
            components.
        local_stl_components (Tuple[Tensor, Tensor, Tensor]):
            A tuple containing the estimated trend, seasonal, and residual
            components obtained from the STL decomposition.
    Returns:
        float: The MSE between the true and predicted STL components.
    """
    true_trend, true_seasonal, true_residual = true_stl_components
    pred_trend, pred_seasonal, pred_residual = pred_stl_components

    true_trend = ensure_2d(true_trend)
    true_seasonal = ensure_2d(true_seasonal)
    true_residual = ensure_2d(true_residual)

    pred_trend = ensure_2d(pred_trend)
    pred_seasonal = ensure_2d(pred_seasonal)
    pred_residual = ensure_2d(pred_residual)

    # Scale true STL components - We don't have to do this because we already normalized the data
    # ? Should we use the scaling factor instead of z-score normalization?
    # true_trend = z_score_normalize(true_trend)
    # true_seasonal = z_score_normalize(true_seasonal)
    # true_residual = z_score_normalize(true_residual)

    # Compute loss between each component
    trend_loss = criterion(true_trend.float(), pred_trend.float())
    seasonal_loss = criterion(true_seasonal.float(), pred_seasonal.float())
    residual_loss = criterion(true_residual.float(), pred_residual.float())

    decomposition_loss = trend_loss + seasonal_loss + residual_loss

    return decomposition_loss
