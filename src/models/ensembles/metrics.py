import warnings

import numpy as np

from src.data.utils import QUANTILE_LEVELS

def mse_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def mae_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def smape_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    numerator = 2 * np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    return float(np.mean(numerator / denominator))


def simple_mase_fn(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray | None = None,
    seasonality: int = 1,
) -> float:
    return _mase_fn(y_true, y_pred, y_train, seasonality)


def _mase_fn(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray | None = None,
    seasonality: int = 1,
) -> float:
    """
    Compute Mean Absolute Scaled Error (MASE).

    MASE = MAE / MAE_naive_seasonal

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        y_train: Training data for computing seasonal naive baseline (if None,
        uses simple naive)
        seasonality: Seasonal period (1 for non-seasonal, 12 for monthly, etc.)

    Returns:
        float: MASE value
    """
    mae_value = mae_fn(y_true, y_pred)

    if y_train is not None and len(y_train) > seasonality:
        # Use seasonal naive baseline from training data
        if seasonality == 1:
            # Simple naive: use last training value
            naive_errors = np.abs(y_train[1:] - y_train[:-1])
        else:
            # Seasonal naive: use values from seasonality periods ago
            naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
        seasonal_naive_mae_value = np.mean(naive_errors)
    else:
        # Fallback: use simple differences in the actual data
        if len(y_true) > 1:
            if seasonality == 1 or len(y_true) <= seasonality:
                naive_errors = np.abs(y_true[1:] - y_true[:-1])
            else:
                naive_errors = np.abs(y_true[seasonality:] - y_true[:-seasonality])
            seasonal_naive_mae_value = np.mean(naive_errors)
        else:
            seasonal_naive_mae_value = 1.0  # Avoid division by zero

    # Avoid division by zero
    seasonal_naive_mae_value = max(seasonal_naive_mae_value, 1e-8)

    return float(mae_value / seasonal_naive_mae_value)


def crps_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Approximation of CRPS (Continuous Ranked Probability Score) using mean
    weighted sum of quantile losses.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        float: CRPS value
    """
    if y_pred.ndim == 1:
        # For point predictions, we can't compute CRPS properly, so return MAE
        warnings.warn(
            f"Expected {len(QUANTILE_LEVELS)} quantiles, got 1 (point prediction). Returning MAE.",
            RuntimeWarning,
            stacklevel=2,
        )
        # Reshape from (n_samples,) to (n_samples, n_quantiles) by repeating
        # each sample
        y_pred = np.tile(y_pred.reshape(-1, 1), (1, len(QUANTILE_LEVELS)))

    # Ensure we have the right number of quantiles
    if (num_quantiles := y_pred.shape[1]) != len(QUANTILE_LEVELS):
        warnings.warn(
            f"Expected {len(QUANTILE_LEVELS)} quantiles, got {num_quantiles}",
            RuntimeWarning,
            stacklevel=2,
        )

    # Compute quantile loss for each quantile level
    quantile_losses = []
    for i, tau in enumerate(QUANTILE_LEVELS):
        errors = y_true - y_pred[:, i]
        ql = np.where(errors >= 0, tau * errors, (tau - 1) * errors)
        quantile_losses.append(ql)

    # Stack and compute mean weighted sum. shape: (n_samples, n_quantiles)
    quantile_losses = np.stack(quantile_losses, axis=1)

    # Compute weights (uniform weighting for now)
    weights = np.ones(len(QUANTILE_LEVELS)) / len(QUANTILE_LEVELS)

    # Weighted sum across quantiles, then mean across samples
    weighted_losses = quantile_losses @ weights
    return float(np.mean(weighted_losses))
