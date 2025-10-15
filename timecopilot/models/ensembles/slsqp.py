from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize
from sklearn.isotonic import IsotonicRegression

from ... import TimeCopilotForecaster
from ..utils.forecaster import Forecaster, QuantileConverter

MetricName = Literal["mse", "mae", "smape", "crps", "mase"]


def _mse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean((y - yhat) ** 2))


def _mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yhat)))


def _smape(y: np.ndarray, yhat: np.ndarray) -> float:
    denom = np.abs(y) + np.abs(yhat) + 1e-8
    return float(np.mean(2 * np.abs(y - yhat) / denom))


def _mase(
    y: np.ndarray,
    yhat: np.ndarray,
    y_train: np.ndarray | None = None,
    seasonality: int = 1,
) -> float:
    """
    Compute Mean Absolute Scaled Error (MASE).

    MASE = MAE / MAE_naive_seasonal

    Args:
        y: Ground truth values
        yhat: Predicted values
        y_train: Training data for computing seasonal naive baseline (if None,
        uses simple naive) seasonality: Seasonal period (1 for non-seasonal, 12
        for monthly, etc.)

    Returns:
        float: MASE value
    """
    mae = np.mean(np.abs(y - yhat))

    if y_train is not None and len(y_train) > seasonality:
        # Use seasonal naive baseline from training data
        if seasonality == 1:
            # Simple naive: use last training value
            naive_errors = np.abs(y_train[1:] - y_train[:-1])
        else:
            # Seasonal naive: use values from seasonality periods ago
            naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])

        mae_naive = np.mean(naive_errors)
    else:
        # Fallback: use simple differences in the actual data
        if len(y) > 1:
            if seasonality == 1 or len(y) <= seasonality:
                naive_errors = np.abs(y[1:] - y[:-1])
            else:
                naive_errors = np.abs(y[seasonality:] - y[:-seasonality])
            mae_naive = np.mean(naive_errors)
        else:
            mae_naive = 1.0  # Avoid division by zero

    # Avoid division by zero
    mae_naive = max(mae_naive, 1e-8)

    return float(mae / mae_naive)


def _mase_simple(y: np.ndarray, yhat: np.ndarray) -> float:
    """
    Simplified MASE that uses naive differences in the validation data.
    This version can be used in optimization without requiring training data.
    """
    return _mase(y, yhat, y_train=None, seasonality=1)


def _mean_weighted_sum_quantile_loss(y: np.ndarray, yhat: np.ndarray) -> float:
    """
    Compute Mean Weighted Sum Quantile Loss (CRPS approximation).

    Args:
        y: Ground truth values, shape (n_samples,)
        yhat: Predicted values, shape (n_samples, n_quantiles) or (n_samples,)
        for point prediction

    Returns:
        float: CRPS value
    """
    from ...gift_eval.utils import QUANTILE_LEVELS

    # print(f"DEBUG: yhat.shape: {yhat.shape}, expected: ({len(y)}, {len(QUANTILE_LEVELS)})")
    # If yhat is 1D, assume it's a point prediction and create quantiles around it
    if yhat.ndim == 1:
        # For point predictions, we can't compute CRPS properly
        # Return a fallback metric (MAE)
        # add warning
        warnings.warn(
            f"Expected {len(QUANTILE_LEVELS)} quantiles, got 1 (point prediction)",
            RuntimeWarning,
        )
        # Reshape from (n_samples,) to (n_samples, n_quantiles) by repeating each sample
        yhat = np.tile(yhat.reshape(-1, 1), (1, len(QUANTILE_LEVELS)))
        # return float(np.mean(np.abs(y - yhat)))

    # Ensure we have the right number of quantiles
    if yhat.shape[1] != len(QUANTILE_LEVELS):
        # make the warning level to WARNING
        warnings.warn(
            f"Expected {len(QUANTILE_LEVELS)} quantiles, got {yhat.shape[1]}",
            RuntimeWarning,
        )
        # # make the shape of yhat to be (n_samples, len(QUANTILE_LEVELS))
        # yhat = np.concatenate([yhat, np.zeros((yhat.shape[0], len(QUANTILE_LEVELS) - yhat.shape[1]))], axis=1)
        # raise ValueError(f"Expected {len(QUANTILE_LEVELS)} quantiles, got {yhat.shape[1]}")

    # Method 1: Quantile Loss based CRPS (current implementation)
    # Compute quantile loss for each quantile level
    quantile_losses = []
    for i, tau in enumerate(QUANTILE_LEVELS):
        errors = y - yhat[:, i]
        ql = np.where(errors >= 0, tau * errors, (tau - 1) * errors)
        quantile_losses.append(ql)

    # Stack and compute mean weighted sum
    quantile_losses = np.stack(
        quantile_losses, axis=1
    )  # shape: (n_samples, n_quantiles)

    # Compute weights (uniform weighting for now)
    weights = np.ones(len(QUANTILE_LEVELS)) / len(QUANTILE_LEVELS)

    # Weighted sum across quantiles, then mean across samples
    weighted_losses = quantile_losses @ weights
    return float(np.mean(weighted_losses))

    # Method 2: Simple CRPS approximation (alternative - uncomment to use)
    # This is a simpler approximation: CRPS ≈ mean(|quantiles - y|) across all quantiles
    # crps_values = []
    # for i in range(len(y)):
    #     y_true = y[i]
    #     quantiles = yhat[i, :]
    #     crps_values.append(np.mean(np.abs(quantiles - y_true)))
    # return float(np.mean(crps_values))


def _crps_empirical_alternative(y: np.ndarray, yhat: np.ndarray) -> float:
    """
    Alternative CRPS implementation using the empirical formula.
    This is more mathematically accurate but computationally expensive.

    CRPS = E[|F^(-1)(U) - y|] - 0.5 * E[|F^(-1)(U) - F^(-1)(V)|]
    where F^(-1) is the quantile function and U,V are independent uniform random variables.

    For discrete quantile levels, this approximates to:
    CRPS ≈ Σ_i w_i * |q_i - y| - 0.5 * Σ_i Σ_j w_i * w_j * |q_i - q_j|
    """
    from ...gift_eval.utils import QUANTILE_LEVELS

    if yhat.ndim == 1:
        return float(np.mean(np.abs(y - yhat)))

    if yhat.shape[1] != len(QUANTILE_LEVELS):
        raise ValueError(
            f"Expected {len(QUANTILE_LEVELS)} quantiles, got {yhat.shape[1]}"
        )

    # Uniform weights for quantile levels
    weights = np.array(QUANTILE_LEVELS[1:]) - np.array(QUANTILE_LEVELS[:-1])
    weights = np.concatenate(
        [[QUANTILE_LEVELS[0]], weights]
    )  # Add weight for first quantile
    weights = weights / np.sum(weights)  # Normalize

    crps_values = []
    for sample_idx in range(len(y)):
        y_true = y[sample_idx]
        quantiles = yhat[sample_idx, :]

        # First term: E[|F^(-1)(U) - y|]
        term1 = np.sum(weights * np.abs(quantiles - y_true))

        # Second term: 0.5 * E[|F^(-1)(U) - F^(-1)(V)|]
        term2 = 0.0
        for i in range(len(quantiles)):
            for j in range(len(quantiles)):
                term2 += weights[i] * weights[j] * np.abs(quantiles[i] - quantiles[j])
        term2 *= 0.5

        crps_values.append(term1 - term2)

    return float(np.mean(crps_values))


METRICS: dict[MetricName, Callable[[np.ndarray, np.ndarray], float]] = {
    "mse": _mse,
    "mae": _mae,
    "smape": _smape,
    "crps": _mean_weighted_sum_quantile_loss,
    "mase": _mase_simple,
}


@dataclass
class SLSQPOptimizationResult:
    weights: list[float]
    metric: MetricName
    metric_value: float
    success: bool
    message: str | None
    scipy_result: OptimizeResult | None = None


class SLSQPEnsemble(Forecaster):
    """Weighted ensemble with SLSQP constrained optimization.

    This ensemble finds non-negative weights that sum to 1 for the underlying
    base model forecasts by minimizing a chosen error metric over
    cross-validation (CV) forecasts produced on the training data.

    Workflow (inside `forecast` when no explicit weights are supplied):
      1. Run model cross-validation (configurable number of windows).
      2. Collect each model's point forecasts and the actual `y`.
      3. Solve:  min_w  metric(y, X w)  subject to  w_i >= 0, sum w_i = 1.
      4. Apply the optimized weights to future (horizon) forecasts.

    Quantile & level forecasts:
      - If quantiles are requested, the same weights are applied per quantile.
      - Monotonicity across quantiles is enforced via isotonic regression.

    Parameters controlling weight optimization (all optional):
      - `weights`: manually pass weights (length = n_models) to skip optimization.
      - `opt_metric`: one of {"mse", "mae", "smape", "crps", "mase"}. Default: 
        "mse".
      - `opt_n_windows`: number of CV windows used for weighting (default 1).
      - `opt_step_size`: step size between CV windows (defaults to h if None).
      - `opt_h`: horizon to use for CV during weighting (defaults to `h`).

    Notes:
      - If optimization fails, the ensemble safely falls back to equal weights.
      - With a single model, the weight is trivially [1.0].
    """

    def __init__(
        self,
        models: list[Forecaster],
        opt_metric: MetricName = "mse",
        batch_size: int = 64,
        n_windows: int = 1,
    ) -> None:
        self.tcf = TimeCopilotForecaster(models=models, fallback_model=None)
        self.weights_: list[float] | None = None
        self.opt_result_: SLSQPOptimizationResult | None = None
        self.batch_size = batch_size
        self.opt_metric = opt_metric
        self.n_windows = n_windows
        self.alias = self.format_alias()
        
    def format_alias(self) -> str:
        num_models_str = f"{len(self.tcf.models)}-models"
        num_windows_str = f"{self.n_windows}-windows"
        opt_metric_str = f"opt-{self.opt_metric}"
        return f"SLSQPEnsemble_{num_models_str}_{opt_metric_str}_{num_windows_str}"
        
    @property
    def model_aliases(self) -> list[str]:
        return [m.alias for m in self.tcf.models]

    def _optimize_weights(
        self,
        y: np.ndarray,
        model_matrix: np.ndarray,
        metric: MetricName,
        init: np.ndarray | None = None,
    ) -> SLSQPOptimizationResult:
        n_models = model_matrix.shape[1]
        if n_models == 1:
            return SLSQPOptimizationResult(
                weights=[1.0],
                metric=metric,
                metric_value=METRICS[metric](y, model_matrix[:, 0]),
                success=True,
                message="Single model - trivial weights",
            )
        if init is None:
            init = np.full(n_models, 1.0 / n_models)

        metric_fn = METRICS[metric]

        def objective(w: np.ndarray) -> float:
            yhat = model_matrix @ w
            return metric_fn(y, yhat)

        constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
        bounds = [(0.0, 1.0)] * n_models

        try:
            res = minimize(
                objective,
                init,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 500, "disp": True},
            )
            if not res.success:
                warnings.warn(
                    f"SLSQP optimization failed: {res.message}. Defaulting to "
                    "equal weights.",
                    RuntimeWarning,
                )
                w = np.full(n_models, 1.0 / n_models)
                mv = metric_fn(y, model_matrix @ w)
                return SLSQPOptimizationResult(
                    weights=w.tolist(),
                    metric=metric,
                    metric_value=mv,
                    success=False,
                    message=str(res.message),
                    scipy_result=res,
                )
            w = res.x
            mv = metric_fn(y, model_matrix @ w)
            return SLSQPOptimizationResult(
                weights=w.tolist(),
                metric=metric,
                metric_value=mv,
                success=True,
                message=str(res.message),
                scipy_result=res,
            )
        except Exception as e:  # pragma: no cover (robustness)
            warnings.warn(
                    f"SLSQP optimization failed: {res.message}. Defaulting to "
                    "equal weights.",
                    RuntimeWarning,
                )
            w = np.full(n_models, 1.0 / n_models)
            mv = metric_fn(y, model_matrix @ w)
            return SLSQPOptimizationResult(
                weights=w.tolist(),
                metric=metric,
                metric_value=mv,
                success=False,
                message=str(e),
                scipy_result=None,
            )

    def _optimize_weights_quantile(
        self,
        y: np.ndarray,
        quantile_matrix: np.ndarray,
        metric: MetricName,
        init: np.ndarray | None = None,
    ) -> SLSQPOptimizationResult:
        """
        Optimize weights for quantile predictions (e.g., CRPS).

        Args:
            y: Ground truth values, shape (n_samples,)
            quantile_matrix: Quantile predictions, shape (n_samples, n_models, n_quantiles)
            metric: Metric name (should be "crps")
            init: Initial weights

        Returns:
            SLSQPOptimizationResult with optimized weights
        """
        n_samples, n_models, n_quantiles = quantile_matrix.shape

        if n_models == 1:
            # Single model case - compute CRPS directly
            single_model_quantiles = quantile_matrix[
                :, 0, :
            ]  # shape: (n_samples, n_quantiles)
            metric_value = METRICS[metric](y, single_model_quantiles)
            return SLSQPOptimizationResult(
                weights=[1.0],
                metric=metric,
                metric_value=metric_value,
                success=True,
                message="Single model - trivial weights",
            )

        if init is None:
            init = np.full(n_models, 1.0 / n_models)

        metric_fn = METRICS[metric]

        def objective(w: np.ndarray) -> float:
            # Compute weighted ensemble quantile predictions
            # quantile_matrix @ w gives shape (n_samples, n_quantiles)
            yhat_quantiles = np.einsum("ijk,j->ik", quantile_matrix, w)
            return metric_fn(y, yhat_quantiles)

        constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
        bounds = [(0.0, 1.0)] * n_models

        try:
            res = minimize(
                objective,
                init,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 500, "disp": False},
            )
            if not res.success:
                warnings.warn(
                    f"SLSQP quantile optimization failed: {res.message}. Falling back to equal weights.",
                    RuntimeWarning,
                )
                w = np.full(n_models, 1.0 / n_models)
                yhat_quantiles = np.einsum("ijk,j->ik", quantile_matrix, w)
                mv = metric_fn(y, yhat_quantiles)
                return SLSQPOptimizationResult(
                    weights=w.tolist(),
                    metric=metric,
                    metric_value=mv,
                    success=False,
                    message=str(res.message),
                    scipy_result=res,
                )
            w = res.x
            yhat_quantiles = np.einsum("ijk,j->ik", quantile_matrix, w)
            mv = metric_fn(y, yhat_quantiles)
            return SLSQPOptimizationResult(
                weights=w.tolist(),
                metric=metric,
                metric_value=mv,
                success=True,
                message=str(res.message),
                scipy_result=res,
            )
        except Exception as e:  # pragma: no cover (robustness)
            print(f"DEBUG: SLSQP quantile optimization exception: {e}")

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
        weights: list[float] | None = None,
        opt_n_windows: int = 1,
        opt_step_size: int | None = None,
        opt_h: int | None = None,
        use_analogue: bool = False,
        verbose: bool = False,
    ) -> pd.DataFrame:
        qc = QuantileConverter(level=level, quantiles=quantiles)

        if weights is not None and len(weights) != len(self.tcf.models):
            raise ValueError(
                "Number of weights must match number of models. Got "
                f"{len(weights)} weights vs {len(self.tcf.models)} models."
            )

        # Optimize weights if they're not given
        if weights is None:
            kwargs = {
                "attr": (
                    "cross_validation"
                    if not use_analogue
                    else "cross_validation_analogue"
                ),
                "merge_on": ["unique_id", "ds", "cutoff"],
                "df": df,
                "h": h if opt_h is None else opt_h,
                "freq": freq,
                "level": None,
                "quantiles": qc.quantiles,
                "n_windows": opt_n_windows,
                "step_size": opt_step_size,
            }
            cv_df = self.tcf._call_models(**kwargs)
            if "y" not in cv_df:
                raise RuntimeError(
                    "Cross-validation did not return 'y' column; cannot "
                    "optimize weights."
                )

            # Column names used to access each model's point forecasts
            model_cols = [m.alias for m in self.tcf.models]
            
            # ! This might not work with the new timecopilot version            
            # Use Toto's median forecasts
            model_cols = [
                col.replace("Toto", "Toto-q-50") if col == "Toto" else col
                for col in model_cols
            ]

            # Ensure all base models have cross-validation forecasts
            missing = [c for c in model_cols if c not in cv_df]
            if missing:
                raise RuntimeError(
                    "Missing model columns in CV dataframe: " f"{missing}."
                )

            # Get the ground truth values
            y = cv_df["y"].to_numpy(dtype=float)

            # Handle special metrics that need additional processing
            if self.opt_metric == "crps" and qc.quantiles is not None:
                # Get quantile columns for each model
                quantile_data = []
                model_cols = [m.alias for m in self.tcf.models]

                for model in model_cols:
                    model_quantiles = []
                    for q in sorted(qc.quantiles):
                        pct = int(q * 100)
                        q_col = f"{model}-q-{pct}"
                        if q_col not in cv_df:
                            raise RuntimeError(
                                f"Missing quantile column: {q_col}.",
                            )
                        model_quantiles.append(cv_df[q_col].to_numpy(dtype=float))

                    # Stack quantiles for this model: shape (n_samples,
                    # n_quantiles)
                    model_quantiles = np.stack(model_quantiles, axis=1)
                    quantile_data.append(model_quantiles)

                # Stack all models: shape (n_samples, n_models, n_quantiles)
                X_quantiles = np.stack(quantile_data, axis=1)

                # guard against NaNs
                mask = ~np.isnan(y) & ~np.any(
                    np.isnan(X_quantiles.reshape(X_quantiles.shape[0], -1)),
                    axis=1,
                )
                y_clean = y[mask]

                # shape: (n_clean_samples, n_models, n_quantiles)
                X_clean = X_quantiles[mask]

                if y_clean.size == 0:
                    raise RuntimeError(
                        "No valid rows available for weight optimization."
                    )

                print(
                    f"DEBUG: CRPS optimization - y_clean.shape: "
                    f"{y_clean.shape}, X_clean.shape: {X_clean.shape}"
                )

                # Optimize weights for quantile predictions
                opt_res = self._optimize_weights_quantile(
                    y_clean,
                    X_clean,
                    self.opt_metric,
                )

            elif self.opt_metric == "mase":
                # Enhanced MASE calculation with training data context
                X = cv_df[model_cols].to_numpy(dtype=float)

                # Try to extract training data for better MASE calculation
                # Group by unique_id to get per-series training data
                enhanced_mase_data = []
                for uid in cv_df["unique_id"].unique():
                    uid_data = cv_df[cv_df["unique_id"] == uid].sort_values(
                        "ds",
                    )
                    uid_y = uid_data["y"].values
                    uid_X = uid_data[model_cols].values

                    # Use the validation data as both training context and
                    # target.
                    # This is a limitation of the CV approach - we don't have
                    # separate training data
                    enhanced_mase_data.append((uid_y, uid_X))

                # For now, fall back to simple MASE since we don't have clean
                # training/validation split
                mask = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
                y_clean = y[mask]
                X_clean = X[mask]
                if y_clean.size == 0:
                    raise RuntimeError(
                        "No valid rows available for weight optimization."
                    )

                print(
                    f"DEBUG: MASE optimization - y_clean.shape: "
                    f"{y_clean.shape}, X_clean.shape: {X_clean.shape}"
                )
                opt_res = self._optimize_weights(y_clean, X_clean, self.opt_metric)

            else:
                # Original logic for point predictions
                X = cv_df[model_cols].to_numpy(dtype=float)
                # guard against NaNs
                mask = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)

                y_clean = y[mask]  # Ground truth
                X_clean = X[mask]  # Predictions

                if y_clean.size == 0:
                    raise RuntimeError(
                        "No valid rows available for weight optimization."
                    )
                opt_res = self._optimize_weights(y_clean, X_clean, self.opt_metric)

            weights = opt_res.weights
            self.opt_result_ = opt_res
        else:
            self.opt_result_ = None

        self.weights_ = [float(w) for w in weights]

        # Save base models' weights
        new_weights = {
            m: w
            for m, w in zip(
                self.model_aliases,
                weights,
                strict=False,
            )
        }
        self.weights_df = pd.DataFrame([new_weights], columns=self.model_aliases),
        model_cols = [m.alias for m in self.tcf.models]
        if verbose:
            weights_str = ", ".join(
                f"{m}:{w:.4f}"
                for m, w in zip(
                    model_cols,
                    self.weights_,
                    strict=False,
                )
            )
            if self.opt_result_ is not None:
                logging.info(
                    f"[{self.alias}] optimized weights ({self.opt_metric}, "
                    f"value={self.opt_result_.metric_value:.6f}): "
                    f"{weights_str}"
                )
            else:
                logging.info(f"[{self.alias}] provided weights: {weights_str}")

        # Forecast future horizon for each base model
        _fcst_df = self.tcf._call_models(
            "forecast",
            merge_on=["unique_id", "ds"],
            df=df,
            h=h,
            freq=freq,
            level=None,  # always use quantiles path then convert if needed
            quantiles=qc.quantiles,
        )

        out = _fcst_df[["unique_id", "ds"]].copy()
        # reuse model_cols from above
        W = np.array(self.weights_)
        # point forecasts
        out[self.alias] = (_fcst_df[model_cols].to_numpy(dtype=float) @ W).astype(float)

        if qc.quantiles is not None:
            qs = sorted(qc.quantiles)
            q_cols = []
            for q in qs:
                pct = int(q * 100)
                base_cols = [f"{col}-q-{pct}" for col in model_cols]
                missing = [c for c in base_cols if c not in _fcst_df]

                if missing:
                    raise RuntimeError(
                        f"Quantile columns missing for pct {pct}: {missing}."
                    )
                vals = _fcst_df[base_cols].to_numpy(dtype=float) @ W
                q_col = f"{self.alias}-q-{pct}"
                out[q_col] = vals
                q_cols.append(q_col)
            # enforce monotonicity across quantiles per row
            ir = IsotonicRegression(increasing=True)

            def _mono(row: pd.Series) -> np.ndarray:
                return ir.fit_transform(qs, row.values)

            mono_vals = out[q_cols].apply(_mono, axis=1, result_type="expand")
            out[q_cols] = mono_vals
            if 0.5 in qc.quantiles:
                out[self.alias] = out[f"{self.alias}-q-50"].values
            out = qc.maybe_convert_quantiles_to_level(out, models=[self.alias])

        return out


    def get_weights_df(self) -> pd.DataFrame:
        """
        Returns a DataFrame with model aliases and their weights and each
        weight configuration's cross-validation error.

        Returns:
            pd.DataFrame: DataFrame with model aliases and their current
                weights.
        """
        self.weights_df[self.opt_metric] = self.ensemble_cv_error_list
        return self.weights_df

    def print_weights(self) -> None:
        """Print current weights in a readable format (forecasts must have
        been run)."""
        if self.weights_ is None:
            logging.info(f"[{self.alias}] weights not available yet - run forecast().")
            return
        message = ", ".join(
            f"{m}:{w:.4f}"
            for m, w in zip(
                self.model_aliases,
                self.weights_,
                strict=False,
            )
        )
        logging.info(f"[{self.alias}] last cross-validation window's weights: {message}")



__all__ = ["SLSQPEnsemble", "SLSQPOptimizationResult"]
