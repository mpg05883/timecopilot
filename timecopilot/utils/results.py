import csv
from pathlib import Path

import pandas as pd
from gluonts.ev.metrics import (
    MAE,
    MAPE,
    MASE,
    MSE,
    MSIS,
    ND,
    NRMSE,
    RMSE,
    SMAPE,
    BaseMetricDefinition,
    MeanWeightedSumQuantileLoss,
)


def get_gift_eval_metrics() -> list[BaseMetricDefinition]:
    """
    Returns evaluation metrics to compute between the forecasted and ground
    truth values.

    returns:
        list[BaseMetricDefinition]: List of metric definitions from
            gluonts.ev.metrics.
    """
    return [
        MSE(forecast_type="mean"),
        MSE(forecast_type=0.5),
        MAE(),
        MASE(),
        MAPE(),
        SMAPE(),
        MSIS(),
        RMSE(),
        NRMSE(),
        ND(),
        MeanWeightedSumQuantileLoss(
            quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ),
    ]


def get_fieldnames() -> list[str]:
    """
    Returns the column fieldnames for a results file.
    """
    return [
        "dataset",
        "model",
        "eval_metrics/MSE[mean]",
        "eval_metrics/MSE[0.5]",
        "eval_metrics/MAE[0.5]",
        "eval_metrics/MASE[0.5]",
        "eval_metrics/MAPE[0.5]",
        "eval_metrics/sMAPE[0.5]",
        "eval_metrics/MSIS",
        "eval_metrics/RMSE[mean]",
        "eval_metrics/NRMSE[mean]",
        "eval_metrics/ND[0.5]",
        "eval_metrics/mean_weighted_sum_quantile_loss",
        "domain",
        "num_variates",
    ]


def save_results(
    file_path: str | Path,
    dataset: str,
    model: str,
    results: pd.DataFrame,
    domain: str,
    num_variates: int,
) -> None:
    """
    Saves the evaluation results to a CSV file.
    
    Args:
        file_path (Path): Path to the CSV file where results will be saved.
        dataset (str): Name of the dataset used for evaluation.
        model (str): Name of the model used for evaluation.
        results (pd.DataFrame): DataFrame containing evaluation metrics.
        domain (str): Domain of the dataset (e.g., "time_series").
        num_variates (int, optional): Number of variates in the dataset.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not file_path.exists():
        with open(file_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(get_fieldnames())

    with open(file_path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                dataset,
                model,
                results["MSE[mean]"].iloc[0],
                results["MSE[0.5]"].iloc[0],
                results["MAE[0.5]"].iloc[0],
                results["MASE[0.5]"].iloc[0],
                results["MAPE[0.5]"].iloc[0],
                results["sMAPE[0.5]"].iloc[0],
                results["MSIS"].iloc[0],
                results["RMSE[mean]"].iloc[0],
                results["NRMSE[mean]"].iloc[0],
                results["ND[0.5]"].iloc[0],
                results["mean_weighted_sum_quantile_loss"].iloc[0],
                domain,
                num_variates,
            ]
        )
