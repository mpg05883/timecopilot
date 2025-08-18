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
    file_path: Path,
    dataset: str,
    model_name: str,
    results: pd.DataFrame,
    domain: str,
    num_variates: int = 1,
) -> None:
    """
    Appends test set results and relevant metadata to a CSV file.

    The output row includes:
    - A timestamp for when the results were logged.
    - Dataset configuration name (e.g., "web_short").
    - Domain the dataset belongs to.
    - Model name (e.g., "tempo_prob").
    - Number of variates (1 for univariate, >1 for multivariate).
    - Evaluation metrics from the `results` DataFrame.
    - Optional notes (e.g., hyperparameters, experimental comments).

    If the CSV file does not exist, a header is first written using `write_header`.

    Args:
        results (DataFrame): Evaluation results (e.g., CRPS, NRMSE) as a
            single-row DataFrame.
        file_path (Path): Path to the CSV file where the row should be appended.
        dataset (str): Identifier of the dataset config used for this
            experiment.
        domain (Domain): Enum representing the domain this dataset belongs to.
        model_name (str, optional): Name of the forecasting model. Defaults to
            "tempo_prob".
        num_variates (int, optional): Number of input variates. Defaults to 1.
        notes (Optional[str], optional): Optional experiment notes. Defaults to
            None.
        verbose (bool, optional): Set to True to print the model's MAPE and
            CRPS.
    """
    if not file_path.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(get_fieldnames())

    with open(file_path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                dataset,
                model_name,
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
