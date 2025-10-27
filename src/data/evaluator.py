import logging
from pathlib import Path

import pandas as pd
from gluonts.model import evaluate_model

from src.data.dataset import Dataset
from src.data.utils import get_metrics
from src.models.gluonts_predictor import GluonTSPredictor

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluator for GIFT-Eval datasets.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        verbose: bool = True,
    ) -> None:
        """
        Initialize an Evaluator instance for a specific dataset.

        Args:
            dataset (Dataset): The dataset to evaluate on.
            batch_size (int): The batch size to use for evaluation.
            verbose (bool): Whether to print progress information.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.verbose = verbose

    def evaluate(self, predictor: GluonTSPredictor) -> None:
        """
        Evaluate a GluonTS predictor on a dataset and save results.

        NOTE: Follow the conventions outlined in the GIFT-Eval repo to remain
        compatible with their leaderboard. See here for more details:
        https://github.com/SalesforceAIResearch/gift-eval?tab=readme-ov-file#evaluation

        Args:
            predictor (GluonTSPredictor): The predictor to evaluate.
        """
        res = evaluate_model(
            predictor,
            test_data=self.dataset.test_data,
            metrics=get_metrics(),
            batch_size=self.batch_size,
            axis=None,
            mask_invalid_label=True,
            allow_nan_forecast=False,
            seasonality=self.seasonality,
        )

        # Prepare the results for the CSV file
        model_name = (
            predictor.__class__.__name__
            if not isinstance(predictor, GluonTSPredictor)
            else predictor.alias
        )
        results_data = [
            [
                self.ds_config,
                model_name,
                res["MSE[mean]"][0],
                res["MSE[0.5]"][0],
                res["MAE[0.5]"][0],
                res["MASE[0.5]"][0],
                res["MAPE[0.5]"][0],
                res["sMAPE[0.5]"][0],
                res["MSIS"][0],
                res["RMSE[mean]"][0],
                res["NRMSE[mean]"][0],
                res["ND[0.5]"][0],
                res["mean_weighted_sum_quantile_loss"][0],
                self.dataset_properties_map[self.ds_key]["domain"],
                self.dataset_properties_map[self.ds_key]["num_variates"],
            ]
        ]

        if self.verbose:
            mase = res["MASE[0.5]"][0]
            crps = res["mean_weighted_sum_quantile_loss"][0]
            logging.info(f"MASE: {mase:.4f}, CRPS: {crps:.4f}")

        results_df = pd.DataFrame(
            results_data,
            columns=[
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
            ],
        )
        csv_file_path = Path(self.output_path) / "results.csv"
        csv_file_path.parent.mkdir(parents=True, exist_ok=True)
        if csv_file_path.exists():
            results_df = pd.concat(
                [pd.read_csv(csv_file_path), results_df],
                ignore_index=True,
            )
        results_df.to_csv(csv_file_path, index=False)
