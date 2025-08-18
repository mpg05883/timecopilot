import time
from pathlib import Path

import hydra
import torch
import wandb
from dotenv import load_dotenv
from gluonts.model import evaluate_forecasts
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import seed_everything

from timecopilot.gift_eval.data import Dataset
from timecopilot.gift_eval.gluonts_predictor import GluonTSPredictor
from timecopilot.models.foundational.tempo import TEMPOForecaster
from timecopilot.utils.common import format_elapsed_time, timestamp_info
from timecopilot.utils.model import find_best_checkpoint
from timecopilot.utils.results import get_gift_eval_metrics, save_results
from timecopilot.utils.wandb import (
    get_checkpoint_artifact_kwargs,
    get_results_artifact_kwargs,
    get_slurm_config,
    get_tempo_eval_run_kwargs,
)

load_dotenv()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed, cfg.workers, cfg.verbose)
    torch.set_float32_matmul_precision(cfg.precision)

    config = OmegaConf.to_container(cfg, resolve=True) | get_slurm_config()
    run_kwargs = get_tempo_eval_run_kwargs(cfg)
    run = wandb.init(
        config=config,
        **run_kwargs,
    )

    dataset_name, term = cfg.dataset_name, cfg.term
    timestamp_info(f"Loading dataset: {dataset_name} ({term}-term)")
    dataset = Dataset(dataset_name, term)

    # Format checkpoints directory paths as
    # results/checkpoints/gift_split/dataset/config/model/type where type
    # specifies whether the model was fit on only the test data, training data
    # with test data leakage, etc.
    # E.g. checkpoints/gift_split/loop_seattle/5T/short/leak_test_data
    checkpoint_dirpath_parts = [
        cfg.results.checkpoints_dir,
        cfg.gift_split,
        dataset.config,
        cfg.model.type,
    ]
    checkpoint_dirpath = to_absolute_path(str(Path(*checkpoint_dirpath_parts)))

    timestamp_info(f"Looking for checkpoints in {checkpoint_dirpath}")
    checkpoint_path = find_best_checkpoint(checkpoint_dirpath)

    timestamp_info(f"Loading checkpoint from: {checkpoint_path}")
    wandb.summary["checkpoint_path"] = checkpoint_path
    forecaster = TEMPOForecaster(checkpoint_path, cfg.batch_size)

    checkpoint_artifact_kwargs = get_checkpoint_artifact_kwargs(
        model_name=cfg.model.name,
        dataset_name=dataset_name,
        term=term,
    )
    checkpoint_artifact = wandb.Artifact(**checkpoint_artifact_kwargs)
    checkpoint_artifact.add_file(str(checkpoint_path))
    run.log_artifact(checkpoint_artifact)

    rolling_mean_value = "gluonts.transform.feature.RollingMeanValueImputation"

    # If using rolling mean value imputation, set the window size to the
    # dataset's seasonality
    if cfg.imputation_method._target_ == rolling_mean_value:
        with open_dict(cfg.imputation_method):
            cfg.imputation_method.window_size = dataset.seasonality

    predictor = GluonTSPredictor(
        forecaster=forecaster,
        h=dataset.prediction_length,
        freq=dataset.freq,
        imputation_method=instantiate(cfg.imputation_method),
        batch_size=cfg.batch_size,
    )

    timestamp_info("Generating test set forecasts...")
    start_time = time.time()

    forecasts = predictor.predict(dataset)

    end_time = time.time()
    elapsed_time = format_elapsed_time(start_time, end_time)
    timestamp_info(
        f"Finished generating test set forecasts! Time taken: {elapsed_time}"
    )

    timestamp_info("Evaluating forecasts...")
    start_time = time.time()

    results_df = evaluate_forecasts(
        forecasts=forecasts,
        test_data=dataset.test_data,
        metrics=get_gift_eval_metrics(),
        batch_size=cfg.batch_size,
        seasonality=dataset.seasonality,
    )

    end_time = time.time()
    elapsed_time = format_elapsed_time(start_time, end_time)
    timestamp_info(f"Finished evaluation! Time taken: {elapsed_time}")

    mase = results_df["MASE[0.5]"].iloc[0]
    crps = results_df["mean_weighted_sum_quantile_loss"].iloc[0]

    timestamp_info(f"MASE: {mase:.4f}, CRPS: {crps:.4f}")

    wandb.summary["MASE"] = mase
    wandb.summary["CRPS"] = crps

    # Format results paths as
    # results/results/dataset_name/freq/term/model_type/results.csv
    # E.g. results/results/loop_seattle/5T/short/leak_test_data/results.csv
    results_path_parts = [
        cfg.results.results_dir,
        dataset.config,
        cfg.model.type,
        "results.csv",
    ]
    results_path = Path(*results_path_parts)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    save_results(
        file_path=results_path,
        dataset=dataset.config,
        model=cfg.model.name,
        results=results_df,
        domain=dataset.domain,
        num_variates=dataset.target_dim,
    )
    timestamp_info(f"Results saved to {results_path}")

    table = wandb.Table(dataframe=results_df)
    wandb.log({"results": table})

    results_artifact_kwargs = get_results_artifact_kwargs(
        model_name=cfg.model.name,
        dataset_name=dataset_name,
        term=term,
    )
    table_artifact = wandb.Artifact(**results_artifact_kwargs)
    table_artifact.add(table, results_artifact_kwargs["type"])
    run.log_artifact(table_artifact)

    run.finish()


if __name__ == "__main__":
    main()
