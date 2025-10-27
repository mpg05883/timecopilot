import logging
import os

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from src.gift_eval.data import Dataset
from src.gift_eval.eval import Evaluator
from src.gift_eval.gluonts_predictor import GluonTSPredictor
from src.models.ensembles import SLSQPEnsemble


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Set seed, precision, and logging
    seed_everything(seed=cfg.seed, workers=cfg.workers, verbose=cfg.verbose)
    torch.set_float32_matmul_precision(cfg.precision)
    logging.basicConfig(**cfg.logging)

    # Load models
    models = [
        instantiate(
            model,
            batch_size=cfg.batch_size,
        )
        for model in cfg.models
    ]

    # Ensemble the models
    logging.info(
        f"Ensembling {len(models)} models: {', '.join([m.alias for m in models])}",
    )
    forecaster = SLSQPEnsemble(models=models, **cfg.ensemble)

    # Prepare the ensemble for evaluation
    predictor = GluonTSPredictor(
        forecaster=forecaster,
        batch_size=cfg.batch_size,
    )

    # Load list of dataset cfgs and use SLURM_ARRAY_TASK_ID to index the list.
    # Defaults to 38, which is the M4 Hourly dataset (short-term).
    cfg.data = cfg.data[int(os.environ.get("SLURM_ARRAY_TASK_ID", 38))]
    dataset_name, term = cfg.data.name, cfg.data.term

    logging.info(f"Loading dataset: {dataset_name} ({term}-term)")
    dataset = Dataset(name=dataset_name, term=term)

    evaluator = Evaluator(
        dataset=dataset,
        batch_size=cfg.batch_size,
        verbose=cfg.verbose,
    )

    # Evaluate the ensemble and save the results
    evaluator.evaluate(predictor=predictor)

    # Display the ensemble weights across all cross-validation windows
    forecaster.print_weights()


if __name__ == "__main__":
    main()
