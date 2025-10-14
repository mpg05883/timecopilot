import argparse
import logging
import os

from pytorch_lightning import seed_everything

from timecopilot.gift_eval.data import Dataset
from timecopilot.gift_eval.eval import GIFTEval
from timecopilot.gift_eval.gluonts_predictor import GluonTSPredictor
from timecopilot.gift_eval.utils import DATASETS_WITH_TERMS, NUM_DATASETS
from timecopilot.models.ensembles import SLSQPEnsemble
from timecopilot.models.foundation import Moirai, Sundial, TimesFM, Toto
from timecopilot.utils.path import resolve_output_path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%b %d, %Y %I:%M:%S%p",
)

seed_everything(seed=42, workers=True, verbose=True)


def main(args):
    print("Command-line arguments:")
    for key, value in vars(args).items():
        print(f"- {key}: {value}")

    m4_hourly_task_id = 5
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", m4_hourly_task_id))
    name, term = DATASETS_WITH_TERMS[task_id % NUM_DATASETS]
    logging.info(f"Loading dataset: {name} ({term})")
    dataset = Dataset(name=name, term=term)

    models = [
        Moirai(
            repo_id="Salesforce/moirai-2.0-R-small",
            batch_size=args.batch_size,
        ),
        Sundial(
            repo_id="thuml/sundial-base-128m",
            batch_size=args.batch_size,
        ),
        TimesFM(
            repo_id="google/timesfm-2.5-200m-pytorch",
            batch_size=args.batch_size,
        ),
        Toto(batch_size=args.batch_size),
    ]

    forecaster = SLSQPEnsemble(
        models=models,
        opt_metric=args.opt_metric,
        batch_size=args.batch_size,
    )

    predictor = GluonTSPredictor(
        forecaster=forecaster,
        batch_size=args.batch_size,
    )

    output_path = resolve_output_path(
        alias=forecaster.alias,
        dataset_config=dataset.config,
    )

    gifteval = GIFTEval(dataset_name=name, term=term, output_path=output_path)
    gifteval.evaluate_predictor(
        predictor=predictor,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size to use during cross-validation and evaluation",
    )
    parser.add_argument(
        "--opt_metric",
        choices=["mse", "mae", "smape", "mase", "crps"],
        default="mse",
        help="Metric to optimize when tuning ensemble during cross-validation",
    )
    args = parser.parse_args()
    main(args)
