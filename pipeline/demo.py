import argparse
import logging

from pytorch_lightning import seed_everything

from timecopilot.gift_eval.eval import GIFTEval
from timecopilot.gift_eval.gluonts_predictor import GluonTSPredictor
from timecopilot.models.foundation import TimesFM
from timecopilot.utils import resolve_output_path, resolve_storage_path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%b %d, %Y %I:%M:%S%p",
)

seed_everything(42)


def main(args):
    logging.info(f"Downloading TimesFM model from {args.repo_id}")
    forecaster = TimesFM(repo_id=args.repo_id)

    logging.info(f"Creating predictor with batch size {args.batch_size}")
    predictor = GluonTSPredictor(
        forecaster=forecaster,
        batch_size=args.batch_size,
    )

    logging.info(f"Loading {args.name} dataset ({args.term})")
    gift_eval = GIFTEval(
        dataset_name=args.name,
        term=args.term,
        storage_path=resolve_storage_path(storage_env_var=args.storage_env_var),
        output_path=resolve_output_path(output_dir=args.output_dir),
    )

    logging.info("Starting evaluation...")
    gift_eval.evaluate_predictor(
        predictor,
        batch_size=args.batch_size,
        overwrite_results=args.overwrite_results,
        verbose=args.verbose,
    )
    logging.info("Finished evaluation!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        default="m4_hourly",
        help="Name of the dataset to load",
    )
    parser.add_argument(
        "--term",
        choices=["short", "medium", "long"],
        default="short",
        help="Forecasting term to use with the dataset",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="google/timesfm-2.0-500m-pytorch",
        help="Hugging Face Hub model ID to load the TimesFM model from",
    )
    parser.add_argument(
        "--storage_env_var",
        type=str,
        default="GIFT_EVAL",
        help="Environment variable that points to the data storage location",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size to use for inference",
    )
    parser.add_argument(
        "--overwrite_results",
        action="store_true",
        help="Whether to overwrite existing evaluation results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print progress statements during evaluation",
    )
    args = parser.parse_args()
    main(args)
