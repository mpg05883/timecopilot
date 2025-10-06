import argparse
import logging

from pytorch_lightning import seed_everything
from timecopilot.gift_eval.data import Dataset
from timecopilot.gift_eval.gluonts_predictor import GluonTSPredictor
from timecopilot.models.ensembles import MedianEnsemble
from timecopilot.models.foundation import Moirai, Sundial, TimesFM, Toto
from timecopilot.utils.path import resolve_output_path, resolve_storage_path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%b %d, %Y %I:%M:%S%p",
)

def main(args):
    seed_everything(args.seed)
    logging.info(f"Loading dataset: {args.name} ({args.term})")
    dataset = Dataset(
        name=args.name,
        term=args.term,
        storage_path=resolve_storage_path(args.storage_env_var),
    )
    
    logging.info("Creating ensemble forecaster")
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
            repo_id="google/timesfm-2.0-500m-pytorch",
            batch_size=args.batch_size,
        ),
        Toto(batch_size=args.batch_size),
    ]
    forecaster = MedianEnsemble(models=models)
    
    logging.info("Wrapping forecaster in GluonTS predictor")    
    predictor = GluonTSPredictor(
        forecaster=forecaster,
        batch_size=args.batch_size,
    )
    
    logging.info("Starting cross-validation...")
    df = predictor.cross_validation(dataset=dataset, n_windows=args.n_windows)
    
    output_dir = resolve_output_path(
        output_dir=args.output_dir,
        dataset_config=dataset.config,
    )
    output_path = output_dir / "cross_validation.csv"
    
    logging.info(f"Finished cross-validation! Saving results to {output_path}")
    df.to_csv(output_path, index=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="ett1/15T",
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
        "--batch_size",
        type=int,
        default=64,
        help="Batch size to use for inference",
    )
    parser.add_argument(
        "--n_windows",
        type=int,
        default=1,
        help="Number of windows to use for cross-validation"
    )
    parser.add_argument(
        "--include_input",
        type=bool,
        default=True,
        help="Whether to include input window in cross-validation results",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save both cross-validation and evaluation results",
    )
    args = parser.parse_args()
    main(args)
