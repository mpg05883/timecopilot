import argparse
import logging

from pytorch_lightning import seed_everything
from timecopilot.gift_eval.data import Dataset
from timecopilot.gift_eval.eval import GIFTEval
from timecopilot.gift_eval.utils import DATASETS_WITH_TERMS, NUM_DATASETS
from timecopilot.gift_eval.gluonts_predictor import GluonTSPredictor
from timecopilot.models.ensembles import MedianEnsemble
from timecopilot.models.foundation import Moirai, Sundial, TimesFM, Toto
from timecopilot.utils.path import resolve_output_path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%b %d, %Y %I:%M:%S%p",
)

seed_everything(seed=42, workers=True, verbose=True)

def main(args):
    name, term = DATASETS_WITH_TERMS[args.task_id % NUM_DATASETS]
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
    
    forecaster = MedianEnsemble(models=models)
    
    predictor = GluonTSPredictor(
        forecaster=forecaster,
        batch_size=args.batch_size,
    )
    
    output_path = resolve_output_path(
        alias=forecaster.alias,
        dataset_config=dataset.config,
    )
    
    logging.info("Starting cross-validation...")
    predictor.cross_validation(dataset=dataset, output_path=output_path)
    
    logging.info("Starting evaluation...")
    gifteval = GIFTEval(dataset_name=name, term=term, output_path=output_path)
    gifteval.evaluate_predictor(
        predictor=predictor,
        batch_size=args.batch_size,
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_id",
        type=int,
        default=36,  # Defaults to the ett1/15T (short) dataset
        help="SLURM ARRAY TASK ID used to select dataset name and term. ",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size to use during cross-validation and evaluation",
    )
    args = parser.parse_args()
    main(args)
