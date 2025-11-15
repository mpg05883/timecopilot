import argparse

from timecopilot.gift_eval.eval import GIFTEval
from timecopilot.gift_eval.gluonts_predictor import GluonTSPredictor
from timecopilot.models.ensembles.median import MedianEnsemble
from timecopilot.models.foundation.chronos import Chronos
from timecopilot.models.foundation.timesfm import TimesFM
from timecopilot.models.foundation.flowstate import FlowState

def main(args) -> None:
    chronos = Chronos(
        repo_id="amazon/chronos-2",
        batch_size=args.batch_size,
    )
    timesfm = TimesFM(
        repo_id="google/timesfm-2.5-200m-pytorch",
        batch_size=args.batch_size,
    )
    flowstate = FlowState(
        repo_id="ibm-research/flowstate",
        batch_size=args.batch_size,
    )
    
    forecaster = MedianEnsemble(
        models=[timesfm, chronos, flowstate],
        alias="TimeCopilot",
    )
    
    predictor = GluonTSPredictor(
        forecaster=forecaster,
        max_length=args.max_length,
        batch_size=args.data_batch_size,  
    )
    
    gifteval = GIFTEval(
        dataset_name=args.dataset_name,
        term=args.term,
        output_path=args.output_path,
        storage_path=args.storage_path,
    )
    gifteval.evaluate_predictor(predictor, batch_size=args.eval_batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Timecopilot demo")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ett1/15T",
        help="The name of the dataset to evaluate",
    )
    parser.add_argument(
        "--term",
        type=str,
        default="short",
        help="The term to evaluate",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results",
        help="The directory to save the results",
    )
    parser.add_argument(
        "--storage_path",
        type=str,
        default="data",
        help="The directory were the GIFT-Eval datasets are stored",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="The batch size to use with the models",
    )
    parser.add_argument(
        "--data_batch_size",
        type=int,
        default=1024,
        help="The batch size to use with the data",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=512,
        help="The batch size to use during evaluation",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
        help="The maximum input length for the models",
    )
    args = parser.parse_args()
    main(args)