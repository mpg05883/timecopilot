import logging
from typing import Annotated

import typer

from timecopilot.gift_eval.eval import GIFTEval
from timecopilot.gift_eval.gluonts_predictor import GluonTSPredictor
from timecopilot.models.ensembles.median import MedianEnsemble
from timecopilot.models.foundation.moirai import Moirai
from timecopilot.models.foundation.sundial import Sundial
from timecopilot.models.foundation.toto import Toto

logging.basicConfig(level=logging.INFO)


app = typer.Typer()


@app.command()
def run_timecopilot(
    dataset_name: Annotated[
        str,
        typer.Option(help="The name of the dataset to evaluate"),
    ],
    term: Annotated[
        str,
        typer.Option(help="The term to evaluate"),
    ],
    output_path: Annotated[
        str,
        typer.Option(help="The directory to save the results"),
    ],
    storage_path: Annotated[
        str,
        typer.Option(help="The directory were the GIFT data is stored"),
    ],
):
    logging.info(f"Running {dataset_name} {term} {output_path}")
    batch_size = 64
    predictor = GluonTSPredictor(
        forecaster=MedianEnsemble(
            models=[
                Moirai(
                    repo_id="Salesforce/moirai-1.1-R-large",
                    batch_size=batch_size,
                ),
                Sundial(batch_size=batch_size),
                Toto(
                    context_length=1_024,
                    batch_size=batch_size,
                ),
            ],
            alias="TimeCopilot",
        ),
        max_length=4_096,
        # data batch size
        batch_size=1_024,
    )
    gifteval = GIFTEval(
        dataset_name=dataset_name,
        term=term,
        output_path=output_path,
        storage_path=storage_path,
    )
    gifteval.evaluate_predictor(predictor, batch_size=512)


if __name__ == "__main__":
    app()
