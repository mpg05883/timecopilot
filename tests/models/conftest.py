import sys

from timecopilot.models.benchmarks import (
    ADIDA,
    AutoARIMA,
    Prophet,
    SeasonalNaive,
    ZeroModel,
)
from timecopilot.models.ensembles.median import MedianEnsemble
from timecopilot.models.foundational.chronos import Chronos
from timecopilot.models.foundational.moirai import Moirai
from timecopilot.models.foundational.sundial import Sundial
from timecopilot.models.foundational.timesfm import TimesFM
from timecopilot.models.foundational.toto import Toto

models = [
    AutoARIMA(),
    SeasonalNaive(),
    ZeroModel(),
    ADIDA(),
    Prophet(),
]
if sys.version_info >= (3, 11):
    from timecopilot.models.foundational.tirex import TiRex

    models.append(TiRex())

if sys.version_info < (3, 13):
    from tabpfn_time_series import TabPFNMode

    from timecopilot.models.foundational.tabpfn import TabPFN

    models.append(TabPFN(mode=TabPFNMode.MOCK))

models.extend(
    [
        Chronos(repo_id="amazon/chronos-t5-tiny", alias="Chronos-T5"),
        Chronos(repo_id="amazon/chronos-bolt-tiny", alias="Chronos-Bolt"),
        Toto(context_length=256, batch_size=2),
        Moirai(
            context_length=256,
            batch_size=2,
            repo_id="Salesforce/moirai-1.1-R-small",
        ),
        Moirai(
            context_length=256,
            batch_size=2,
            repo_id="Salesforce/moirai-moe-1.0-R-small",
            alias="Moirai-MoE",
        ),
        TimesFM(
            repo_id="google/timesfm-1.0-200m-pytorch",
            context_length=256,
        ),
        MedianEnsemble(
            models=[
                Chronos(repo_id="amazon/chronos-t5-tiny", alias="Chronos-T5"),
                Chronos(repo_id="amazon/chronos-bolt-tiny", alias="Chronos-Bolt"),
                SeasonalNaive(),
            ],
        ),
        Sundial(context_length=256),
    ]
)
