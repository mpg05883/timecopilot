import sys

import pytest

from timecopilot.models.benchmarks import (
    ADIDA,
    AutoARIMA,
    Prophet,
    SeasonalNaive,
    ZeroModel,
)
from timecopilot.models.benchmarks.ml import AutoLGBM
from timecopilot.models.benchmarks.neural import AutoNHITS, AutoTFT
from timecopilot.models.ensembles.median import MedianEnsemble
from timecopilot.models.foundational.chronos import Chronos
from timecopilot.models.foundational.moirai import Moirai
from timecopilot.models.foundational.timesfm import TimesFM
from timecopilot.models.foundational.toto import Toto


@pytest.fixture(autouse=True)
def disable_mps_session(monkeypatch):
    # Make torch.backends.mps report unavailable
    try:
        import torch

        monkeypatch.setattr(
            torch.backends.mps, "is_available", lambda: False, raising=False
        )
        monkeypatch.setattr(
            torch.backends.mps, "is_built", lambda: False, raising=False
        )
    except Exception:
        # torch might not be installed in some envs; ignore
        pass


models = [
    AutoLGBM(num_samples=2, cv_n_windows=2),
    AutoNHITS(
        num_samples=2,
        config=dict(
            max_steps=1,
            val_check_steps=1,
            input_size=12,
            mlp_units=3 * [[8, 8]],
        ),
    ),
    AutoTFT(
        num_samples=2,
        config=dict(
            max_steps=1,
            val_check_steps=1,
            input_size=12,
            hidden_size=8,
        ),
    ),
    AutoARIMA(),
    SeasonalNaive(),
    ZeroModel(),
    ADIDA(),
    Prophet(),
    Chronos(repo_id="amazon/chronos-bolt-tiny", alias="Chronos-Bolt"),
    Toto(context_length=256, batch_size=2),
    Moirai(
        context_length=256,
        batch_size=2,
        repo_id="Salesforce/moirai-1.1-R-small",
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
    Moirai(
        context_length=256,
        batch_size=2,
        repo_id="Salesforce/moirai-2.0-R-small",
    ),
]
if sys.version_info >= (3, 11):
    from timecopilot.models.foundational.tirex import TiRex

    models.append(TiRex())

if sys.version_info < (3, 13):
    from tabpfn_time_series import TabPFNMode

    from timecopilot.models.foundational.sundial import Sundial
    from timecopilot.models.foundational.tabpfn import TabPFN

    models.append(TabPFN(mode=TabPFNMode.MOCK))
    models.append(Sundial(context_length=256, num_samples=10))
