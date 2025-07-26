from pathlib import Path

import pandas as pd
import pytest
from huggingface_hub import snapshot_download


@pytest.fixture(scope="session")
def cache_dir() -> Path:
    cache_dir = Path(".pytest_cache") / "gift_eval"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@pytest.fixture(scope="session")
def all_results_df(cache_dir: Path) -> pd.DataFrame:
    all_results_file = cache_dir / "seasonal_naive_all_results.csv"
    if not all_results_file.exists():
        all_results_df = pd.read_csv(
            "https://huggingface.co/spaces/Salesforce/GIFT-Eval/raw/main/results/seasonal_naive/all_results.csv"
        )
        all_results_df.to_csv(all_results_file, index=False)
    return pd.read_csv(all_results_file)


@pytest.fixture(scope="session")
def gift_eval_dir(cache_dir: Path) -> Path:
    snapshot_download(
        repo_id="Salesforce/GiftEval",
        repo_type="dataset",
        local_dir=cache_dir,
    )
    return cache_dir


@pytest.fixture(autouse=True)
def gift_eval_env(monkeypatch, gift_eval_dir):
    monkeypatch.setenv("GIFT_EVAL", gift_eval_dir)
