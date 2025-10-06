import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv


def resolve_storage_path(
    storage_env_var: str = "GIFT_EVAL",
    split_name: Literal["train_test", "pretrain"] = "train_test",
) -> Path:
    load_dotenv()

    # Get the data directory's name from the environment variable
    data_dir = os.getenv(storage_env_var, "data")

    # Resolve the path to the root directory
    root_dir = Path(__file__).resolve().parent.parent.parent
    
    # Return the full path to ./<data_dir>/<split_name>     
    return root_dir / data_dir / split_name


def resolve_output_path(
        output_dir: str = "results",
        dataset_config: str | None = None,
    ) -> Path:
    # Resolve the path to the root directory
    root_dir = Path(__file__).resolve().parent.parent.parent

    # Create a path to ./<output_dir> and ensure it exists
    output_path = root_dir / output_dir
    if dataset_config is not None:
        output_path = output_path / dataset_config
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Return the full path to ./<output_dir> or ./<output_dir>/<dataset_config>
    return output_path
