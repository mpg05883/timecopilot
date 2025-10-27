import os
from pathlib import Path

from dotenv import load_dotenv


def resolve_storage_path(storage_env_var: str = "GIFT_EVAL") -> Path:
    """
    Resolve the directory path where GIFT-Eval datasets are stored.

    Args:
        storage_env_var (str): Name of the environment variable that contains
            the data directory's name. Defaults to "GIFT_EVAL".

    Returns:
        Path: Full path to the dataset storage directory.
    """
    # Resolve the path to the root directory
    root_dir = Path(__file__).resolve().parent.parent.parent

    # Load the storange environment variable from the .env file
    load_dotenv()

    # Get the data directory's name from the environment variable
    data_dir = os.getenv(storage_env_var, "data")

    # Return the full path to ./<data_dir>
    return root_dir / data_dir


def resolve_metadata_path() -> Path:
    """
    Resolve the directory path where GIFT-Eval dataset metadata is stored.
    """
    # Resolve path to src directory
    src_dir = Path(__file__).resolve().parent.parent
    return src_dir / "data" / "meta" / "metadata.json"

def resolve_dataset_properties_path() -> Path:
    """
    Resolve the directory path where GIFT-Eval dataset properties are stored.
    """
    # Resolve path to src directory
    src_dir = Path(__file__).resolve().parent.parent
    return src_dir / "data" / "meta" / "dataset_properties.json"

def resolve_output_path(
    alias: str,
    dataset_config: str,
    output_dir: str = "results",
) -> Path:
    """
    Resolve the directory path where results are saved.

    Args:
        alias (str): Name of the forecaster.
        dataset_config (str): The dataset's configuration formatted as:
            <name>/<freq>/<term>.
        output_dir (str): Directory to save results. Defaults to "results".

    Returns:
        Path: Full path to the output directory.
    """
    # Resolve the path to the root directory
    root_dir = Path(__file__).resolve().parent.parent.parent

    # Create a directory path to ./output_dir/alias/dataset_config
    output_path = root_dir / output_dir / alias / dataset_config
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path
