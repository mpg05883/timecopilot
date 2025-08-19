import os
from typing import Any, Literal

from omegaconf import DictConfig


def get_slurm_config() -> dict[str, str]:
    """
    Returns a dictionary with SLURM configuration parameters.
    """
    return {
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID", "N/A"),
            "job_name": os.environ.get("SLURM_JOB_NAME", "N/A"),
            "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", "N/A"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", "N/A"),
            "partition": os.environ.get("SLURM_JOB_PARTITION", "N/A"),
        }
    }


def get_tempo_eval_run_kwargs(cfg: DictConfig, dataset_name: str, term: Literal["short", "medium", "long"],) -> dict[str, str | list[str]]:
    """
    Returns a dictionary with initialization arguments for a TEMPO evaluation
    run.
    """
    return {
        "name": f"{cfg.model.name}-eval-{dataset_name.replace('/', '_')}-{term}",
        "tags": [
            f"model={cfg.model.name.split('_', 1)[0]}",
            cfg.model.type,
            f"dataset={dataset_name}",
            f"term={term}",
        ],
        "job_type": "eval",
        "group": "tempo_eval",
    }
    
def get_checkpoint_artifact_kwargs(
    model_name: str,
    dataset_name: str,
    term: Literal["short", "medium", "long"],
) -> dict[str, str]:
    """
    Returns a dictionary with the arguments for the checkpoint artifact.
    
    Args:
        model_name (str): Name of the model.
        dataset_name (str): Name of the dataset.
        term (Literal["short", "medium", "long"]): Term of the dataset the 
            checkpoint was trained on.
            
    Returns:
        dict[str, str]: A dictionary containing the artifact name, type,
            description, and metadata.
    """
    return {
        "name": f"{model_name}-{dataset_name.replace('/', '_')}-{term}-checkpoint",
        "type": "checkpoint",
        "description": f"Checkpoint for {model_name} on {dataset_name} ({term}-term)",
        "metadata": {
            "dataset": dataset_name,
            "term": term,
            "model": model_name,
        }
    }

def get_results_artifact_kwargs(
    model_name: str,
    dataset_name: str,
    term: Literal["short", "medium", "long"],
) -> dict[str, str]:
    """
    Returns a dictionary with the arguments for the results artifact.
    
    Args:
        model_name (str): Name of the model.
        dataset_name (str): Name of the dataset.
        term (Literal["short", "medium", "long"]): Term for which the results
            are generated.
            
    Returns:
        dict[str, str]: A dictionary containing the artifact name, type,
            description, and metadata.
    """
    return {
        "name": f"{model_name}-{dataset_name}-{term}-results",
        "type": "results",
        "description": f"Evaluation results for {model_name} on {dataset_name} ({term}-term)",
        "metadata": {
            "dataset": dataset_name,
            "term": term,
            "model": model_name,
        }
    }   
