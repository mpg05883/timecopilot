import os
from typing import Any

from omegaconf import DictConfig


def get_slurm_config() -> dict[str, str]:
    """
    Returns a dictionary with SLURM configuration parameters.
    """
    return {
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID", "-1"),
            "job_name": os.environ.get("SLURM_JOB_NAME", "bash"),
            "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", "-1"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", "0"),
            "partition": os.environ.get("SLURM_JOB_PARTITION", "local"),
        }
    }


def get_tempo_eval_run_kwargs(cfg: DictConfig) -> dict[str, Any]:
    """
    Returns a dictionary with initialization arguments for a TEMPO evaluation
    run.
    """
    return {
        "name": f"{cfg.model.name}-eval-{cfg.dataset_name}-{cfg.term}",
        "tags": [
            f"model={cfg.model.name.split('_', 1)[0]}",
            cfg.model.type,
            f"dataset={cfg.dataset_name}",
            f"term={cfg.term}",
        ],
        "job_type": "eval",
        "group": "tempo_eval",
    }
