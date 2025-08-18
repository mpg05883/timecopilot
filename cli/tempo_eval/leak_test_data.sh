#!/bin/bash
#SBATCH --job-name=train_test_leak_test_data
#SBATCH --array=0-96
#SBATCH --partition=gpuA40x4     
#SBATCH --mem=200G     
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16   
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bcqc-delta-gpu  
#SBATCH --time=6:00:00
#SBATCH --output=output/logs/%x/out/%A/%a.out
#SBATCH --error=output/logs/%x/err/%A/%a.err
#SBATCH --mail-user=mpgee@usc.edu
#SBATCH --mail-type=BEGIN,END,FAIL

mkdir -p ./output/logs
source ./cli/utils/utils.sh
activate_tempo_env
log_info "Starting $(get_slurm_message)"

imputation_method=dummy_value

model_type=leak_test_data

notes=""
mode=disabled

export WANDB_NOTES="$notes"
export WANDB_MODE="$mode"

if python -m scripts.tempo_eval.py ../conf \
    "data=train_test/data" \
    "imputation_method=${imputation_method}" \
    "model=tempo/${model_type}" \
    "task_id=${SLURM_ARRAY_TASK_ID:-0}"; then
    
    log_info "Successfully finished $(get_slurm_message)!"
    log_error "No errors!"
    echo "[$(get_timestamp)] Done with $(get_slurm_message)" >"$(get_done_file)"
else
    log_error "Job failed for $NAME ($TERM)" >&2
fi
