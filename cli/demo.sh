#!/bin/bash
#SBATCH --job-name=slsqp_ensemble_eval
#SBATCH --array=0-194
#SBATCH --partition=gpuA40x4     
#SBATCH --mem=200G     
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16   
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bdem-delta-gpu  
#SBATCH --time=24:00:00
#SBATCH --output=output/logs/%x/out/%A/%a.out
#SBATCH --error=output/logs/%x/err/%A/%a.err
#SBATCH --mail-user=mpgee@usc.edu
#SBATCH --mail-type=BEGIN,END,FAIL

mkdir -p ./output/logs
source ./cli/utils.sh
activate_timecopilot_env
log_info "Starting $(get_slurm_message)"

# Set SLURM_ARRAY_TASK_ID
ETT1_15T_SHORT_TASK_ID=36
M4_HOURLY_TASK_ID=5
DEFAULT_TASK_ID=$M4_HOURLY_TASK_ID
SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-$DEFAULT_TASK_ID}
export SLURM_ARRAY_TASK_ID

# Set opt_metric based on the task ID
opt_metric="mae"

if python -m pipeline.demo --opt_metric="${opt_metric}"; then
    log_info "Successfully finished $(get_slurm_message)!"
    log_error "No errors!"
    echo "[$(get_timestamp)] Done with $(get_slurm_message)" >"$(get_done_file)"
    exit 0
else
    log_error "Job failed for $(get_slurm_message)!" >&2
    exti 1
fi