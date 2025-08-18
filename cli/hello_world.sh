#!/bin/bash

mkdir -p ./output/logs
source ./cli/utils/common.sh

source /sw/external/python/anaconda3/etc/profile.d/conda.sh
conda activate tcp

log_info "Starting hello_world.sh..."
python -m scripts.hello_world 