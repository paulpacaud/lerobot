#!/bin/bash

DATASETS=(
    "put_cube_in_spot_pointact"
    "stack_cups_pointact"
    "open_microwave_pointact"
    "put_banana_and_toy_in_plates_pointact"
    "put_socks_into_drawer_pointact"
)

for dataset in "${DATASETS[@]}"; do
    echo "Submitting job for dataset: $dataset"
    sbatch --export=ALL,INPUT_DATASET="$dataset" jobs/preprocessing/run_reformatting.sh
done
