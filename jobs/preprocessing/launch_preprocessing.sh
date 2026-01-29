#!/bin/bash

DATASETS=(
    "put_cube_in_spot"
    "stack_cups"
    "open_microwave"
    "put_banana_and_toy_in_plates"
    "put_socks_into_drawer"
    "move_plates_from_rack_to_box"
    "put_items_in_box"
)

for dataset in "${DATASETS[@]}"; do
    echo "Submitting job for dataset: $dataset"
    sbatch --export=ALL,INPUT_DATASET="$dataset" jobs/preprocessing/run_full_pipeline.sh
done
