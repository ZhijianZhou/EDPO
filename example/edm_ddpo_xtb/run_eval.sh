#!/bin/bash
cd ./Model/EDM

MODEL_PATHS=(
    "../../exp/edm-xtb"
)

STEP_SETS=(
    "1000"
)

MODEL_NAME=(
    "generative_model_ema.npy"
)

N_SAMPLES=100

for model_path in "${MODEL_PATHS[@]}"; do
    for steps in "${STEP_SETS[@]}"; do
        for model_name in "${MODEL_NAME[@]}"; do
            echo "Running eval_analyze.py with model: $model_path, steps: $steps, MODEL_NAME $model_name"
            python eval_analyze.py --model_path "$model_path" --n_samples "$N_SAMPLES" --steps "$steps" --save_to_xyz True --model_name "${model_name}"
        done
    done
done

echo "All evaluations completed!"