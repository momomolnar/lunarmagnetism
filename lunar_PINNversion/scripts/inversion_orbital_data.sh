#!/bin/bash

python activate pytorchenv
cd /home/memolnar/Projects/lunarmagnetism/lunar_PINNversion

# Hardcoded list of YAML config files
# Just add/remove paths as needed!
CONFIGS=(
    "experiments/your_experiment1.yaml"
    "experiments/your_experiment2.yaml"
    "experiments/your_surface_test.yaml"
    # Add further YAML files here
)

# Loop through each config file and run inversion.py
for CONFIG in "${CONFIGS[@]}"
do
    echo "==== Running inversion with config: $CONFIG ===="
    python inversion.py "$CONFIG"
    echo ""
done

