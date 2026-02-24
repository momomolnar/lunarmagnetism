#!/bin/bash
conda init
conda activate pytorch_env

cd /home/memolnar/Projects/lunarmagnetism

# Hardcoded list of YAML config files
# Just add/remove paths as needed!
CONFIGS=(
    "/home/memolnar/Projects/lunarmagnetism/lunar_PINNversion/config/inversion_combined_mag_ER.yaml"
    # Add further YAML files here
)

# Loop through each config file and run inversion_orbital_data.py
for CONFIG in "${CONFIGS[@]}"
do
    echo "==== Running inversion with config: $CONFIG ===="
    python -i -m lunar_PINNversion.inversion_orbital_data "$CONFIG"
    echo ""
done

