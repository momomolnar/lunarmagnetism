# Lunar Magnetic Field PINN Inversion

This repository contains a flexible, modular framework for performing **physics-informed neural network (PINN) inversions** of the lunar magnetic field with multiple heterogeneous data sources. It enables simultaneously using orbital vector measurements, surface amplitude measurements, and surface magnetic vector measurements from multiple missions and datasets.

---

## Features

- **Supports Multiple Data Sources:** LRO, Kaguya, Apollo, MagROVER, etc.
- **Data Modalities:** Orbital vector, surface amplitude, surface vector, each with a `source` descriptor for every file.
- **Configurable Experiments:** All experiment parameters are specified in a YAML configuration file.
- **Customizable Data Loaders:** Dispatcher pattern for source-specific loading logic.
- **Reproducible Results:** Experiments and outputs tracked via YAML config and output directories.
- **Scalable Design:** Add new sources or data types with minimal code change.

---

## Codebase Structure
```text 
.
├── inversion.py                   # Main inversion pipeline using YAML config
├── data_loader_dispatcher.py      # Dispatches custom logic for each data type/source
├── lunar_PINNversion/
│   ├── model.py                   # PINN model definition
│   ├── dataloader/
│   │   └── dataLoader.py          # Data loaders for each mission/type
│   └── ...                        # Utility, plotting, evaluation, etc.
├── experiments/
│   └── your_config.yaml           # Example YAML configs for running inversions
├── run_inversion.sh               # Shell script to run one or more batch experiments
└── README.txt                     # You are here
```
---

## Getting Started

### 1. Prerequisites

- Python 3.8+
- PyTorch >= 1.9
- numpy, pyyaml, matplotlib, tqdm, (optional: wandb, etc.)
- (Recommended) Use a virtual environment

pip install torch numpy pyyaml matplotlib tqdm wandb

### 2. Prepare Your Data

- Place data files (orbital, surface amplitude, surface vector) anywhere you like.
- Supported sources: LRO, Kaguya, Apollo, MagROVER, etc.
- See the [YAML Configuration](#yaml-experiment-configuration) example below for file specifications.

### 3. Write a YAML Experiment Configuration

Example (experiments/my_lunar_inv.yaml):

device: "cuda"
output_dir: "./Outputs/exp1/"
R_lunar: 1737e3
batch_size: 65536

data:
  orbital:
    - path: "/data/Moon_Mag_100km_LRO.txt"
      source: "LRO"
    - path: "/data/Moon_Mag_100km_Kaguya.txt"
      source: "Kaguya"
  surface_amplitude:
    - path: "/data/surface_amp_Apollo.txt"
      source: "Apollo"
  surface_vector:
    - path: "/data/surface_vec_MagROVER.txt"
      source: "MagROVER"

collocation:
  n_points: 100000
  r_offset: 0

model:
  w0_initial: 20
  w0: 15
  pe_num_freqs: 6
  base_freq: 1.4
  hidden_dim: 128
  num_hidden_layers: 5

train:
  epochs: 15000
  lambda_bc: 10.0
  lambda_domain: 1.0
  n_boundary_samples: 100000
  n_colloc_samples: 100000
  resample_boundary_every: 1000
  resample_colloc_every: 1000
  initial_lr: 1e-3
  target_lr: 1e-6
  checkpoint_every: 1000
  period_eval: 200

> See the experiments/ folder for more samples.

---

## Running an Inversion

### Single experiment

python inversion.py experiments/my_lunar_inv.yaml

### Batch mode: multiple YAMLs

List your YAML files in run_inversion.sh, then run:

bash run_inversion.sh

---

## Adding New Data Sources/Types

- **New spacecraft/mission:**  
  Implement or adapt a loader in lunar_PINNversion/dataloader/dataLoader.py,  
  register it in data_loader_dispatcher.py  
  and refer to it by a unique `source` name in the YAML `data` section.

- **New data modality:**  
  Update the YAML and expand the dispatcher logic to handle the new type.  
  Adjust training/model code if needed.

---

## Notes

- **Collocation points** are generated according to the config's collocation section.
- **Outputs, checkpoints, and logs** are saved to output_dir specified in your YAML.
- Open a GitHub issue for help or bug reports.

---

## Citation

If you use this codebase in your research, please cite and acknowledge the contributors (to be filled in).

---

## License

MIT license. 

---

_For further details, see code docstrings and example configs in the experiments/ folder._
