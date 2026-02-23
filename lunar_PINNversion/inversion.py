import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

# ---- Import your data loaders & model as before ----
from lunar_PINNversion.model import PINN
from lunar_PINNversion.dataloader.dataLoader import Lunar_data_loader, Lunar_surface_data_loader
from lunar_PINNversion.dataloader.util import spherical_to_cartesian


# ==== Data Loader Utilities ====

def load_orbital_data(file_path, R_lunar, device):
    loader = Lunar_data_loader(filename=file_path)
    pts = np.stack([loader.x_coord, loader.y_coord, loader.z_coord], axis=-1) / R_lunar
    B = np.stack([loader.b_x, loader.b_y, loader.b_z], axis=-1)
    return torch.tensor(pts, dtype=torch.float32).to(device), torch.tensor(B, dtype=torch.float32).to(device)


def load_surface_amp_data(file_path, R_lunar, device):
    loader = Lunar_surface_data_loader(filename=file_path)
    pts = np.stack([loader.x_coord, loader.y_coord, loader.z_coord], axis=-1) / R_lunar
    amp = torch.tensor(loader.B, dtype=torch.float32).to(device)
    return torch.tensor(pts, dtype=torch.float32).to(device), amp


def load_surface_vector_data(file_path, R_lunar, device):
    loader = Lunar_surface_data_loader(filename=file_path)
    pts = np.stack([loader.x_coord, loader.y_coord, loader.z_coord], axis=-1) / R_lunar
    # You might have to adapt b_x, b_y, b_z source depending on the file!
    B = np.stack([loader.b_x, loader.b_y, loader.b_z], axis=-1)
    return torch.tensor(pts, dtype=torch.float32).to(device), torch.tensor(B, dtype=torch.float32).to(device)


def make_loader_from_points(points, targets, batch_size, shuffle=True):
    dataset = TensorDataset(points, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def build_all_data_loaders(config, device):
    R_lunar = config['R_lunar']
    batch_size = config['batch_size']
    data_spec = config['data']

    # Orbital data
    orbital_loaders = []
    for file in data_spec.get('orbital', []):
        pts, vecs = load_orbital_data(file, R_lunar, device)
        orbital_loaders.append(make_loader_from_points(pts, vecs, batch_size, shuffle=True))

    # Surface field amplitude data
    surface_amp_loaders = []
    for file in data_spec.get('surface_amplitude', []):
        pts, amps = load_surface_amp_data(file, R_lunar, device)
        surface_amp_loaders.append(make_loader_from_points(pts, amps, batch_size, shuffle=True))

    # Surface vector data
    surface_vec_loaders = []
    for file in data_spec.get('surface_vector', []):
        pts, vecs = load_surface_vector_data(file, R_lunar, device)
        surface_vec_loaders.append(make_loader_from_points(pts, vecs, batch_size, shuffle=True))

    return {
        'orbital': orbital_loaders,
        'surface_amplitude': surface_amp_loaders,
        'surface_vector': surface_vec_loaders,
    }


def generate_collocation_loader(n_points, R_lunar, spherical_to_cartesian, device, batch_size, r_offset=0):
    domain = np.random.rand(n_points, 3)
    domain[:, 0] = domain[:, 0] * 1e5 + R_lunar + r_offset
    domain[:, 1] = domain[:, 1] * np.pi - np.pi / 2
    domain[:, 2] = domain[:, 2] * 2 * np.pi - np.pi
    domain_xyz = np.array([spherical_to_cartesian(el[0] / R_lunar, el[1], el[2]) for el in domain])
    domain_xyz = torch.tensor(domain_xyz, dtype=torch.float32).to(device)
    dataset = TensorDataset(domain_xyz)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def concat_loaders(loader_list, batch_size):
    if not loader_list:
        return None
    all_data = ConcatDataset([dl.dataset for dl in loader_list])
    return DataLoader(all_data, batch_size=batch_size, shuffle=True)


# ========== Main Script Logic ==========

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(config):
    # --- Device/dirs
    device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    R_lunar = config['R_lunar']
    batch_size = config['batch_size']

    # --- Data Loaders
    loaders = build_all_data_loaders(config, device)
    colloc_loader = generate_collocation_loader(
        n_points=config['collocation']['n_points'],
        R_lunar=R_lunar,
        spherical_to_cartesian=spherical_to_cartesian,
        device=device,
        batch_size=batch_size
    )
    # Combine loaders if desired (optional), else keep as lists
    if loaders['orbital']:
        boundary_loader = concat_loaders(loaders['orbital'], batch_size)
    else:
        boundary_loader = None
    if loaders['surface_amplitude']:
        surface_amp_loader = concat_loaders(loaders['surface_amplitude'], batch_size)
    else:
        surface_amp_loader = None
    if loaders['surface_vector']:
        surface_vec_loader = concat_loaders(loaders['surface_vector'], batch_size)
    else:
        surface_vec_loader = None

    # --- Model Init (adaptable for PINN or SIREN style)
    model_cfg = config['model']
    pinn = PINN(
        w0_initial=model_cfg.get('w0_initial', 30),
        w0=model_cfg.get('w0', 30),
        pe_num_freqs=model_cfg.get('pe_num_freqs', 6),
        base_freq=model_cfg.get('base_freq', 1.0),
        device=device
    ).to(device)

    # --- Training kwargs
    train_args = config['train']
    train_kwargs = dict(
        epochs=train_args['epochs'],
        lambda_bc=train_args['lambda_bc'],
        lambda_domain=train_args['lambda_domain'],
        n_boundary_samples=train_args.get('n_boundary_samples', batch_size),
        n_colloc_samples=train_args.get('n_colloc_samples', batch_size),
        resample_boundary_every=train_args.get('resample_boundary_every', 1000),
        resample_colloc_every=train_args.get('resample_colloc_every', 1000),
        initial_lr=train_args['initial_lr'],
        target_lr=train_args['target_lr'],
        checkpoint_every=train_args['checkpoint_every'],
        period_eval=train_args.get('period_eval', 500),
        output_dir=output_dir,
        batch_size=batch_size
    )

    # --- Model Training: Pass relevant DataLoaders
    # Here, we give all 3 types; adjust as needed for your train_pinn function!
    # For surface loaders: supply both amplitude and vector if available
    pinn.train_pinn(
        inner_loader=colloc_loader,
        boundary_loader=boundary_loader,
        surface_amp_loader=surface_amp_loader,
        surface_vec_loader=surface_vec_loader,
        **train_kwargs
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config .yaml file")
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)