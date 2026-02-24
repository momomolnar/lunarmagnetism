import os
import torch

from lunar_PINNversion.PINNmodel.model import PINN
from lunar_PINNversion.dataloader.dataLoader import (spherical_to_cartesian,
                                               concat_data_loaders,
                                               build_data_loaders,
                                               generate_collocation_loader,
                                               load_config)


def main(config):
    device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    R_lunar = config['R_lunar']
    batch_size = config['batch_size']
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # === DataLoader Setup ===
    orbital_loaders, surf_amp_loaders, surf_vec_loaders = build_data_loaders(
        config['data'], R_lunar, device, batch_size
    )

    # Optionally concatenate all datasets of each type
    orbital_loader = concat_data_loaders(orbital_loaders, batch_size)
    surface_amp_loader = concat_data_loaders(surf_amp_loaders, batch_size)
    surface_vec_loader = concat_data_loaders(surf_vec_loaders, batch_size)

    # === Model ===
    mcfg = config['model']
    pinn = PINN(
        w0_initial=mcfg.get('w0_initial', 30),
        w0=mcfg.get('w0', 30),
        hidden_dim=mcfg.get('hidden_dim', 128),
        num_hidden_layers=mcfg.get('num_hidden_layers', 5),
        device=device
    ).to(device)

    train_args = config['train']
    train_args['initial_lr'] = float(train_args['initial_lr'])
    train_args['target_lr']= float(train_args['target_lr'])

    # You may want to adapt arguments to your preferred convention!

    pinn.train_pinn(
        boundary_loader=orbital_loader,
        **train_args,
        output_dir=output_dir,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config .yaml file")
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)