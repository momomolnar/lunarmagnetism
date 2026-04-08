import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from lunar_PINNversion.PINNmodel.model import PINN
from lunar_PINNversion.dataloader.util import spherical_to_cartesian
import wandb

if torch.cuda.is_available():
    device = torch.device("cuda")  # Select GPU
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")  # Fallback to CPU
    print("Using CPU")


domain_xyz = np.array([spherical_to_cartesian(el[0] / (R_lunar), el[1], el[2]) for
                       el in domain])
domain_xyz = torch.tensor(domain_xyz, dtype=torch.float32).to(device)

boundary_points_full = np.stack((Lunar_data_loader1.x_coord,
                                        Lunar_data_loader1.y_coord,
                                        Lunar_data_loader1.z_coord), axis=-1) / (R_lunar)

boundary_points_full = torch.tensor(boundary_points_full, dtype=torch.float32).to(device)

B_measured_full = np.stack((Lunar_data_loader1.b_x,
                             Lunar_data_loader1.b_y,
                             Lunar_data_loader1.b_z), axis=-1)
B_measured_full = torch.tensor(B_measured_full, dtype=torch.float32).to(device)

print(f"Boundary points shape: {boundary_points_full.shape}")
print(f"B measured shape: {B_measured_full.shape}")
# ============================================
# SAMPLE BOUNDARY DATA
# ============================================
def sample_boundary_data(boundary_points_full, B_measured_full, n_samples=10000):
    """Randomly sample boundary observations"""
    total_points = len(boundary_points_full)
    indices = torch.randperm(total_points)[:n_samples]  # Random sampling on GPU
    return boundary_points_full[indices], B_measured_full[indices]

# Initial sampling
boundary_dataset = TensorDataset(boundary_points_full, B_measured_full)
boundary_loader = DataLoader(boundary_dataset, batch_size=4096, shuffle=True)


# ============================================
# COLLOCATION POINTS
# ============================================
def generate_collocation_points(n_points=10000):
    """Generate random collocation points"""
    domain = np.random.rand(n_points, 3)
    domain[:, 0] = domain[:, 0] * 1e5 + R_lunar
    domain[:, 1] = domain[:, 1] * np.pi - np.pi / 2
    domain[:, 2] = domain[:, 2] * 2 * np.pi - np.pi

    domain_xyz = np.array([spherical_to_cartesian(el[0] / R_lunar, el[1], el[2])
                           for el in domain])
    return torch.tensor(domain_xyz, dtype=torch.float32).to(device)

n_colloc = 4000
domain_xyz = generate_collocation_points(n_points=n_colloc)
inner_dataset = TensorDataset(domain_xyz)
inner_loader = DataLoader(inner_dataset, batch_size=4096, shuffle=True)

pinn = PINN(pe_num_freqs=6, base_freq = 1.4, device=device)
pinn = pinn.to(device)

pinn.train_pinn(inner_loader, boundary_loader,
                Lunar_data_loader1,
                epochs=10000,
                lambda_bc=1.0, lambda_domain=1.,
                boundary_points_full = boundary_points_full,
                B_measured_full = B_measured_full,
                initial_lr=1e-3, target_lr=1e-6,
                output_dir = "/home/memolnar/Projects/lunarmagnetism/Outputs/real_data/surface_only_v1/")
wandb.finish()