import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from lunar_PINNversion.PINNmodel.model import PINN
from lunar_PINNversion.dataloader.dataLoader import Lunar_data_loader, Lunar_surface_data_loader
from lunar_PINNversion.dataloader.util import spherical_to_cartesian


def generate_collocation_points(n_points=60000):
    """Generate random collocation points"""
    domain = np.random.rand(n_points, 3)
    domain[:, 0] = domain[:, 0] * 1e5 + 1
    domain[:, 1] = domain[:, 1] * np.pi - np.pi / 2
    domain[:, 2] = domain[:, 2] * 2 * np.pi - np.pi

    domain_xyz = np.array([spherical_to_cartesian(el[0] / R_lunar, el[1], el[2])
                           for el in domain])
    return torch.tensor(domain_xyz, dtype=torch.float32).to(device)

def sample_boundary_data(boundary_points_full, B_measured_full, n_samples=60000):
    """Randomly sample boundary observations"""
    total_points = len(boundary_points_full)
    indices = torch.randperm(total_points)[:n_samples]  # Random sampling on GPU
    return boundary_points_full[indices], B_measured_full[indices]

if torch.cuda.is_available():
    device = torch.device("cuda")  # Select GPU
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")  # Fallback to CPU
    print("Using CPU")

R_lunar = 1737e3 # lunar radius
height_obs = 1e5 # height of observation
batch_size = 100000
num_sample_points = int(1e5)
data_filename = '/home/memolnar/Projects/lunarmagnetism/data/Moon_Mag_100km.txt'
surface_data_filename = '/home/memolnar/Projects/lunarmagnetism/data/surface_measurements.txt'

Lunar_data_loader1 = Lunar_data_loader(filename=data_filename)
Lunar_surface_data_loader1 = Lunar_surface_data_loader(filename=surface_data_filename)

# Random points inside the domain [0, 1]^3
domain = np.random.rand(100000, 3) # Random points inside the domain [0, 1]^3

domain[:, 0] = domain[:, 0] * 1e5 + R_lunar # r in kkm
domain[:, 1] = domain[:, 1] * np.pi  - np.pi/2# theta
domain[:, 2] = domain[:, 2] * 2 * np.pi  - np.pi# theta

domain_xyz = np.array([spherical_to_cartesian(el[0] / (R_lunar), el[1], el[2]) for
                       el in domain])
domain_xyz = torch.tensor(domain_xyz, dtype=torch.float32).to(device)


# BC from orbit
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

#Surface points:
boundary_points_surface = np.stack((Lunar_surface_data_loader1.x_coord,
                                        Lunar_surface_data_loader1.y_coord,
                                        Lunar_surface_data_loader1.z_coord), axis=-1) / (R_lunar)
boundary_points_surface = torch.tensor(boundary_points_surface, dtype=torch.float32).to(device)
B_measured_surface      = torch.tensor(Lunar_surface_data_loader1.data[2, :],
                               dtype=torch.float32).to(device)
M = boundary_points_surface.shape[0]
repeats = (int(num_sample_points) + M - 1) // M
boundary_points_surface_full = boundary_points_surface.repeat(repeats, 1)[:int(num_sample_points)]
B_measured_surface_full = B_measured_surface.repeat(repeats)[:int(num_sample_points)]
surface_dataset = TensorDataset(boundary_points_surface_full, B_measured_surface_full)
surface_loader = DataLoader(surface_dataset, batch_size=batch_size, shuffle=True)


# ============================================
# SAMPLE BOUNDARY DATA
# ============================================

# Initial sampling
boundary_dataset = TensorDataset(boundary_points_full, B_measured_full)
boundary_loader = DataLoader(boundary_dataset, batch_size=batch_size, shuffle=True)

# ============================================
# COLLOCATION POINTS
# ============================================

n_colloc = 60000
domain_xyz = generate_collocation_points(n_points=n_colloc)
inner_dataset = TensorDataset(domain_xyz)
inner_loader = DataLoader(inner_dataset, batch_size=batch_size, shuffle=True)

pinn = PINN(w0_initial=20, w0=15,
            device=device)
pinn = pinn.to(device)

pinn.train_pinn_with_surface_data(inner_loader, boundary_loader,
                                  surface_loader,
                Lunar_data_loader1,
                epochs=15000,
                lambda_bc=10.0, lambda_domain=1.,
                boundary_points_full = boundary_points_full,
                B_measured_full = B_measured_full,
                n_boundary_samples=num_sample_points,
                n_colloc_samples=num_sample_points,
                resample_colloc_every=1000, period_eval=200,
                resample_boundary_every=1000,
                initial_lr=1e-3, target_lr=1e-6, batch_size=batch_size,
                checkpoint_every=1000,  # Save checkpoint every N epochs
                resume_from=None,  # Path to checkpoint to resume from
                output_dir = "/home/memolnar/Projects/lunarmagnetism/Outputs/real_data/with_surface_data_v2_lr-3/")