import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def create_collocation_data(x_c, y_c, z_c):
    """Create a collocation tensor from x/y/z arrays."""
    return torch.tensor(np.hstack([x_c, y_c, z_c]), requires_grad=True, dtype=torch.float32)

def create_boundary_data(x_b, y_b, z_b, B_bc_vals):
    """Create boundary point and boundary field tensors."""
    bc_pts = torch.tensor(np.hstack([x_b, y_b, z_b]), requires_grad=True, dtype=torch.float32)
    bc_vals = torch.tensor(B_bc_vals, dtype=torch.float32, requires_grad=True)
    return bc_pts, bc_vals

def colloc_data_loader(colloc_data, batch_size=32):
    """Wrap collocation points in a DataLoader."""
    dataset = TensorDataset(colloc_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def boundary_data_loader(bc_data, bc_vals, batch_size=32):
    """Wrap boundary points and labels in a DataLoader."""
    dataset = TensorDataset(bc_data, bc_vals)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def create_dataloaders(h=1.0, N_colloc=10000, N_bc = 4096, batch_size=4096):
    """Generate synthetic training sets and return collocation/boundary loaders."""
    x_c, y_c, z_c, x_b, y_b, z_b, B_bc_vals = create_synthetic_set(h, N_colloc, N_bc)

    colloc_data = create_collocation_data(x_c, y_c, z_c)
    train_colloc_loader = colloc_data_loader(colloc_data, batch_size=batch_size)

    bc_data, bc_vals = create_boundary_data(x_b, y_b, z_b, B_bc_vals)
    train_boundary_loader = boundary_data_loader(bc_data, bc_vals, batch_size=batch_size)

    return train_colloc_loader, train_boundary_loader

def true_phi(x, y, z, kx=4.0, ky=4.0):
    """Analytical scalar potential used to synthesize training data."""
    return np.exp(-np.sqrt(kx**2 + ky**2) * z) * np.sin(kx*x) * np.cos(ky*y)

def true_B(x, y, z, kx=4.0, ky=4.0):
    """Analytical magnetic field derived from the synthetic scalar potential."""
    k = np.sqrt(kx**2 + ky**2)
    Bx =  (kx * np.cos(kx*x) * np.cos(ky*y)) * np.exp(-k*z)
    By =  - (ky * np.sin(kx*x) * np.sin(ky*y)) * np.exp(-k*z)
    Bz =  - k * np.sin(kx*x) * np.cos(ky*y) * np.exp(-k*z)

    # Bx = By = Bz = np.ones_like((Bx))

    return np.stack([Bx / np.std(Bx),
                     By / np.std(By),
                     Bz / np.std(Bz)], axis=-1)

def create_synthetic_set(h=1.0, N_colloc=3000, N_bc = 10000):
    """Sample synthetic collocation and boundary sets with analytical boundary labels."""

    # Collocation points
    x_c = np.random.rand(N_colloc, 1)
    y_c = np.random.rand(N_colloc, 1)
    z_c = h * np.random.rand(N_colloc, 1)

    # Boundary points
    x_b = np.random.rand(N_bc, 1)
    y_b = np.random.rand(N_bc, 1)
    z_b = h * np.ones((N_bc, 1))

    # True values at boundary
    B_bc_vals = true_B(x_b, y_b, z_b)

    return x_c, y_c, z_c, x_b, y_b, z_b, B_bc_vals
