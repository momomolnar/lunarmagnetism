import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from torch.optim.lr_scheduler import ExponentialLR

import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

criterion = nn.CrossEntropyLoss()
scaler = GradScaler()

if torch.cuda.is_available():
    device = torch.device("cuda")  # Select GPU
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")  # Fallback to CPU
    print("Using CPU")

resume_training = False

# -------------------------
# True potential parameters
# -------------------------
kx = 12
ky = 12
kz = np.sqrt(kx**2 + ky**2)

def true_phi_np(x, y, z):
    return np.cos(kx*x) * np.sin(ky*y) * np.exp(-kz*z)

def true_phi_torch(xyz):
    x = xyz[:,0:1]
    y = xyz[:,1:2]
    z = xyz[:,2:3]
    return (
        torch.cos(kx*x)
        * torch.sin(ky*y)
        * torch.exp(-kz*z)
    )

def true_H_torch(xyz):
    xyz.requires_grad_(True)
    phi = true_phi_torch(xyz)
    grad = torch.autograd.grad(phi, xyz, torch.ones_like(phi),
                               create_graph=True)[0]
    H = -grad
    return H


# -------------------------
# PINN Model
# -------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, in_dim=3, num_frequencies=4):
        super().__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        freq_bands = 1.2 ** torch.arange(num_frequencies)
        self.register_buffer("freq_bands", freq_bands)
    def forward(self, x):
        """
        x: (N,3)
        """
        enc = [x]
        for f in self.freq_bands:
            enc.append(torch.sin(2 * np.pi * f * x))
            enc.append(torch.cos(2 * np.pi * f * x))
        return torch.cat(enc, dim=-1)

    def out_dim(self):
        return self.in_dim * (1 + 2 * self.num_frequencies)


    def out_dim(self):
        return self.in_dim * (1 + 2 * self.num_frequencies)

class PINN(nn.Module):
    def __init__(self, pe_num_freqs =8,
                 layers=[64,128,128,128]):
        super().__init__()

        self.pe = PositionalEncoding(in_dim=3, num_frequencies=pe_num_freqs)

        in_dim = self.pe.out_dim()

        layer_dims = [in_dim] + layers + [1]

        net = []
        for i in range(len(layer_dims) - 1):
            net.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                net.append(nn.Tanh())
        self.net = nn.Sequential(*net)

    def forward(self, xyz):
        xyz_pe = self.pe(xyz)
        return self.net(xyz_pe)

model = PINN()

# -------------------------
# Laplacian
# -------------------------
def laplacian_phi(model, xyz):
    xyz.requires_grad_(True)
    phi = model(xyz)

    grad = torch.autograd.grad(phi, xyz, torch.ones_like(phi), create_graph=True)[0]
    phix, phiy, phiz = grad[:,0:1], grad[:,1:2], grad[:,2:3]

    phixx = torch.autograd.grad(phix, xyz, torch.ones_like(phix), create_graph=True)[0][:,0:1]
    phiyy = torch.autograd.grad(phiy, xyz, torch.ones_like(phiy), create_graph=True)[0][:,1:2]
    phizz = torch.autograd.grad(phiz, xyz, torch.ones_like(phiz), create_graph=True)[0][:,2:3]

    return phixx + phiyy + phizz


# -------------------------
# Training data
# -------------------------
N_f = 10000
xyz_f = torch.rand(N_f,3)  # (0,1)^3 collocation points
# xyz_f[:, 2] = 1

# Boundary points on z=0
N_b = 4000
x = torch.rand(N_b,1)
y = torch.rand(N_b,1)
z = torch.zeros_like(x) + 0.25
xyz_b = torch.cat([x,y,z], dim=1)

# True magnetic field on bc
H_true_b = true_H_torch(xyz_b).detach()
# Move model to GPU


# Move training data to GPU
xyz_f = xyz_f.to(device)
xyz_b = xyz_b.to(device)
H_true_b = H_true_b.to(device)

# -------------------------
# Training
# -------------------------

n_iterations = 1500000
target_lr = 1e-8
initial_lr = 1e-3
gamma = (target_lr / initial_lr) ** (1 / n_iterations)  # Decay factor per iteration


if resume_training:  # If you're resuming training
    # Load the model from a checkpoint
    n_iterations = 2000000
    target_lr = 1e-7
    initial_lr = 1e-6
    gamma = (target_lr / initial_lr) ** (1 / n_iterations)  # Decay factor per iteration

    checkpoint = torch.load("trained_model_checkpoint.pth", map_location=device)

    # Recreate the model, optimizer, and scheduler
    model = PINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = ExponentialLR(optimizer, gamma=gamma)

    # Load checkpoint data
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_iteration = checkpoint['iteration']

    print(f"Resuming training from iteration {start_iteration}")
else:
    # Train model from scratch
    model = PINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = ExponentialLR(optimizer, gamma=gamma)
    start_iteration = 0

for it in range(start_iteration, n_iterations):
    optimizer.zero_grad()

    # PDE loss
    lap = laplacian_phi(model, xyz_f)
    loss_pde = torch.mean(lap**2)

    # Boundary loss: match H = -∇Φ on z=0
    xyz_b.requires_grad_(True)
    phi_b = model(xyz_b)
    grad_b = torch.autograd.grad(phi_b, xyz_b, torch.ones_like(phi_b), create_graph=True)[0]
    H_pred_b = -grad_b
    loss_bc = torch.mean((H_pred_b - H_true_b)**2) / (torch.mean(torch.abs(H_true_b)) + 1e-6)

    loss = loss_pde + loss_bc
    loss.backward()
    optimizer.step()
    scheduler.step()

    optimizer.zero_grad()
    if it % 500 == 0:
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"iter {it}, loss = {loss.item():.4e}, PDE={loss_pde.item():.4e}, BC={loss_bc.item():.4e}, LR={current_lr:.1e}")

    # Save a checkpoint every 5000 iterations
    if it > 0 and it % 5000 == 0:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'iteration': it
        }, "trained_model_checkpoint.pth")
        print(f"Checkpoint saved at iteration {it}.")

print("Training complete.")

# Move the trained model to CPU
model_cpu = model.to("cpu")
print("Model has been moved back to CPU.")

torch.save(model_cpu.state_dict(), "trained_model.pth")
print("Model saved to trained_model.pth")



# Make grid
N = 80
x = np.linspace(0,1,N)
y = np.linspace(0,1,N)
X, Y = np.meshgrid(x, y)

# Evaluate PINN solution on z=0 and z=1
def evaluate_phi(z_value):
    pts = np.stack([X.flatten(), Y.flatten(), z_value*np.ones_like(X.flatten())], axis=1)
    pts_t = torch.tensor(pts, dtype=torch.float32)
    with torch.no_grad():
        phi_pred = model(pts_t).numpy().reshape(N, N)
    phi_true = true_phi_np(X, Y, z_value)
    return phi_pred, phi_true

phi_pred_0, phi_true_0 = evaluate_phi(0.0)
phi_pred_1, phi_true_1 = evaluate_phi(0.5)

# ---------------------------------------
# Plot 4 panels
# ---------------------------------------
plt.figure(figsize=(14,10))

# --- Bottom face z=0 ---
plt.subplot(2,2,1)
plt.title("PINN  Φ(x,y,0)")
plt.pcolormesh(X, Y, phi_pred_0, shading='auto')
plt.colorbar()

plt.subplot(2,2,2)
plt.title("True  Φ(x,y,0)")
plt.pcolormesh(X, Y, phi_true_0, shading='auto')
plt.colorbar()

# --- Top face z=1 ---
plt.subplot(2,2,3)
plt.title("PINN  Φ(x,y,0.5)")
plt.pcolormesh(X, Y, phi_pred_1, shading='auto')
plt.colorbar()

plt.subplot(2,2,4)
plt.title("True  Φ(x,y,0.5)")
plt.pcolormesh(X, Y, phi_true_1, shading='auto')
plt.colorbar()

plt.tight_layout()
plt.savefig('../Outputs/example_new.png')
plt.show()

def plot_magnetic_field_comparison(model, X, Y, z_value, N=80, save_as=None):
    """
    Function to plot the magnetic field components (H = -∇Φ) at a given height z_value
    for the true solution, the PINN solution, and their difference.

    Args:
        model (torch.nn.Module): The trained PINN model.
        X (np.ndarray): Meshgrid for x-coordinates.
        Y (np.ndarray): Meshgrid for y-coordinates.
        z_value (float): Height at which to compute the magnetic field.
        N (int): Resolution of the grid. Default is 80.
        save_as (str): Optional. Filename to save the plot. If None, the plot is only displayed.

    Returns:
        None: Displays or saves a figure with 9 subplots.
    """

    # ---------------------------------------
    # Compute the XYZ grid at height z_value
    # ---------------------------------------
    pts = np.stack([X.flatten(), Y.flatten(), z_value * np.ones_like(X.flatten())], axis=1)
    pts_t = torch.tensor(pts, dtype=torch.float32)

    # Compute magnetic field components
    # True magnetic field from the analytical potential
    Hx_true, Hy_true, Hz_true = compute_H_from_phi(true_phi_torch, pts_t)


    # Flatten each grid to 1D arrays
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z = z_value * np.ones_like(X)  # Fixed height z=0.5
    Z_flat = Z.flatten()

    # Combine into a single (N, 3) array
    xyz_np = np.stack([X_flat, Y_flat, Z_flat], axis=1)  # Shape: (N, 3)

    # Convert to PyTorch tensor
    xyz_b = torch.tensor(xyz_np, dtype=torch.float32)

    xyz_b.requires_grad_(True)
    phi_b = model(xyz_b)
    grad_b = torch.autograd.grad(phi_b, xyz_b, torch.ones_like(phi_b), create_graph=True)[0]
    H_pred = -grad_b
    Hx_pred = H_pred[:, 0]
    Hy_pred = H_pred[:, 1]
    Hz_pred = H_pred[:, 2]

    # Difference between true and predicted magnetic fields
    Hx_diff = Hx_true - Hx_pred
    Hy_diff = Hy_true - Hy_pred
    Hz_diff = Hz_true - Hz_pred

    # Reshape all components to grid (N, N)
    Hx_true_grid = Hx_true.reshape(N, N).detach().numpy()
    Hy_true_grid = Hy_true.reshape(N, N).detach().numpy()
    Hz_true_grid = Hz_true.reshape(N, N).detach().numpy()

    Hx_pred_grid = Hx_pred.reshape(N, N).detach().numpy()
    Hy_pred_grid = Hy_pred.reshape(N, N).detach().numpy()
    Hz_pred_grid = Hz_pred.reshape(N, N).detach().numpy()

    Hx_diff_grid = Hx_diff.reshape(N, N).detach().numpy()
    Hy_diff_grid = Hy_diff.reshape(N, N).detach().numpy()
    Hz_diff_grid = Hz_diff.reshape(N, N).detach().numpy()

    # --------- Plotting 3x3 Figure ---------

    plt.figure(figsize=(15, 12))

    # First row: True magnetic field components
    plt.subplot(3, 3, 1)
    plt.title("True Hx (z={})".format(z_value))
    plt.pcolormesh(X, Y, Hx_true_grid,
                   shading='auto', cmap='seismic',
                   vmin=np.nanquantile(Hx_true_grid, 0.02),
                   vmax=np.nanquantile(Hx_true_grid, 0.98))
    plt.colorbar()

    plt.subplot(3, 3, 2)
    plt.title("True Hy (z={})".format(z_value))
    plt.pcolormesh(X, Y, Hy_true_grid,
                   shading='auto', cmap='seismic',
                   vmin=np.nanquantile(Hy_true_grid, 0.02),
                   vmax=np.nanquantile(Hy_true_grid, 0.98))
    plt.colorbar()

    plt.subplot(3, 3, 3)
    plt.title("True Hz (z={})".format(z_value))
    plt.pcolormesh(X, Y, Hz_true_grid,
                   shading='auto', cmap='seismic',
                   vmin=np.nanquantile(Hz_true_grid, 0.02),
                   vmax=np.nanquantile(Hz_true_grid, 0.98))
    plt.colorbar()

    # Second row: PINN-predicted magnetic field components
    plt.subplot(3, 3, 4)
    plt.title("PINN Hx (z={})".format(z_value))
    plt.pcolormesh(X, Y, Hx_pred_grid,
                   shading='auto', cmap='seismic',
                   vmin=np.nanquantile(Hx_pred_grid, 0.02),
                   vmax=np.nanquantile(Hx_pred_grid, 0.98))
    plt.colorbar()

    plt.subplot(3, 3, 5)
    plt.title("PINN Hy (z={})".format(z_value))
    plt.pcolormesh(X, Y, Hy_pred_grid,
                   shading='auto', cmap='seismic',
                   vmin=np.nanquantile(Hy_pred_grid, 0.02),
                   vmax=np.nanquantile(Hy_pred_grid, 0.98))
    plt.colorbar()

    plt.subplot(3, 3, 6)
    plt.title("PINN Hz (z={})".format(z_value))
    plt.pcolormesh(X, Y, Hz_pred_grid,
                   shading='auto', cmap='seismic',
                   vmin=np.nanquantile(Hz_pred_grid, 0.02),
                   vmax=np.nanquantile(Hz_pred_grid, 0.98))
    plt.colorbar()

    # Third row: Difference between true and PINN-predicted magnetic field components
    plt.subplot(3, 3, 7)
    plt.title("Difference Hx (z={})".format(z_value))
    plt.pcolormesh(X, Y, Hx_diff_grid / np.abs(Hx_true_grid),
                   shading='auto', cmap='seismic',
                   norm=colors.SymLogNorm(linthresh=0.1, linscale=0.03,
                                          vmin=-5, vmax=5, base=10))
    plt.colorbar()

    plt.subplot(3, 3, 8)
    plt.title("Difference Hy (z={})".format(z_value))
    plt.pcolormesh(X, Y, Hy_diff_grid / np.abs(Hy_true_grid),
                   shading='auto', cmap='seismic',
                   norm=colors.SymLogNorm(linthresh=0.1, linscale=0.03,
                                          vmin=-5, vmax=5, base=10))
    plt.colorbar()

    plt.subplot(3, 3, 9)
    plt.title("Difference Hz (z={})".format(z_value))
    plt.pcolormesh(X, Y, Hz_diff_grid / np.abs(Hz_true_grid),
                   shading='auto', cmap='seismic',
                   norm=colors.SymLogNorm(linthresh=0.1, linscale=0.03,
                                          vmin=-5, vmax=5, base=10))
    plt.colorbar()

    plt.tight_layout()

    # Save or display the figure
    if save_as:
        plt.savefig(save_as)
    else:
        plt.show()

# ---------------------------------------
# Usage Example
# ---------------------------------------
def compute_H_from_phi(phi_fn, xyz):
    """
    Compute magnetic field components H = -∇Φ.
    Args:
        phi_fn (callable): Function to compute scalar potential Φ, either analytical or from PINN.
        xyz (torch.Tensor): Input XYZ grid points.
    Returns:
        Hx, Hy, Hz: Magnetic field components.
    """
    # Ensure the input tensor requires gradients
    xyz = xyz.clone().detach().requires_grad_(True)

    # Compute the scalar potential
    phi = phi_fn(xyz)

    # Compute gradients of the potential
    grad = torch.autograd.grad(outputs=phi, inputs=xyz,
                               grad_outputs=torch.ones_like(phi),
                               create_graph=True)[0]

    # Magnetic field H = -∇Φ
    H = -grad  # Negative gradient of scalar potential
    Hx, Hy, Hz = H[:, 0], H[:, 1], H[:, 2]
    return Hx, Hy, Hz
# Visualize magnetic field components at a specific height z=0.5

heights = np.linspace(0, 1, num=11)
for el in heights:
    plot_magnetic_field_comparison(
        model=model,
        X=X,
        Y=Y,
        z_value=el,
        N=80,
        save_as=f"../Outputs/magnetic_field_comparison_z_{el}.png"  # Set to None if you don't want to save
    )

