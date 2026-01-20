import torch
import copy
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from torch.optim.lr_scheduler import ExponentialLR
from evaluation.mollweide_plot import plot_mollweide_map
from lunar_PINNversion.dataloader.dataLoader import Lunar_data_loader, Lunar_surface_data_loader
from lunar_PINNversion.dataloader.util import spherical_to_cartesian
from torch.utils.data import DataLoader, TensorDataset


R_lunar = 1701e3 # lunar radius
height_obs = 1e5 # height of observation

if torch.cuda.is_available():
    device = torch.device("cuda")  # Select GPU
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")  # Fallback to CPU
    print("Using CPU")


data_filename = '/home/memolnar/Projects/lunarmagnetism/data/Moon_Mag_100km.txt'
surface_data_filename = '/home/memolnar/Projects/lunarmagnetism/data/surface_measurements.txt'

Lunar_data_loader1 = Lunar_data_loader(filename=data_filename)
# Lunar_surface_data_loader1 = Lunar_surface_data_loader(filename=surface_data_filename)

# Random points inside the domain [0, 1]^3

domain = np.random.rand(1000, 3) # Random points inside the domain [0, 1]^3

domain[:, 0] = domain[:, 0] * 1e5 + R_lunar # r in kkm
domain[:, 1] = domain[:, 1] * np.pi  - np.pi/2# theta
domain[:, 2] = domain[:, 2] * 2 * np.pi  - np.pi# theta

domain_xyz = np.array([spherical_to_cartesian(el[0] / (R_lunar), el[1], el[2]) for
                       el in domain])
domain_xyz = torch.tensor(domain_xyz, dtype=torch.float32).to(device)

boundary_points = np.stack((Lunar_data_loader1.x_coord,
                            Lunar_data_loader1.y_coord,
                            Lunar_data_loader1.z_coord), axis=-1) / (R_lunar)

boundary_points = torch.tensor(boundary_points, dtype=torch.float32).to(device)


B_surface_measured = np.stack((Lunar_surface_data_loader1.B), axis=-1)
B_surface_measured = torch.tensor(B_surface_measured,
                                  dtype=torch.float32).to(device)



boundary_surface_points = np.stack((Lunar_surface_data_loader1.x_coord,
                                          Lunar_surface_data_loader1.y_coord,
                                          Lunar_surface_data_loader1.z_coord),
                                    axis=-1) / (R_lunar)

boundary_surface_points = torch.tensor(boundary_surface_points,
                                       dtype=torch.float32).to(device)

B_measured = np.stack((Lunar_data_loader1.b_x,
                             Lunar_data_loader1.b_y,
                             Lunar_data_loader1.b_z), axis=-1)
B_measured = torch.tensor(B_measured, dtype=torch.float32).to(device)

inner_dataset = TensorDataset(domain_xyz)
inner_loader = DataLoader(inner_dataset, batch_size=24024,shuffle=True)

boundary_dataset = TensorDataset(boundary_points, B_measured)
boundary_loader = DataLoader(boundary_dataset, batch_size=24024, shuffle=True)

boundary_surface_dataset = TensorDataset(boundary_surface_points,
                                         B_surface_measured)
boundary_surface_loader = DataLoader(boundary_surface_dataset,
                                     batch_size=24024, shuffle=True)



resume_training = False


# -------------------------
# PINN Model
# -------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, in_dim=3, num_frequencies=4, base_freq = 0.98):
        super().__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        freq_bands = base_freq ** torch.arange(num_frequencies)
        print(f"freq bands are: {freq_bands}")
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
    def __init__(self, pe_num_freqs =6, base_freq = 1.3,
                 layers=[64,128,128,128,128]):
        super().__init__()
        # in_dim = 3

        self.pe = PositionalEncoding(in_dim=3, num_frequencies=pe_num_freqs, base_freq=base_freq)

        in_dim = self.pe.out_dim()

        layer_dims = [in_dim] + layers + [1]

        net = []
        for i in range(len(layer_dims) - 1):
            net.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                net.append(nn.Tanh())
        self.net = nn.Sequential(*net)

    def forward(self, xyz):
        xyz_pe = self.pe.forward(xyz)
        return self.net(xyz_pe)
        # return self.net(xyz)

# Evaluate PINN solution on z=0 and z=1

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

model = PINN(pe_num_freqs =6, base_freq = 1.4,)

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

# Move model to GPU

n_iterations = 60010
target_lr = 1e-5
initial_lr = 3e-3
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
    xyz_b1.requires_grad_(True)
    phi_b1 = model(xyz_b1)
    grad_b1 = torch.autograd.grad(phi_b1, xyz_b1, torch.ones_like(phi_b1), create_graph=True)[0]
    H_pred_b1 = -grad_b1
    loss_bc1 = torch.mean((H_pred_b1 - H_true_b1)**2) / (torch.mean(torch.abs(H_true_b1)) + 1e-6)


    loss = loss_pde + loss_bc1
    loss.backward()
    optimizer.step()
    scheduler.step()

    optimizer.zero_grad()
    if it % 500 == 0:
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"iter {it}, loss = {loss.item():.4e}, PDE={loss_pde.item():.4e}, ",
              f"BC1={loss_bc1.item():.4e},",
              f"LR={current_lr:.1e}")

    # Save a checkpoint every 5000 iterations
    if it > 0 and it % 5000 == 0:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'iteration': it
        }, "trained_model_checkpoint.pth")
        print(f"Checkpoint saved at iteration {it}.")

        # Create a new instance of the same model
        model_cpu = copy.deepcopy(model)  # Deep copy the model structure
        model_cpu = model_cpu.to('cpu')  # Move the new copy to the CPU

        # Copy the state_dict (weights and biases)
        model_cpu.load_state_dict(model.state_dict())

        # At this point:
        # - model is on the GPU
        # - model_cpu is a replica of model, but on the CPU
        # Make grid
        N = 80
        x = np.linspace(0, 1, N)
        y = np.linspace(0, 1, N)
        X, Y = np.meshgrid(x, y)

        phi_pred_0, phi_true_0 = evaluate_phi(model_cpu, 0.0)
        phi_pred_1, phi_true_1 = evaluate_phi(model_cpu, 0.5)

        heights = np.linspace(0, 1, num=11)
        for el in heights:
            plot_magnetic_field_comparison(
                model=model_cpu,
                X=X,
                Y=Y,
                z_value=el,
                N=80,
                save_as=f"../Outputs/Toy_model_2height_{bc_height1}_{bc_height2}/B_comp_z_{el}_it_{it}.png"  # Set to None if you don't want to save
            )

print("Training complete.")

# Move the trained model to CPU
model_cpu = model.to("cpu")
print("Model has been moved back to CPU.")

torch.save(model_cpu.state_dict(), "trained_model.pth")
print("Model saved to trained_model.pth")

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


