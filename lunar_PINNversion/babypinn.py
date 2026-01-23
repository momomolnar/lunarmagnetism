import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as pl

class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs, d_input, max_freq=8):
        super().__init__()
        frequencies = 2 ** torch.linspace(0, max_freq, num_freqs)
        self.frequencies = nn.Parameter(
            frequencies[None, :, None],
            requires_grad=False,
        )
        self.d_output = d_input * (num_freqs * 2)

    def forward(self, x):
        encoded = x[:, None, :] * torch.pi * self.frequencies
        encoded = encoded.reshape(x.shape[0], -1)
        encoded = torch.cat([torch.sin(encoded), torch.cos(encoded)], -1)
        return encoded

def create_collocation_data(x_c, y_c, z_c):
    return torch.tensor(
        np.hstack([x_c, y_c, z_c]),
        requires_grad=True,
        dtype=torch.float32,
    )

# CHANGES: changed np.hstack to np.vstack, take .T of torch.tensor
def create_boundary_data_pts(
    x_b: torch.Tensor,
    y_b: torch.Tensor,
    z_b: torch.Tensor,
) -> torch.Tensor:
    bc_pts = torch.tensor(
        np.vstack([x_b, y_b, z_b]),
        requires_grad=True,
        dtype=torch.float32,
    ).T
    return bc_pts

def create_boundary_data(B_bc_vals):
    bc_vals = torch.tensor(B_bc_vals, dtype=torch.float32, requires_grad=True)
    return bc_vals

def colloc_data_loader(colloc_data, batch_size=32):
    dataset = TensorDataset(colloc_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def boundary_data_loader(bc_vals, batch_size=32):
    dataset = TensorDataset(bc_vals)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def create_dataloaders(
    h=1.0,
    N_colloc=10000,
    N_bc=4096,
    batch_size=4096,
):
    (
        x_c, y_c, z_c,
        x_b, y_b, z_b,
        B_bc_vals,
    ) = create_synthetic_set(h, N_colloc, N_bc)

    colloc_data = create_collocation_data(x_c, y_c, z_c)
    train_colloc_loader = colloc_data_loader(
        colloc_data,
        batch_size=batch_size,
    )

    bc_pts = create_boundary_data_pts(x_b, y_b, z_b)
    train_boundary_loader_pts = boundary_data_loader(
        bc_pts,
        batch_size=batch_size,
    )

    bc_vals = create_boundary_data(B_bc_vals)
    train_boundary_loader = boundary_data_loader(
        bc_vals,
        batch_size=batch_size,
    )

    return colloc_data, bc_pts, bc_vals

def true_phi(x, y, z, kx=4, ky=4):
    kz = np.sqrt(kx**2 + ky**2)
    return np.exp(kz * z) * np.sin(kx * x) * np.cos(ky * y)

def true_B(x, y, z, kx=torch.tensor(4), ky=torch.tensor(4)):
    kx = torch.tensor(kx)
    ky = torch.tensor(ky)
    kz = torch.sqrt((kx**2 + ky**2))
    Bx = (kx * torch.cos(kx * x) * torch.cos(ky * y)) * torch.exp(-kz * z)
    By = - (ky * torch.sin(kx * x) * torch.sin(ky * y)) * torch.exp(-kz * z)
    Bz = - kz * torch.sin(kx * x) * torch.cos(ky * y) * torch.exp(-kz * z)

    # Bx = By = Bz = np.ones_like((Bx))

    return torch.stack([Bx,
                        By,
                        Bz], axis=-1)

def create_synthetic_set(h=1.0, N_colloc=3000, N_bc=10000):
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


# Define the neural network model
class PINN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_freqs,
        max_freq=8,
    ):
        super(PINN, self).__init__()
        self.positional_encoding = PositionalEncoding(
            num_freqs,
            input_size,
            max_freq,
        )
        self.hidden = nn.Sequential(
            nn.Linear(self.positional_encoding.d_output, hidden_size),
            # nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        x_encoded = self.positional_encoding(x)
        # x_encoded = x
        return self.hidden(x_encoded)

# Compute the Laplacian using automatic differentiation
def compute_laplacian(model, inputs):
    # inputs should require grad
    inputs = inputs.requires_grad_(True)
    phi = model(inputs)                 # shape (N, 1)
    # Sum second derivatives:
    grads = torch.autograd.grad(outputs=phi, inputs=inputs,
                                grad_outputs=torch.ones_like(phi),
                                create_graph=True, retain_graph=True)[0]   # shape (N, 3)

    # second derivatives for each input dim:
    d2 = []
    for i in range(inputs.shape[1]):
        grad_i = grads[:, i:i+1]   # shape (N,1)
        d2_i = torch.autograd.grad(outputs=grad_i, inputs=inputs,
                                   grad_outputs=torch.ones_like(grad_i),
                                   create_graph=True, retain_graph=True)[0][:, i:i+1]  # shape (N,1)
        d2.append(d2_i)

    lap = d2[0] + d2[1] + d2[2]   # shape (N,1)
    laplacian_loss = torch.mean(lap ** 2)   # MSE of laplacian to zero
    return laplacian_loss

# Define the boundary condition loss function
def boundary_condition_loss(model, inputs, B_measured):
    phi = model(inputs.requires_grad_(True))

    grad_phi = torch.autograd.grad(
        outputs=phi,
        inputs=inputs,
        grad_outputs=torch.ones_like(phi),
        create_graph=True,
    )[0]
    B_pred = -1 * grad_phi

    return torch.mean((B_measured - B_pred) ** 2)

# Training the PINN
def train_pinn(
    model,
    x_inner,
    x_boundary,
    B_measured,
    epochs,
    lr,
    lambda_domain=1,
    lambda_bc=1,
    period_log=1000,
    period_eval=5000,
    step_size=1000,
    gamma=0.95,
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma,
    )  # Example scheduler
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    for epoch in range(epochs):
        optimizer.zero_grad()
        laplacian_loss = lambda_domain * compute_laplacian(model, x_inner)
        boundary_loss = lambda_bc * boundary_condition_loss(model, x_boundary, B_measured)
        total_loss = laplacian_loss + boundary_loss
        total_loss.backward()
        optimizer.step()
        if epoch % period_log == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f'Epoch {epoch}, Total Loss: {total_loss.item():.3e}, Laplacian Loss: {laplacian_loss.item():.3e}, Boundary Loss: {boundary_loss.item():.3e}, LR: {current_lr:.3e}')
        if epoch % period_eval == 0:
            evaluate_model(model, epoch)
        scheduler.step()  # Update the learning rate at the end of each epoch

def evaluate_model(model, epoch):
    # Predict the potential and field after training

    kx = ky = 4
    x_test = np.linspace(0, 1, 100)
    y_test = np.linspace(0, 1, 100)
    z_test = np.linspace(0, .75, 100)
    X, Y, Z = np.meshgrid(x_test, y_test, z_test)
    test_points = torch.tensor(np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T,
                               dtype=torch.float32, requires_grad=True).to(device)
    # Example usage

    phi_pred = model(test_points)
    grad_phi = torch.autograd.grad(outputs=phi_pred, inputs=test_points,
                                   grad_outputs=torch.ones_like(phi_pred),
                                   create_graph=False)[0]

    B_pred = (-1 * grad_phi).cpu().detach().numpy()

    fig, ax = pl.subplots(1, 3, figsize=(6, 2))

    im1 = ax[0].imshow(B_pred[..., 0].reshape(100, 100, 100)[:, :, 0], cmap='seismic')
    ax[0].set_title("B_pred from PINN")
    pl.colorbar(im1, shrink=0.4)

    B_tru = true_B(X, Y, Z, kx =kx, ky=ky)
    print(B_tru.shape)
    ax[1].set_title("B true")
    im1 = ax[1].imshow(B_tru[:, :, :, 0].reshape(100, 100, 100)[:, :, 0], cmap='seismic')
    pl.colorbar(im1, shrink=0.4)

    ax[2].set_title("$\\delta$B")
    deltaB_B = (
        torch.tensor(B_pred[..., 0]).reshape(100, 100, 100)[:, :, 0]
        - torch.tensor(B_tru[:, :, :, 0]).reshape(100, 100, 100)[:, :, 0]
    ) / torch.tensor(B_tru[:, :, :, 0]).reshape(100, 100, 100)[:, :, 0]

    im1 = ax[2].imshow(deltaB_B, cmap='seismic', vmin=-0.1, vmax=0.1)
    pl.colorbar(im1, shrink=0.4)
    pl.tight_layout()
    pl.savefig(f"/home/memolnar/Projects/lunarmagnetism/Outputs/test_cart_data/pred_{epoch}.png")
    pl.show()
    pl.close()


    fig, ax = pl.subplots(1, 3, figsize=(6, 2))
    ind = 20
    im1 = ax[0].imshow(B_pred[..., 0].reshape(100, 100, 100)[:, :, -1], cmap='seismic')
    ax[0].set_title("B_pred from PINN @ z = 1")
    pl.colorbar(im1)

    B_tru = true_B(X, Y, Z, kx = kx, ky=ky)
    print(B_tru.shape)
    ax[1].set_title("B true @ z = 1")
    im1 = ax[1].imshow(B_tru[:, :, :, 0].reshape(100, 100, 100)[:, :, -1], cmap='seismic')
    pl.colorbar(im1)

    ax[2].set_title("$\\delta$B @ z = 1")
    deltaB_B = (
        torch.tensor(B_pred[..., 0]).reshape(100, 100, 100)[:, :, -1]
        - torch.tensor(B_tru[:, :, :, 0]).reshape(100, 100, 100)[:, :, -1]
    ) / torch.tensor(B_tru[:, :, :, 0]).reshape(100, 100, 100)[:, :, -1]

    im1 = ax[2].imshow(deltaB_B, cmap='seismic', vmin=-0.1, vmax=0.1)
    pl.colorbar(im1)
    pl.tight_layout()
    pl.savefig(f"/home/memolnar/Projects/lunarmagnetism/Outputs/test_cart_data/eval_{epoch:d}.png")
    pl.show()
    pl.close()

# Training script configuration
hidden_size = 128
num_freqs = 6  # Number of frequencies for positional encoding
max_freq = 1
input_size = 3  # for (x, y, z) coordinates
output_size = 1  # for the scalar magnetic potential
z_loc = .5

pinn = PINN(input_size, hidden_size, output_size, num_freqs, max_freq)
pinn = pinn.to(device)

domain = torch.tensor(np.random.rand(5000, 3), dtype=torch.float32).to(
    device)  # Random points inside the domain [0, 1]^3
domain[:, -1] = domain[:, -1] * z_loc

boundary_points = torch.tensor(np.random.rand(1000, 3), dtype=torch.float32).to(device)
boundary_points[:, -1] = z_loc  # Points on the face z = constant


B_measured = true_B(boundary_points[:, 0],
                    boundary_points[:, 1],
                    boundary_points[:, 2])

train_pinn(pinn, domain, boundary_points, B_measured,
           epochs=100000, lr=5e-3,
           lambda_bc=1.0, lambda_domain=1, period_eval=4000,
           step_size=1000, gamma=.98)
