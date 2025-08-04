import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from dataloader.synthetic_data_generator import true_B
from torch.distributions import Normal

from evaluation.visualization_tools import plot_magnetic_field
# SIREN Layer Definition
# class SIRENLayer(nn.Module):
#     def __init__(self, in_features, out_features, is_first=False, omega_0=30, dtype=torch.float32):
#         super().__init__()
#         self.in_features = in_features
#         self.is_first = is_first
#         self.omega_0 = omega_0
#         self.dtype = dtype
#
#         self.linear = nn.Linear(in_features, out_features).to(dtype)
#         self._initialize_weights()
#
#     def _initialize_weights(self):
#         with torch.no_grad():
#             if self.is_first:
#                 self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
#             else:
#                 self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
#                                             np.sqrt(6 / self.in_features) / self.omega_0)
#
#     def forward(self, x):
#         return torch.sin(self.omega_0 * self.linear(x))

# BVectNetFourierSIREN Model Definition

class PositionalEncoding(nn.Module):
    def __init__(self, in_dim, num_frequencies):
        super(PositionalEncoding, self).__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies

    def forward(self, x):
        # Generate frequency bands
        freq_bands = 2.0 ** torch.arange(self.num_frequencies, device=x.device, dtype=x.dtype) * torch.pi
        out = [x]
        for i in range(x.shape[1]):
            for freq in freq_bands:
                out.append(torch.sin(freq * x[:, i:i + 1]))
                out.append(torch.cos(freq * x[:, i:i + 1]))
        return torch.cat(out, dim=-1)

    @property
    def size(self):
        return self.in_dim * (2 * self.num_frequencies + 1)


class BVectNetFourierSIREN(nn.Module):
    def __init__(self, in_dim=3, out_dim=1, num_frequencies=6, hidden_layers=[256, 256, 256, 256, 256, 256],
                 device=torch.device("cpu"), lr=1e-3):
        super().__init__()

        self.positional_encoding = PositionalEncoding(in_dim, num_frequencies)
        input_dim = self.positional_encoding.size

        self.device = device
        self.dtype = torch.float32

        layer_sizes = [input_dim] + hidden_layers + [out_dim]

        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                self.layers.append(nn.Tanh())  # Add tanh activation for hidden layers

        self.to(self.device)

    def forward(self, x):
        x = x.to(self.dtype)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x)
        return x

    @staticmethod
    def laplacian(phi, coords):
        grads = torch.autograd.grad(phi, coords, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
        d2 = []
        for i in range(coords.shape[1]):
            grad2 = -1 * torch.autograd.grad(grads[:, i], coords, grad_outputs=torch.ones_like(grads[:, i]), create_graph=True)[0][:, i]
            d2.append(grad2.unsqueeze(1))
        return torch.cat(d2, dim=1).sum(dim=1)

    def compute_b_field(self, phi, coords):
        grads = -1 * torch.autograd.grad(phi, coords, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
        return grads

    def train_model(self, train_colloc_loader, train_boundary_loader, val_colloc_loader, val_boundary_loader,
                    epochs=100, lambda_pde=1.0, lambda_bc=10.0,
                    log_interval=1, val_interval=20, lr=1e-3,
                    step_size=10, gamma=0.1):

        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)  # Initialize the scheduler

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            # with tqdm(total=len(train_colloc_loader) + len(train_boundary_loader), desc=f"Epoch {epoch}", leave=True) as pbar:
            with tqdm(total=len(train_colloc_loader) + len(train_boundary_loader), desc=f"Epoch {epoch}", leave=True) as pbar:
                # Train on collocation data

                # Train on boundary data
                for bc_batch in train_boundary_loader:
                    bc_pts, bc_vals = bc_batch
                    bc_pts, bc_vals = bc_pts.to(self.device), bc_vals.to(self.device)

                    optimizer.zero_grad()
                    phi_bc = self(bc_pts)
                    B_inferred = self.compute_b_field(phi_bc, bc_pts)
                    loss_bc = torch.mean((B_inferred - bc_vals)**2)
                    loss_bc.backward()
                    optimizer.step()
                    epoch_loss += loss_bc.item()
                    pbar.set_postfix({"loss_bc": loss_bc.item()})
                    pbar.update(1)

                    if epoch % log_interval == 0:
                        wandb.log({
                            'train_loss_bc': loss_bc.item(),
                            'epoch': epoch
                        })

                # for colloc_batch in train_colloc_loader:
                #     colloc_pts = colloc_batch[0].clone().requires_grad_(True).to(self.device)
                #
                #     optimizer.zero_grad()
                #     phi_c = self(colloc_pts)  # shape (N, 3)
                #     loss_pde = torch.mean(self.laplacian(phi_c, colloc_pts)**2)
                #
                #     loss_pde.backward()
                #     optimizer.step()
                #     epoch_loss += lambda_pde*loss_pde.item()
                #     pbar.set_postfix({"loss_pde": lambda_pde*loss_pde.item()})
                #     pbar.update(1)
                #
                #     if epoch % log_interval == 0:
                #         wandb.log({
                #             'train_loss_pde': loss_pde.item(),
                #             'epoch': epoch
                #         })

                # Combined epoch loss
                if epoch % log_interval == 0:
                    wandb.log({
                        'train_epoch_loss': epoch_loss,
                        'epoch': epoch
                    })

                # Validation step
                if epoch % val_interval == 0:
                    self.evaluate_model(val_colloc_loader, val_boundary_loader, epoch)

    def create_xyz_plane_torch(self, x_range, y_range, z_value, step, device="cuda", dtype=torch.float32):
        """
        Create a batch of (x, y, z) points on z=0 plane for PyTorch models.

        Parameters:
            x_range (tuple): (xmin, xmax) range for x values.
            y_range (tuple): (ymin, ymax) range for y values.
            step (float): spacing between points.
            device (str): "cpu" or "cuda" for tensor location.
            dtype: PyTorch dtype (default float32).

        Returns:
            torch.Tensor: shape (N, 3) tensor of (x, y, z) points.
        """
        x = np.arange(x_range[0], x_range[1] + step, step)
        y = np.arange(y_range[0], y_range[1] + step, step)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X) + z_value

        points_np = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        points_torch = torch.tensor(points_np, dtype=dtype, device=device, requires_grad=True)

        return points_torch

    def plot_vector_components(self, vec, n, cmap='viridis', title="blah"):

        if vec.shape != (n ** 2, 3):
            raise ValueError(f"Expected shape ({n ** 2}, 3), got {vec.shape}")

        # Reshape each component into an n x n grid
        comp_x = vec[:, 0].reshape(n, n)
        comp_y = vec[:, 1].reshape(n, n)
        comp_z = vec[:, 2].reshape(n, n)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        comps = [comp_x, comp_y, comp_z]
        titles = ['X-component', 'Y-component', 'Z-component']

        for ax, comp, title in zip(axs, comps, titles):
            im = ax.imshow(comp, origin='lower', cmap=cmap)
            ax.set_title(title)
            fig.colorbar(im, ax=ax)
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()
        return fig

    def plot_colormesh_3plot(self, X, Y, vec, cmap='viridis', title="blah"):

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        titles = ['X-component', 'Y-component', 'Z-component']

        for ax, comp, title, i in zip(axs, vec.T, titles, range(3)):
            im = ax.tripcolor(X, Y, comp, shading='flat', cmap=cmap)
            ax.set_title(title)
            fig.colorbar(im, ax=ax)
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()
        return fig

    def evaluate_model(self, val_colloc_loader, val_boundary_loader, epoch):
        self.eval()
        val_loss_pde = 0.0
        val_loss_bc = 0.0

        # # Validation on collocation data
        # for colloc_batch in val_colloc_loader:
        #     colloc_pts = colloc_batch[0].clone().requires_grad_(True).to(self.device)
        #     phi_c = self(colloc_pts)
        #     loss_pde = sum(torch.mean(self.laplacian(phi_c[:, j:j + 1], colloc_pts)**2) for j in range(3))
        #     val_loss_pde += loss_pde.item()

        # Validation on boundary data
        for bc_batch in val_boundary_loader:
            bc_pts, bc_vals = bc_batch
            bc_pts, bc_vals = bc_pts.to(self.device), bc_vals.to(self.device)
            phi_b = self(bc_pts)
            B_b = self.compute_b_field(phi_b, bc_pts)
            loss_bc = torch.mean((B_b - bc_vals)**2)
            val_loss_bc += loss_bc.item()

        # Combined validation loss
        val_loss = val_loss_pde + val_loss_bc

        # Logging validation metric
        wandb.log({
            'val_total_loss': val_loss,
            'val_pde_loss': val_loss_pde,
            'val_bc_loss': val_loss_bc,
            'epoch': epoch
        })

        bc_vals = bc_vals.detach().cpu().numpy()
        bc_pts = bc_pts.detach().cpu().numpy()

        fig_bc_vals = self.plot_colormesh_3plot(bc_pts[:, 0], bc_pts[:, 1], bc_vals.squeeze(), cmap='seismic',
                                                title=f"Magnetic Field at z = 1")

        wandb.log({"Magnetic Field BC": wandb.Image(fig_bc_vals)})

        # Example of plotting magnetic field (assuming `plot_magnetic_field` function exists)
        pts_torch = self.create_xyz_plane_torch((0, 1), (0, 1), 1, 0.025)
        inferred_phi = self(pts_torch)
        inferred_B = self.compute_b_field(inferred_phi, pts_torch)
        inferred_B = inferred_B.detach().cpu().numpy()
        # Example usage
        # print(pts_torch.shape)  # (N, 3)
        fig_upper_boundary = self.plot_vector_components(inferred_B, 41, cmap='seismic',
                                                         title=f"Magnetic Field at z = 1 at Epoch {epoch}")
        wandb.log({"Magnetic Field @ z=0": wandb.Image(fig_upper_boundary)})

        # Example of plotting magnetic field (assuming `plot_magnetic_field` function exists)
        pts_torch = self.create_xyz_plane_torch((0, 1), (0, 1), 0, 0.025)
        inferred_phi = self(pts_torch)
        inferred_B = self.compute_b_field(inferred_phi, pts_torch)
        inferred_B = inferred_B.detach().cpu().numpy()
        fig_lower_boundary = self.plot_vector_components(inferred_B, 41, cmap='seismic',
                                          title=f"Magnetic Field at z = 0 at Epoch {epoch}")
        wandb.log({"Magnetic Field@ z=1": wandb.Image(fig_lower_boundary)})