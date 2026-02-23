import os

import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import torch
import math
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR
from lunar_PINNversion.evaluation.mollweide_plot import plot_three_component_mollweide, plot_four_component_mollweide
from lunar_PINNversion.dataloader.util import spherical_to_cartesian
import wandb

R_lunar = 1737e3 # m

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

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, w0=30.0, is_first=False):
        super().__init__()
        self.w0 = w0
        self.linear = nn.Linear(in_features, out_features)

        self.init_weights(is_first)

    def init_weights(self, is_first):
        with torch.no_grad():
            if is_first:
                # First layer (very important)
                bound = 1 / self.linear.in_features
            else:
                bound = math.sqrt(6 / self.linear.in_features) / self.w0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.zero_()

    def forward(self, x):
        return torch.sin(self.w0 * self.linear(x))

class SirenNet(nn.Module):
    def __init__(
        self,
        in_dim=3,
        hidden_dim=128,
        num_hidden_layers=6,
        w0=35.0,
        w0_initial=40.0,
    ):
        super().__init__()

        layers = []

        # First layer
        layers.append(
            SineLayer(in_dim, hidden_dim, w0=w0_initial, is_first=True)
        )

        # Hidden layers
        for _ in range(num_hidden_layers):
            layers.append(
                SineLayer(hidden_dim, hidden_dim, w0=w0, is_first=False)
            )

        # Final linear layer (NO sine)
        final = nn.Linear(hidden_dim, 1)
        with torch.no_grad():
            bound = math.sqrt(6 / hidden_dim) / w0
            final.weight.uniform_(-bound, bound)
            final.bias.zero_()

        layers.append(final)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class PINN(nn.Module):

    def __init__(self, hidden_dim=128,
                 num_hidden_layers=5,
                 w0=35.0, w0_initial=40.0, device=None):
        super().__init__()
        #
        # self.pe = PositionalEncoding(in_dim=3, num_frequencies=pe_num_freqs,
        #                              base_freq=base_freq)
        self.device=device

        self.net = SirenNet(
            in_dim=3,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            w0=w0,
            w0_initial=w0_initial,
        )

        # Store hyperparameters for checkpoint
        self.hparams = {
            'hidden_dim': hidden_dim,
            'num_hidden_layers': num_hidden_layers,
            'w0': w0,
            'w0_initial': w0_initial
        }
        # # in_dim = self.pe.out_dim()
        # layer_dims = [in_dim] + layers + [1]
        #
        # net = []
        # for i in range(len(layer_dims) - 1):
        #     net.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
        #     if i < len(layer_dims) - 2:
        #         net.append(nn.Tanh())
        # self.net = nn.Sequential(*net)

    def forward(self, xyz):
        # xyz_pe = self.pe.forward(xyz)
        # return self.net(xyz_pe)
        return self.net(xyz)

    def generate_collocation_points(self, n_points=5000):
        """Generate random collocation points"""
        domain = np.random.rand(int(n_points), 3)
        domain[:, 0] = domain[:, 0] * 1e5 + R_lunar
        domain[:, 1] = domain[:, 1] * np.pi
        domain[:, 2] = domain[:, 2] * 2 * np.pi - np.pi

        domain_xyz = np.array([spherical_to_cartesian(el[0] / R_lunar, el[1], el[2])
                               for el in domain])
        return torch.tensor(domain_xyz, dtype=torch.float32).to(self.device)

    # Compute the Laplacian using automatic differentiation

    def compute_laplacian(self, xyz):
        xyz = xyz.requires_grad_(True)
        phi = self(xyz)

        # Compute all first derivatives at once
        grad_outputs = torch.ones_like(phi)
        grad = torch.autograd.grad(phi, xyz, grad_outputs, create_graph=True)[0]

        # Compute divergence of gradient (Laplacian)
        laplacian = 0
        for i in range(3):
            grad_i = grad[:, i:i + 1]
            grad_grad = torch.autograd.grad(
                grad_i, xyz, grad_outputs, create_graph=True
            )[0]
            laplacian += grad_grad[:, i:i + 1]

        return laplacian

    def compute_total_B_field_loss(self, inputs, B_measured_magnitude):
        phi = self(inputs.requires_grad_(True))
        grad_phi = torch.autograd.grad(outputs=phi, inputs=inputs,
                                       grad_outputs=torch.ones_like(phi),
                                       create_graph=True)[0]
        B_pred = -1 * grad_phi
        B_pred_magnitude = torch.sqrt(B_pred[...,0]**2
                                      +B_pred[...,1]**2
                                      +B_pred[...,2]**2)

        return torch.mean((B_measured_magnitude - B_pred_magnitude)**2)

    def boundary_condition_loss(self, inputs, B_measured):
        phi = self(inputs.requires_grad_(True))
        grad_phi = torch.autograd.grad(outputs=phi, inputs=inputs, grad_outputs=torch.ones_like(phi),
                                       create_graph=True)[0]
        B_pred = -1 * grad_phi

        return torch.mean((B_measured - B_pred) ** 2)

    def save_checkpoint(self, epoch, optimizer, scheduler, scaler, output_dir,
                        best_loss=None, is_best=False):
        """
        Save training checkpoint

        Parameters:
        -----------
        epoch : int
            Current epoch number
        optimizer : torch.optim.Optimizer
            Optimizer state
        scheduler : torch.optim.lr_scheduler
            Learning rate scheduler state
        scaler : torch.cuda.amp.GradScaler or None
            Gradient scaler for mixed precision
        output_dir : str
            Directory to save checkpoint
        best_loss : float, optional
            Best loss achieved so far
        is_best : bool
            Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'hparams': self.hparams,
            'best_loss': best_loss,
        }

        if scaler is not None:
            checkpoint['scaler_state_dict'] = scaler.state_dict()

        # Save regular checkpoint
        checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

        # Save as latest checkpoint
        latest_path = os.path.join(output_dir, 'checkpoint_latest.pt')
        torch.save(checkpoint, latest_path)

        # Save best model if this is the best
        if is_best:
            best_path = os.path.join(output_dir, 'checkpoint_best.pt')
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")

    @classmethod
    def load_checkpoint(cls, checkpoint_path, device=None):
        """
        Load model from checkpoint

        Parameters:
        -----------
        checkpoint_path : str
            Path to checkpoint file
        device : torch.device, optional
            Device to load model to

        Returns:
        --------
        model : PINN
            Loaded model
        checkpoint : dict
            Full checkpoint dictionary with optimizer state, etc.
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Create model with saved hyperparameters
        model = cls(
            hidden_dim=checkpoint['hparams']['hidden_dim'],
            num_hidden_layers=checkpoint['hparams']['num_hidden_layers'],
            w0=checkpoint['hparams']['w0'],
            w0_initial=checkpoint['hparams']['w0_initial'],
            device=device
        )

        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        if checkpoint.get('best_loss') is not None:
            print(f"Best loss: {checkpoint['best_loss']:.6e}")

        return model, checkpoint

    def train_pinn(
            self,
            inner_loader,
            boundary_loader,
            lunar_data,
            epochs,
            lambda_domain=1.0,
            lambda_bc=100.0,
            period_eval=100,
            checkpoint_every=500,  # Save checkpoint every N epochs
            boundary_points_full=None,
            B_measured_full=None,
            n_boundary_samples=60000,
            n_colloc_samples = 10,
            resample_boundary_every=10,
            resample_colloc_every=5,
            initial_lr=1e-3,
            target_lr=1e-6,
            use_amp=False,  # Enable mixed precision
            output_dir="",
            resume_from=None,  # Path to checkpoint to resume from
            batch_size = 8096,
    ):
        start_epoch = 0
        best_loss = float('inf')

        if resume_from is not None and os.path.exists(resume_from):
            print(f"Resuming training from {resume_from}")
            _, checkpoint = self.load_checkpoint(resume_from, self.device)
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint.get('best_loss', float('inf'))

            # Restore optimizer
            optimizer = optim.Adam(self.parameters(), lr=initial_lr)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Restore scheduler
            gamma = (target_lr / initial_lr) ** (1 / epochs)
            scheduler = ExponentialLR(optimizer, gamma=gamma)
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            # Restore scaler if using AMP
            scaler = None
            if use_amp:
                scaler = GradScaler()
                if 'scaler_state_dict' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])

            print(f"Resuming from epoch {start_epoch}")
        else:
            optimizer = optim.Adam(self.parameters(), lr=initial_lr)
            gamma = (target_lr / initial_lr) ** (1 / epochs)
            scheduler = ExponentialLR(optimizer, gamma=gamma)
            scaler = GradScaler() if use_amp else None

        # ✅ Initialize gradient scaler for mixed precision
        scaler = GradScaler() if use_amp else None

        if use_amp:
            print("Training with Automatic Mixed Precision (AMP)")

        epoch_bar = tqdm(range(epochs), desc="Training", unit="epoch")

        for epoch in epoch_bar:
            lambda_domain = min(1.0, epoch / 100000)
            # Resample boundary data periodically
            if epoch % resample_boundary_every == 0 and boundary_points_full is not None:
                indices = torch.randperm(len(boundary_points_full))[:n_boundary_samples]
                boundary_points = boundary_points_full[indices]
                B_measured = B_measured_full[indices]
                boundary_dataset = TensorDataset(boundary_points, B_measured)
                boundary_loader = DataLoader(boundary_dataset, batch_size=batch_size, shuffle=True)

            # Resample collocation points periodically
            if epoch % resample_colloc_every == 0:
                domain_xyz = self.generate_collocation_points(n_points=n_colloc_samples)
                inner_dataset = TensorDataset(domain_xyz)
                inner_loader = DataLoader(inner_dataset, batch_size=batch_size, shuffle=True)

            epoch_losses = {'bc': 0.0, 'pde': 0.0}
            n_iters = 0

            # ✅ BEST: Use zip - clean and efficient
            # Automatically stops when shortest loader is exhausted
            for (x_bc, B_bc), (x_inner,) in zip(boundary_loader, inner_loader):
                optimizer.zero_grad()

                # ✅ Mixed precision forward pass
                if use_amp:
                    with autocast():
                        bc_loss = self.boundary_condition_loss(x_bc, B_bc)
                        lap = self.compute_laplacian(x_inner)
                        pde_loss = torch.mean(lap ** 2)
                        total_loss = lambda_bc * bc_loss + lambda_domain * pde_loss

                    # Scaled backward and optimizer step
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard precision
                    bc_loss = self.boundary_condition_loss(x_bc, B_bc)
                    lap = self.compute_laplacian(x_inner)
                    pde_loss = torch.mean(lap ** 2)
                    total_loss = lambda_bc * bc_loss + lambda_domain * pde_loss

                    total_loss.backward()
                    optimizer.step()

                # Track for logging
                epoch_losses['bc'] += bc_loss.item()
                epoch_losses['pde'] += pde_loss.item()
                n_iters += 1

            scheduler.step()


            epoch_bar.set_postfix({
                "bc": f"{bc_loss.item():.3e}",
                "pde": f"{pde_loss.item():.3e}",
                "lr": f"{optimizer.param_groups[0]['lr']:.1e}"
            })

            if epoch % period_eval == 0:
                self.evaluate_model(epoch, lunar_data, output_dir)

            # if epoch % 10 == 0:
            #     print(f"Epoch {epoch}:")
            #     print(f"  Establ:     {epoch - t0:.3f}")
            #     print(f"  BC loss0:    {t1 - t01:.3f}s")
            #     print(f"  BC loss:    {t1 - t0:.3f}s")
            #     print(f"  PDE loss:   {t2 - t1:.3f}s")
            #     print(f"  Backward:   {t3 - t2:.3f}s")
            #     print(f"  Opt step:   {t4 - t3:.3f}s")
            #     print(f"  Total:      {t0 - epoch_start:.3f}s")

                # Save checkpoint periodically
            if epoch % checkpoint_every == 0 and epoch > 0:
                is_best = total_loss < best_loss
                if is_best:
                    best_loss = total_loss
                self.save_checkpoint(
                    epoch=epoch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    output_dir=output_dir,
                    best_loss=best_loss,
                    is_best=is_best
                )

            # Save final checkpoint
        self.save_checkpoint(
            epoch=epochs - 1,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            output_dir=output_dir,
            best_loss=best_loss,
            is_best=False
        )

        def train_pinn(
                self,
                inner_loader,
                boundary_loader,
                lunar_data,
                epochs,
                lambda_domain=1.0,
                lambda_bc=100.0,
                period_eval=100,
                checkpoint_every=500,  # Save checkpoint every N epochs
                boundary_points_full=None,
                B_measured_full=None,
                n_boundary_samples=60000,
                n_colloc_samples=10,
                resample_boundary_every=10,
                resample_colloc_every=5,
                initial_lr=1e-3,
                target_lr=1e-6,
                use_amp=False,  # Enable mixed precision
                output_dir="",
                resume_from=None,  # Path to checkpoint to resume from
                batch_size=8096,
        ):
            start_epoch = 0
            best_loss = float('inf')

            if resume_from is not None and os.path.exists(resume_from):
                print(f"Resuming training from {resume_from}")
                _, checkpoint = self.load_checkpoint(resume_from, self.device)
                start_epoch = checkpoint['epoch'] + 1
                best_loss = checkpoint.get('best_loss', float('inf'))

                # Restore optimizer
                optimizer = optim.Adam(self.parameters(), lr=initial_lr)
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                # Restore scheduler
                gamma = (target_lr / initial_lr) ** (1 / epochs)
                scheduler = ExponentialLR(optimizer, gamma=gamma)
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

                # Restore scaler if using AMP
                scaler = None
                if use_amp:
                    scaler = GradScaler()
                    if 'scaler_state_dict' in checkpoint:
                        scaler.load_state_dict(checkpoint['scaler_state_dict'])

                print(f"Resuming from epoch {start_epoch}")
            else:
                optimizer = optim.Adam(self.parameters(), lr=initial_lr)
                gamma = (target_lr / initial_lr) ** (1 / epochs)
                scheduler = ExponentialLR(optimizer, gamma=gamma)
                scaler = GradScaler() if use_amp else None

            # ✅ Initialize gradient scaler for mixed precision
            scaler = GradScaler() if use_amp else None

            if use_amp:
                print("Training with Automatic Mixed Precision (AMP)")

            epoch_bar = tqdm(range(epochs), desc="Training", unit="epoch")

            for epoch in epoch_bar:
                lambda_domain = min(1.0, epoch / 100000)
                # Resample boundary data periodically
                if epoch % resample_boundary_every == 0 and boundary_points_full is not None:
                    indices = torch.randperm(len(boundary_points_full))[:n_boundary_samples]
                    boundary_points = boundary_points_full[indices]
                    B_measured = B_measured_full[indices]
                    boundary_dataset = TensorDataset(boundary_points, B_measured)
                    boundary_loader = DataLoader(boundary_dataset, batch_size=batch_size, shuffle=True)

                # Resample collocation points periodically
                if epoch % resample_colloc_every == 0:
                    domain_xyz = self.generate_collocation_points(n_points=n_colloc_samples)
                    inner_dataset = TensorDataset(domain_xyz)
                    inner_loader = DataLoader(inner_dataset, batch_size=batch_size, shuffle=True)

                epoch_losses = {'bc': 0.0, 'pde': 0.0}
                n_iters = 0

                # ✅ BEST: Use zip - clean and efficient
                # Automatically stops when shortest loader is exhausted
                for (x_bc, B_bc), (x_inner,) in zip(boundary_loader, inner_loader):
                    optimizer.zero_grad()

                    # ✅ Mixed precision forward pass
                    if use_amp:
                        with autocast():
                            bc_loss = self.boundary_condition_loss(x_bc, B_bc)
                            lap = self.compute_laplacian(x_inner)
                            pde_loss = torch.mean(lap ** 2)
                            total_loss = lambda_bc * bc_loss + lambda_domain * pde_loss

                        # Scaled backward and optimizer step
                        scaler.scale(total_loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # Standard precision
                        bc_loss = self.boundary_condition_loss(x_bc, B_bc)
                        lap = self.compute_laplacian(x_inner)
                        pde_loss = torch.mean(lap ** 2)
                        total_loss = lambda_bc * bc_loss + lambda_domain * pde_loss

                        total_loss.backward()
                        optimizer.step()

                    # Track for logging
                    epoch_losses['bc'] += bc_loss.item()
                    epoch_losses['pde'] += pde_loss.item()
                    n_iters += 1

                scheduler.step()

                epoch_bar.set_postfix({
                    "bc": f"{bc_loss.item():.3e}",
                    "pde": f"{pde_loss.item():.3e}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.1e}"
                })

                if epoch % period_eval == 0:
                    self.evaluate_model(epoch, lunar_data, output_dir)

                # if epoch % 10 == 0:
                #     print(f"Epoch {epoch}:")
                #     print(f"  Establ:     {epoch - t0:.3f}")
                #     print(f"  BC loss0:    {t1 - t01:.3f}s")
                #     print(f"  BC loss:    {t1 - t0:.3f}s")
                #     print(f"  PDE loss:   {t2 - t1:.3f}s")
                #     print(f"  Backward:   {t3 - t2:.3f}s")
                #     print(f"  Opt step:   {t4 - t3:.3f}s")
                #     print(f"  Total:      {t0 - epoch_start:.3f}s")

                # Save checkpoint periodically
                if epoch % checkpoint_every == 0 and epoch > 0:
                    is_best = total_loss < best_loss
                    if is_best:
                        best_loss = total_loss
                    self.save_checkpoint(
                        epoch=epoch,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        output_dir=output_dir,
                        best_loss=best_loss,
                        is_best=is_best
                    )

                # Save final checkpoint
            self.save_checkpoint(
                epoch=epochs - 1,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                output_dir=output_dir,
                best_loss=best_loss,
                is_best=False
            )

    def train_pinn_with_surface_data(
            self,
            inner_loader,
            boundary_loader,
            surface_loader,
            lunar_data,
            epochs,
            lambda_domain=1.0,
            lambda_bc=100.0,
            period_eval=100,
            checkpoint_every=500,  # Save checkpoint every N epochs
            boundary_points_full=None,
            B_measured_full=None,
            n_boundary_samples=60000,
            n_colloc_samples = 10,
            resample_boundary_every=10,
            resample_colloc_every=5,
            initial_lr=1e-3,
            target_lr=1e-6,
            use_amp=False,  # Enable mixed precision
            output_dir="",
            resume_from=None,  # Path to checkpoint to resume from
            batch_size = 8096,
    ):
        start_epoch = 0
        best_loss = float('inf')

        wandb.init(
            project="lunar-pinn",
            name=f"SIREN_hidden{self.hparams['hidden_dim']}_layers{self.hparams['num_hidden_layers']}",
            config={
                **self.hparams,
                "lambda_bc": lambda_bc,
                "lambda_domain": lambda_domain,
                "initial_lr": initial_lr,
                "target_lr": target_lr,
                "batch_size": batch_size,
                "use_amp": use_amp,
            },
            resume="allow" if resume_from is not None else False,
        )

        if resume_from is not None and os.path.exists(resume_from):
            print(f"Resuming training from {resume_from}")
            _, checkpoint = self.load_checkpoint(resume_from, self.device)
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint.get('best_loss', float('inf'))

            # Restore optimizer
            optimizer = optim.Adam(self.parameters(), lr=initial_lr)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Restore scheduler
            gamma = (target_lr / initial_lr) ** (1 / epochs)
            scheduler = ExponentialLR(optimizer, gamma=gamma)
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            # Restore scaler if using AMP
            scaler = None
            if use_amp:
                scaler = GradScaler()
                if 'scaler_state_dict' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])

            print(f"Resuming from epoch {start_epoch}")
        else:
            optimizer = optim.Adam(self.parameters(), lr=initial_lr)
            gamma = (target_lr / initial_lr) ** (1 / epochs)
            scheduler = ExponentialLR(optimizer, gamma=gamma)
            scaler = GradScaler() if use_amp else None

        # ✅ Initialize gradient scaler for mixed precision
        scaler = GradScaler() if use_amp else None

        if use_amp:
            print("Training with Automatic Mixed Precision (AMP)")

        epoch_bar = tqdm(range(epochs), desc="Training", unit="epoch")

        for epoch in epoch_bar:
            pde_start_epoch = 100
            if epoch < pde_start_epoch:
                lambda_surface = 0
                lambda_domain = 0

            else:
                lambda_domain = min(1.0, epoch / 1e5)
                lambda_surface = lambda_domain / 1e3
            # Resample boundary data periodically
            if epoch % resample_boundary_every == 0 and boundary_points_full is not None:
                indices = torch.randperm(len(boundary_points_full))[:int(n_boundary_samples)]
                boundary_points = boundary_points_full[indices]
                B_measured = B_measured_full[indices]
                boundary_dataset = TensorDataset(boundary_points, B_measured)
                boundary_loader = DataLoader(boundary_dataset, batch_size=batch_size, shuffle=True)

            # Resample collocation points periodically
            if epoch % resample_colloc_every == 0:
                domain_xyz = self.generate_collocation_points(n_points=n_colloc_samples)
                inner_dataset = TensorDataset(domain_xyz)
                inner_loader = DataLoader(inner_dataset, batch_size=batch_size, shuffle=True)

            epoch_losses = {'bc': 0.0, 'pde': 0.0, 'surf':0}
            n_iters = 0

            for (x_bc, B_bc), (x_inner,), (x_surface, B_surface) in zip(boundary_loader, inner_loader, surface_loader):
                optimizer.zero_grad()

                # ✅ Mixed precision forward pass
                if use_amp:
                    with autocast():
                        bc_loss = self.boundary_condition_loss(x_bc, B_bc)
                        lap = self.compute_laplacian(x_inner)
                        pde_loss = torch.mean(lap ** 2)
                        total_loss = lambda_bc * bc_loss + lambda_domain * pde_loss

                    # Scaled backward and optimizer step
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard precision
                    if epoch < pde_start_epoch:
                        bc_loss = self.boundary_condition_loss(x_bc, B_bc)
                        total_loss = lambda_bc * bc_loss
                        pde_loss = bc_loss
                        surface_loss = bc_loss
                    else:
                        bc_loss = self.boundary_condition_loss(x_bc, B_bc)
                        surface_loss = self.compute_total_B_field_loss(x_surface, B_surface)
                        lap = self.compute_laplacian(x_inner)
                        pde_loss = torch.mean(lap ** 2)
                        total_loss = lambda_bc * bc_loss + lambda_domain * pde_loss + surface_loss * lambda_surface

                    total_loss.backward()
                    optimizer.step()

                # Track for logging
                epoch_losses['bc'] += bc_loss.item()
                epoch_losses['pde'] += pde_loss.item()
                epoch_losses['surf'] += surface_loss.item()

                n_iters += 1
            wandb.log({
                "epoch": epoch,
                "loss/bc": bc_loss,
                "loss/pde": pde_loss,
                "loss/surface": surface_loss,
                "loss/total": total_loss,
                "lambda/domain": lambda_domain,
                "lambda/surface": lambda_surface,
                "lambda/bc": lambda_bc,
                "lr": optimizer.param_groups[0]['lr'],
            })
            scheduler.step()


            epoch_bar.set_postfix({
                "bc": f"{bc_loss.item():.3e}",
                "pde": f"{pde_loss.item():.3e}",
                "surf": f"{surface_loss.item():.3e}",
                'ld': f"{lambda_domain:.3e}",
                "lr": f"{optimizer.param_groups[0]['lr']:.1e}"
            })

            if epoch % period_eval == 0:
                self.evaluate_model(epoch, lunar_data, output_dir)


            if epoch % checkpoint_every == 0 and epoch > 0:
                is_best = total_loss < best_loss
                if is_best:
                    best_loss = total_loss
                self.save_checkpoint(
                    epoch=epoch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    output_dir=output_dir,
                    best_loss=best_loss,
                    is_best=is_best
                )

            # Save final checkpoint

        self.save_checkpoint(
            epoch=epochs - 1,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            output_dir=output_dir,
            best_loss=best_loss,
            is_best=False
        )
    def evaluate_model(self, epoch, lunar_data, output_dir):
        # Predict the potential and field after training
        self.plot_B_eval(epoch, lunar_data, output_dir)

        wandb.log({
            "eval/surface_B": wandb.Image(f"{output_dir}/eval_surface_{epoch:d}.png"),
            "eval/BC_B": wandb.Image(f"{output_dir}/eval_BC_{epoch:d}.png"),
        })

        if epoch == 0:
            wandb.log({
                "eval/true_BC": wandb.Image(f"{output_dir}/true_BC_{epoch:d}.png")
            })
    def plot_B_eval(self, epoch, lunar_data, output_dir,
                    num_pts=400, height_obs=1e5):

        def spherical_to_cartesian(r, theta, phi):
            x = r * torch.cos(theta) * torch.cos(phi)
            y = r * torch.cos(theta) * torch.sin(phi)
            z = r * torch.sin(theta)
            return x, y, z

        theta_linspace = torch.linspace(-torch.pi/2, torch.pi/2,num_pts)
        phi_linspace = torch.linspace(-np.pi, torch.pi, num_pts)
        r_lunar_surface = torch.ones(1)
        r_BC_surface = torch.ones(1) + height_obs/R_lunar

        # Create meshgrid in spherical coordinates
        # meshgrid(..., indexing='ij') gives shape [r, theta, phi]
        R_lunar_surface, Theta, Phi = torch.meshgrid(r_lunar_surface, theta_linspace, phi_linspace,
                                                     indexing='ij')
        R_orbit_surface, Theta, Phi = torch.meshgrid(r_BC_surface, theta_linspace, phi_linspace,
                                                     indexing='ij')
        # Convert spherical to Cartesian (vectorized)
        X_0, Y_0, Z_0 = spherical_to_cartesian(R_lunar_surface, Theta, Phi)
        X_BC, Y_BC, Z_BC = spherical_to_cartesian(R_orbit_surface, Theta, Phi)
        # Stack into single tensor if needed
        grid_mesh_eval_xyz_0 = torch.stack((X_0.ravel(), Y_0.ravel(), Z_0.ravel()), dim=-1)
        grid_mesh_eval_xyz_0 = torch.tensor(grid_mesh_eval_xyz_0, dtype=torch.float32,
                               requires_grad=True).to(self.device)
        phi_pred_0 = self(grid_mesh_eval_xyz_0)
        grad_phi_0 = torch.autograd.grad(outputs=phi_pred_0, inputs=grid_mesh_eval_xyz_0,
                                         grad_outputs=torch.ones_like(phi_pred_0),
                                         create_graph=True)[0]

        B_pred_0 = (-1 * grad_phi_0).cpu().detach().numpy()

        grid_mesh_eval_xyz_BC = torch.stack((X_BC.ravel(), Y_BC.ravel(), Z_BC.ravel()), dim=-1)
        grid_mesh_eval_xyz_BC = torch.tensor(grid_mesh_eval_xyz_BC, dtype=torch.float32,
                               requires_grad=True).to(self.device)

        phi_pred_BC = self(grid_mesh_eval_xyz_BC)
        grad_phi_BC = torch.autograd.grad(outputs=phi_pred_BC, inputs=grid_mesh_eval_xyz_BC,
                                         grad_outputs=torch.ones_like(phi_pred_BC),
                                         create_graph=True)[0]

        B_pred_BC = (-1 * grad_phi_BC).cpu().detach().numpy()
        labels = ['B$_{x}$', 'B$_{y}$', "B$_{z}$"]
        Bx_surf = B_pred_0[..., 0].reshape(num_pts, num_pts)
        By_surf = B_pred_0[..., 1].reshape(num_pts, num_pts)
        Bz_surf = B_pred_0[..., 2].reshape(num_pts, num_pts)
        Btotal_surf = np.sqrt(Bx_surf**2 + By_surf**2 + Bz_surf**2)

        plot_four_component_mollweide(Phi, Theta,
                                      Bx_surf, By_surf, Bz_surf, Btotal_surf,
                                       output_file=f"{output_dir}/eval_surface_{epoch:d}.png",
                                       cnorm='symlog',
                                       figsize=(10, 10),
                                       titles=None,
                                       cbar_label="B field [nTesla]",
                                       share_colorbar=False)

        Bx_BC = B_pred_BC[..., 0].reshape(num_pts, num_pts)
        By_BC = B_pred_BC[..., 1].reshape(num_pts, num_pts)
        Bz_BC = B_pred_BC[..., 2].reshape(num_pts, num_pts)
        Btot_BC  = np.sqrt(Bx_BC ** 2 + By_BC ** 2 + Bz_BC ** 2)
        plot_four_component_mollweide(Phi, Theta,
                                      Bx_BC, By_BC, Bz_BC, Btot_BC,
                                       output_file=f"{output_dir}/eval_BC_{epoch:d}.png",
                                       cnorm='symlog',
                                       figsize=(10, 10),
                                       titles=None,
                                       cbar_label="B field [nTesla]",
                                       share_colorbar=False)

        if epoch == 0:
            plot_four_component_mollweide(lunar_data.phi[::2],
                                       lunar_data.theta[::2],
                                       lunar_data.b_x[::2],
                                       lunar_data.b_y[::2],
                                       lunar_data.b_z[::2],
                                       np.sqrt(lunar_data.b_z[::2]**2 + lunar_data.b_x[::2]**2 +lunar_data.b_y[::2]**2),
                                       output_file=f"{output_dir}/true_BC_{epoch:d}.png",
                                       cnorm='symlog',
                                       figsize=(10, 10),
                                       titles=None,
                                       cbar_label="B field [nTesla]",
                                       share_colorbar=False)


# Example usage functions
def train_new_model():
    """Example: Start new training"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = PINN(hidden_dim=128, num_hidden_layers=5, device=device)

    # Your training data setup here
    # boundary_points_full = ...
    # B_measured_full = ...

    model.train_pinn(
        inner_loader=None,
        boundary_loader=None,
        lunar_data=None,
        epochs=10000,
        output_dir="./checkpoints",
        checkpoint_every=500,  # Save every 500 epochs
        resume_from=None  # Start from scratch
    )


def resume_training():
    """Example: Resume training from checkpoint"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model from checkpoint
    model, _ = PINN.load_checkpoint('./checkpoints/checkpoint_latest.pt', device=device)

    # Continue training
    model.train_pinn(
        inner_loader=None,
        boundary_loader=None,
        lunar_data=None,
        epochs=10000,
        output_dir="./checkpoints",
        checkpoint_every=500,
        resume_from='./checkpoints/checkpoint_latest.pt'  # Resume from here
    )


def evaluate_from_checkpoint():
    """Example: Load model and evaluate without training"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load best model
    model, checkpoint = PINN.load_checkpoint('./checkpoints/checkpoint_best.pt', device=device)

    # Put in eval mode
    model.eval()

    # Run evaluation
    with torch.no_grad():
        # Your evaluation code here
        # model.plot_B_eval(checkpoint['epoch'], lunar_data, './results')
        pass