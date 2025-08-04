import torch
from model import BVectNetFourierSIREN  # save the model above as bvect_pinn.py
import wandb
from dataloader.synthetic_data_generator import create_dataloaders

torch.set_grad_enabled(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# Create data loaders
train_colloc_loader, train_boundary_loader = create_dataloaders(batch_size=8196)

wandb.init(project='lunarmagnetism')

# Initialize model
model = BVectNetFourierSIREN(in_dim=3, out_dim=1, hidden_layers=[64, 64, 64],
                             device=device, num_frequencies=5)

# Train the model
model.train_model(train_colloc_loader, train_boundary_loader, train_colloc_loader, train_boundary_loader,
                  lambda_pde=0, lambda_bc=1, epochs=3000, lr=1e-3)