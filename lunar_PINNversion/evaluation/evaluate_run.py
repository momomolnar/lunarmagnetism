import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
import torch, os
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as pl
from lunar_PINNversion.model import PINN
from lunar_PINNversion.dataloader.dataLoader import Lunar_data_loader, Lunar_surface_data_loader
from lunar_PINNversion.dataloader.util import spherical_to_cartesian
from lunar_PINNversion.evaluation.mollweide_plot import plot_four_component_mollweide
def evaluate_on_latlon_grid(
    model,
    num_pts=360,
    height_obs=1e5,
    device="cpu"
):
    """
    Returns:
        lon [rad], lat [rad]
        Bx, By, Bz, |B| with shape (num_pts, num_pts)
    """
    model.eval()
    R_lunar = 1738e3

    # Latitude [-pi/2, pi/2], Longitude [-pi, pi]
    lat = torch.linspace(-torch.pi/2, torch.pi/2, num_pts, device=device)
    lon = torch.linspace(-torch.pi, torch.pi, num_pts, device=device)
    r = torch.tensor(1.0 + height_obs / R_lunar, device=device)

    Lat, Lon = torch.meshgrid(lat, lon, indexing="ij")

    # Spherical → Cartesian (unit radius + altitude)
    x = r * torch.cos(Lat) * torch.cos(Lon)
    y = r * torch.cos(Lat) * torch.sin(Lon)
    z = r * torch.sin(Lat)

    xyz = torch.stack(
        (x.reshape(-1), y.reshape(-1), z.reshape(-1)),
        dim=-1
    ).requires_grad_(True)

    # Forward pass
    phi = model(xyz)

    # Gradient
    grad_phi = torch.autograd.grad(
        outputs=phi,
        inputs=xyz,
        grad_outputs=torch.ones_like(phi),
        create_graph=False
    )[0]

    B = -grad_phi

    Bx = B[:, 0].reshape(num_pts, num_pts).detach().cpu().numpy()
    By = B[:, 1].reshape(num_pts, num_pts).detach().cpu().numpy()
    Bz = B[:, 2].reshape(num_pts, num_pts).detach().cpu().numpy()
    Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)

    return (
        lon.cpu().numpy(),
        lat.cpu().numpy(),
        Bx, By, Bz, Bmag
    )

def save_Bfield_csv_numpy(
    lon,
    lat,
    Bx,
    By,
    Bz,
    Bmag,
    filename
):
    import numpy as np
    import os

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    Lon, Lat = np.meshgrid(lon, lat, indexing="xy")

    data = np.column_stack([
        Lon.ravel(),
        Lat.ravel(),
        Bx.ravel(),
        By.ravel(),
        Bz.ravel(),
        Bmag.ravel(),
    ])

    header = "lon_rad,lat_rad,Bx,By,Bz,Bmag"

    np.savetxt(
        filename,
        data,
        delimiter=",",
        header=header,
        comments=""
    )

def plot_B_eval_from_arrays(
    lon,
    lat,
    Bx,
    By,
    Bz,
    Bmag,
    output_dir,
    tag="eval",
    epoch=None,
    cnorm="symlog",
    figsize=(10, 10),
    cbar_label="B field [nTesla]",
):
    """
    Plot B-field Mollweide maps from precomputed arrays.

    Parameters
    ----------
    lon, lat : 1D arrays [rad]
        Longitude and latitude arrays
    Bx, By, Bz, Bmag : 2D arrays
        Magnetic field components on (lat, lon) grid
    output_dir : str
        Directory for output images
    tag : str
        Label used in filename (e.g. 'surface', 'BC')
    epoch : int or None
        Epoch number for filename
    """

    import os
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)

    # Meshgrid for Mollweide plotting
    Lon, Lat = np.meshgrid(lon, lat, indexing="xy")

    # Filename
    if epoch is None:
        fname = f"{tag}.png"
    else:
        fname = f"{tag}_{epoch:05d}.png"

    plot_four_component_mollweide(
        Lon,
        Lat,
        Bx,
        By,
        Bz,
        Bmag,
        output_file=os.path.join(output_dir, fname),
        cnorm=cnorm,
        figsize=figsize,
        titles=None,
        cbar_label=cbar_label,
        share_colorbar=False, vlims=[1e-1, 1e2]
    )


output_dir = "/home/memolnar/Projects/lunarmagnetism/Outputs/real_data/LRO_ER_surface_data_v1/"
checkpoint_path = os.path.join(output_dir, 'checkpoint_latest.pt')

device: object = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, checkpoint = PINN.load_checkpoint(
    checkpoint_path,
    device=device
)

lon_surface, lat_surface, Bx_surface, By_surface, Bz_surface, Bmag_surface = evaluate_on_latlon_grid(
    model,
    num_pts=360,
    height_obs=0,
    device=device
)

lon_orbit, lat_orbit, Bx_orbit, By_orbit, Bz_orbit, Bmag_orbit = evaluate_on_latlon_grid(
    model,
    num_pts=360,
    height_obs=1e5,
    device=device
)

plot_B_eval_from_arrays(
    lon_surface, lat_surface, Bx_surface, By_surface, Bz_surface, Bmag_surface,
    output_dir=output_dir,
    tag="eval_surface",
    epoch=checkpoint["epoch"]
)

plot_B_eval_from_arrays(
    lon_orbit, lat_orbit, Bx_orbit, By_orbit, Bz_orbit, Bmag_orbit,
    output_dir=output_dir,
    tag="eval_orbit",
    epoch=checkpoint["epoch"]
)

save_Bfield_csv_numpy(
    lon_orbit, lat_orbit, Bx_orbit, By_orbit, Bz_orbit, Bmag_orbit,
    os.path.join(output_dir, "Lunar_Bfield_orbit.csv"),
)

save_Bfield_csv_numpy(
    lon_surface, lat_surface, Bx_surface, By_surface, Bz_surface, Bmag_surface,
    os.path.join(output_dir, "Lunar_Bfield_surface.csv"),
)