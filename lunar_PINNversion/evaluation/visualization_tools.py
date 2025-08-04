import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_component_comparison(model, true_B_func, component=2, nx=50, ny=50, device="cpu",
                              name='a'):
    """
    Visualize true, predicted, and difference for one magnetic field component at z=0.

    Args:
        model: trained BVectNet model
        true_B_func: function (x, y, z) -> [Bx, By, Bz]
        component: 0 for Bx, 1 for By, 2 for Bz
        nx, ny: grid resolution
        device: 'cpu' or 'cuda'
    """
    # Grid points at surface
    X, Y = np.meshgrid(np.linspace(0, 2*np.pi, nx),
                       np.linspace(0, 2*np.pi, ny))
    Z = np.zeros_like(X)

    # Prepare evaluation points
    eval_pts = torch.tensor(np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T,
                            dtype=torch.float32, requires_grad=False).to(device)

    # Predicted field
    B_pred = model(eval_pts).detach().cpu().numpy()
    B_pred_comp = B_pred[:, component].reshape(nx, ny)

    # True field
    B_true = true_B_func(X, Y, Z)
    B_true_comp = B_true[..., component]

    # Difference
    B_diff = B_pred_comp - B_true_comp

    # --- Plot ---
    fig, axes = plt.subplots(3, 1, figsize=(6, 12), constrained_layout=True)

    labels = ["$B_x$", "$B_y$", "$B_z$"]
    label = labels[component] if component < 3 else f"Component {component}"

    # True
    im0 = axes[0].pcolormesh(X, Y, B_true_comp, cmap="viridis", shading="auto")
    axes[0].set_title(f"True {label} at Surface")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0])

    # Predicted
    im1 = axes[1].pcolormesh(X, Y, B_pred_comp, cmap="viridis", shading="auto")
    axes[1].set_title(f"Predicted {label} at Surface")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1])

    # Difference
    im2 = axes[2].pcolormesh(X, Y, B_diff, cmap="RdBu_r", shading="auto")
    axes[2].set_title(f"Difference (Predicted - True) {label}")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    fig.colorbar(im2, ax=axes[2])
    plt.savefig(f"{name}.png")
    plt.show()


def plot_magnetic_field(points, B_values, title="Magnetic Field"):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.quiver(points[:, 0], points[:, 1], B_values[:, 0], B_values[:, 1])
    ax.set_title(title)
    plt.close(fig)  # Close figure to avoid duplicate display in Jupyter or interactive environments
    return fig