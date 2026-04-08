"""Plot LRO combined ER data in a 4-panel Mollweide projection."""

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cmcrameri.cm as cmc
from matplotlib.gridspec import GridSpec

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def load_lro_combined(path):
    """Read whitespace-separated LRO combined file and return ER fields."""
    rows = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip() or line.lstrip().startswith("==>"):
                continue
            cols = line.split()
            if len(cols) < 5:
                continue
            try:
                rows.append([float(cols[1]), float(cols[2]), float(cols[3]), float(cols[4])])
            except ValueError:
                continue

    arr = np.asarray(rows, dtype=np.float64)
    phi = np.deg2rad(arr[:, 0])
    phi = (phi + np.pi) % (2.0 * np.pi) - np.pi
    theta = np.deg2rad(arr[:, 1])
    b_sc = arr[:, 2]
    alpha_c = np.deg2rad(arr[:, 3])
    b_er = b_sc / np.sin(alpha_c) ** 2

    return SimpleNamespace(phi=phi, theta=theta, B_sc=b_sc, alpha_c=alpha_c, B=b_er)


def build_four_panel_fields(lro_data):
    """Build the four plotted fields from loaded LRO ER data."""
    b_sc = np.asarray(lro_data.B_sc, dtype=np.float64)
    sin_alpha = np.sin(np.asarray(lro_data.alpha_c, dtype=np.float64))

    b_er = np.asarray(lro_data.B, dtype=np.float64)
    # Keep panel-3 and panel-4 semantics explicit for readability.
    b_total = b_er

    return b_sc, sin_alpha, b_er, b_total


def main():
    """Load LRO combined data and save a 4-panel Mollweide plot."""
    parser = argparse.ArgumentParser(
        description="Plot LRO combined ER data in a 4-panel Mollweide projection."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="LRO_combined_220V.csv",
        help="Path to LRO combined file (default: LRO_combined_220V.csv).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/LRO_combined_220V_mollweide.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=1.0,
        help="Lower scale used by the Mollweide helper.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=100.0,
        help="Upper scale used by the Mollweide helper.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lro_data = load_lro_combined(str(input_path))
    b_sc, sin_alpha, b_er, b_total = build_four_panel_fields(lro_data)

    lon = lro_data.phi
    lat = lro_data.theta
    field_cmap = cmc.batlow

    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(100, 100, figure=fig, hspace=0.3, wspace=0.1)
    axes = []
    for i in range(4):
        ax = fig.add_subplot(
            gs[(i // 2) * 50:(i // 2) * 50 + 45, (i % 2) * 50:(i % 2) * 50 + 45],
            projection="mollweide",
        )
        axes.append(ax)

    # Panel 1: B_sc with same style/range as total field panel (log-scaled positive field)
    axes[0].set_facecolor("black")
    im1 = axes[0].scatter(
        lon,
        lat,
        c=b_sc,
        s=1,
        cmap=field_cmap,
        rasterized=True,
        norm=colors.LogNorm(vmin=args.vmin, vmax=args.vmax),
    )

    # Panel 2: sine(alpha) is unitless and bounded in [-1, 1]
    im2 = axes[1].scatter(
        lon,
        lat,
        c=sin_alpha,
        s=1,
        cmap=cmc.romaO,
        rasterized=True,
        norm=colors.Normalize(vmin=-1.0, vmax=1.0),
    )

    # Panel 3: same style/range as panel 1
    axes[2].set_facecolor("black")
    im3 = axes[2].scatter(
        lon,
        lat,
        c=b_er,
        s=1,
        cmap=field_cmap,
        rasterized=True,
        norm=colors.LogNorm(vmin=args.vmin, vmax=args.vmax),
    )

    # Panel 4: unchanged (already correct)
    axes[3].set_facecolor("black")
    im4 = axes[3].scatter(
        lon,
        lat,
        c=b_total,
        s=1,
        cmap=field_cmap,
        rasterized=True,
        norm=colors.LogNorm(vmin=args.vmin, vmax=args.vmax),
    )

    ims = [im1, im2, im3, im4]
    titles = ["$B_{sc}$ [nT]", "$\\sin(\\alpha_{sc})$", "$B_{ER}$ [nT]", "$B_{total}$ [nT]"]
    cbar_labels = ["B field [nT]", "sin(alpha) [unitless]", "B field [nT]", "B field [nT]"]

    for ax, im, title, cbar_label in zip(axes, ims, titles, cbar_labels):
        ax.grid(True)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(title, fontsize=16, pad=10)
        ax.xaxis.set_ticks_position("bottom")
        ax.xaxis.set_label_position("bottom")
        plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.15, label=cbar_label)

    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
