"""Plot a 3x3 Mollweide comparison of boundary data and two PINN runs.

The figure layout is:
    row 1: boundary-condition data
    row 2: first PINN surface model
    row 3: second PINN surface model

The three columns are:
    1. radial
    2. toroidal
    3. poloidal

File-based inputs can provide either:
    - direct spherical components: lon, lat, radial, toroidal, poloidal
    - Cartesian vectors: lon, lat, bx, by, bz

Each PINN row can be supplied as either a file or a checkpoint. When a checkpoint is
given, the script evaluates the model on a latitude/longitude surface grid and converts
Bx/By/Bz to (radial, toroidal, poloidal).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import cmcrameri.cm as cmc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


COMPONENT_TITLES = ["Radial", "Toroidal", "Poloidal"]
DEFAULT_ROW_LABELS = ["Boundary Conditions", "PINN Run 1", "PINN Run 2"]


@dataclass
class SurfaceDataset:
    """Container for the three plotted components on one shell."""

    lon: np.ndarray
    lat: np.ndarray
    radial: np.ndarray
    toroidal: np.ndarray
    poloidal: np.ndarray


def is_float(token: str) -> bool:
    """Return True when the token can be parsed as a floating-point value."""
    try:
        float(token)
    except ValueError:
        return False
    return True


def sniff_table_format(path: Path) -> tuple[str | None, bool]:
    """Infer delimiter and header presence from the first data-like line."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            delimiter = "," if "," in stripped else None
            tokens = stripped.split(",") if delimiter == "," else stripped.split()
            has_header = any(not is_float(token) for token in tokens)
            return delimiter, has_header
    raise ValueError(f"No readable rows found in {path}")


def load_numeric_table(path: str):
    """Load a numeric text table with or without a header."""
    file_path = Path(path)
    delimiter, has_header = sniff_table_format(file_path)
    if has_header:
        data = np.genfromtxt(
            file_path,
            delimiter=delimiter,
            names=True,
            dtype=float,
            encoding="utf-8",
            autostrip=True,
            comments="#",
        )
        if data.size == 0:
            raise ValueError(f"No numeric rows found in {path}")
        return data

    data = np.loadtxt(file_path, delimiter=delimiter, comments="#", ndmin=2)
    if data.size == 0:
        raise ValueError(f"No numeric rows found in {path}")
    return data


def normalize_name(name: str) -> str:
    """Normalize a header name for fuzzy matching."""
    return "".join(ch.lower() for ch in name if ch.isalnum())


def structured_column_names(table) -> list[str]:
    """Return structured-array column names or an empty list."""
    dtype_names = getattr(table.dtype, "names", None)
    return list(dtype_names) if dtype_names else []


def resolve_column_name(table, selector: str | None, aliases: Iterable[str]) -> str | int:
    """Resolve a column selector against a structured or plain ndarray table."""
    if selector is not None and selector.isdigit():
        return int(selector)

    names = structured_column_names(table)
    if names:
        normalized = {normalize_name(name): name for name in names}

        candidates = []
        if selector:
            candidates.append(selector)
        candidates.extend(aliases)

        for candidate in candidates:
            key = normalize_name(candidate)
            if key in normalized:
                return normalized[key]

        raise ValueError(
            f"Could not resolve column '{selector}' in columns {names}"
        )

    if selector is None:
        raise ValueError("Numeric tables without headers require explicit column indices.")
    if selector.isdigit():
        return int(selector)
    raise ValueError(f"Column '{selector}' requires a header row but the file has none.")


def extract_column(table, selector: str | None, aliases: Iterable[str]) -> np.ndarray:
    """Return one numeric column as a float64 vector."""
    resolved = resolve_column_name(table, selector, aliases)
    if isinstance(resolved, int):
        array = np.asarray(table, dtype=np.float64)
        if resolved < 0 or resolved >= array.shape[1]:
            raise IndexError(f"Column index {resolved} is out of bounds for shape {array.shape}")
        return array[:, resolved].astype(np.float64, copy=False)

    return np.asarray(table[resolved], dtype=np.float64)


def infer_angles_in_radians(lon: np.ndarray, lat: np.ndarray, unit: str) -> tuple[np.ndarray, np.ndarray]:
    """Return longitudes and latitudes in radians."""
    if unit == "degrees":
        return np.deg2rad(lon), np.deg2rad(lat)
    if unit == "radians":
        return lon, lat

    lon_scale = np.nanmax(np.abs(lon))
    lat_scale = np.nanmax(np.abs(lat))
    if lon_scale > 2.0 * np.pi + 1e-6 or lat_scale > np.pi / 2.0 + 1e-6:
        return np.deg2rad(lon), np.deg2rad(lat)
    return lon, lat


def wrap_longitude(lon: np.ndarray) -> np.ndarray:
    """Wrap longitudes into [-pi, pi]."""
    return (lon + np.pi) % (2.0 * np.pi) - np.pi


def cartesian_to_surface_components(
    lon: np.ndarray,
    lat: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    bz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert Bx/By/Bz into radial, toroidal, and poloidal components.

    The script uses:
        radial    = B dot e_r
        toroidal  = B dot e_phi      (east-west / azimuthal)
        poloidal  = B dot e_theta    (north-south / meridional)

    Latitude is assumed to follow the project convention: [-pi/2, pi/2].
    """
    cos_lat = np.cos(lat)
    sin_lat = np.sin(lat)
    cos_lon = np.cos(lon)
    sin_lon = np.sin(lon)

    radial = bx * cos_lat * cos_lon + by * cos_lat * sin_lon + bz * sin_lat
    poloidal = -bx * sin_lat * cos_lon - by * sin_lat * sin_lon + bz * cos_lat
    toroidal = -bx * sin_lon + by * cos_lon
    return radial, toroidal, poloidal


def load_component_dataset(
    path: str,
    *,
    lon_col: str | None,
    lat_col: str | None,
    radial_col: str | None,
    toroidal_col: str | None,
    poloidal_col: str | None,
    angle_unit: str,
) -> SurfaceDataset:
    """Load direct radial/toroidal/poloidal components from a text table."""
    table = load_numeric_table(path)
    lon = extract_column(table, lon_col, ["lon", "longitude", "phi", "lonrad", "phirad"])
    lat = extract_column(table, lat_col, ["lat", "latitude", "theta", "latrad", "thetarad"])
    radial = extract_column(table, radial_col, ["radial", "br", "b_r", "bradial"])
    toroidal = extract_column(table, toroidal_col, ["toroidal", "bphi", "b_phi", "azimuthal", "east"])
    poloidal = extract_column(table, poloidal_col, ["poloidal", "btheta", "b_theta", "meridional", "north"])
    lon, lat = infer_angles_in_radians(lon, lat, angle_unit)
    return SurfaceDataset(
        lon=wrap_longitude(lon),
        lat=lat,
        radial=radial,
        toroidal=toroidal,
        poloidal=poloidal,
    )


def load_cartesian_dataset(
    path: str,
    *,
    lon_col: str | None,
    lat_col: str | None,
    bx_col: str | None,
    by_col: str | None,
    bz_col: str | None,
    angle_unit: str,
) -> SurfaceDataset:
    """Load lon/lat/Bx/By/Bz and convert them to spherical components."""
    table = load_numeric_table(path)
    lon = extract_column(table, lon_col, ["lon", "longitude", "phi", "lonrad", "phirad"])
    lat = extract_column(table, lat_col, ["lat", "latitude", "theta", "latrad", "thetarad"])
    bx = extract_column(table, bx_col, ["bx", "b_x"])
    by = extract_column(table, by_col, ["by", "b_y"])
    bz = extract_column(table, bz_col, ["bz", "b_z"])
    lon, lat = infer_angles_in_radians(lon, lat, angle_unit)
    lon = wrap_longitude(lon)
    radial, toroidal, poloidal = cartesian_to_surface_components(lon, lat, bx, by, bz)
    return SurfaceDataset(
        lon=lon,
        lat=lat,
        radial=radial,
        toroidal=toroidal,
        poloidal=poloidal,
    )


def evaluate_pinn_checkpoint(checkpoint_path: str, grid_size: int, device: str) -> SurfaceDataset:
    """Evaluate a PINN checkpoint on the lunar surface and return spherical components."""
    import torch

    from lunar_PINNversion.PINNmodel.model import PINN

    torch_device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, _ = PINN.load_checkpoint(checkpoint_path, device=torch_device)
    model.eval()

    lat = torch.linspace(-torch.pi / 2.0, torch.pi / 2.0, grid_size, device=torch_device)
    lon = torch.linspace(-torch.pi, torch.pi, grid_size, device=torch_device)
    lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing="ij")

    x = torch.cos(lat_grid) * torch.cos(lon_grid)
    y = torch.cos(lat_grid) * torch.sin(lon_grid)
    z = torch.sin(lat_grid)
    xyz = torch.stack((x.reshape(-1), y.reshape(-1), z.reshape(-1)), dim=-1).requires_grad_(True)

    phi_pred = model(xyz)
    grad_phi = torch.autograd.grad(
        outputs=phi_pred,
        inputs=xyz,
        grad_outputs=torch.ones_like(phi_pred),
        create_graph=False,
    )[0]
    bxyz = (-grad_phi).detach().cpu().numpy()

    lon_np = lon_grid.detach().cpu().numpy().reshape(-1)
    lat_np = lat_grid.detach().cpu().numpy().reshape(-1)
    radial, toroidal, poloidal = cartesian_to_surface_components(
        lon_np,
        lat_np,
        bxyz[:, 0],
        bxyz[:, 1],
        bxyz[:, 2],
    )

    return SurfaceDataset(
        lon=lon_np,
        lat=lat_np,
        radial=radial,
        toroidal=toroidal,
        poloidal=poloidal,
    )


def finite_abs_quantile(values: np.ndarray, quantile: float) -> float:
    """Return an absolute-value quantile while skipping NaN/Inf."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    magnitude = np.nanquantile(np.abs(finite), quantile)
    return float(magnitude) if magnitude > 0.0 else 1.0


def draw_panel(ax, lon: np.ndarray, lat: np.ndarray, values: np.ndarray, vmax: float, point_size: float):
    """Draw one Mollweide scatter panel."""
    linthresh = max(vmax * 1e-3, 1e-9)
    image = ax.scatter(
        lon,
        lat,
        c=values,
        s=point_size,
        cmap=cmc.vik,
        rasterized=True,
        norm=colors.SymLogNorm(
            linthresh=linthresh,
            linscale=1.0,
            vmin=-vmax,
            vmax=vmax,
            base=10,
        ),
    )
    ax.grid(True)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    return image


def plot_surface_comparison(
    boundary_data: SurfaceDataset,
    pinn1_data: SurfaceDataset,
    pinn2_data: SurfaceDataset,
    *,
    output_path: str,
    row_labels: list[str],
    point_size: float,
) -> None:
    """Render and save the requested 3x3 Mollweide comparison figure."""
    rows = [boundary_data, pinn1_data, pinn2_data]
    column_values = [
        [boundary_data.radial, pinn1_data.radial, pinn2_data.radial],
        [boundary_data.toroidal, pinn1_data.toroidal, pinn2_data.toroidal],
        [boundary_data.poloidal, pinn1_data.poloidal, pinn2_data.poloidal],
    ]
    vmaxs = [
        finite_abs_quantile(np.concatenate([np.ravel(values) for values in col]), 0.98)
        for col in column_values
    ]

    fig = plt.figure(figsize=(15, 11))
    gs = GridSpec(4, 3, figure=fig, height_ratios=[1.0, 1.0, 1.0, 0.08], hspace=0.28, wspace=0.2)
    axes = [[fig.add_subplot(gs[row_idx, col_idx], projection="mollweide") for col_idx in range(3)] for row_idx in range(3)]
    cbar_axes = [fig.add_subplot(gs[3, col_idx]) for col_idx in range(3)]

    images = []
    for row_idx, dataset in enumerate(rows):
        row_components = [dataset.radial, dataset.toroidal, dataset.poloidal]
        for col_idx, component in enumerate(row_components):
            image = draw_panel(
                axes[row_idx][col_idx],
                np.ravel(dataset.lon),
                np.ravel(dataset.lat),
                np.ravel(component),
                vmax=vmaxs[col_idx],
                point_size=point_size,
            )
            images.append(image)
            if row_idx == 0:
                axes[row_idx][col_idx].set_title(COMPONENT_TITLES[col_idx], fontsize=14, pad=12)

    for row_idx, row_label in enumerate(row_labels):
        fig.text(
            0.035,
            0.83 - row_idx * 0.30,
            row_label,
            rotation=90,
            va="center",
            ha="center",
            fontsize=13,
        )

    for col_idx, cax in enumerate(cbar_axes):
        colorbar = fig.colorbar(images[col_idx], cax=cax, orientation="horizontal")
        colorbar.set_label(f"{COMPONENT_TITLES[col_idx]} field")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def add_file_dataset_arguments(parser: argparse.ArgumentParser, prefix: str, label: str) -> None:
    """Add shared CLI arguments for one file-based dataset."""
    option_prefix = prefix.replace("_", "-")
    parser.add_argument(f"--{option_prefix}", dest=prefix, type=str, help=f"Path to the {label} table.")
    parser.add_argument(
        f"--{option_prefix}-format",
        dest=f"{prefix}_format",
        choices=["components", "cartesian"],
        default="components",
        help=f"Interpret the {label} table as direct components or Cartesian vectors.",
    )
    parser.add_argument(
        f"--{option_prefix}-angles",
        dest=f"{prefix}_angles",
        choices=["auto", "degrees", "radians"],
        default="auto",
        help=f"Longitude/latitude unit for the {label} table.",
    )
    parser.add_argument(f"--{option_prefix}-lon-col", dest=f"{prefix}_lon_col", type=str, default=None, help=f"Longitude column name or index for {label}.")
    parser.add_argument(f"--{option_prefix}-lat-col", dest=f"{prefix}_lat_col", type=str, default=None, help=f"Latitude column name or index for {label}.")
    parser.add_argument(f"--{option_prefix}-radial-col", dest=f"{prefix}_radial_col", type=str, default=None, help=f"Radial-component column name or index for {label}.")
    parser.add_argument(f"--{option_prefix}-toroidal-col", dest=f"{prefix}_toroidal_col", type=str, default=None, help=f"Toroidal-component column name or index for {label}.")
    parser.add_argument(f"--{option_prefix}-poloidal-col", dest=f"{prefix}_poloidal_col", type=str, default=None, help=f"Poloidal-component column name or index for {label}.")
    parser.add_argument(f"--{option_prefix}-bx-col", dest=f"{prefix}_bx_col", type=str, default=None, help=f"Bx column name or index for {label}.")
    parser.add_argument(f"--{option_prefix}-by-col", dest=f"{prefix}_by_col", type=str, default=None, help=f"By column name or index for {label}.")
    parser.add_argument(f"--{option_prefix}-bz-col", dest=f"{prefix}_bz_col", type=str, default=None, help=f"Bz column name or index for {label}.")


def load_dataset_from_args(args: argparse.Namespace, prefix: str) -> SurfaceDataset:
    """Load one file-based dataset from CLI arguments."""
    path = getattr(args, prefix)
    if not path:
        raise ValueError(f"--{prefix} is required for this dataset.")

    data_format = getattr(args, f"{prefix}_format")
    angle_unit = getattr(args, f"{prefix}_angles")

    if data_format == "components":
        return load_component_dataset(
            path,
            lon_col=getattr(args, f"{prefix}_lon_col"),
            lat_col=getattr(args, f"{prefix}_lat_col"),
            radial_col=getattr(args, f"{prefix}_radial_col"),
            toroidal_col=getattr(args, f"{prefix}_toroidal_col"),
            poloidal_col=getattr(args, f"{prefix}_poloidal_col"),
            angle_unit=angle_unit,
        )

    return load_cartesian_dataset(
        path,
        lon_col=getattr(args, f"{prefix}_lon_col"),
        lat_col=getattr(args, f"{prefix}_lat_col"),
        bx_col=getattr(args, f"{prefix}_bx_col"),
        by_col=getattr(args, f"{prefix}_by_col"),
        bz_col=getattr(args, f"{prefix}_bz_col"),
        angle_unit=angle_unit,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Create a 3x3 Mollweide comparison of boundary data and two PINN runs."
    )
    add_file_dataset_arguments(parser, "bc", "boundary-condition")
    add_file_dataset_arguments(parser, "pinn1_file", "first PINN")
    add_file_dataset_arguments(parser, "pinn2_file", "second PINN")
    parser.add_argument(
        "--pinn1-checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint for row 2. If provided, it replaces --pinn1-file.",
    )
    parser.add_argument(
        "--pinn2-checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint for row 3. If provided, it replaces --pinn2-file.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device used when evaluating a PINN checkpoint.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=181,
        help="Latitude/longitude grid size for PINN checkpoint evaluation.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/surface_model_comparison.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=3.0,
        help="Scatter point size used in each Mollweide panel.",
    )
    parser.add_argument("--bc-label", type=str, default=DEFAULT_ROW_LABELS[0], help="Label for row 1.")
    parser.add_argument("--pinn1-label", type=str, default=DEFAULT_ROW_LABELS[1], help="Label for row 2.")
    parser.add_argument("--pinn2-label", type=str, default=DEFAULT_ROW_LABELS[2], help="Label for row 3.")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI combinations before doing any work."""
    if args.pinn1_checkpoint and args.pinn1_file:
        raise ValueError("Use either --pinn1-checkpoint or --pinn1-file, not both.")
    if args.pinn2_checkpoint and args.pinn2_file:
        raise ValueError("Use either --pinn2-checkpoint or --pinn2-file, not both.")
    if not args.pinn1_checkpoint and not args.pinn1_file:
        raise ValueError("Provide either --pinn1-checkpoint or --pinn1-file.")
    if not args.pinn2_checkpoint and not args.pinn2_file:
        raise ValueError("Provide either --pinn2-checkpoint or --pinn2-file.")
    if args.grid_size < 2:
        raise ValueError("--grid-size must be at least 2.")
    if args.point_size <= 0:
        raise ValueError("--point-size must be positive.")


def main() -> None:
    """CLI entry point."""
    parser = build_argument_parser()
    args = parser.parse_args()
    validate_args(args)

    boundary_data = load_dataset_from_args(args, "bc")

    if args.pinn1_checkpoint:
        pinn1_data = evaluate_pinn_checkpoint(args.pinn1_checkpoint, args.grid_size, args.device)
    else:
        pinn1_data = load_dataset_from_args(args, "pinn1_file")

    if args.pinn2_checkpoint:
        pinn2_data = evaluate_pinn_checkpoint(args.pinn2_checkpoint, args.grid_size, args.device)
    else:
        pinn2_data = load_dataset_from_args(args, "pinn2_file")

    plot_surface_comparison(
        boundary_data,
        pinn1_data,
        pinn2_data,
        output_path=args.output,
        row_labels=[args.bc_label, args.pinn1_label, args.pinn2_label],
        point_size=args.point_size,
    )
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
