"""Fourier-based downward continuation of magnetic fields on Cartesian XY grids.

This module implements 2D FFT continuation between two constant-z planes:
    - unregularized continuation (exact inverse in noise-free periodic settings)
    - regularized continuation (spectral Tikhonov damping of high-wavenumber gain)

Assumptions:
    - measurements are on a uniform Cartesian grid in x/y
    - all points in one file belong to one constant measurement height z=z_obs
    - continuation is done in a source-free half-space between z_target and z_obs
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np


def _is_float(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _sniff_table_format(path: Path) -> tuple[str | None, bool]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            delimiter = "," if "," in stripped else None
            tokens = stripped.split(",") if delimiter == "," else stripped.split()
            has_header = any(not _is_float(token) for token in tokens)
            return delimiter, has_header
    raise ValueError(f"No readable rows found in {path}")


def _load_numeric_table(path: Path):
    delimiter, has_header = _sniff_table_format(path)
    if has_header:
        data = np.genfromtxt(
            path,
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

    data = np.loadtxt(path, delimiter=delimiter, comments="#", ndmin=2)
    if data.size == 0:
        raise ValueError(f"No numeric rows found in {path}")
    return data


def _normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def _structured_column_names(table) -> list[str]:
    dtype_names = getattr(table.dtype, "names", None)
    return list(dtype_names) if dtype_names else []


def _resolve_column_name(table, selector: str | None, aliases: Iterable[str]) -> str | int:
    if selector is not None and selector.isdigit():
        return int(selector)

    names = _structured_column_names(table)
    if names:
        normalized = {_normalize_name(name): name for name in names}
        candidates = []
        if selector:
            candidates.append(selector)
        candidates.extend(aliases)
        for candidate in candidates:
            key = _normalize_name(candidate)
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


def _extract_column(table, selector: str | None, aliases: Iterable[str]) -> np.ndarray:
    resolved = _resolve_column_name(table, selector, aliases)
    if isinstance(resolved, int):
        array = np.asarray(table, dtype=np.float64)
        if resolved < 0 or resolved >= array.shape[1]:
            raise IndexError(f"Column index {resolved} is out of bounds for shape {array.shape}")
        return array[:, resolved].astype(np.float64, copy=False)
    return np.asarray(table[resolved], dtype=np.float64)


def _assert_uniform_spacing(axis: np.ndarray, name: str) -> None:
    if axis.ndim != 1 or axis.size < 2:
        raise ValueError(f"{name} must be a 1D axis with at least 2 samples.")
    diffs = np.diff(axis)
    if not np.allclose(diffs, diffs[0], rtol=1e-6, atol=1e-12):
        raise ValueError(f"{name} axis is not uniformly spaced, which FFT continuation requires.")


def _grid_from_scattered(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    bz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    bx = np.asarray(bx, dtype=np.float64).reshape(-1)
    by = np.asarray(by, dtype=np.float64).reshape(-1)
    bz = np.asarray(bz, dtype=np.float64).reshape(-1)

    n = x.size
    if not (y.size == z.size == bx.size == by.size == bz.size == n):
        raise ValueError("x, y, z, bx, by, bz must have the same number of samples.")

    z_mean = float(np.mean(z))
    z_tol = max(1e-8, 1e-8 * max(1.0, abs(z_mean)))
    if np.max(np.abs(z - z_mean)) > z_tol:
        raise ValueError("Input measurements must be on one constant-z plane.")

    x_unique = np.unique(x)
    y_unique = np.unique(y)
    nx = x_unique.size
    ny = y_unique.size
    if nx * ny != n:
        raise ValueError(
            "Input points do not form a complete regular x/y grid "
            f"(n={n}, unique_x={nx}, unique_y={ny})."
        )

    ix = np.searchsorted(x_unique, x)
    iy = np.searchsorted(y_unique, y)
    linear_idx = iy * nx + ix
    if np.unique(linear_idx).size != linear_idx.size:
        raise ValueError("Duplicate x/y sample locations detected.")

    def fill(values: np.ndarray, name: str) -> np.ndarray:
        grid = np.full((ny, nx), np.nan, dtype=np.float64)
        grid[iy, ix] = values
        if np.any(~np.isfinite(grid)):
            raise ValueError(f"Missing x/y cells in {name} grid.")
        return grid

    bx_grid = fill(bx, "bx")
    by_grid = fill(by, "by")
    bz_grid = fill(bz, "bz")

    _assert_uniform_spacing(x_unique, "x")
    _assert_uniform_spacing(y_unique, "y")
    return x_unique, y_unique, z_mean, bx_grid, by_grid, bz_grid


def load_xyz_field_grid(
    path: str,
    *,
    x_col: str | None = None,
    y_col: str | None = None,
    z_col: str | None = None,
    bx_col: str | None = None,
    by_col: str | None = None,
    bz_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """Load x/y/z/Bx/By/Bz points and reshape into a uniform XY grid.

    Returns
    -------
    x, y : 1D arrays
        Sorted unique axis values.
    z_plane : float
        Constant z for all input samples.
    bx, by, bz : 2D arrays, shape (ny, nx)
        Field components on the (y, x) grid.
    """
    table = _load_numeric_table(Path(path))
    names = _structured_column_names(table)

    if names:
        x = _extract_column(table, x_col, ["x", "xcoord", "x_coord"])
        y = _extract_column(table, y_col, ["y", "ycoord", "y_coord"])
        z = _extract_column(table, z_col, ["z", "zcoord", "z_coord"])
        bx = _extract_column(table, bx_col, ["bx", "b_x"])
        by = _extract_column(table, by_col, ["by", "b_y"])
        bz = _extract_column(table, bz_col, ["bz", "b_z"])
    else:
        arr = np.asarray(table, dtype=np.float64)
        if arr.shape[1] < 6:
            raise ValueError(
                f"{path} has {arr.shape[1]} columns; expected at least 6 "
                "(x, y, z, bx, by, bz) when no header is present."
            )
        x, y, z, bx, by, bz = [arr[:, idx] for idx in range(6)]

    return _grid_from_scattered(x, y, z, bx, by, bz)


def save_xyz_field_grid_csv(
    path: str,
    x: np.ndarray,
    y: np.ndarray,
    z: float,
    bx: np.ndarray,
    by: np.ndarray,
    bz: np.ndarray,
) -> None:
    """Save one (x,y,z,bx,by,bz) grid to CSV with a header row."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    bx = np.asarray(bx, dtype=np.float64)
    by = np.asarray(by, dtype=np.float64)
    bz = np.asarray(bz, dtype=np.float64)
    if bx.shape != by.shape or bx.shape != bz.shape:
        raise ValueError("bx, by, bz must have the same shape.")
    if bx.shape != (y.size, x.size):
        raise ValueError(
            f"Field shape {bx.shape} does not match axis sizes (ny, nx)=({y.size}, {x.size})."
        )

    x_grid, y_grid = np.meshgrid(x, y, indexing="xy")
    z_grid = np.full_like(x_grid, float(z), dtype=np.float64)
    data = np.column_stack(
        (
            x_grid.reshape(-1),
            y_grid.reshape(-1),
            z_grid.reshape(-1),
            bx.reshape(-1),
            by.reshape(-1),
            bz.reshape(-1),
        )
    )

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output_path,
        data,
        delimiter=",",
        header="x,y,z,bx,by,bz",
        comments="",
    )


def _spectral_wavenumbers(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    kx_1d = 2.0 * np.pi * np.fft.fftfreq(x.size, d=dx)
    ky_1d = 2.0 * np.pi * np.fft.fftfreq(y.size, d=dy)
    kx, ky = np.meshgrid(kx_1d, ky_1d, indexing="xy")
    k = np.sqrt(kx ** 2 + ky ** 2)
    return kx, ky, k


def _downward_gain(
    k: np.ndarray,
    delta_z: float,
    *,
    lambda_reg: float = 0.0,
    reg_power: float = 2.0,
    max_gain: float | None = None,
) -> np.ndarray:
    if delta_z < 0.0:
        raise ValueError("delta_z must be >= 0 for downward continuation gain.")
    if lambda_reg < 0.0:
        raise ValueError("lambda_reg must be >= 0.")
    if reg_power < 0.0:
        raise ValueError("reg_power must be >= 0.")

    kz = k * float(delta_z)
    growth = np.exp(np.clip(kz, a_min=-700.0, a_max=700.0))

    if lambda_reg <= 0.0:
        gain = growth
    else:
        k_ref = float(np.max(k))
        if k_ref <= 0.0:
            reg_weight = np.zeros_like(k)
        else:
            reg_weight = (k / k_ref) ** reg_power
        growth_sq = np.exp(np.clip(2.0 * kz, a_min=-700.0, a_max=700.0))
        gain = growth / (1.0 + lambda_reg * reg_weight * growth_sq)

    if max_gain is not None:
        if max_gain <= 0.0:
            raise ValueError("max_gain must be > 0 when provided.")
        gain = np.minimum(gain, max_gain)

    return gain


def _apply_fft_filter(field_xy: np.ndarray, filt: np.ndarray) -> np.ndarray:
    field_fft = np.fft.fft2(field_xy)
    filtered = np.fft.ifft2(field_fft * filt)
    return np.real(filtered)


def downward_continue_xyz_fft(
    x: np.ndarray,
    y: np.ndarray,
    bx_obs: np.ndarray,
    by_obs: np.ndarray,
    bz_obs: np.ndarray,
    *,
    z_observation: float,
    z_target: float = 0.0,
    lambda_reg: float = 0.0,
    reg_power: float = 2.0,
    max_gain: float | None = None,
) -> dict[str, np.ndarray | float]:
    """Downward continue Bx/By/Bz from z_observation to z_target using 2D FFTs.

    Parameters
    ----------
    x, y : 1D arrays
        Uniform grid axes.
    bx_obs, by_obs, bz_obs : 2D arrays
        Field components at z=z_observation with shape (len(y), len(x)).
    z_observation, z_target : float
        Heights for continuation. Requires z_observation >= z_target.
    lambda_reg : float
        Regularization strength. Use 0 for unregularized continuation.
    reg_power : float
        Spectral power for regularization weight (k/k_max)^reg_power.
    max_gain : float or None
        Optional cap on continuation gain to control amplification.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    bx_obs = np.asarray(bx_obs, dtype=np.float64)
    by_obs = np.asarray(by_obs, dtype=np.float64)
    bz_obs = np.asarray(bz_obs, dtype=np.float64)

    expected_shape = (y.size, x.size)
    if bx_obs.shape != expected_shape or by_obs.shape != expected_shape or bz_obs.shape != expected_shape:
        raise ValueError(
            "Field component shapes must all equal (len(y), len(x)); "
            f"got bx={bx_obs.shape}, by={by_obs.shape}, bz={bz_obs.shape}, expected={expected_shape}."
        )

    _assert_uniform_spacing(x, "x")
    _assert_uniform_spacing(y, "y")

    delta_z = float(z_observation) - float(z_target)
    if delta_z < -1e-12:
        raise ValueError("z_observation must be >= z_target for downward continuation.")
    if abs(delta_z) <= 1e-15:
        return {
            "x": x.copy(),
            "y": y.copy(),
            "z": float(z_target),
            "bx": bx_obs.copy(),
            "by": by_obs.copy(),
            "bz": bz_obs.copy(),
        }

    _, _, k = _spectral_wavenumbers(x, y)
    gain = _downward_gain(
        k,
        delta_z,
        lambda_reg=lambda_reg,
        reg_power=reg_power,
        max_gain=max_gain,
    )

    return {
        "x": x.copy(),
        "y": y.copy(),
        "z": float(z_target),
        "bx": _apply_fft_filter(bx_obs, gain),
        "by": _apply_fft_filter(by_obs, gain),
        "bz": _apply_fft_filter(bz_obs, gain),
    }


def upward_continue_xyz_fft(
    x: np.ndarray,
    y: np.ndarray,
    bx_source: np.ndarray,
    by_source: np.ndarray,
    bz_source: np.ndarray,
    *,
    z_source: float,
    z_target: float,
) -> dict[str, np.ndarray | float]:
    """Forward continue Bx/By/Bz from z_source to z_target using 2D FFTs."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    bx_source = np.asarray(bx_source, dtype=np.float64)
    by_source = np.asarray(by_source, dtype=np.float64)
    bz_source = np.asarray(bz_source, dtype=np.float64)

    expected_shape = (y.size, x.size)
    if bx_source.shape != expected_shape or by_source.shape != expected_shape or bz_source.shape != expected_shape:
        raise ValueError(
            "Field component shapes must all equal (len(y), len(x)); "
            f"got bx={bx_source.shape}, by={by_source.shape}, bz={bz_source.shape}, expected={expected_shape}."
        )

    _assert_uniform_spacing(x, "x")
    _assert_uniform_spacing(y, "y")

    delta_z = float(z_target) - float(z_source)
    if delta_z < -1e-12:
        raise ValueError("z_target must be >= z_source for upward continuation.")
    if abs(delta_z) <= 1e-15:
        return {
            "x": x.copy(),
            "y": y.copy(),
            "z": float(z_target),
            "bx": bx_source.copy(),
            "by": by_source.copy(),
            "bz": bz_source.copy(),
        }

    _, _, k = _spectral_wavenumbers(x, y)
    decay = np.exp(np.clip(-k * delta_z, a_min=-700.0, a_max=700.0))

    return {
        "x": x.copy(),
        "y": y.copy(),
        "z": float(z_target),
        "bx": _apply_fft_filter(bx_source, decay),
        "by": _apply_fft_filter(by_source, decay),
        "bz": _apply_fft_filter(bz_source, decay),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FFT-based downward continuation from one constant-z measurement plane."
    )
    parser.add_argument("--input", type=str, required=True, help="Input CSV/table with x,y,z,bx,by,bz.")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path for the continued field.")
    parser.add_argument(
        "--z-observation",
        type=float,
        default=None,
        help="Optional override for measurement plane height. If omitted, uses z from the input file.",
    )
    parser.add_argument("--z-target", type=float, default=0.0, help="Target continuation height.")
    parser.add_argument(
        "--lambda-reg",
        type=float,
        default=0.0,
        help="Regularization strength. Use 0 for unregularized continuation.",
    )
    parser.add_argument(
        "--reg-power",
        type=float,
        default=2.0,
        help="Spectral regularization power in (k/k_max)^reg_power.",
    )
    parser.add_argument(
        "--max-gain",
        type=float,
        default=None,
        help="Optional cap for spectral continuation gain.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    x, y, z_from_file, bx_obs, by_obs, bz_obs = load_xyz_field_grid(args.input)
    z_obs = float(args.z_observation) if args.z_observation is not None else float(z_from_file)

    continued = downward_continue_xyz_fft(
        x,
        y,
        bx_obs,
        by_obs,
        bz_obs,
        z_observation=z_obs,
        z_target=float(args.z_target),
        lambda_reg=float(args.lambda_reg),
        reg_power=float(args.reg_power),
        max_gain=args.max_gain,
    )

    save_xyz_field_grid_csv(
        args.output,
        continued["x"],
        continued["y"],
        float(continued["z"]),
        continued["bx"],
        continued["by"],
        continued["bz"],
    )
    print(
        "Saved Fourier downward-continued field to "
        f"{args.output} (z_obs={z_obs:.6g}, z_target={float(args.z_target):.6g}, lambda={float(args.lambda_reg):.6g})"
    )


if __name__ == "__main__":
    main()
