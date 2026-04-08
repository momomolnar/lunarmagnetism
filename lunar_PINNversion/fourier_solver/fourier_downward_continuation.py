#!/usr/bin/env python3
"""
fourier_downward_continuation.py
================================

A documented reference implementation of 2-D Fourier downward continuation for
potential magnetic fields in a source-free region.

This script is intended as a simple comparison model for PINN-based lunar
magnetic inversions. It assumes a **local Cartesian patch** with horizontal
coordinates ``x`` and ``y`` and a vertical coordinate ``z``. The top boundary is
at ``z = z_top`` and the lower boundary is at ``z = z_bottom = z_top - dz``.

The method assumes that the field in the layer between the two boundaries is
potential:

    B = -∇Φ
    ∇²Φ = 0

For a horizontal Fourier mode ``exp(i k_x x + i k_y y)``, Laplace's equation
reduces to

    d²Φ_hat/dz² - k_h² Φ_hat = 0,

where

    k_h = sqrt(k_x² + k_y²).

The vertically decaying solution above buried sources is

    Φ_hat(z) = Φ_hat(z0) * exp[-k_h (z - z0)].

Therefore, downward continuation by a distance ``dz > 0`` amplifies each mode by

    exp(k_h dz).

This is the classic instability of downward continuation: high-wavenumber modes
blow up exponentially. For that reason, the script includes regularization.

Why this implementation uses the scalar potential
-----------------------------------------------
Although each Cartesian component of a potential magnetic field is harmonic,
blindly continuing ``B_x``, ``B_y``, and ``B_z`` independently does not enforce
``∇·B = 0`` and does not guarantee mutual consistency. This implementation first
estimates the scalar potential in Fourier space and then reconstructs the vector
field from that potential. That gives you a cleaner baseline comparison against
PINN solutions that also enforce the source-free physics.

Conventions
-----------
1. ``z`` increases upward, away from the source region.
2. ``dz`` is positive and means "move downward by dz", i.e.

       z_bottom = z_top - dz.

3. The Fourier convention is NumPy's ``fft2`` / ``ifft2`` convention.
4. The method is local-planar. For a truly global lunar shell problem, a
   spherical-harmonic continuation is the natural analogue.

Inputs
------
You may supply either:

A) Only ``Bz_top``
   In that case, the potential is estimated from

       Bz_hat = k_h * Φ_hat,

   for all nonzero horizontal wavenumbers.

B) ``Bx_top``, ``By_top``, and ``Bz_top``
   In that case, the script uses a least-squares estimate of ``Φ_hat`` at each
   nonzero Fourier mode:

       Bx_hat = -i kx Φ_hat
       By_hat = -i ky Φ_hat
       Bz_hat =  k_h Φ_hat

   which yields

       Φ_hat = [i kx Bx_hat + i ky By_hat + k_h Bz_hat] / [2 k_h²].

This is exact for noiseless potential-field data and behaves like a simple
projection onto the potential-field subspace when the top boundary data are
noisy or not perfectly self-consistent.

Regularization options
----------------------
1. method='none'
   Pure inverse upward continuation. This is mathematically direct but usually
   unstable unless the data are very smooth and the continuation distance is
   small.

2. method='tikhonov'
   Uses the mode-by-mode inverse of upward continuation with Tikhonov damping:

       D_reg(k_h) = exp(k_h dz) / [1 + λ exp(2 k_h dz)]

   where ``λ = reg_param``.

3. method='cutoff'
   Hard spectral cutoff:

       D_reg = exp(k_h dz),   if k_h <= kmax
              0,              otherwise

Practical notes
---------------
- Downward continuation assumes periodicity because it uses FFTs. If your patch
  has strong edge discontinuities, wrap-around artefacts can contaminate the
  result.
- The optional cosine taper can reduce those edge artefacts, at the cost of
  damping the boundary values near the edges.
- The mean (k=0) mode is special. For a potential field, the k=0 part is not
  described by the decaying exponential branch. In this script the mean field is
  simply preserved between the two levels if ``preserve_mean=True``.
- The returned fields are real-valued arrays obtained from the inverse FFT.
- The input arrays should all be in consistent units. For example, if ``dx``,
  ``dy``, and ``dz`` are in meters, then the wavenumbers are in rad/m and the
  magnetic field units are preserved.

Example (import as a module)
----------------------------
>>> import numpy as np
>>> from fourier_downward_continuation import downward_continue_magnetic_field
>>> data = np.load("top_boundary_field.npz")
>>> out = downward_continue_magnetic_field(
...     Bz_top=data["Bz_top"],
...     Bx_top=data["Bx_top"],
...     By_top=data["By_top"],
...     dx=10_000.0,
...     dy=10_000.0,
...     dz=50_000.0,
...     regularization="tikhonov",
...     reg_param=1e-8,
...     taper_fraction=0.05,
... )
>>> np.savez("bottom_boundary_field.npz", **out)

Example (command line)
----------------------
python fourier_downward_continuation.py \
    --input top_boundary_field.npz \
    --output bottom_boundary_field.npz \
    --dx 10000 --dy 10000 --dz 50000 \
    --regularization tikhonov --reg-param 1e-8 \
    --taper-fraction 0.05

The input NPZ file must contain at least ``Bz_top``. If it also contains
``Bx_top`` and ``By_top``, those will be used automatically.
"""

from __future__ import print_function

import argparse
import json
from typing import Any, Dict, Optional, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# FFT helpers
# -----------------------------------------------------------------------------

def _fft2_realspace(arr: np.ndarray) -> np.ndarray:
    """
    Return the 2-D discrete Fourier transform of a real-space array.

    Parameters
    ----------
    arr : np.ndarray
        Real-space 2-D array with shape (ny, nx).

    Returns
    -------
    np.ndarray
        Complex Fourier coefficients with the same shape.
    """
    return np.fft.fft2(arr)


def _ifft2_realspace(arr_hat: np.ndarray) -> np.ndarray:
    """
    Return the inverse 2-D Fourier transform.

    Parameters
    ----------
    arr_hat : np.ndarray
        Complex Fourier coefficients.

    Returns
    -------
    np.ndarray
        Complex spatial-domain array. In this script the physical fields are
        taken as the real part of this inverse transform.
    """
    return np.fft.ifft2(arr_hat)


# -----------------------------------------------------------------------------
# Spectral grids and tapers
# -----------------------------------------------------------------------------

def wavenumber_grids(nx: int, ny: int, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the 2-D horizontal wavenumber grids.

    Parameters
    ----------
    nx, ny : int
        Number of grid points in x and y.
    dx, dy : float
        Grid spacing in x and y.

    Returns
    -------
    kx, ky, kh : tuple of np.ndarray
        2-D arrays of horizontal wavenumbers and their magnitude,

            kh = sqrt(kx^2 + ky^2).

    Notes
    -----
    NumPy's FFT convention is used, so the returned wavenumbers are in angular
    units (radians per unit length).
    """
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="xy")
    kh = np.sqrt(kx_grid ** 2 + ky_grid ** 2)
    return kx_grid, ky_grid, kh


def _cosine_taper_1d(n: int, fraction: float) -> np.ndarray:
    """
    Create a 1-D cosine taper near the two ends of an array.

    Parameters
    ----------
    n : int
        Number of samples.
    fraction : float
        Fraction of the array length to taper on each side. Must satisfy
        0 <= fraction < 0.5.

    Returns
    -------
    np.ndarray
        1-D taper array of length `n`.
    """
    if fraction <= 0.0:
        return np.ones(n, dtype=float)
    if fraction >= 0.5:
        raise ValueError("taper_fraction must be < 0.5.")

    m = int(np.floor(fraction * n))
    if m == 0:
        return np.ones(n, dtype=float)

    w = np.ones(n, dtype=float)
    idx = np.arange(m, dtype=float)
    ramp = 0.5 * (1.0 - np.cos(np.pi * (idx + 1.0) / (m + 1.0)))
    w[:m] = ramp
    w[-m:] = ramp[::-1]
    return w


def make_2d_taper(shape: Tuple[int, int], taper_fraction: float = 0.0) -> np.ndarray:
    """
    Create a separable 2-D cosine taper.

    Parameters
    ----------
    shape : tuple of int
        Array shape (ny, nx).
    taper_fraction : float, optional
        Fraction of each edge to taper.

    Returns
    -------
    np.ndarray
        2-D taper with the same shape.
    """
    ny, nx = shape
    wx = _cosine_taper_1d(nx, taper_fraction)
    wy = _cosine_taper_1d(ny, taper_fraction)
    return np.outer(wy, wx)


# -----------------------------------------------------------------------------
# Physics helpers
# -----------------------------------------------------------------------------

def downward_operator(
    kh: np.ndarray,
    dz: float,
    method: str = "tikhonov",
    reg_param: float = 1e-8,
    kmax: Optional[float] = None,
) -> np.ndarray:
    """
    Construct the mode-by-mode downward continuation operator.

    Parameters
    ----------
    kh : np.ndarray
        Horizontal wavenumber magnitude.
    dz : float
        Downward continuation distance, defined to be positive:

            dz = z_top - z_bottom > 0.

    method : {'none', 'tikhonov', 'cutoff'}, optional
        Regularization strategy.
    reg_param : float, optional
        Regularization strength for Tikhonov damping.
    kmax : float, optional
        Spectral cutoff used when `method='cutoff'`.

    Returns
    -------
    np.ndarray
        Real amplification factor for each spectral mode.

    Notes
    -----
    Without regularization the operator is simply

        exp(kh * dz),

    which grows exponentially with wavenumber. This is why downward
    continuation is ill-conditioned.
    """
    if dz < 0.0:
        raise ValueError("dz must be positive. Use dz = z_top - z_bottom.")

    kh = np.asarray(kh, dtype=float)
    D = np.exp(kh * dz)
    method = method.lower()

    if method == "none":
        Dreg = D
    elif method == "tikhonov":
        lam = float(reg_param)
        Dreg = D / (1.0 + lam * np.exp(2.0 * kh * dz))
    elif method == "cutoff":
        if kmax is None:
            raise ValueError("kmax must be supplied when method='cutoff'.")
        Dreg = np.where(kh <= kmax, D, 0.0)
    else:
        raise ValueError("Unknown regularization method: {!r}".format(method))

    # The k=0 mode is preserved rather than amplified. This corresponds to
    # keeping the mean field the same on both planes.
    Dreg = np.where(kh == 0.0, 1.0, Dreg)
    return Dreg


def estimate_phi_hat_from_top_boundary(
    Bz_top_hat: np.ndarray,
    Bx_top_hat: Optional[np.ndarray],
    By_top_hat: Optional[np.ndarray],
    kx: np.ndarray,
    ky: np.ndarray,
    kh: np.ndarray,
) -> np.ndarray:
    """
    Estimate the scalar potential on the top boundary in Fourier space.

    Parameters
    ----------
    Bz_top_hat : np.ndarray
        Fourier coefficients of the vertical component on the top boundary.
    Bx_top_hat, By_top_hat : np.ndarray or None
        Optional Fourier coefficients of the horizontal field.
    kx, ky, kh : np.ndarray
        Wavenumber grids.

    Returns
    -------
    np.ndarray
        Estimated Fourier coefficients of the scalar potential on the top plane.

    Notes
    -----
    For nonzero wavenumbers, a potential magnetic field satisfies

        Bx_hat = -i kx Phi_hat
        By_hat = -i ky Phi_hat
        Bz_hat =  kh Phi_hat

    If only `Bz_top_hat` is supplied, this routine uses

        Phi_hat = Bz_hat / kh.

    If the full vector field is supplied, it computes the least-squares fit
    mode-by-mode:

        Phi_hat = [i kx Bx_hat + i ky By_hat + kh Bz_hat] / [2 kh^2].

    The k=0 mode is set to zero here and handled separately later by preserving
    the mean field directly in the magnetic components.
    """
    phi_hat = np.zeros_like(Bz_top_hat, dtype=np.complex128)
    mask = kh > 0.0

    if Bx_top_hat is None or By_top_hat is None:
        phi_hat[mask] = Bz_top_hat[mask] / kh[mask]
    else:
        numerator = (
            1j * kx[mask] * Bx_top_hat[mask]
            + 1j * ky[mask] * By_top_hat[mask]
            + kh[mask] * Bz_top_hat[mask]
        )
        denominator = 2.0 * kh[mask] ** 2
        phi_hat[mask] = numerator / denominator

    return phi_hat


def field_from_phi_hat(
    phi_hat: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    kh: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct the magnetic field from the scalar potential in Fourier space.

    Parameters
    ----------
    phi_hat : np.ndarray
        Fourier coefficients of the scalar potential.
    kx, ky, kh : np.ndarray
        Wavenumber grids.

    Returns
    -------
    Bx_hat, By_hat, Bz_hat : tuple of np.ndarray
        Fourier coefficients of the magnetic field components.
    """
    Bx_hat = -1j * kx * phi_hat
    By_hat = -1j * ky * phi_hat
    Bz_hat = kh * phi_hat
    return Bx_hat, By_hat, Bz_hat


def _rms(arr: np.ndarray) -> float:
    """Return the root-mean-square amplitude of an array."""
    return float(np.sqrt(np.mean(np.abs(arr) ** 2)))


# -----------------------------------------------------------------------------
# Main high-level API
# -----------------------------------------------------------------------------

def downward_continue_magnetic_field(
    Bz_top: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    Bx_top: Optional[np.ndarray] = None,
    By_top: Optional[np.ndarray] = None,
    regularization: str = "tikhonov",
    reg_param: float = 1e-8,
    kmax: Optional[float] = None,
    taper_fraction: float = 0.0,
    preserve_mean: bool = True,
) -> Dict[str, Any]:
    """
    Downward continue a potential magnetic field from a top plane to a bottom plane.

    Parameters
    ----------
    Bz_top : np.ndarray
        Vertical magnetic field on the upper boundary, shape (ny, nx).
    dx, dy : float
        Horizontal grid spacing.
    dz : float
        Positive downward continuation distance:

            z_bottom = z_top - dz.

    Bx_top, By_top : np.ndarray, optional
        Optional horizontal magnetic field components on the upper boundary.
        If supplied, both must be supplied.
    regularization : {'none', 'tikhonov', 'cutoff'}, optional
        Spectral stabilization used in the downward operator.
    reg_param : float, optional
        Tikhonov damping parameter when `regularization='tikhonov'`.
    kmax : float, optional
        Spectral cutoff when `regularization='cutoff'`.
    taper_fraction : float, optional
        Fraction of the domain edge to cosine-taper before Fourier transforming.
        This helps reduce periodic wrap-around artefacts.
    preserve_mean : bool, optional
        If True, preserve the spatial mean of the supplied magnetic components.

    Returns
    -------
    dict
        Dictionary containing the downward-continued bottom field, projected top
        field, spectral arrays, and diagnostics. The main outputs are

        - `Bx_bottom`, `By_bottom`, `Bz_bottom`
        - `Bx_top_projected`, `By_top_projected`, `Bz_top_projected`
        - `phi_top_hat`, `phi_bottom_hat`
        - `kx`, `ky`, `kh`
        - `downward_operator`
        - `diagnostics`

    Notes
    -----
    The returned `*_top_projected` arrays are useful diagnostics. They show the
    potential field implied by the supplied top boundary after projection onto
    the Fourier potential-field model. Comparing them to the raw inputs lets you
    estimate how incompatible the measured boundary is with the assumed physics.
    """
    Bz_top = np.asarray(Bz_top, dtype=float)
    if Bz_top.ndim != 2:
        raise ValueError("Bz_top must be a 2-D array with shape (ny, nx).")

    if Bx_top is not None:
        Bx_top = np.asarray(Bx_top, dtype=float)
        if Bx_top.shape != Bz_top.shape:
            raise ValueError("Bx_top must have the same shape as Bz_top.")

    if By_top is not None:
        By_top = np.asarray(By_top, dtype=float)
        if By_top.shape != Bz_top.shape:
            raise ValueError("By_top must have the same shape as Bz_top.")

    if (Bx_top is None) != (By_top is None):
        raise ValueError("Provide both Bx_top and By_top, or neither.")

    ny, nx = Bz_top.shape

    taper = make_2d_taper(Bz_top.shape, taper_fraction=taper_fraction)
    Bz_work = Bz_top * taper
    Bx_work = None if Bx_top is None else Bx_top * taper
    By_work = None if By_top is None else By_top * taper

    Bz_top_hat = _fft2_realspace(Bz_work)
    Bx_top_hat = None if Bx_work is None else _fft2_realspace(Bx_work)
    By_top_hat = None if By_work is None else _fft2_realspace(By_work)

    kx, ky, kh = wavenumber_grids(nx=nx, ny=ny, dx=dx, dy=dy)

    phi_top_hat = estimate_phi_hat_from_top_boundary(
        Bz_top_hat=Bz_top_hat,
        Bx_top_hat=Bx_top_hat,
        By_top_hat=By_top_hat,
        kx=kx,
        ky=ky,
        kh=kh,
    )

    Dreg = downward_operator(
        kh=kh,
        dz=dz,
        method=regularization,
        reg_param=reg_param,
        kmax=kmax,
    )

    phi_bottom_hat = phi_top_hat * Dreg

    Bx_bottom_hat, By_bottom_hat, Bz_bottom_hat = field_from_phi_hat(
        phi_hat=phi_bottom_hat,
        kx=kx,
        ky=ky,
        kh=kh,
    )

    Bx_top_proj_hat, By_top_proj_hat, Bz_top_proj_hat = field_from_phi_hat(
        phi_hat=phi_top_hat,
        kx=kx,
        ky=ky,
        kh=kh,
    )

    if preserve_mean:
        Bz_bottom_hat[0, 0] = Bz_top_hat[0, 0]
        Bz_top_proj_hat[0, 0] = Bz_top_hat[0, 0]

        if Bx_top_hat is not None:
            Bx_bottom_hat[0, 0] = Bx_top_hat[0, 0]
            Bx_top_proj_hat[0, 0] = Bx_top_hat[0, 0]
        else:
            Bx_bottom_hat[0, 0] = 0.0
            Bx_top_proj_hat[0, 0] = 0.0

        if By_top_hat is not None:
            By_bottom_hat[0, 0] = By_top_hat[0, 0]
            By_top_proj_hat[0, 0] = By_top_hat[0, 0]
        else:
            By_bottom_hat[0, 0] = 0.0
            By_top_proj_hat[0, 0] = 0.0

    Bx_bottom = _ifft2_realspace(Bx_bottom_hat).real
    By_bottom = _ifft2_realspace(By_bottom_hat).real
    Bz_bottom = _ifft2_realspace(Bz_bottom_hat).real

    Bx_top_projected = _ifft2_realspace(Bx_top_proj_hat).real
    By_top_projected = _ifft2_realspace(By_top_proj_hat).real
    Bz_top_projected = _ifft2_realspace(Bz_top_proj_hat).real

    diagnostics = {
        "input_type": "full_vector" if (Bx_top is not None and By_top is not None) else "Bz_only",
        "max_amplification_factor": float(np.max(np.abs(Dreg))),
        "mean_Bz_top": float(np.mean(Bz_top)),
        "kh_max": float(np.max(kh)),
        "taper_fraction": float(taper_fraction),
        "regularization": regularization,
        "reg_param": float(reg_param),
        "preserve_mean": bool(preserve_mean),
        "grid_shape": [int(ny), int(nx)],
        "dx": float(dx),
        "dy": float(dy),
        "dz": float(dz),
    }

    if Bx_top is not None:
        diagnostics["top_projection_rms_misfit_Bx"] = _rms(Bx_top - Bx_top_projected)
        diagnostics["top_projection_rms_misfit_By"] = _rms(By_top - By_top_projected)
    diagnostics["top_projection_rms_misfit_Bz"] = _rms(Bz_top - Bz_top_projected)

    return {
        "Bx_bottom": Bx_bottom,
        "By_bottom": By_bottom,
        "Bz_bottom": Bz_bottom,
        "Bx_top_projected": Bx_top_projected,
        "By_top_projected": By_top_projected,
        "Bz_top_projected": Bz_top_projected,
        "phi_top_hat": phi_top_hat,
        "phi_bottom_hat": phi_bottom_hat,
        "kx": kx,
        "ky": ky,
        "kh": kh,
        "downward_operator": Dreg,
        "diagnostics": diagnostics,
    }


# -----------------------------------------------------------------------------
# Demonstration and CLI
# -----------------------------------------------------------------------------

def synthetic_demo(
    nx: int = 128,
    ny: int = 128,
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 0.5,
    regularization: str = "none",
    reg_param: float = 1e-8,
) -> Dict[str, Any]:
    """
    Generate a synthetic potential field, upward continue it, then recover the
    bottom boundary with the downward continuation routine.

    This is intended only as a sanity-check and example workflow.
    """
    kx, ky, kh = wavenumber_grids(nx=nx, ny=ny, dx=dx, dy=dy)

    phi_bottom_hat_true = np.exp(-(kh / (0.25 * np.max(kh) + 1e-15)) ** 2).astype(np.complex128)
    phi_bottom_hat_true[0, 0] = 0.0

    Bx_bottom_hat_true, By_bottom_hat_true, Bz_bottom_hat_true = field_from_phi_hat(
        phi_bottom_hat_true, kx, ky, kh
    )

    upward_factor = np.exp(-kh * dz)
    Bx_top_hat = Bx_bottom_hat_true * upward_factor
    By_top_hat = By_bottom_hat_true * upward_factor
    Bz_top_hat = Bz_bottom_hat_true * upward_factor

    Bx_top = np.fft.ifft2(Bx_top_hat).real
    By_top = np.fft.ifft2(By_top_hat).real
    Bz_top = np.fft.ifft2(Bz_top_hat).real

    out = downward_continue_magnetic_field(
        Bz_top=Bz_top,
        Bx_top=Bx_top,
        By_top=By_top,
        dx=dx,
        dy=dy,
        dz=dz,
        regularization=regularization,
        reg_param=reg_param,
    )

    out["Bx_bottom_true"] = np.fft.ifft2(Bx_bottom_hat_true).real
    out["By_bottom_true"] = np.fft.ifft2(By_bottom_hat_true).real
    out["Bz_bottom_true"] = np.fft.ifft2(Bz_bottom_hat_true).real
    out["demo_rms_error_Bz"] = _rms(out["Bz_bottom"] - out["Bz_bottom_true"])
    return out


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fourier downward continuation of a potential magnetic field from a top plane "
            "to a bottom plane in a source-free region."
        )
    )
    parser.add_argument("--input", type=str, default=None,
                        help="Input NPZ file containing at least Bz_top, and optionally Bx_top and By_top.")
    parser.add_argument("--output", type=str, default="downward_continuation_output.npz",
                        help="Output NPZ file for the downward-continued result.")
    parser.add_argument("--dx", type=float, default=None, help="Grid spacing in x.")
    parser.add_argument("--dy", type=float, default=None, help="Grid spacing in y.")
    parser.add_argument("--dz", type=float, default=None,
                        help="Positive downward continuation distance: dz = z_top - z_bottom.")
    parser.add_argument("--regularization", type=str, default="tikhonov",
                        choices=["none", "tikhonov", "cutoff"],
                        help="Regularization strategy for downward continuation.")
    parser.add_argument("--reg-param", type=float, default=1e-8,
                        help="Regularization strength for Tikhonov damping.")
    parser.add_argument("--kmax", type=float, default=None,
                        help="Spectral cutoff when --regularization cutoff is used.")
    parser.add_argument("--taper-fraction", type=float, default=0.0,
                        help="Cosine taper fraction on each boundary edge before FFT.")
    parser.add_argument("--no-preserve-mean", action="store_true",
                        help="Do not preserve the mean (k=0) magnetic-field components.")
    parser.add_argument("--demo", action="store_true",
                        help="Run an internal synthetic demo instead of loading an input file.")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.demo:
        if args.dx is None:
            args.dx = 1.0
        if args.dy is None:
            args.dy = 1.0
        if args.dz is None:
            args.dz = 0.5

        out = synthetic_demo(
            dx=args.dx,
            dy=args.dy,
            dz=args.dz,
            regularization=args.regularization,
            reg_param=args.reg_param,
        )
        np.savez(args.output, **out)
        print("Saved synthetic demo output to {}".format(args.output))
        print("Demo RMS error in Bz:", out["demo_rms_error_Bz"])
        print(json.dumps(out["diagnostics"], indent=2))
        return

    if args.input is None:
        parser.error("Either provide --input or use --demo.")
    if args.dx is None or args.dy is None or args.dz is None:
        parser.error("--dx, --dy, and --dz are required unless --demo is used.")

    data = np.load(args.input)
    if "Bz_top" not in data:
        raise KeyError("Input NPZ file must contain an array named 'Bz_top'.")

    Bz_top = data["Bz_top"]
    Bx_top = data["Bx_top"] if "Bx_top" in data else None
    By_top = data["By_top"] if "By_top" in data else None

    out = downward_continue_magnetic_field(
        Bz_top=Bz_top,
        Bx_top=Bx_top,
        By_top=By_top,
        dx=args.dx,
        dy=args.dy,
        dz=args.dz,
        regularization=args.regularization,
        reg_param=args.reg_param,
        kmax=args.kmax,
        taper_fraction=args.taper_fraction,
        preserve_mean=(not args.no_preserve_mean),
    )

    np.savez(args.output, **out)
    print("Saved downward-continued result to {}".format(args.output))
    print(json.dumps(out["diagnostics"], indent=2))


if __name__ == "__main__":
    main()
