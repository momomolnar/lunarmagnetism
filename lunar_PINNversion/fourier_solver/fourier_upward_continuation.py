#!/usr/bin/env python3
"""
fourier_upward_continuation.py
================================

A documented reference implementation of 2-D Fourier upward continuation for
potential magnetic fields in a source-free Cartesian layer.

This script is intended as a simple, transparent comparison model for PINN-based
lunar magnetic inversions on a **local Cartesian patch**. It assumes that the
magnetic field in the region above the lower boundary is potential:

    B = -∇Φ
    ∇²Φ = 0

with horizontal coordinates ``x`` and ``y`` and vertical coordinate ``z``.
The lower boundary is at ``z = z_bottom``. The user specifies an upward
continuation distance ``dz > 0``, so the upper boundary is

    z_top = z_bottom + dz.

For each horizontal Fourier mode,

    Φ(x, y, z) = Φ_hat(z) exp(i k_x x + i k_y y),

Laplace's equation becomes

    d²Φ_hat/dz² - k_h² Φ_hat = 0,

where

    k_h = sqrt(k_x² + k_y²).

The solution that decays with height above the source region is

    Φ_hat(z) = Φ_hat(z0) exp[-k_h (z - z0)].

Therefore, upward continuation by a positive distance ``dz`` damps each
horizontal Fourier mode by

    exp(-k_h dz).

This is the stable counterpart of downward continuation. High-wavenumber modes
attenuate rapidly with height, which is physically why small-scale magnetic
structure fades away as one moves upward from the source boundary.

Why this implementation uses the scalar potential
-----------------------------------------------
Although each Cartesian component of a potential field is harmonic, it is better
not to upward continue ``B_x``, ``B_y``, and ``B_z`` independently. Doing so
would not explicitly enforce the consistency relations implied by

    B = -∇Φ
    ∇·B = 0.

Instead, this script:

1. infers the scalar potential ``Φ_hat`` on the lower boundary in Fourier space,
2. propagates that potential upward mode-by-mode with ``exp(-k_h dz)``, and
3. reconstructs the magnetic field components from the propagated potential.

That provides a cleaner baseline for comparison with PINN solutions that also
enforce the source-free field equations.

Conventions
-----------
1. ``z`` increases upward, away from the source region.
2. ``dz`` is positive and means "propagate upward by dz", i.e.

       z_top = z_bottom + dz.

3. The Fourier convention is NumPy's ``fft2`` / ``ifft2`` convention.
4. The method is local-planar. For a truly global lunar problem on a shell, a
   spherical-harmonic continuation is the more natural global analogue.

Inputs
------
You may supply either:

A) Only ``Bz_bottom``
   In that case, the lower-boundary potential is estimated from

       Bz_hat = k_h Φ_hat

   for all nonzero horizontal wavenumbers.

B) ``Bx_bottom``, ``By_bottom``, and ``Bz_bottom``
   In that case, the script uses a mode-by-mode least-squares estimate of the
   boundary potential:

       Bx_hat = -i kx Φ_hat
       By_hat = -i ky Φ_hat
       Bz_hat =  k_h Φ_hat

   which yields

       Φ_hat = [i kx Bx_hat + i ky By_hat + k_h Bz_hat] / [2 k_h²].

For noiseless potential-field data this is exact. If the supplied lower
boundary is not perfectly potential-consistent, this acts as a simple spectral
projection onto the nearest potential-field representation.

Outputs
-------
The script can return:

1. The upward-continued magnetic field at a single target height ``z = dz``.
2. Optionally, a full 3-D stack of magnetic-field planes between the lower and
   upper boundaries for comparison with interior PINN solutions.

The main outputs are:

- ``Bx_top``, ``By_top``, ``Bz_top``
    the field on the upper boundary,
- ``Bx_bottom_projected``, ``By_bottom_projected``, ``Bz_bottom_projected``
    the potential-field projection of the supplied lower boundary,
- ``phi_bottom_hat`` and ``phi_top_hat``
    the lower and upper boundary scalar potentials in Fourier space,
- optional 3-D arrays ``Bx_volume``, ``By_volume``, ``Bz_volume`` if requested.

Notes on the k = 0 mode
-----------------------
The horizontally uniform mode requires special handling. The exponentially
decaying branch ``exp(-k_h z)`` becomes unity for ``k_h = 0``, so the mean
component is not damped by the upward operator. In this script the mean of each
supplied magnetic component is preserved if ``preserve_mean=True``.

This is a practical convention for a local patch. Depending on your scientific
setup, you may wish to treat the mean mode differently.

FFT periodicity and edge artefacts
----------------------------------
Because the method uses FFTs, it implicitly assumes periodicity in ``x`` and
``y``. Strong discontinuities at the patch edges can therefore wrap around and
contaminate the spectral continuation. To mitigate this, the script includes an
optional cosine taper that smoothly damps the edges before the Fourier
transform.

No regularization is included
-----------------------------
Unlike downward continuation, upward continuation is spectrally stable because
it multiplies each mode by ``exp(-k_h dz)``. Therefore, this script does **not**
include regularization. Small-scale power is naturally attenuated with height.

Example (import as a module)
----------------------------
>>> import numpy as np
>>> from fourier_upward_continuation import upward_continue_magnetic_field
>>> data = np.load("bottom_boundary_field.npz")
>>> out = upward_continue_magnetic_field(
...     Bz_bottom=data["Bz_bottom"],
...     Bx_bottom=data.get("Bx_bottom"),
...     By_bottom=data.get("By_bottom"),
...     dx=10_000.0,
...     dy=10_000.0,
...     dz=50_000.0,
...     taper_fraction=0.05,
... )
>>> np.savez("top_boundary_field.npz", **out)

Example (return the full Cartesian volume)
------------------------------------------
>>> out = upward_continue_magnetic_field(
...     Bz_bottom=data["Bz_bottom"],
...     Bx_bottom=data.get("Bx_bottom"),
...     By_bottom=data.get("By_bottom"),
...     dx=10_000.0,
...     dy=10_000.0,
...     dz=50_000.0,
...     z_levels=np.linspace(0.0, 50_000.0, 11),
... )
>>> out["Bz_volume"].shape
(11, ny, nx)

Example (command line)
----------------------
python fourier_upward_continuation.py \
    --input bottom_boundary_field.npz \
    --output top_boundary_field.npz \
    --dx 10000 --dy 10000 --dz 50000 \
    --taper-fraction 0.05

To also write a full 3-D volume sampled on ``nz`` planes from ``z=0`` to
``z=dz``:

python fourier_upward_continuation.py \
    --input bottom_boundary_field.npz \
    --output top_boundary_field.npz \
    --dx 10000 --dy 10000 --dz 50000 \
    --nz 21

The input NPZ file must contain at least ``Bz_bottom``. If it also contains
``Bx_bottom`` and ``By_bottom``, those will be used automatically.
"""

from __future__ import print_function

import argparse
import json
from typing import Any, Dict, Optional, Sequence, Tuple

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
        Real-space 2-D array with shape ``(ny, nx)``.

    Returns
    -------
    np.ndarray
        Complex Fourier coefficients with the same shape.
    """
    return np.fft.fft2(arr)



def _ifft2_realspace(arr_hat: np.ndarray) -> np.ndarray:
    """
    Return the inverse 2-D discrete Fourier transform.

    Parameters
    ----------
    arr_hat : np.ndarray
        Complex Fourier coefficients.

    Returns
    -------
    np.ndarray
        Complex spatial-domain array. Physical fields are taken as the real
        part of the inverse transform.
    """
    return np.fft.ifft2(arr_hat)


# -----------------------------------------------------------------------------
# Spectral grids and tapers
# -----------------------------------------------------------------------------

def wavenumber_grids(nx: int, ny: int, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the horizontal wavenumber grids.

    Parameters
    ----------
    nx, ny : int
        Number of grid points in the x and y directions.
    dx, dy : float
        Grid spacing in x and y.

    Returns
    -------
    kx, ky, kh : tuple of np.ndarray
        Two-dimensional grids of horizontal wavenumbers and their magnitude,

            kh = sqrt(kx^2 + ky^2).

    Notes
    -----
    NumPy's FFT convention is used, so the returned wavenumbers are angular
    wavenumbers in radians per unit length.
    """
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="xy")
    kh = np.sqrt(kx_grid ** 2 + ky_grid ** 2)
    return kx_grid, ky_grid, kh



def _cosine_taper_1d(n: int, fraction: float) -> np.ndarray:
    """
    Create a one-dimensional cosine taper near the array ends.

    Parameters
    ----------
    n : int
        Number of samples.
    fraction : float
        Fraction of the domain length to taper on each side. Must satisfy
        ``0 <= fraction < 0.5``.

    Returns
    -------
    np.ndarray
        One-dimensional taper of length ``n``.
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
    Create a separable two-dimensional cosine taper.

    Parameters
    ----------
    shape : tuple of int
        Array shape ``(ny, nx)``.
    taper_fraction : float, optional
        Fraction of each edge to taper.

    Returns
    -------
    np.ndarray
        Two-dimensional taper with the same shape as the boundary arrays.
    """
    ny, nx = shape
    wx = _cosine_taper_1d(nx, taper_fraction)
    wy = _cosine_taper_1d(ny, taper_fraction)
    return np.outer(wy, wx)


# -----------------------------------------------------------------------------
# Physics helpers
# -----------------------------------------------------------------------------

def upward_operator(kh: np.ndarray, dz: float) -> np.ndarray:
    """
    Construct the mode-by-mode upward continuation operator.

    Parameters
    ----------
    kh : np.ndarray
        Horizontal wavenumber magnitude.
    dz : float
        Positive upward continuation distance,

            dz = z_top - z_bottom > 0.

    Returns
    -------
    np.ndarray
        Real attenuation factor for each spectral mode,

            U(kh, dz) = exp(-kh dz).

    Notes
    -----
    This is the Fourier-space propagator for the upward-decaying branch of the
    harmonic scalar potential above the source region.
    """
    if dz < 0.0:
        raise ValueError("dz must be positive. Use dz = z_top - z_bottom.")
    kh = np.asarray(kh, dtype=float)
    return np.exp(-kh * dz)



def estimate_phi_hat_from_bottom_boundary(
    Bz_bottom_hat: np.ndarray,
    Bx_bottom_hat: Optional[np.ndarray],
    By_bottom_hat: Optional[np.ndarray],
    kx: np.ndarray,
    ky: np.ndarray,
    kh: np.ndarray,
) -> np.ndarray:
    """
    Estimate the scalar potential on the lower boundary in Fourier space.

    Parameters
    ----------
    Bz_bottom_hat : np.ndarray
        Fourier coefficients of the vertical field on the lower boundary.
    Bx_bottom_hat, By_bottom_hat : np.ndarray or None
        Optional Fourier coefficients of the horizontal field.
    kx, ky, kh : np.ndarray
        Wavenumber grids.

    Returns
    -------
    np.ndarray
        Fourier coefficients of the scalar potential on the lower plane.

    Notes
    -----
    For each nonzero horizontal wavenumber, a potential magnetic field obeys

        Bx_hat = -i kx Phi_hat
        By_hat = -i ky Phi_hat
        Bz_hat =  kh Phi_hat.

    If only ``Bz_bottom_hat`` is provided, this routine uses

        Phi_hat = Bz_hat / kh.

    If the full vector field is provided, it computes the least-squares estimate

        Phi_hat = [i kx Bx_hat + i ky By_hat + kh Bz_hat] / [2 kh^2].

    The ``k=0`` mode is set to zero here and the spatial mean of the magnetic
    field components is handled later by direct preservation if requested.
    """
    phi_hat = np.zeros_like(Bz_bottom_hat, dtype=np.complex128)
    mask = kh > 0.0

    if Bx_bottom_hat is None or By_bottom_hat is None:
        phi_hat[mask] = Bz_bottom_hat[mask] / kh[mask]
    else:
        numerator = (
            1j * kx[mask] * Bx_bottom_hat[mask]
            + 1j * ky[mask] * By_bottom_hat[mask]
            + kh[mask] * Bz_bottom_hat[mask]
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

def upward_continue_magnetic_field(
    Bz_bottom: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    Bx_bottom: Optional[np.ndarray] = None,
    By_bottom: Optional[np.ndarray] = None,
    taper_fraction: float = 0.0,
    preserve_mean: bool = True,
    z_levels: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Upward continue a potential magnetic field from a lower boundary to a
    higher boundary, and optionally through the whole Cartesian layer.

    Parameters
    ----------
    Bz_bottom : np.ndarray
        Vertical magnetic field on the lower boundary, shape ``(ny, nx)``.
    dx, dy : float
        Horizontal grid spacing.
    dz : float
        Positive upward continuation distance,

            z_top = z_bottom + dz.

    Bx_bottom, By_bottom : np.ndarray, optional
        Optional horizontal magnetic field components on the lower boundary.
        If supplied, both must be supplied.
    taper_fraction : float, optional
        Fraction of the domain edge to cosine-taper before the Fourier
        transform. This reduces wrap-around artefacts for nonperiodic patches.
    preserve_mean : bool, optional
        If True, preserve the spatial mean of each supplied magnetic component.
    z_levels : sequence of float, optional
        Heights above the lower boundary at which to evaluate the field. If
        omitted, only the upper boundary at ``z = dz`` is returned. If supplied,
        the routine also returns volume arrays sampled at these heights.

    Returns
    -------
    dict
        Dictionary containing the upward-continued upper boundary field,
        projected lower boundary field, spectral arrays, and diagnostics.

        Main keys are:

        - ``Bx_top``, ``By_top``, ``Bz_top``
        - ``Bx_bottom_projected``, ``By_bottom_projected``, ``Bz_bottom_projected``
        - ``phi_bottom_hat``, ``phi_top_hat``
        - ``kx``, ``ky``, ``kh``
        - ``upward_operator``
        - ``diagnostics``

        If ``z_levels`` is provided, the dictionary also contains:

        - ``z_levels``
        - ``Bx_volume``, ``By_volume``, ``Bz_volume``

        each with shape ``(nz, ny, nx)``.

    Notes
    -----
    The returned ``*_bottom_projected`` arrays are useful diagnostics. They show
    the potential-field representation implied by the supplied boundary after
    projection into the Fourier potential-field model. Comparing them to the raw
    input lets you quantify how incompatible the supplied boundary is with the
    potential-field assumption.
    """
    Bz_bottom = np.asarray(Bz_bottom, dtype=float)
    if Bz_bottom.ndim != 2:
        raise ValueError("Bz_bottom must be a 2-D array with shape (ny, nx).")

    if (Bx_bottom is None) ^ (By_bottom is None):
        raise ValueError("Either supply both Bx_bottom and By_bottom, or neither.")

    if Bx_bottom is not None:
        Bx_bottom = np.asarray(Bx_bottom, dtype=float)
        By_bottom = np.asarray(By_bottom, dtype=float)
        if Bx_bottom.shape != Bz_bottom.shape or By_bottom.shape != Bz_bottom.shape:
            raise ValueError("Bx_bottom, By_bottom, and Bz_bottom must have identical shapes.")

    ny, nx = Bz_bottom.shape
    kx, ky, kh = wavenumber_grids(nx=nx, ny=ny, dx=dx, dy=dy)
    taper = make_2d_taper(Bz_bottom.shape, taper_fraction=taper_fraction)

    Bz_bottom_tapered = Bz_bottom * taper
    Bz_bottom_hat = _fft2_realspace(Bz_bottom_tapered)

    if Bx_bottom is not None:
        Bx_bottom_tapered = Bx_bottom * taper
        By_bottom_tapered = By_bottom * taper
        Bx_bottom_hat = _fft2_realspace(Bx_bottom_tapered)
        By_bottom_hat = _fft2_realspace(By_bottom_tapered)
    else:
        Bx_bottom_tapered = None
        By_bottom_tapered = None
        Bx_bottom_hat = None
        By_bottom_hat = None

    phi_bottom_hat = estimate_phi_hat_from_bottom_boundary(
        Bz_bottom_hat=Bz_bottom_hat,
        Bx_bottom_hat=Bx_bottom_hat,
        By_bottom_hat=By_bottom_hat,
        kx=kx,
        ky=ky,
        kh=kh,
    )

    # Reconstruct the projected lower-boundary field implied by the potential model.
    Bx_bottom_proj_hat, By_bottom_proj_hat, Bz_bottom_proj_hat = field_from_phi_hat(
        phi_hat=phi_bottom_hat,
        kx=kx,
        ky=ky,
        kh=kh,
    )

    Bx_bottom_projected = np.real(_ifft2_realspace(Bx_bottom_proj_hat))
    By_bottom_projected = np.real(_ifft2_realspace(By_bottom_proj_hat))
    Bz_bottom_projected = np.real(_ifft2_realspace(Bz_bottom_proj_hat))

    if preserve_mean:
        Bz_bottom_projected += float(np.mean(Bz_bottom))
        if Bx_bottom is not None:
            Bx_bottom_projected += float(np.mean(Bx_bottom))
            By_bottom_projected += float(np.mean(By_bottom))

    # Propagate the scalar potential to the requested upper boundary.
    U = upward_operator(kh=kh, dz=dz)
    phi_top_hat = phi_bottom_hat * U

    Bx_top_hat, By_top_hat, Bz_top_hat = field_from_phi_hat(
        phi_hat=phi_top_hat,
        kx=kx,
        ky=ky,
        kh=kh,
    )

    Bx_top = np.real(_ifft2_realspace(Bx_top_hat))
    By_top = np.real(_ifft2_realspace(By_top_hat))
    Bz_top = np.real(_ifft2_realspace(Bz_top_hat))

    if preserve_mean:
        Bz_top += float(np.mean(Bz_bottom))
        if Bx_bottom is not None:
            Bx_top += float(np.mean(Bx_bottom))
            By_top += float(np.mean(By_bottom))

    diagnostics: Dict[str, Any] = {
        "nx": int(nx),
        "ny": int(ny),
        "dx": float(dx),
        "dy": float(dy),
        "dz": float(dz),
        "taper_fraction": float(taper_fraction),
        "preserve_mean": bool(preserve_mean),
        "kh_min_nonzero": float(np.min(kh[kh > 0.0])) if np.any(kh > 0.0) else 0.0,
        "kh_max": float(np.max(kh)),
        "operator_min": float(np.min(U)),
        "operator_max": float(np.max(U)),
        "rms_Bz_bottom_input": _rms(Bz_bottom),
        "rms_Bz_bottom_projected": _rms(Bz_bottom_projected),
        "rms_Bz_top": _rms(Bz_top),
    }

    if Bx_bottom is not None:
        diagnostics.update({
            "rms_Bx_bottom_input": _rms(Bx_bottom),
            "rms_By_bottom_input": _rms(By_bottom),
            "rms_Bx_bottom_projected": _rms(Bx_bottom_projected),
            "rms_By_bottom_projected": _rms(By_bottom_projected),
            "rms_Bx_top": _rms(Bx_top),
            "rms_By_top": _rms(By_top),
        })

    result: Dict[str, Any] = {
        "Bx_top": Bx_top,
        "By_top": By_top,
        "Bz_top": Bz_top,
        "Bx_bottom_projected": Bx_bottom_projected,
        "By_bottom_projected": By_bottom_projected,
        "Bz_bottom_projected": Bz_bottom_projected,
        "phi_bottom_hat": phi_bottom_hat,
        "phi_top_hat": phi_top_hat,
        "kx": kx,
        "ky": ky,
        "kh": kh,
        "upward_operator": U,
        "taper": taper,
        "diagnostics": diagnostics,
    }

    # Optional interior stack.
    if z_levels is not None:
        z_levels = np.asarray(z_levels, dtype=float)
        if z_levels.ndim != 1:
            raise ValueError("z_levels must be a one-dimensional sequence of heights.")
        if np.any(z_levels < 0.0):
            raise ValueError("z_levels must all be >= 0, measured upward from the bottom boundary.")
        if np.any(z_levels > dz):
            raise ValueError("All z_levels must satisfy z_levels <= dz.")

        nz = z_levels.size
        Bx_volume = np.empty((nz, ny, nx), dtype=float)
        By_volume = np.empty((nz, ny, nx), dtype=float)
        Bz_volume = np.empty((nz, ny, nx), dtype=float)

        for iz, zlev in enumerate(z_levels):
            Uz = upward_operator(kh=kh, dz=float(zlev))
            phi_z_hat = phi_bottom_hat * Uz
            Bx_z_hat, By_z_hat, Bz_z_hat = field_from_phi_hat(
                phi_hat=phi_z_hat,
                kx=kx,
                ky=ky,
                kh=kh,
            )

            bx = np.real(_ifft2_realspace(Bx_z_hat))
            by = np.real(_ifft2_realspace(By_z_hat))
            bz = np.real(_ifft2_realspace(Bz_z_hat))

            if preserve_mean:
                bz += float(np.mean(Bz_bottom))
                if Bx_bottom is not None:
                    bx += float(np.mean(Bx_bottom))
                    by += float(np.mean(By_bottom))

            Bx_volume[iz] = bx
            By_volume[iz] = by
            Bz_volume[iz] = bz

        result.update({
            "z_levels": z_levels,
            "Bx_volume": Bx_volume,
            "By_volume": By_volume,
            "Bz_volume": Bz_volume,
        })

    return result


# -----------------------------------------------------------------------------
# CLI helpers
# -----------------------------------------------------------------------------

def _load_input_npz(path: str) -> Dict[str, np.ndarray]:
    """
    Load an NPZ file containing lower-boundary magnetic-field arrays.

    Expected keys
    -------------
    Required:
        - ``Bz_bottom``

    Optional:
        - ``Bx_bottom``
        - ``By_bottom``
    """
    with np.load(path) as data:
        out = {key: data[key] for key in data.files}
    if "Bz_bottom" not in out:
        raise KeyError("Input NPZ file must contain 'Bz_bottom'.")
    return out



def _save_output_npz(path: str, result: Dict[str, Any]) -> None:
    """Save the result dictionary to an NPZ file."""
    to_save: Dict[str, Any] = {}
    for key, value in result.items():
        if key == "diagnostics":
            to_save["diagnostics_json"] = np.array(json.dumps(value, indent=2), dtype=object)
        else:
            to_save[key] = value
    np.savez(path, **to_save)



def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Upward continue a lower-boundary magnetic field using a Fourier "
            "potential-field model on a local Cartesian patch."
        )
    )
    parser.add_argument("--input", required=True, help="Input NPZ containing at least Bz_bottom.")
    parser.add_argument("--output", required=True, help="Output NPZ filename.")
    parser.add_argument("--dx", type=float, required=True, help="Grid spacing in x.")
    parser.add_argument("--dy", type=float, required=True, help="Grid spacing in y.")
    parser.add_argument("--dz", type=float, required=True, help="Positive upward continuation distance.")
    parser.add_argument(
        "--taper-fraction",
        type=float,
        default=0.0,
        help="Fraction of each edge to cosine-taper before the FFT.",
    )
    parser.add_argument(
        "--no-preserve-mean",
        action="store_true",
        help="Do not preserve the mean magnetic-field components.",
    )
    parser.add_argument(
        "--nz",
        type=int,
        default=None,
        help=(
            "If supplied, also evaluate the field on nz evenly spaced planes from "
            "z=0 to z=dz and save Bx_volume, By_volume, Bz_volume."
        ),
    )
    return parser.parse_args()



def main() -> None:
    """Run the command-line interface."""
    args = _parse_args()
    data = _load_input_npz(args.input)

    if args.nz is not None:
        if args.nz < 2:
            raise ValueError("If --nz is supplied, it must be >= 2.")
        z_levels = np.linspace(0.0, float(args.dz), int(args.nz))
    else:
        z_levels = None

    result = upward_continue_magnetic_field(
        Bz_bottom=data["Bz_bottom"],
        Bx_bottom=data.get("Bx_bottom"),
        By_bottom=data.get("By_bottom"),
        dx=float(args.dx),
        dy=float(args.dy),
        dz=float(args.dz),
        taper_fraction=float(args.taper_fraction),
        preserve_mean=not args.no_preserve_mean,
        z_levels=z_levels,
    )

    _save_output_npz(args.output, result)

    print("Saved upward-continued field to:", args.output)
    print(json.dumps(result["diagnostics"], indent=2))
    if z_levels is not None:
        print("Saved interior volume with {} z-levels.".format(len(z_levels)))


if __name__ == "__main__":
    main()
