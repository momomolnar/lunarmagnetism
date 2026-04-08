#!/usr/bin/env python3
"""
synthetic_potential_field_box.py
================================

Generate a synthetic 3-D magnetic field in a Cartesian box that contains a few
localized strong-field regions separated by weaker-field regions, while
satisfying the source-free conditions inside the computational domain.

This script is designed as a clean comparison / test-bed for PINN-based lunar
or solar magnetic-field inversions.  The construction is intentionally simple:

1. A bottom-boundary normal field Bz(x, y, z=0) is prescribed as a sum of a few
   2-D Gaussian spots.
2. That boundary field is Fourier-decomposed in the horizontal directions.
3. Each horizontal Fourier mode is upward propagated into the volume according
   to the potential-field (Laplace) solution.
4. The full magnetic field B = (Bx, By, Bz) is reconstructed throughout the box.

Because the field is derived from a scalar potential Phi satisfying

    ∇² Phi = 0

in the volume z > 0, the magnetic field

    B = -∇Phi

satisfies

    ∇·B = 0
    ∇×B = 0

throughout the domain (up to numerical FFT / finite-difference errors).

Why not just sum arbitrary 3-D Gaussians for Bx, By, Bz?
---------------------------------------------------------
A generic sum of Gaussians for the vector components will *not* in general obey
both divergence-freeness and source-freeness.  By instead prescribing only the
bottom-boundary normal field and solving the potential problem above it, we get
Gaussian-like flux concentrations together with a mathematically consistent
3-D field.

Mathematical summary
--------------------
Let the bottom boundary be Bz(x, y, 0).  Its 2-D Fourier transform is

    B̂z(kx, ky, 0).

For a potential field in the half space z >= 0,

    Phî(kx, ky, z) = Phî(kx, ky, 0) exp(-kh z),

where

    kh = sqrt(kx² + ky²).

Using B = -∇Phi, the Fourier-space field at height z is

    B̂z(kx, ky, z) = B̂z(kx, ky, 0) exp(-kh z)
    B̂x(kx, ky, z) = -i (kx / kh) B̂z(kx, ky, 0) exp(-kh z)
    B̂y(kx, ky, z) = -i (ky / kh) B̂z(kx, ky, 0) exp(-kh z)

for kh > 0.  The kh = 0 mode corresponds to a uniform vertical field:

    Bx = By = 0,
    Bz = constant.

Boundary-model remarks
----------------------
The bottom boundary is constructed as

    Bz(x, y, 0) = Σ_j A_j G_j(x, y),

where each G_j is an anisotropic rotated 2-D Gaussian patch.  This gives a few
strong-field regions with weaker field in between.  The script supports both:

- deterministic spot placement (the default), and
- random spot generation with a fixed RNG seed.

Caveats
-------
1. The FFT formulation assumes periodicity in x and y.  If the field is strong
   near the horizontal boundaries, periodic wrap-around artefacts can occur.
   To mitigate this, keep the Gaussian spots well away from the edges and/or
   use a box larger than the visually interesting region.
2. This is a *planar* Cartesian potential-field construction.  It is not a
   substitute for a global spherical-harmonic lunar continuation method.
3. For even nx or ny, Nyquist-mode effects can slightly contaminate derivative-
   based diagnostics at the boundary plane z=0.  Using odd horizontal grid
   sizes avoids this issue.
4. The field is source-free only inside the modeled volume z >= 0.  The bottom
   boundary Bz is prescribed rather than derived from physical currents or
   internal magnetization below the plane.

Outputs
-------
The main output is a .npz file containing:

    x, y, z            1-D coordinate arrays [same length units as input]
    X, Y               2-D boundary mesh grids
    Bz_bottom          bottom-boundary normal field, shape (ny, nx)
    Bx, By, Bz         full 3-D field, shape (nz, ny, nx)
    divB_modal         modal divergence residual, shape (nz, ny, nx)
                       (should be near machine precision)
    spot_table         array describing the Gaussian spots used

An optional quick-look figure can also be saved, showing:

- the prescribed Bz boundary,
- Bx, By, Bz on the top boundary,
- and the modal divergence residual at mid-height.

Example usage
-------------
Deterministic default configuration:

    python synthetic_potential_field_box.py \
        --output synthetic_field.npz \
        --plot synthetic_field.png

Randomize the Gaussian spots:

    python synthetic_potential_field_box.py \
        --random-spots \
        --nspots 5 \
        --seed 42 \
        --output synthetic_random_field.npz

Increase the box size and resolution:

    python synthetic_potential_field_box.py \
        --nx 256 --ny 256 --nz 64 \
        --lx 4.0 --ly 4.0 --lz 1.0 \
        --output field_large.npz

Import as a module:

    from synthetic_potential_field_box import (
        GaussianSpot,
        build_gaussian_boundary,
        potential_field_volume_from_bz,
    )

    # define your own boundary spots
    spots = [
        GaussianSpot(x0=-0.3, y0=0.1, amplitude=1.0, sigma_x=0.12, sigma_y=0.08),
        GaussianSpot(x0=0.2,  y0=-0.2, amplitude=-0.9, sigma_x=0.10, sigma_y=0.10),
    ]

    x = np.linspace(-1, 1, 128)
    y = np.linspace(-1, 1, 128)
    z = np.linspace(0, 0.5, 32)

    X, Y, Bz0 = build_gaussian_boundary(x, y, spots, zero_mean=False)
    Bx, By, Bz = potential_field_volume_from_bz(Bz0, x, y, z)

Author intent
-------------
This script is meant to provide a mathematically self-consistent synthetic
volume field with visually localized magnetic structures, suitable for testing,
benchmarking, and comparison against PINN reconstructions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np


@dataclass
class GaussianSpot:
    """
    Parameters describing one Gaussian spot in the bottom-boundary Bz field.

    Parameters
    ----------
    x0, y0 : float
        Center of the spot in the same coordinate units as x and y.
    amplitude : float
        Peak amplitude of the Gaussian contribution to Bz at the boundary.
        Positive values produce outward normal field; negative values produce
        inward normal field.
    sigma_x, sigma_y : float
        Standard deviations of the Gaussian along its principal axes.
    theta_deg : float, optional
        Rotation angle in degrees of the principal axes, measured
        counterclockwise from the +x axis.
    """

    x0: float
    y0: float
    amplitude: float
    sigma_x: float
    sigma_y: float
    theta_deg: float = 0.0


# -----------------------------------------------------------------------------
# Boundary construction utilities
# -----------------------------------------------------------------------------


def gaussian_2d_rotated(
    X: np.ndarray,
    Y: np.ndarray,
    spot: GaussianSpot,
) -> np.ndarray:
    """
    Evaluate one rotated anisotropic 2-D Gaussian on a mesh.

    The Gaussian is

        G(x, y) = A exp[-0.5 ((x'/σx)^2 + (y'/σy)^2)]

    where (x', y') are coordinates rotated into the principal-axis frame of the
    spot.

    Parameters
    ----------
    X, Y : ndarray, shape (ny, nx)
        2-D coordinate grids.
    spot : GaussianSpot
        Definition of the Gaussian patch.

    Returns
    -------
    ndarray, shape (ny, nx)
        The Gaussian contribution on the mesh.
    """
    theta = np.deg2rad(spot.theta_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    dx = X - spot.x0
    dy = Y - spot.y0

    xp = cos_t * dx + sin_t * dy
    yp = -sin_t * dx + cos_t * dy

    return spot.amplitude * np.exp(
        -0.5 * ((xp / spot.sigma_x) ** 2 + (yp / spot.sigma_y) ** 2)
    )



def build_gaussian_boundary(
    x: np.ndarray,
    y: np.ndarray,
    spots: Sequence[GaussianSpot],
    zero_mean: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct the bottom-boundary Bz field as a sum of Gaussian spots.

    Parameters
    ----------
    x, y : ndarray
        1-D coordinate arrays.  The boundary plane is defined on the Cartesian
        product of these arrays.
    spots : sequence of GaussianSpot
        Spot definitions to be summed.
    zero_mean : bool, optional
        If True, subtract the spatial mean of the boundary Bz field after the
        Gaussian sum is constructed.  This removes the kh=0 mode, i.e. removes
        the uniform vertical background field from the potential solution.

    Returns
    -------
    X, Y : ndarray, shape (ny, nx)
        2-D coordinate meshes.
    Bz0 : ndarray, shape (ny, nx)
        Bottom-boundary normal field.
    """
    X, Y = np.meshgrid(x, y, indexing="xy")
    Bz0 = np.zeros_like(X, dtype=float)

    for spot in spots:
        Bz0 += gaussian_2d_rotated(X, Y, spot)

    if zero_mean:
        Bz0 = Bz0 - np.mean(Bz0)

    return X, Y, Bz0



def default_spots(lx: float, ly: float) -> List[GaussianSpot]:
    """
    Return a small deterministic collection of Gaussian spots.

    The default configuration is chosen to create a few localized stronger
    patches with weaker field between them.  The spots are intentionally kept
    away from the boundaries to reduce FFT periodic wrap-around artefacts.

    Parameters
    ----------
    lx, ly : float
        Full box lengths in x and y.

    Returns
    -------
    list of GaussianSpot
        Deterministic spot configuration.
    """
    return [
        GaussianSpot(
            x0=-0.28 * lx,
            y0=0.12 * ly,
            amplitude=1.00,
            sigma_x=0.08 * lx,
            sigma_y=0.06 * ly,
            theta_deg=20.0,
        ),
        GaussianSpot(
            x0=0.18 * lx,
            y0=-0.18 * ly,
            amplitude=-0.85,
            sigma_x=0.07 * lx,
            sigma_y=0.09 * ly,
            theta_deg=-10.0,
        ),
        GaussianSpot(
            x0=0.26 * lx,
            y0=0.24 * ly,
            amplitude=0.75,
            sigma_x=0.09 * lx,
            sigma_y=0.07 * ly,
            theta_deg=35.0,
        ),
        GaussianSpot(
            x0=-0.05 * lx,
            y0=-0.02 * ly,
            amplitude=0.35,
            sigma_x=0.14 * lx,
            sigma_y=0.12 * ly,
            theta_deg=0.0,
        ),
    ]



def random_spots(
    lx: float,
    ly: float,
    nspots: int,
    seed: int,
    amplitude_range: tuple[float, float] = (0.4, 1.1),
    sigma_frac_range: tuple[float, float] = (0.05, 0.12),
    balanced_polarities: bool = True,
) -> List[GaussianSpot]:
    """
    Create a random collection of Gaussian spots.

    Parameters
    ----------
    lx, ly : float
        Full box lengths in x and y.
    nspots : int
        Number of spots to generate.
    seed : int
        Random seed for reproducibility.
    amplitude_range : tuple(float, float), optional
        Range of absolute amplitudes.
    sigma_frac_range : tuple(float, float), optional
        Range of Gaussian widths as fractions of lx and ly.
    balanced_polarities : bool, optional
        If True, alternate the signs of the amplitudes to avoid a strongly
        one-signed boundary field.

    Returns
    -------
    list of GaussianSpot
        Randomized Gaussian spot configuration.
    """
    rng = np.random.default_rng(seed)
    spots: List[GaussianSpot] = []

    for i in range(nspots):
        x0 = rng.uniform(-0.30 * lx, 0.30 * lx)
        y0 = rng.uniform(-0.30 * ly, 0.30 * ly)
        amp = rng.uniform(*amplitude_range)
        if balanced_polarities:
            amp *= 1.0 if (i % 2 == 0) else -1.0
        sx = rng.uniform(*sigma_frac_range) * lx
        sy = rng.uniform(*sigma_frac_range) * ly
        theta_deg = rng.uniform(-45.0, 45.0)
        spots.append(
            GaussianSpot(
                x0=x0,
                y0=y0,
                amplitude=amp,
                sigma_x=sx,
                sigma_y=sy,
                theta_deg=theta_deg,
            )
        )

    return spots


# -----------------------------------------------------------------------------
# Potential-field volume reconstruction
# -----------------------------------------------------------------------------


def horizontal_wavenumbers(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct the 2-D horizontal Fourier wavenumber arrays.

    Parameters
    ----------
    x, y : ndarray
        1-D coordinate arrays.  Uniform spacing is assumed.

    Returns
    -------
    kx, ky, kh : ndarray, shape (ny, nx)
        Horizontal wavenumber arrays in radians per unit length.
        `kh = sqrt(kx^2 + ky^2)`.
    """
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1-D coordinate arrays.")
    if len(x) < 2 or len(y) < 2:
        raise ValueError("x and y must each contain at least two points.")

    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])

    if not np.allclose(np.diff(x), dx):
        raise ValueError("x must be uniformly spaced for the FFT formulation.")
    if not np.allclose(np.diff(y), dy):
        raise ValueError("y must be uniformly spaced for the FFT formulation.")

    kx_1d = 2.0 * np.pi * np.fft.fftfreq(len(x), d=dx)
    ky_1d = 2.0 * np.pi * np.fft.fftfreq(len(y), d=dy)
    kx, ky = np.meshgrid(kx_1d, ky_1d, indexing="xy")
    kh = np.sqrt(kx**2 + ky**2)

    return kx, ky, kh



def potential_field_volume_from_bz(
    Bz0: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct the full 3-D potential magnetic field from a bottom-boundary Bz.

    This solves the standard potential-field continuation problem in a Cartesian
    half-space, using 2-D FFTs in the horizontal directions and exponential mode
    decay in z.

    Parameters
    ----------
    Bz0 : ndarray, shape (ny, nx)
        Bottom-boundary vertical field Bz(x, y, z=0).
    x, y, z : ndarray
        1-D coordinate arrays.  x and y must be uniformly spaced.  z gives the
        heights at which the field should be reconstructed, with z >= 0.

    Returns
    -------
    Bx, By, Bz : ndarray, shape (nz, ny, nx)
        Reconstructed magnetic-field components throughout the volume.

    Notes
    -----
    In Fourier space, for kh > 0,

        B̂z(z) = B̂z(0) exp(-kh z)
        B̂x(z) = -i (kx / kh) B̂z(0) exp(-kh z)
        B̂y(z) = -i (ky / kh) B̂z(0) exp(-kh z)

    The kh = 0 mode is treated separately as a uniform vertical field:

        Bx = By = 0
        Bz = constant = mean(Bz0)
    """
    if Bz0.ndim != 2:
        raise ValueError("Bz0 must be a 2-D array with shape (ny, nx).")
    if np.any(z < 0.0):
        raise ValueError("This continuation formula assumes z >= 0.")

    ny, nx = Bz0.shape
    if len(x) != nx or len(y) != ny:
        raise ValueError(
            "Bz0 shape must be (len(y), len(x)). Got "
            f"Bz0.shape={Bz0.shape}, len(x)={len(x)}, len(y)={len(y)}."
        )

    kx, ky, kh = horizontal_wavenumbers(x, y)
    mask = kh > 0.0

    Bz0_hat = np.fft.fft2(Bz0)

    Bx = np.empty((len(z), ny, nx), dtype=float)
    By = np.empty((len(z), ny, nx), dtype=float)
    Bz = np.empty((len(z), ny, nx), dtype=float)

    for iz, zz in enumerate(z):
        decay = np.exp(-kh * zz)
        Bz_hat = Bz0_hat * decay

        Bx_hat = np.zeros_like(Bz_hat, dtype=complex)
        By_hat = np.zeros_like(Bz_hat, dtype=complex)

        Bx_hat[mask] = -1j * (kx[mask] / kh[mask]) * Bz_hat[mask]
        By_hat[mask] = -1j * (ky[mask] / kh[mask]) * Bz_hat[mask]

        Bx[iz] = np.fft.ifft2(Bx_hat).real
        By[iz] = np.fft.ifft2(By_hat).real
        Bz[iz] = np.fft.ifft2(Bz_hat).real

    return Bx, By, Bz


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------


def modal_divergence_residual(
    Bx: np.ndarray,
    By: np.ndarray,
    Bz: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """
    Compute the divergence residual using the modal potential-field relation.

    This is a stricter and more appropriate diagnostic for this FFT-based
    construction than a finite-difference z-derivative.  In Fourier space, a
    potential mode obeys

        ∂/∂z -> -kh,

    so the divergence residual is

        R_div = i kx B̂x + i ky B̂y - kh B̂z.

    For an exactly reconstructed potential field this residual is zero, up to
    floating-point roundoff.

    Parameters
    ----------
    Bx, By, Bz : ndarray, shape (nz, ny, nx)
        Field components on the Cartesian grid.
    x, y : ndarray
        1-D horizontal coordinate arrays.

    Returns
    -------
    residual : ndarray, shape (nz, ny, nx)
        Real-space modal divergence residual.
    """
    kx, ky, kh = horizontal_wavenumbers(x, y)
    residual = np.empty_like(Bx)

    for iz in range(Bx.shape[0]):
        Bxh = np.fft.fft2(Bx[iz])
        Byh = np.fft.fft2(By[iz])
        Bzh = np.fft.fft2(Bz[iz])
        Rh = 1j * kx * Bxh + 1j * ky * Byh - kh * Bzh
        residual[iz] = np.fft.ifft2(Rh).real

    return residual


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------


def spots_to_array(spots: Sequence[GaussianSpot]) -> np.ndarray:
    """
    Convert a list of GaussianSpot objects into a numeric table for saving.

    Returns
    -------
    ndarray, shape (nspots, 6)
        Columns are:
        [x0, y0, amplitude, sigma_x, sigma_y, theta_deg]
    """
    return np.array(
        [
            [s.x0, s.y0, s.amplitude, s.sigma_x, s.sigma_y, s.theta_deg]
            for s in spots
        ],
        dtype=float,
    )


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def save_quicklook_plot(
    filename: str,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    Bz_bottom: np.ndarray,
    Bx: np.ndarray,
    By: np.ndarray,
    Bz: np.ndarray,
    divB_modal: np.ndarray,
) -> None:
    """
    Save a quick-look diagnostic figure.

    This function is optional and only imported when needed, so that the script
    does not require matplotlib unless a plot is requested.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm

    extent = [x.min(), x.max(), y.min(), y.max()]
    iz_mid = len(z) // 2
    iz_top = len(z) - 1

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)

    panels = [
        (Bz_bottom, r"$B_z(x,y,0)$ bottom boundary"),
        (Bx[iz_top], rf"$B_x(x,y,z={z[iz_top]:.3g})$"),
        (By[iz_top], rf"$B_y(x,y,z={z[iz_top]:.3g})$"),
        (Bz[iz_top], rf"$B_z(x,y,z={z[iz_top]:.3g})$"),
        (divB_modal[iz_mid], rf"modal div residual at z={z[iz_mid]:.3g}"),
        (
            np.sqrt(Bx[iz_mid] ** 2 + By[iz_mid] ** 2 + Bz[iz_mid] ** 2),
            rf"$|B|$ at z={z[iz_mid]:.3g}",
        ),
    ]

    for ax, (img, title) in zip(axes.flat, panels):
        if "div" in title:
            vmax = np.nanmax(np.abs(img))
            norm = SymLogNorm(linthresh=max(vmax * 1e-6, 1e-14), vmin=-vmax, vmax=vmax)
            im = ax.imshow(img, origin="lower", extent=extent, aspect="auto", norm=norm)
        else:
            vmax = np.nanmax(np.abs(img))
            im = ax.imshow(
                img,
                origin="lower",
                extent=extent,
                aspect="auto",
                vmin=-vmax,
                vmax=vmax,
            )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.9)

    fig.suptitle("Synthetic divergence-free potential magnetic field", fontsize=14)
    fig.savefig(filename, dpi=180)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a synthetic 3-D divergence-free potential magnetic field "
            "in a Cartesian box from a Gaussian-spot bottom boundary."
        )
    )

    # Grid / domain
    parser.add_argument("--nx", type=int, default=191, help="Number of x points. Odd values avoid Nyquist-mode artefacts in diagnostics.")
    parser.add_argument("--ny", type=int, default=191, help="Number of y points. Odd values avoid Nyquist-mode artefacts in diagnostics.")
    parser.add_argument("--nz", type=int, default=48, help="Number of z points.")
    parser.add_argument("--lx", type=float, default=2.0, help="Full x extent.")
    parser.add_argument("--ly", type=float, default=2.0, help="Full y extent.")
    parser.add_argument("--lz", type=float, default=0.6, help="Top height of the box.")

    # Spot generation
    parser.add_argument(
        "--random-spots",
        action="store_true",
        help="Use random Gaussian spots instead of the deterministic default set.",
    )
    parser.add_argument("--nspots", type=int, default=4, help="Number of random spots.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for random spots.")
    parser.add_argument(
        "--zero-mean",
        action="store_true",
        help=(
            "Subtract the mean of the bottom-boundary Bz field so that the kh=0 "
            "mode (uniform vertical background field) is removed."
        ),
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="synthetic_potential_field_box.npz",
        help="Output .npz filename.",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Optional quick-look image filename (e.g. quicklook.png).",
    )

    return parser.parse_args()



def main() -> None:
    """Entry point for standalone script usage."""
    args = parse_args()

    x = np.linspace(-0.5 * args.lx, 0.5 * args.lx, args.nx)
    y = np.linspace(-0.5 * args.ly, 0.5 * args.ly, args.ny)
    z = np.linspace(0.0, args.lz, args.nz)

    if args.random_spots:
        spots = random_spots(
            lx=args.lx,
            ly=args.ly,
            nspots=args.nspots,
            seed=args.seed,
        )
    else:
        spots = default_spots(args.lx, args.ly)

    X, Y, Bz_bottom = build_gaussian_boundary(x, y, spots, zero_mean=args.zero_mean)
    Bx, By, Bz = potential_field_volume_from_bz(Bz_bottom, x, y, z)
    divB_modal = modal_divergence_residual(Bx, By, Bz, x, y)

    np.savez_compressed(
        args.output,
        x=x,
        y=y,
        z=z,
        X=X,
        Y=Y,
        Bz_bottom=Bz_bottom,
        Bx=Bx,
        By=By,
        Bz=Bz,
        divB_modal=divB_modal,
        spot_table=spots_to_array(spots),
    )

    print(f"Saved synthetic field cube to: {args.output}")
    print(f"Grid shape: nx={args.nx}, ny={args.ny}, nz={args.nz}")
    print(f"Domain: x in [{x.min()}, {x.max()}], y in [{y.min()}, {y.max()}], z in [{z.min()}, {z.max()}]")
    print(f"Number of spots: {len(spots)}")
    print(f"Boundary Bz min/max: {Bz_bottom.min():.6g} / {Bz_bottom.max():.6g}")
    if len(z) > 1:
        div_resid_max = np.nanmax(np.abs(divB_modal[1:]))
        print(f"Max |modal divergence residual| for z>0: {div_resid_max:.6e}")
    else:
        div_resid_max = np.nanmax(np.abs(divB_modal))
        print(f"Max |modal divergence residual|: {div_resid_max:.6e}")

    if args.plot is not None:
        save_quicklook_plot(args.plot, x, y, z, Bz_bottom, Bx, By, Bz, divB_modal)
        print(f"Saved quick-look figure to: {args.plot}")


if __name__ == "__main__":
    main()
