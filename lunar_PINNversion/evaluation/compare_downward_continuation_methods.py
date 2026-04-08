"""Compare Fourier downward continuation (regularized/unregularized) vs PINN.

Workflow:
1) Load magnetic-field measurements on one constant-z plane from a regular XY grid.
2) Downward continue to z_target with:
   - unregularized Fourier continuation
   - regularized Fourier continuation
3) Optionally evaluate a PINN checkpoint on the same XY grid at z_target and z_obs.
4) Compute metrics:
   - measurement-fit metrics at z_obs (always available)
   - boundary-truth metrics at z_target when a truth file is provided
5) Save method outputs and metrics under output_dir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fourier_downward_continuation_solver import (  # noqa: E402
    downward_continue_xyz_fft,
    load_xyz_field_grid,
    save_xyz_field_grid_csv,
    upward_continue_xyz_fft,
)


def _stack_vector(bx: np.ndarray, by: np.ndarray, bz: np.ndarray) -> np.ndarray:
    return np.stack((bx, by, bz), axis=-1)


def _error_stats(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    err = pred - truth
    denom = np.linalg.norm(truth.reshape(-1)) + 1e-12
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "rel_l2": float(np.linalg.norm(err.reshape(-1)) / denom),
        "max_abs": float(np.max(np.abs(err))),
    }


def _field_metrics(
    pred_bx: np.ndarray,
    pred_by: np.ndarray,
    pred_bz: np.ndarray,
    true_bx: np.ndarray,
    true_by: np.ndarray,
    true_bz: np.ndarray,
) -> dict[str, dict[str, float]]:
    metrics = {
        "vector": _error_stats(
            _stack_vector(pred_bx, pred_by, pred_bz),
            _stack_vector(true_bx, true_by, true_bz),
        ),
        "bx": _error_stats(pred_bx, true_bx),
        "by": _error_stats(pred_by, true_by),
        "bz": _error_stats(pred_bz, true_bz),
    }
    return metrics


def _assert_same_grid(
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    x_other: np.ndarray,
    y_other: np.ndarray,
    *,
    label: str,
) -> None:
    if x_ref.shape != x_other.shape or y_ref.shape != y_other.shape:
        raise ValueError(f"{label} grid shape does not match measurement grid shape.")
    if not np.allclose(x_ref, x_other, rtol=0.0, atol=1e-12):
        raise ValueError(f"{label} x-grid values do not match measurement x-grid.")
    if not np.allclose(y_ref, y_other, rtol=0.0, atol=1e-12):
        raise ValueError(f"{label} y-grid values do not match measurement y-grid.")


def _choose_torch_device(device_arg: str):
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _load_pinn(checkpoint_path: str, device_arg: str):
    device = _choose_torch_device(device_arg)
    try:
        from lunar_PINNversion.PINNmodel.model import PINN
    except ImportError:
        from PINNmodel.model import PINN

    model, checkpoint = PINN.load_checkpoint(checkpoint_path, device=device)
    model.eval()
    return model, checkpoint, device


def _evaluate_pinn_on_plane(
    model,
    device,
    x: np.ndarray,
    y: np.ndarray,
    z_eval: float,
    scale_xyz: tuple[float, float, float],
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0.")

    x_grid, y_grid = np.meshgrid(x, y, indexing="xy")
    z_grid = np.full_like(x_grid, float(z_eval), dtype=np.float64)
    coords = np.column_stack((x_grid.reshape(-1), y_grid.reshape(-1), z_grid.reshape(-1)))
    scale = np.asarray(scale_xyz, dtype=np.float64).reshape(1, 3)
    if np.any(scale == 0.0):
        raise ValueError("All values in scale_xyz must be non-zero.")

    coords = coords / scale
    n_pts = coords.shape[0]
    b_parts = []

    for start in range(0, n_pts, chunk_size):
        end = min(start + chunk_size, n_pts)
        xyz = torch.tensor(coords[start:end], dtype=torch.float32, device=device, requires_grad=True)
        phi_pred = model(xyz)
        grad_phi = torch.autograd.grad(
            outputs=phi_pred,
            inputs=xyz,
            grad_outputs=torch.ones_like(phi_pred),
            create_graph=False,
        )[0]
        bxyz = (-grad_phi).detach().cpu().numpy()
        b_parts.append(bxyz)

    b_all = np.vstack(b_parts)
    ny = y.size
    nx = x.size
    bx = b_all[:, 0].reshape(ny, nx)
    by = b_all[:, 1].reshape(ny, nx)
    bz = b_all[:, 2].reshape(ny, nx)
    return bx, by, bz


def _metrics_to_lines(metrics: dict) -> list[str]:
    lines: list[str] = []
    for section_name, section in metrics.items():
        if not isinstance(section, dict):
            lines.append(f"{section_name}: {section}")
            continue
        lines.append(f"[{section_name}]")
        for method_name, method_metrics in section.items():
            lines.append(f"  {method_name}")
            for comp_name, comp_metrics in method_metrics.items():
                lines.append(
                    "    "
                    f"{comp_name}: rmse={comp_metrics['rmse']:.6e}, "
                    f"mae={comp_metrics['mae']:.6e}, "
                    f"rel_l2={comp_metrics['rel_l2']:.6e}, "
                    f"max_abs={comp_metrics['max_abs']:.6e}"
                )
    return lines


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Fourier downward continuation (regularized/unregularized) and PINN on one XY grid."
    )
    parser.add_argument("--measurements", type=str, required=True, help="Input measurements at z=z_obs.")
    parser.add_argument(
        "--truth-boundary",
        type=str,
        default=None,
        help="Optional truth field at z=z_target for boundary error metrics.",
    )
    parser.add_argument(
        "--pinn-checkpoint",
        type=str,
        default=None,
        help="Optional PINN checkpoint path for PINN-vs-Fourier comparison.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation/downward_continuation_comparison",
        help="Directory where outputs and metrics are written.",
    )
    parser.add_argument(
        "--z-observation",
        type=float,
        default=None,
        help="Optional override for measurement height. If omitted, uses z from measurements file.",
    )
    parser.add_argument("--z-target", type=float, default=0.0, help="Boundary height to reconstruct.")
    parser.add_argument(
        "--lambda-reg",
        type=float,
        default=1e-3,
        help="Regularization strength for regularized Fourier continuation.",
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
        help="Optional cap for downward continuation spectral gain.",
    )
    parser.add_argument(
        "--pinn-device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Torch device for PINN checkpoint evaluation.",
    )
    parser.add_argument(
        "--pinn-scale",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        metavar=("SX", "SY", "SZ"),
        help="Coordinate scale before PINN inference: [x/SX, y/SY, z/SZ].",
    )
    parser.add_argument(
        "--pinn-chunk-size",
        type=int,
        default=200000,
        help="Number of grid points per PINN autograd chunk.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x, y, z_from_file, bx_obs, by_obs, bz_obs = load_xyz_field_grid(args.measurements)
    z_obs = float(args.z_observation) if args.z_observation is not None else float(z_from_file)

    fourier_unreg = downward_continue_xyz_fft(
        x,
        y,
        bx_obs,
        by_obs,
        bz_obs,
        z_observation=z_obs,
        z_target=float(args.z_target),
        lambda_reg=0.0,
        reg_power=float(args.reg_power),
        max_gain=args.max_gain,
    )
    fourier_reg = downward_continue_xyz_fft(
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
        str(output_dir / "fourier_unregularized_boundary.csv"),
        x,
        y,
        float(args.z_target),
        fourier_unreg["bx"],
        fourier_unreg["by"],
        fourier_unreg["bz"],
    )
    save_xyz_field_grid_csv(
        str(output_dir / "fourier_regularized_boundary.csv"),
        x,
        y,
        float(args.z_target),
        fourier_reg["bx"],
        fourier_reg["by"],
        fourier_reg["bz"],
    )

    # Reproject Fourier reconstructions back up to z_obs for data-fit metrics.
    fourier_unreg_reproj = upward_continue_xyz_fft(
        x,
        y,
        fourier_unreg["bx"],
        fourier_unreg["by"],
        fourier_unreg["bz"],
        z_source=float(args.z_target),
        z_target=z_obs,
    )
    fourier_reg_reproj = upward_continue_xyz_fft(
        x,
        y,
        fourier_reg["bx"],
        fourier_reg["by"],
        fourier_reg["bz"],
        z_source=float(args.z_target),
        z_target=z_obs,
    )

    save_xyz_field_grid_csv(
        str(output_dir / "fourier_unregularized_reprojected_measurement_plane.csv"),
        x,
        y,
        z_obs,
        fourier_unreg_reproj["bx"],
        fourier_unreg_reproj["by"],
        fourier_unreg_reproj["bz"],
    )
    save_xyz_field_grid_csv(
        str(output_dir / "fourier_regularized_reprojected_measurement_plane.csv"),
        x,
        y,
        z_obs,
        fourier_reg_reproj["bx"],
        fourier_reg_reproj["by"],
        fourier_reg_reproj["bz"],
    )

    metrics: dict[str, dict] = {
        "config": {
            "measurements": str(Path(args.measurements).resolve()),
            "truth_boundary": None if args.truth_boundary is None else str(Path(args.truth_boundary).resolve()),
            "pinn_checkpoint": None if args.pinn_checkpoint is None else str(Path(args.pinn_checkpoint).resolve()),
            "z_observation": z_obs,
            "z_target": float(args.z_target),
            "lambda_reg": float(args.lambda_reg),
            "reg_power": float(args.reg_power),
            "max_gain": None if args.max_gain is None else float(args.max_gain),
            "pinn_scale": [float(s) for s in args.pinn_scale],
        },
        "measurement_fit": {},
    }

    metrics["measurement_fit"]["fourier_unregularized"] = _field_metrics(
        fourier_unreg_reproj["bx"],
        fourier_unreg_reproj["by"],
        fourier_unreg_reproj["bz"],
        bx_obs,
        by_obs,
        bz_obs,
    )
    metrics["measurement_fit"]["fourier_regularized"] = _field_metrics(
        fourier_reg_reproj["bx"],
        fourier_reg_reproj["by"],
        fourier_reg_reproj["bz"],
        bx_obs,
        by_obs,
        bz_obs,
    )

    pinn_boundary = None
    if args.pinn_checkpoint:
        model, checkpoint, device = _load_pinn(args.pinn_checkpoint, args.pinn_device)
        pinn_boundary_bx, pinn_boundary_by, pinn_boundary_bz = _evaluate_pinn_on_plane(
            model=model,
            device=device,
            x=x,
            y=y,
            z_eval=float(args.z_target),
            scale_xyz=tuple(float(s) for s in args.pinn_scale),
            chunk_size=int(args.pinn_chunk_size),
        )
        pinn_boundary = {
            "bx": pinn_boundary_bx,
            "by": pinn_boundary_by,
            "bz": pinn_boundary_bz,
            "epoch": int(checkpoint["epoch"]),
        }
        save_xyz_field_grid_csv(
            str(output_dir / "pinn_boundary.csv"),
            x,
            y,
            float(args.z_target),
            pinn_boundary_bx,
            pinn_boundary_by,
            pinn_boundary_bz,
        )

        pinn_obs_bx, pinn_obs_by, pinn_obs_bz = _evaluate_pinn_on_plane(
            model=model,
            device=device,
            x=x,
            y=y,
            z_eval=z_obs,
            scale_xyz=tuple(float(s) for s in args.pinn_scale),
            chunk_size=int(args.pinn_chunk_size),
        )
        save_xyz_field_grid_csv(
            str(output_dir / "pinn_measurement_plane.csv"),
            x,
            y,
            z_obs,
            pinn_obs_bx,
            pinn_obs_by,
            pinn_obs_bz,
        )

        metrics["measurement_fit"]["pinn"] = _field_metrics(
            pinn_obs_bx,
            pinn_obs_by,
            pinn_obs_bz,
            bx_obs,
            by_obs,
            bz_obs,
        )
        metrics["config"]["pinn_epoch"] = int(checkpoint["epoch"])

    if args.truth_boundary:
        x_truth, y_truth, z_truth, bx_truth, by_truth, bz_truth = load_xyz_field_grid(args.truth_boundary)
        _assert_same_grid(x, y, x_truth, y_truth, label="truth-boundary")
        if abs(float(z_truth) - float(args.z_target)) > 1e-9:
            raise ValueError(
                "truth-boundary file z-plane does not match z_target "
                f"({float(z_truth):.6g} vs {float(args.z_target):.6g})."
            )

        metrics["boundary_truth"] = {}
        metrics["boundary_truth"]["fourier_unregularized"] = _field_metrics(
            fourier_unreg["bx"],
            fourier_unreg["by"],
            fourier_unreg["bz"],
            bx_truth,
            by_truth,
            bz_truth,
        )
        metrics["boundary_truth"]["fourier_regularized"] = _field_metrics(
            fourier_reg["bx"],
            fourier_reg["by"],
            fourier_reg["bz"],
            bx_truth,
            by_truth,
            bz_truth,
        )
        if pinn_boundary is not None:
            metrics["boundary_truth"]["pinn"] = _field_metrics(
                pinn_boundary["bx"],
                pinn_boundary["by"],
                pinn_boundary["bz"],
                bx_truth,
                by_truth,
                bz_truth,
            )

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    summary_lines = _metrics_to_lines(
        {k: v for k, v in metrics.items() if k in ("measurement_fit", "boundary_truth")}
    )
    summary_text = "\n".join(summary_lines)
    (output_dir / "summary.txt").write_text(summary_text + "\n", encoding="utf-8")

    print(f"Saved comparison outputs to {output_dir}")
    print(f"Saved metrics JSON to {metrics_path}")
    if summary_lines:
        print("")
        print(summary_text)


if __name__ == "__main__":
    main()

