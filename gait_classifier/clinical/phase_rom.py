"""Phase-gated knee Range of Motion (ROM) computation."""

import numpy as np
from ..constants import GaitPhase, IMU_YAW_BASELINE


def compute_phase_rom(imu_yaw: np.ndarray, phase_labels: np.ndarray) -> dict:
    """Compute knee ROM metrics per gait phase.

    Uses imu_yaw (flexion axis) to extract:
    - stance_flexion_range: ROM during loading response
    - swing_peak_flexion: peak flexion during initial swing
    - terminal_extension: angle at end of terminal swing (near full extension)
    - swing_clearance: swing_peak / 60° norm (target > 0.90)

    Args:
        imu_yaw: (T,) yaw angle values in degrees
        phase_labels: (T,) gait phase labels

    Returns:
        dict with ROM metrics
    """
    # Convert to knee flexion angle (offset from baseline)
    flexion = imu_yaw - IMU_YAW_BASELINE

    results = {}

    # Stance flexion range during loading response
    loading_mask = phase_labels == GaitPhase.LOADING_RESPONSE
    if loading_mask.any():
        loading_vals = flexion[loading_mask]
        results["stance_flexion_range"] = float(loading_vals.max() - loading_vals.min())
        results["stance_peak_flexion"] = float(loading_vals.max())
    else:
        results["stance_flexion_range"] = float("nan")
        results["stance_peak_flexion"] = float("nan")

    # Swing peak flexion during initial swing
    swing_mask = phase_labels == GaitPhase.INITIAL_SWING
    if swing_mask.any():
        results["swing_peak_flexion"] = float(flexion[swing_mask].max())
    else:
        results["swing_peak_flexion"] = float("nan")

    # Terminal extension
    terminal_mask = phase_labels == GaitPhase.TERMINAL_SWING
    if terminal_mask.any():
        terminal_vals = flexion[terminal_mask]
        results["terminal_extension"] = float(terminal_vals[-1] if len(terminal_vals) > 0
                                              else float("nan"))
    else:
        results["terminal_extension"] = float("nan")

    # Swing clearance relative to 60° norm
    norm_flexion = 60.0
    if not np.isnan(results.get("swing_peak_flexion", float("nan"))):
        results["swing_clearance"] = results["swing_peak_flexion"] / norm_flexion
    else:
        results["swing_clearance"] = float("nan")

    return results
