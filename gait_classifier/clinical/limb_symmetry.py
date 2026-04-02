"""Limb Symmetry Index (LSI) computation."""

import numpy as np
from ..constants import GaitPhase, EMG_QUAD_BASELINE, IMU_YAW_BASELINE


def compute_lsi_bilateral(
    involved_metrics: dict,
    uninvolved_metrics: dict,
) -> dict:
    """Compute LSI from bilateral (two-session) data.

    Primary approach: device worn on each leg in separate sessions.
    LSI = (involved / uninvolved) × 100 for each metric.

    Args:
        involved_metrics: dict with keys like "peak_quad_activation", "swing_rom", "stance_time"
        uninvolved_metrics: same keys for uninvolved (healthy) leg

    Returns:
        dict of LSI values per metric (100 = symmetric, target > 90 for return-to-sport)
    """
    results = {}
    for key in involved_metrics:
        uninvolved_val = uninvolved_metrics.get(key, 0)
        if uninvolved_val and uninvolved_val != 0:
            results[key] = float(involved_metrics[key] / uninvolved_val * 100)
        else:
            results[key] = float("nan")
    return results


def extract_session_metrics(emg_quad: np.ndarray, imu_yaw: np.ndarray,
                            phase_labels: np.ndarray) -> dict:
    """Extract per-session metrics for bilateral LSI comparison.

    Args:
        emg_quad: (T,) quad EMG
        imu_yaw: (T,) yaw angle
        phase_labels: (T,) gait phase labels

    Returns:
        dict with peak_quad_activation, swing_rom, stance_pct
    """
    loading_mask = phase_labels == GaitPhase.LOADING_RESPONSE
    swing_mask = phase_labels == GaitPhase.INITIAL_SWING
    stance_mask = (phase_labels <= GaitPhase.TERMINAL_STANCE)

    peak_quad = float(emg_quad[loading_mask].max() - EMG_QUAD_BASELINE) if loading_mask.any() else 0.0
    flexion = imu_yaw - IMU_YAW_BASELINE
    swing_rom = float(flexion[swing_mask].max() - flexion[swing_mask].min()) if swing_mask.any() else 0.0
    stance_pct = float(stance_mask.sum() / len(phase_labels) * 100) if len(phase_labels) > 0 else 0.0

    return {
        "peak_quad_activation": peak_quad,
        "swing_rom": swing_rom,
        "stance_pct": stance_pct,
    }


def compute_lsi_temporal(phase_labels: np.ndarray,
                         healthy_stance_pct: float = 60.0) -> float:
    """Compute temporal asymmetry LSI from single-leg data.

    Fallback approach: uses stance-to-swing ratio as a proxy.
    LSI = (observed_stance_pct / healthy_reference) × 100

    Args:
        phase_labels: (T,) gait phase labels
        healthy_stance_pct: reference stance percentage (default 60%)

    Returns:
        LSI value (100 = symmetric)
    """
    if len(phase_labels) == 0:
        return float("nan")

    stance_mask = (phase_labels <= GaitPhase.TERMINAL_STANCE)
    observed_stance_pct = stance_mask.sum() / len(phase_labels) * 100

    return float(observed_stance_pct / healthy_stance_pct * 100)
