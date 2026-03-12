"""Hamstring-to-Quadriceps (H:Q) activation ratio calculation."""

import numpy as np
from ..constants import GaitPhase, EMG_QUAD_BASELINE, EMG_HAM_BASELINE


def compute_hq_conventional(emg_quad: np.ndarray, emg_ham: np.ndarray,
                            phase_labels: np.ndarray) -> float:
    """Conventional H:Q ratio: ham_concentric / quad_concentric during mid-stance.

    Target: 0.5-0.8

    Args:
        emg_quad: (T,) raw quad EMG
        emg_ham: (T,) raw hamstring EMG
        phase_labels: (T,) gait phase labels

    Returns:
        Conventional H:Q ratio
    """
    mid_stance_mask = phase_labels == GaitPhase.MID_STANCE
    if not mid_stance_mask.any():
        return float("nan")

    quad_act = max(emg_quad[mid_stance_mask].mean() - EMG_QUAD_BASELINE, 1e-6)
    ham_act = max(emg_ham[mid_stance_mask].mean() - EMG_HAM_BASELINE, 0.0)

    return float(ham_act / quad_act)


def compute_hq_functional(emg_quad: np.ndarray, emg_ham: np.ndarray,
                          phase_labels: np.ndarray) -> float:
    """Functional H:Q ratio: ham_eccentric (terminal swing) / quad_concentric (loading).

    More clinically relevant for injury risk assessment.

    Args:
        emg_quad: (T,) raw quad EMG
        emg_ham: (T,) raw hamstring EMG
        phase_labels: (T,) gait phase labels

    Returns:
        Functional H:Q ratio
    """
    loading_mask = phase_labels == GaitPhase.LOADING_RESPONSE
    terminal_swing_mask = phase_labels == GaitPhase.TERMINAL_SWING

    if not loading_mask.any() or not terminal_swing_mask.any():
        return float("nan")

    quad_loading = max(emg_quad[loading_mask].mean() - EMG_QUAD_BASELINE, 1e-6)
    ham_terminal = max(emg_ham[terminal_swing_mask].mean() - EMG_HAM_BASELINE, 0.0)

    return float(ham_terminal / quad_loading)
