"""Quadriceps Inhibition Index (QII) calculation."""

import numpy as np
from ..constants import GaitPhase, EMG_QUAD_BASELINE


def compute_qii(emg_quad: np.ndarray, phase_labels: np.ndarray,
                healthy_reference: float | None = None) -> float:
    """Compute Quadriceps Inhibition Index.

    QII = 1 - (mean_quad_activation_during_loading / healthy_reference)

    Args:
        emg_quad: (T,) raw quad EMG values
        phase_labels: (T,) gait phase labels
        healthy_reference: reference peak quad activation (healthy norm).
            If None, uses the max activation across all phases as reference.

    Returns:
        QII value in [0, 1]. Higher = more inhibited. Target: < 0.10
    """
    loading_mask = phase_labels == GaitPhase.LOADING_RESPONSE
    if not loading_mask.any():
        return float("nan")

    loading_activation = emg_quad[loading_mask].mean() - EMG_QUAD_BASELINE

    if healthy_reference is None:
        # Use max activation minus baseline as reference
        healthy_reference = emg_quad.max() - EMG_QUAD_BASELINE

    if healthy_reference <= 0:
        return 1.0

    qii = 1.0 - (loading_activation / healthy_reference)
    return float(np.clip(qii, 0.0, 1.0))
