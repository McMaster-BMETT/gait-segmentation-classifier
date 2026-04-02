"""Tests for clinical metric computations."""

import numpy as np
import pytest

from gait_classifier.constants import GaitPhase, EMG_QUAD_BASELINE, EMG_HAM_BASELINE, IMU_YAW_BASELINE
from gait_classifier.clinical.quad_inhibition import compute_qii
from gait_classifier.clinical.hq_ratio import compute_hq_conventional, compute_hq_functional
from gait_classifier.clinical.phase_rom import compute_phase_rom
from gait_classifier.clinical.limb_symmetry import (
    compute_lsi_bilateral, extract_session_metrics, compute_lsi_temporal,
)


def _make_labels(phase_counts: dict) -> np.ndarray:
    """Helper: create label array from {phase: count} dict."""
    parts = []
    for phase in GaitPhase:
        parts.append(np.full(phase_counts.get(phase, 10), int(phase)))
    return np.concatenate(parts)


class TestQII:
    def test_zero_activation_gives_qii_one(self):
        labels = _make_labels({GaitPhase.LOADING_RESPONSE: 20})
        emg = np.full(len(labels), EMG_QUAD_BASELINE, dtype=float)
        # healthy reference = 1000 ADC above baseline
        qii = compute_qii(emg, labels, healthy_reference=1000.0)
        assert qii == pytest.approx(1.0)

    def test_full_activation_gives_qii_zero(self):
        labels = _make_labels({GaitPhase.LOADING_RESPONSE: 20})
        ref = 500.0
        emg = np.full(len(labels), EMG_QUAD_BASELINE + ref, dtype=float)
        qii = compute_qii(emg, labels, healthy_reference=ref)
        assert qii == pytest.approx(0.0, abs=0.01)

    def test_no_loading_response_gives_nan(self):
        labels = np.full(50, int(GaitPhase.MID_STANCE))
        emg = np.full(50, EMG_QUAD_BASELINE + 100.0)
        qii = compute_qii(emg, labels)
        assert np.isnan(qii)


class TestHQRatio:
    def test_conventional_ratio(self):
        labels = _make_labels({GaitPhase.MID_STANCE: 50})
        emg_quad = np.full(len(labels), EMG_QUAD_BASELINE + 200.0)
        emg_ham = np.full(len(labels), EMG_HAM_BASELINE + 100.0)
        ratio = compute_hq_conventional(emg_quad, emg_ham, labels)
        assert ratio == pytest.approx(100.0 / 200.0, rel=0.01)

    def test_functional_ratio(self):
        labels = _make_labels({
            GaitPhase.LOADING_RESPONSE: 20,
            GaitPhase.TERMINAL_SWING: 30,
        })
        emg_quad = np.full(len(labels), EMG_QUAD_BASELINE + 400.0)
        emg_ham = np.full(len(labels), EMG_HAM_BASELINE + 200.0)
        ratio = compute_hq_functional(emg_quad, emg_ham, labels)
        assert ratio == pytest.approx(200.0 / 400.0, rel=0.01)


class TestPhaseROM:
    def test_swing_peak(self):
        labels = _make_labels({GaitPhase.INITIAL_SWING: 50})
        imu_yaw = np.full(len(labels), IMU_YAW_BASELINE + 55.0)
        rom = compute_phase_rom(imu_yaw, labels)
        assert rom["swing_peak_flexion"] == pytest.approx(55.0, abs=1.0)

    def test_swing_clearance(self):
        labels = _make_labels({GaitPhase.INITIAL_SWING: 50})
        imu_yaw = np.full(len(labels), IMU_YAW_BASELINE + 60.0)
        rom = compute_phase_rom(imu_yaw, labels)
        assert rom["swing_clearance"] == pytest.approx(1.0, abs=0.01)


class TestLSI:
    def test_bilateral_symmetric(self):
        involved = {"peak_quad_activation": 500, "swing_rom": 55, "stance_pct": 60}
        uninvolved = {"peak_quad_activation": 500, "swing_rom": 55, "stance_pct": 60}
        lsi = compute_lsi_bilateral(involved, uninvolved)
        for key in lsi:
            assert lsi[key] == pytest.approx(100.0)

    def test_bilateral_asymmetric(self):
        involved = {"peak_quad_activation": 400}
        uninvolved = {"peak_quad_activation": 500}
        lsi = compute_lsi_bilateral(involved, uninvolved)
        assert lsi["peak_quad_activation"] == pytest.approx(80.0)

    def test_temporal_symmetric(self):
        # 60% stance = healthy reference → LSI = 100
        labels = np.concatenate([
            np.zeros(60, dtype=int),   # stance phases
            np.full(40, int(GaitPhase.INITIAL_SWING)),
        ])
        lsi = compute_lsi_temporal(labels, healthy_stance_pct=60.0)
        assert lsi == pytest.approx(100.0, abs=1.0)
