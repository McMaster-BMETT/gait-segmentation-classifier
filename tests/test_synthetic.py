"""Tests for synthetic gait data generation."""

import numpy as np
import pytest

from gait_classifier.data.synthetic import GaitCycleGenerator
from gait_classifier.data.healing_profiles import ALL_PROFILES, STAGE4_HEALTHY, STAGE1_INJURED
from gait_classifier.constants import (
    NUM_CHANNELS, NUM_PHASES, GaitPhase,
    EMG_QUAD_BASELINE, EMG_HAM_BASELINE, EMG_ADC_MIN, EMG_ADC_MAX,
)


@pytest.fixture
def healthy_gen():
    return GaitCycleGenerator(STAGE4_HEALTHY, sampling_rate=50, rng=np.random.default_rng(42))


@pytest.fixture
def injured_gen():
    return GaitCycleGenerator(STAGE1_INJURED, sampling_rate=50, rng=np.random.default_rng(42))


class TestGaitCycleGenerator:
    def test_output_shapes(self, healthy_gen):
        data, labels = healthy_gen.generate_cycle()
        assert data.ndim == 2
        assert data.shape[0] == NUM_CHANNELS
        assert labels.shape == (data.shape[1],)

    def test_label_coverage(self, healthy_gen):
        data, labels = healthy_gen.generate_cycle()
        unique = set(labels.tolist())
        assert unique == set(range(NUM_PHASES)), f"Missing phases: {set(range(NUM_PHASES)) - unique}"

    def test_emg_quad_baseline(self, healthy_gen):
        data, _ = healthy_gen.generate_cycle()
        emg_quad = data[6]
        assert np.median(emg_quad) > EMG_QUAD_BASELINE - 200
        assert np.median(emg_quad) < EMG_QUAD_BASELINE + 500

    def test_emg_ham_baseline(self, healthy_gen):
        data, _ = healthy_gen.generate_cycle()
        emg_ham = data[7]
        assert np.median(emg_ham) > EMG_HAM_BASELINE - 50
        assert np.median(emg_ham) < EMG_HAM_BASELINE + 200

    def test_emg_within_adc_range(self, healthy_gen):
        data, _ = healthy_gen.generate_cycle()
        assert data[6].min() >= EMG_ADC_MIN
        assert data[6].max() <= EMG_ADC_MAX
        assert data[7].min() >= EMG_ADC_MIN
        assert data[7].max() <= EMG_ADC_MAX

    def test_injured_lower_swing_flexion(self, injured_gen, healthy_gen):
        """Injured profile should have lower peak swing flexion than healthy."""
        injured_data, _ = injured_gen.generate_cycle()
        healthy_data, _ = healthy_gen.generate_cycle()
        # imu_yaw is channel 2 (flexion axis)
        assert injured_data[2].max() < healthy_data[2].max()

    def test_session_concatenation(self, healthy_gen):
        data, labels = healthy_gen.generate_session(num_cycles=5)
        assert data.ndim == 2
        assert data.shape[0] == NUM_CHANNELS
        assert labels.shape == (data.shape[1],)
        # Should be longer than a single cycle
        single_data, _ = healthy_gen.generate_cycle()
        assert data.shape[1] > single_data.shape[1]

    def test_all_profiles_generate(self):
        """All healing profiles should generate valid data."""
        for profile in ALL_PROFILES:
            gen = GaitCycleGenerator(profile, rng=np.random.default_rng(0))
            data, labels = gen.generate_cycle()
            assert data.shape[0] == NUM_CHANNELS
            assert labels.shape[0] == data.shape[1]
            assert set(labels.tolist()).issubset(set(range(NUM_PHASES)))

    def test_reproducibility(self):
        gen1 = GaitCycleGenerator(STAGE4_HEALTHY, rng=np.random.default_rng(123))
        gen2 = GaitCycleGenerator(STAGE4_HEALTHY, rng=np.random.default_rng(123))
        d1, l1 = gen1.generate_cycle()
        d2, l2 = gen2.generate_cycle()
        np.testing.assert_array_equal(d1, d2)
        np.testing.assert_array_equal(l1, l2)
