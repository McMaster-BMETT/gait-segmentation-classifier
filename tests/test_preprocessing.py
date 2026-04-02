"""Tests for preprocessing utilities."""

import numpy as np
import pytest

from gait_classifier.data.preprocessing import Normalizer, compute_angular_velocity, compute_emg_rms


class TestNormalizer:
    def test_fit_transform(self):
        data = np.array([[10.0, 20.0, 30.0], [100.0, 200.0, 300.0]])
        norm = Normalizer()
        result = norm.fit_transform(data)
        # Each channel should have mean ~0 and std ~1
        np.testing.assert_allclose(result.mean(axis=1), 0, atol=1e-6)
        np.testing.assert_allclose(result.std(axis=1), 1, atol=1e-6)

    def test_transform_without_fit_raises(self):
        norm = Normalizer()
        with pytest.raises(RuntimeError):
            norm.transform(np.zeros((2, 10)))

    def test_get_params(self):
        data = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
        norm = Normalizer().fit(data)
        params = norm.get_params()
        assert len(params["mean"]) == 2
        assert len(params["std"]) == 2

    def test_constant_channel(self):
        """Constant channel should not cause division by zero."""
        data = np.array([[5.0, 5.0, 5.0], [1.0, 2.0, 3.0]])
        norm = Normalizer()
        result = norm.fit_transform(data)
        assert np.all(np.isfinite(result))


class TestAngularVelocity:
    def test_constant_angle(self):
        angle = np.ones(100) * 45.0
        vel = compute_angular_velocity(angle, dt=0.02)
        np.testing.assert_allclose(vel, 0, atol=1e-10)

    def test_linear_ramp(self):
        angle = np.linspace(0, 90, 100)
        vel = compute_angular_velocity(angle, dt=0.02)
        # Should be approximately constant
        expected = 90 / (99 * 0.02)
        np.testing.assert_allclose(vel[1:-1], expected, rtol=0.05)


class TestEMGRMS:
    def test_output_length(self):
        emg = np.random.randn(100)
        rms = compute_emg_rms(emg, window=5)
        assert len(rms) == len(emg)

    def test_positive(self):
        emg = np.random.randn(100)
        rms = compute_emg_rms(emg, window=5)
        assert np.all(rms >= 0)
