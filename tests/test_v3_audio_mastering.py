"""Regression checks for sharp peaks introduced by V3's former mastering limiter."""

import unittest
from unittest.mock import patch

import numpy as np

from v3.audio import studio_renderer as renderer


class MasteringTests(unittest.TestCase):
    def test_mastering_preserves_waveform_through_loudness_changes(self):
        sr = 44100
        t = np.arange(sr, dtype=np.float64) / sr
        amplitude = np.linspace(0.01, 1.4, sr)
        left = amplitude * np.sin(2 * np.pi * 440 * t)
        right = 0.4 * amplitude * np.sin(2 * np.pi * 330 * t)
        source = np.stack((left, right)).astype(np.float32)
        output = renderer._master_stereo_mix(source, sr)
        # Quiet and loud sections, and both channels, must share one gain.
        mask = np.abs(source) > 0.01
        gains = output[mask] / source[mask]
        self.assertLess(float(np.ptp(gains)), 1e-6)
        self.assertLessEqual(float(np.max(np.abs(output))), 0.860001)

    @unittest.skipUnless(renderer.HAS_PEDALBOARD, "pedalboard is not installed")
    def test_sine_peaks_do_not_acquire_clicks(self):
        sample_rate = 44100
        t = np.arange(4 * sample_rate, dtype=np.float64) / sample_rate
        tone = (0.86 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        stereo = np.stack((tone, tone))

        mastered = renderer._master_stereo_mix(stereo, sample_rate)

        # A smooth 440 Hz tone has gradual slope changes. The former Limiter
        # produced a 0.0546 second difference at a crest in this same signal.
        curvature = float(np.max(np.abs(np.diff(mastered, n=2, axis=1))))
        self.assertLess(curvature, 0.01)
        self.assertLessEqual(float(np.max(np.abs(mastered))), 0.860001)

    def test_silence_stays_silent(self):
        silence = np.zeros((2, 44100), dtype=np.float32)
        mastered = renderer._master_stereo_mix(silence, 44100)
        np.testing.assert_array_equal(mastered, silence)

    def test_fallback_keeps_headroom_without_pedalboard(self):
        t = np.arange(44100, dtype=np.float64) / 44100
        tone = (8 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        with patch.object(renderer, "HAS_PEDALBOARD", False):
            mastered = renderer._master_stereo_mix(np.stack((tone, -tone)), 44100)

        self.assertTrue(np.all(np.isfinite(mastered)))
        self.assertLessEqual(float(np.max(np.abs(mastered))), 0.860001)
        np.testing.assert_allclose(mastered[0], -mastered[1], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
