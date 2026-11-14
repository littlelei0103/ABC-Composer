"""Check attack smoothing without shifting notes or modifying cached voices."""

import unittest
from unittest.mock import patch

import numpy as np

from v3.audio import studio_renderer as renderer


class PianoAttackTests(unittest.TestCase):
    def test_attack_is_continuous_and_body_is_unchanged(self):
        sr = 44100
        cached = np.ones((2, sr), dtype=np.float32)
        kwargs = dict(notes=[(0.1, 0.8, 64, 60)], bank=0, prog=0,
                      total_sec=1.0, sf2_path="unused", sample_rate=sr)
        with patch.object(renderer, "_get_zero_click_single_note_wave", return_value=cached):
            plain = renderer.render_polyphonic_notes_zero_click(**kwargs)
            soft = renderer.render_polyphonic_notes_zero_click(**kwargs, attack_sec=0.035)
        onset = round(0.1 * sr)
        end_attack = onset + round(0.035 * sr)
        np.testing.assert_array_equal(soft[:, :onset + 1], 0.0)
        self.assertGreater(float(np.max(np.abs(soft[:, onset + 1:end_attack]))), 0.1)
        jump = float(np.max(np.abs(np.diff(soft[:, onset - 1:end_attack + 1], axis=1))))
        self.assertLess(jump, 0.002)
        np.testing.assert_array_equal(soft[:, end_attack:], plain[:, end_attack:])
        np.testing.assert_array_equal(cached, 1.0)

    def test_short_note_still_releases_to_silence(self):
        sr = 44100
        with patch.object(renderer, "_get_zero_click_single_note_wave",
                          return_value=np.ones((2, sr), dtype=np.float32)):
            audio = renderer.render_polyphonic_notes_zero_click(
                [(0.0, 0.1, 64, 60)], bank=0, prog=0, total_sec=1.0,
                sf2_path="unused", sample_rate=sr, attack_sec=0.035,
                release_before_onsets=[0.0, 0.08],
            )
        self.assertTrue(np.all(np.isfinite(audio)))
        self.assertGreater(float(np.max(np.abs(audio))), 0.1)
        np.testing.assert_array_equal(audio[:, round(0.075 * sr):], 0.0)


if __name__ == "__main__":
    unittest.main()
