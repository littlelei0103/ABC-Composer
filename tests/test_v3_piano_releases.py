"""Regression checks for audible lead tails overlapping the next melodic note."""

import unittest
from unittest.mock import patch

import numpy as np

from v3.audio import studio_renderer as renderer


class PianoReleaseTests(unittest.TestCase):
    def render_note(self, start=0.0, onsets=None):
        sample_rate = 44100
        t = np.arange(2 * sample_rate, dtype=np.float64) / sample_rate
        tone = np.sin(2 * np.pi * 330 * t).astype(np.float32)
        wave = np.stack((tone, tone))
        with patch.object(renderer, "_get_zero_click_single_note_wave", return_value=wave):
            return renderer.render_polyphonic_notes_zero_click(
                [(start, 1.01, 64, 60)], bank=0, prog=0, total_sec=2.0,
                sf2_path="unused.sf2", sample_rate=sample_rate,
                release_sec=0.20, release_before_onsets=onsets,
            )

    def test_entire_release_ends_before_next_attack(self):
        audio = self.render_note(onsets=[0.0, 1.0])
        self.assertGreater(float(np.max(np.abs(audio[:, 30000:40000]))), 0.1)
        np.testing.assert_array_equal(audio[:, round(0.995 * 44100):], 0.0)

    def test_delayed_double_obeys_all_lead_onsets(self):
        # The next lead can be absent from the optional double's own note list.
        audio = self.render_note(start=0.006, onsets=[0.0, 1.0])
        np.testing.assert_array_equal(audio[:, round(0.995 * 44100):], 0.0)
        self.assertGreater(float(np.max(np.abs(audio[:, 30000:40000]))), 0.1)

    def test_polyphonic_parts_can_keep_their_release(self):
        audio = self.render_note()
        self.assertGreater(float(np.max(np.abs(audio[:, 44100:48000]))), 0.1)


if __name__ == "__main__":
    unittest.main()
