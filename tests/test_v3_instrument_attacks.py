"""Check instrument-specific envelopes and same-pitch retriggers."""

import unittest
from unittest.mock import patch

import numpy as np

from v3.audio import studio_renderer as renderer


class InstrumentAttackTests(unittest.TestCase):
    def test_all_instrument_fades_have_silent_endpoints(self):
        sr = 44100
        # A source with nonzero endpoints exposes a missing envelope immediately.
        source = np.ones((2, sr), dtype=np.float32)
        for name, attack_ms in (("lead", 70), ("accompaniment", 35),
                                ("strings", 160), ("bass", 25),
                                ("kick", 8), ("hihat", 4)):
            with self.subTest(instrument=name):
                audio = renderer._shape_note_segment(source, sr, round(sr * attack_ms / 1000), 4410)
                np.testing.assert_array_equal(audio[:, 0], 0.0)
                np.testing.assert_array_equal(audio[:, -1], 0.0)
                self.assertGreater(float(np.max(audio)), 0.99)
                self.assertLess(float(np.max(np.abs(np.diff(audio[:, :100], axis=1)))), 0.01)
        np.testing.assert_array_equal(source, 1.0)

    def test_hihat_can_use_a_shorter_attack_than_kick(self):
        sr = 44100
        source = np.ones((2, sr), dtype=np.float32)
        with patch.object(renderer, "_get_zero_click_single_note_wave", return_value=source):
            outputs = []
            for pitch in (36, 42):
                outputs.append(renderer.render_polyphonic_notes_zero_click(
                    [(0.0, 0.12, pitch, 48)], bank=128, prog=0, total_sec=1.0,
                    sf2_path="unused", sample_rate=sr, attack_sec=0.008,
                    attack_sec_by_pitch={42: 0.004},
                ))
        at_4ms = round(0.004 * sr)
        self.assertGreater(float(outputs[1][0, at_4ms]), float(outputs[0][0, at_4ms]) * 1.8)
        self.assertGreater(float(np.max(outputs[0])), 0.6)
        self.assertGreater(float(np.max(outputs[1])), 0.6)

    def test_retrigger_finishes_same_pitch_but_preserves_chord(self):
        sr = 44100
        source = np.ones((2, 2 * sr), dtype=np.float32)
        notes = [(0.0, 0.9, 48, 60), (0.3, 0.8, 52, 60), (0.5, 0.9, 48, 60)]
        with patch.object(renderer, "_get_zero_click_single_note_wave", return_value=source):
            audio = renderer.render_polyphonic_notes_zero_click(
                notes, bank=8, prog=0, total_sec=1.5, sf2_path="unused",
                sample_rate=sr, release_before_same_pitch=True,
            )
        # Immediately before the retrigger, E3 still sounds but the old C3 has ended.
        np.testing.assert_allclose(audio[:, round(0.497 * sr)], np.sqrt(0.5), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
