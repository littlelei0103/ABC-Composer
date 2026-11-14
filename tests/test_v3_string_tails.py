"""Check numerical noise and pitch stability in long additive string notes."""

import unittest

import numpy as np

from v3.audio import studio_renderer as renderer


class StringTailTests(unittest.TestCase):
    def test_long_low_note_has_no_broadband_phase_noise(self):
        sample_rate = 44100
        audio = renderer.render_velvet_harmonic_strings(
            [(0.0, 5.0, 48, 80)], total_sec=5.5, sample_rate=sample_rate,
        ).astype(np.float64)

        # All musical partials are below 400 Hz. Windowing suppresses finite
        # buffer leakage so energy above 6 kHz exposes numerical phase noise.
        spectrum = np.abs(np.fft.rfft(audio * np.hanning(audio.shape[1]), axis=1)) ** 2
        frequencies = np.fft.rfftfreq(audio.shape[1], 1.0 / sample_rate)
        noise_ratio = float(spectrum[:, frequencies > 6000].sum() / spectrum.sum())
        self.assertLess(noise_ratio, 1e-12)
        np.testing.assert_array_equal(audio[:, 0], np.zeros(2))
        np.testing.assert_array_equal(audio[:, round(5.28 * sample_rate) - 1], np.zeros(2))
        np.testing.assert_array_equal(audio[:, -1], np.zeros(2))

    def test_vibrato_depth_does_not_grow_in_long_note(self):
        sample_rate = 44100
        audio = renderer.render_velvet_harmonic_strings(
            [(0.0, 5.0, 60, 80)], total_sec=5.5, sample_rate=sample_rate,
        )
        signal = audio[0, sample_rate // 2:int(4.5 * sample_rate)].astype(np.float64)
        crossings = np.flatnonzero((signal[:-1] <= 0) & (signal[1:] > 0))
        times = crossings - signal[crossings] / (signal[crossings + 1] - signal[crossings])
        # The sub-octave partial repeats every two fundamental periods.
        frequencies = 2 * sample_rate / (times[2:] - times[:-2])
        relative_spread = float(np.ptp(frequencies) / np.mean(frequencies))
        self.assertLess(relative_spread, 0.01)


if __name__ == "__main__":
    unittest.main()
