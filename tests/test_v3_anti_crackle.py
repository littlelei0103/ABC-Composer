"""Regression checks ensuring stem leveling, solo voices, and SF2 presets stay crackle-free."""

import unittest
from pathlib import Path

import numpy as np

from v3.audio import studio_renderer as renderer


class AntiCrackleTests(unittest.TestCase):
    def test_stem_loudness_normalization_stays_linear(self):
        sr = 44100
        t = np.arange(sr, dtype=np.float64) / sr
        envelope = np.linspace(0.05, 1.5, sr)
        stereo = np.stack((envelope * np.sin(2 * np.pi * 440 * t),
                           0.6 * envelope * np.sin(2 * np.pi * 330 * t))).astype(np.float32)

        normalized = renderer._normalize_stem_loudness(stereo, target_rms=0.25, soft_ceiling=0.30)

        mask = np.abs(stereo) > 0.05
        gains = normalized[mask] / stereo[mask]
        self.assertLess(float(np.ptp(gains)), 1e-6)
        self.assertLessEqual(float(np.max(np.abs(normalized))), 0.300001)

    def test_bass_lowpass_has_no_circular_wrap_clicks(self):
        sr = 44100
        t = np.arange(sr, dtype=np.float64) / sr
        ramp_in = 0.5 * (1.0 - np.cos(np.pi * np.minimum(t / 0.05, 1.0)))
        bass = (ramp_in * np.sin(2 * np.pi * 110 * t) * 0.4).astype(np.float32)
        stereo = np.stack((bass, bass))

        filtered = renderer._lowpass_bass_stem(stereo, sr, cutoff_hz=1550.0)

        self.assertTrue(np.all(np.isfinite(filtered)))
        # Nonzero tail at t=1.0s must not wrap back into t=0.0s.
        self.assertLess(float(np.max(np.abs(filtered[:, :10]))), 0.01)

    def test_solo_cello_and_flute_have_no_loop_clicks(self):
        sr = 44100
        for prog, pitch in ((42, 67), (73, 79)):
            with self.subTest(prog=prog, pitch=pitch):
                wave = renderer._get_zero_click_single_note_wave(
                    "unused", sr, bank=0, prog=prog, pitch=pitch, vel=64, max_sec=3.0,
                )
                self.assertEqual(wave.shape, (2, 3 * sr))
                self.assertTrue(np.all(np.isfinite(wave)))
                np.testing.assert_array_equal(wave[:, 0], 0.0)
                spec = np.abs(np.fft.rfft(wave.astype(np.float64) * np.hanning(wave.shape[1]), axis=1)) ** 2
                freqs = np.fft.rfftfreq(wave.shape[1], 1.0 / sr)
                hf_ratio = float(spec[:, freqs > 6000].sum() / max(spec.sum(), 1e-12))
                self.assertLess(hf_ratio, 1e-5)

    @unittest.skipUnless(Path(renderer.DEFAULT_SF2).exists(), "MuseScore_General.sf2 not present")
    def test_sf2_guitar_ep_and_hihat_notes_are_declicked(self):
        sr = 44100
        sf2 = renderer.resolve_sf2_path(None)
        for bank, prog, pitch, max_d3_ratio in (
            (0, 24, 60, 0.02),
            (0, 4, 72, 0.02),
            (128, 0, 42, 0.20),
        ):
            with self.subTest(bank=bank, prog=prog, pitch=pitch):
                wave = renderer._get_zero_click_single_note_wave(
                    sf2, sr, bank=bank, prog=prog, pitch=pitch, vel=64, max_sec=3.0,
                )
                pk = max(float(np.max(np.abs(wave))), 1e-9)
                d3 = float(np.max(np.abs(np.diff(wave, n=3, axis=1))))
                self.assertLess(d3 / pk, max_d3_ratio)

    @unittest.skipUnless(Path(renderer.DEFAULT_SF2).exists(), "MuseScore_General.sf2 not present")
    def test_sf2_piano_calibration_tames_pitch_and_pan_spikes(self):
        sr = 44100
        sf2 = renderer.resolve_sf2_path(None)
        win = int(0.50 * sr)
        core_pitches = (48, 55, 60, 62, 64, 67, 69, 72)
        all_pitches = (*core_pitches, 84)
        for bank, prog in ((0, 0), (8, 0), (0, 1)):
            with self.subTest(bank=bank, prog=prog):
                rms_by_pitch = {}
                for pitch in all_pitches:
                    wave = renderer._get_zero_click_single_note_wave(
                        sf2, sr, bank=bank, prog=prog, pitch=pitch, vel=60, max_sec=1.0,
                    )[:, :win].astype(np.float64)
                    rms_l = float(np.sqrt(np.mean(wave[0] ** 2)))
                    rms_r = float(np.sqrt(np.mean(wave[1] ** 2)))
                    rms = float(np.sqrt(0.5 * (rms_l ** 2 + rms_r ** 2)))
                    rms_by_pitch[pitch] = rms
                    pan_ratio = max(rms_l, rms_r) / max(min(rms_l, rms_r), 1e-9)
                    self.assertLess(pan_ratio, 1.45, msg=f"pitch={pitch} pan_ratio={pan_ratio:.3f}")
                core_vals = [rms_by_pitch[p] for p in core_pitches]
                self.assertLess(max(core_vals) / max(min(core_vals), 1e-9), 1.65)
                all_vals = [rms_by_pitch[p] for p in all_pitches]
                self.assertLess(max(all_vals) / max(min(all_vals), 1e-9), 3.0)

    def test_high_register_strings_stay_silky_without_treble_screech(self):
        sr = 44100
        wave = renderer.render_velvet_harmonic_strings(
            [(0.05, 1.65, 84, 82)], total_sec=2.2, sample_rate=sr,
        ).astype(np.float64)
        sustain = wave[:, int(0.35 * sr):int(1.45 * sr)]
        win = np.hanning(sustain.shape[1])
        spec = np.abs(np.fft.rfft(sustain * win, axis=1)) ** 2
        freqs = np.fft.rfftfreq(sustain.shape[1], 1.0 / sr)
        fund_e = float(spec[:, (freqs >= 950) & (freqs <= 1150)].sum())
        harsh_e = float(spec[:, (freqs >= 1900) & (freqs <= 4000)].sum())
        ultra_e = float(spec[:, freqs > 6000].sum())
        self.assertLess(harsh_e / max(fund_e, 1e-12), 0.025)
        self.assertLess(ultra_e / max(float(spec.sum()), 1e-12), 1e-12)
        d2 = float(np.max(np.abs(np.diff(wave, n=2, axis=1))))
        self.assertLess(d2, 0.0025)

    def test_unison_piano_doubles_use_soft_attack_and_level(self):
        for name, orch in renderer.PRESET_ORCHESTRATIONS.items():
            if orch["double"] in ((0, 0), (8, 0)) and orch["double_transpose"] == 0:
                with self.subTest(preset=name):
                    self.assertIsNotNone(orch["double_attack"])
                    self.assertGreaterEqual(orch["double_attack"], 0.06)
                    self.assertLessEqual(orch["double_rms"], 0.022)


if __name__ == "__main__":
    unittest.main()
