"""Verify the audit observes every rendered instrument, including percussion."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import symusic

from v3.audio import studio_renderer as renderer
from v3.audio.audit_onsets import inspect_score


class OnsetAuditTests(unittest.TestCase):
    def test_audit_counts_real_voice_envelopes_for_all_roles(self):
        score = symusic.Score(480)
        for name, program, is_drum, pitches in (
            ("LEAD_HOOK", 0, False, [64]), ("PIANO_ARP", 0, False, [48]),
            ("STRINGS_PAD", 48, False, [60]), ("BASS", 32, False, [36]),
            ("DRUMS", 0, True, [36, 42]),
        ):
            track = symusic.Track(name, program, is_drum)
            for index, pitch in enumerate(pitches):
                track.notes.append(symusic.Note(index * 240, 480, pitch, 72))
            score.tracks.append(track)
        score.tempos.append(symusic.Tempo(0, 72))

        def source(*args, **kwargs):
            # Nonzero source endpoints make the real envelope necessary.
            return np.ones((2, int(kwargs["max_sec"] * kwargs["sample_rate"])), dtype=np.float32)

        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            midi = directory / "score.mid"
            score.dump_midi(str(midi))
            with patch.object(renderer, "HAS_PEDALBOARD", False), \
                 patch.object(renderer, "_get_zero_click_single_note_wave", side_effect=source):
                report = inspect_score(midi, directory / "audio.wav", directory / "reports")
        self.assertEqual(report["voices_checked"], 7)  # Six input notes plus the piano double.
        self.assertEqual(set(report["by_instrument"]),
                         {"lead", "piano_double", "accompaniment", "strings", "bass", "kick", "hihat"})
        self.assertEqual(report["boundary_flags"], 0)
        self.assertEqual(report["nonfinite_voices"], 0)
        self.assertEqual(report["clipped_pcm_samples"], 0)


if __name__ == "__main__":
    unittest.main()
