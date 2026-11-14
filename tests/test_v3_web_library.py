"""Tests for paginated track catalog, bilingual search/filtering/sorting, and metadata persistence."""

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import symusic

from v3 import web_studio


def _write_minimal_midi(path: Path, bpm: float = 72.0) -> None:
    score = symusic.Score()
    score.tempos.append(symusic.Tempo(0, bpm))
    lead = symusic.Track(name="LEAD_HOOK", program=0, is_drum=False)
    lead.notes.append(symusic.Note(0, 480, 64, 80))
    lead.notes.append(symusic.Note(480, 480, 67, 84))
    score.tracks.append(lead)
    score.dump_midi(str(path))


class WebLibraryTests(unittest.TestCase):
    def _make_engine(self, root: Path) -> web_studio.StudioEngine:
        eng = web_studio.StudioEngine.__new__(web_studio.StudioEngine)
        eng.device = "cpu"
        eng.models = {}
        eng.lock = threading.Lock()
        eng.meta_lock = threading.RLock()
        eng.showcase_dir = root / "v3_showcase"
        eng.custom_dir = eng.showcase_dir / "custom"
        eng.custom_dir.mkdir(parents=True, exist_ok=True)
        eng.library_meta_path = eng.showcase_dir / "library_meta.json"
        eng._piano_roll_cache = {}
        eng._summary_cache = {}
        return eng

    def _seed_custom_tracks(self, eng: web_studio.StudioEngine, count: int) -> None:
        styles = [
            ("[Style:Ballad_Slow]", "[Key:C_maj]", "intro_chamber", "温暖晴天前奏"),
            ("[Style:Guofeng_Lyrical]", "[Key:A_min]", "guofeng_silk", "烟雨江南笛声"),
            ("[Style:Urban_Romantic]", "[Key:F_maj]", "urban_rhodes", "霓虹微醺夜色"),
        ]
        for idx in range(count):
            ts = 1794000000 + idx * 100
            slug = f"custom_{ts}_track_{idx:02d}"
            style, key, preset, prompt = styles[idx % len(styles)]
            mid_p = eng.custom_dir / f"{slug}.mid"
            wav_p = eng.custom_dir / f"{slug}.wav"
            json_p = eng.custom_dir / f"{slug}.json"
            _write_minimal_midi(mid_p)
            wav_p.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
            meta = {
                "slug": slug,
                "title": f"《作品{idx:02d}》",
                "style": style,
                "mood": "[Mood:Nostalgic_Warm]",
                "tempo": "[Tempo:72]",
                "tempo_bpm": 72,
                "key": key,
                "bars": 16,
                "chords": ["[Chord:C:maj7]"],
                "model_stage": "phase4_dpo" if idx % 2 == 0 else "phase3_grpo",
                "preset_style": preset,
                "nl_prompt": prompt,
                "duration_sec": 30.0 + idx,
                "created_at": ts,
                "verifier_metrics": {"r_total": round(1.5 + 0.01 * idx, 4)},
                # Simulate bloated legacy piano_roll inside saved JSON
                "piano_roll": {"tracks": [{"notes": [{"pitch": 60}] * 200}], "duration_sec": 30.0 + idx},
            }
            json_p.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    def test_pagination_removes_12_track_cap_and_strips_heavy_piano_roll(self):
        with tempfile.TemporaryDirectory() as tmp:
            eng = self._make_engine(Path(tmp))
            self._seed_custom_tracks(eng, 18)

            with patch.object(web_studio, "SHOWCASE_TRACKS", []):
                page1 = eng.query_tracks({"offset": "0", "limit": "12", "sort": "newest"})
                page2 = eng.query_tracks({"offset": "12", "limit": "12", "sort": "newest"})

                self.assertEqual(page1["total"], 18)
                self.assertEqual(len(page1["items"]), 12)
                self.assertTrue(page1["has_more"])
                self.assertEqual(len(page2["items"]), 6)
                self.assertFalse(page2["has_more"])
                self.assertEqual(page1["counts"]["custom"], 18)
                # List summaries stay lightweight without bulk note arrays
                self.assertNotIn("piano_roll", page1["items"][0])

                # On-demand detail endpoint loads full piano roll from cached MIDI
                detail = eng.get_track_detail(page2["items"][-1]["slug"])
                self.assertIn("piano_roll", detail)
                self.assertEqual(len(detail["piano_roll"]["tracks"]), 4)

    def test_bilingual_search_filters_and_sorting(self):
        with tempfile.TemporaryDirectory() as tmp:
            eng = self._make_engine(Path(tmp))
            self._seed_custom_tracks(eng, 6)

            with patch.object(web_studio, "SHOWCASE_TRACKS", []):
                by_prompt = eng.query_tracks({"q": "烟雨"})
                self.assertEqual(by_prompt["total"], 2)

                by_key_alias = eng.query_tracks({"q": "F大调 霓虹"})
                self.assertEqual(by_key_alias["total"], 2)

                by_stage_score = eng.query_tracks({"stage": "phase3_grpo", "sort": "score"})
                scores = [item["r_total"] for item in by_stage_score["items"]]
                self.assertEqual(by_stage_score["total"], 3)
                self.assertEqual(scores, sorted(scores, reverse=True))

    def test_star_rename_and_notes_persist_for_custom_and_showcase(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eng = self._make_engine(root)
            self._seed_custom_tracks(eng, 2)

            showcase_stub = {
                "slug": "01_showcase_stub",
                "title": "《晴空前的温柔》",
                "style": "[Style:Ballad_Slow]",
                "mood": "[Mood:Nostalgic_Warm]",
                "tempo": "[Tempo:72]",
                "key": "[Key:C_maj]",
                "reg": "[Reg_Mid:E4_C5]",
                "arp": "[Arp:1_5_8_10_Fluid]",
                "bars": 16,
                "chords": ["[Chord:F:maj7]"],
                "preset_style": "intro_chamber",
            }
            _write_minimal_midi(eng.showcase_dir / "01_showcase_stub.mid")
            (eng.showcase_dir / "01_showcase_stub.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

            with patch.object(web_studio, "SHOWCASE_TRACKS", [showcase_stub]):
                custom_slug = "custom_1794000000_track_00"
                updated_custom = eng.update_track_metadata({
                    "slug": custom_slug,
                    "starred": True,
                    "title": "《深夜改名版》",
                    "notes": "12秒很干净",
                })
                self.assertTrue(updated_custom["starred"])
                self.assertEqual(updated_custom["title"], "《深夜改名版》")
                self.assertEqual(updated_custom["notes"], "12秒很干净")

                updated_show = eng.update_track_metadata({
                    "slug": "01_showcase_stub",
                    "starred": True,
                    "notes": "重点对照曲目",
                })
                self.assertTrue(updated_show["starred"])
                self.assertEqual(updated_show["notes"], "重点对照曲目")

                starred_only = eng.query_tracks({"starred": "1"})
                self.assertEqual(starred_only["total"], 2)
                self.assertEqual(starred_only["counts"]["starred"], 2)

                by_note = eng.query_tracks({"q": "很干净"})
                self.assertEqual([item["slug"] for item in by_note["items"]], [custom_slug])

                with self.assertRaises(ValueError):
                    eng.update_track_metadata({"slug": custom_slug, "title": "   "})


if __name__ == "__main__":
    unittest.main()
