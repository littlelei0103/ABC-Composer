#!/usr/bin/env python3
"""
Mandopop-Studio V3 — Interactive Web Studio Server (`v3/web_studio.py`)
=======================================================================
Zero-dependency HTTP + REST API Server powering the Mandopop-Studio V3 DAW UI:
- Browse & play all Flagship Showcase pieces (`outputs/v3_showcase/`) and Stage Evolution comparisons
- Interactive Custom Composition Studio:
    * Choose Style, Mood, Key, Tempo (BPM), Bars (8 / 12 / 16), Register, Arpeggio Texture, and 8-Bar Chord Progression
    * Choose Model Stage (`Phase 4 SimPO/DPO+OPD`, `Phase 3 Dr. GRPO`, `Phase 2 Golden SFT`, `Phase 1 Pretrain`)
    * Adjust Best-of-N Verifier Reranking (`1..6`) and Studio Renderer Envelope Controls (`lead_attack_ms`, `piano_attack_ms`, `bass_attack_ms`, `preset_style`)
- Returns 4-Track Piano Roll note coordinates, `<think>` CoT Blueprint tokens, 6D Aesthetic Verifier Radar metrics, and streamed `.wav` / `.mid` files.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import sys
import threading
import time
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import symusic
import torch
from transformers import LogitsProcessorList

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v3.audio.studio_renderer import (
    apply_pedal_hold_physics,
    enforce_monophonic_legato_lead,
    infer_exact_chord_boundaries,
    prune_sustained_clashes_against_harmony,
    render_v3_score_to_stereo_wav,
)
from v3.compose_v3_studio import SHOWCASE_TRACKS
from v3.model.modeling_mandopop_v3 import (
    MandopopGrammarLogitsProcessor,
    MandopopV3ForCausalLM,
)
from v3.tokenizer.v3_unified_tokenizer import MandopopV3Tokenizer
from v3.train.train_v3_grpo import build_prompt_ids
from v3.verifiers.intro_aesthetic_verifier import MandopopIntroVerifier


CHORD_PRESETS: Dict[str, Dict] = {
    "royal_4536251_C": {
        "label": "黄金 4-5-3-6-2-5-1 (《晴天》《七里香》暖心抒情 · C大调)",
        "key": "[Key:C_maj]",
        "chords": [
            "[Chord:F:maj7]", "[Chord:G:7]", "[Chord:E:min7]", "[Chord:A:min7]",
            "[Chord:D:min7]", "[Chord:G:sus4]", "[Chord:C:add9]", "[Chord:C:maj7]",
        ],
    },
    "guofeng_pentatonic_C": {
        "label": "国风烟雨 4-5-3-6-2-5-6 (《青花瓷》《大鱼》五声落小调 · C大调/A小调)",
        "key": "[Key:C_maj]",
        "chords": [
            "[Chord:F:maj7]", "[Chord:G:sus4]", "[Chord:E:min7]", "[Chord:A:min7]",
            "[Chord:D:min7]", "[Chord:G:7]", "[Chord:A:min7]", "[Chord:A:min7]",
        ],
    },
    "urban_mercury_F": {
        "label": "都市微光 4-5-3-6-2-5-1-6 (《水星记》《交换余生》R&B九和弦 · F大调)",
        "key": "[Key:F_maj]",
        "chords": [
            "[Chord:Bb:maj7]", "[Chord:C:7]", "[Chord:A:min7]", "[Chord:D:min7]",
            "[Chord:G:min7]", "[Chord:C:sus4]", "[Chord:F:maj9]", "[Chord:D:min7]",
        ],
    },
    "poprock_wind_G": {
        "label": "破晓疾风 6-4-1-5 (《起风了》《平凡之路》摇滚推进 · G大调)",
        "key": "[Key:G_maj]",
        "chords": [
            "[Chord:E:min7]", "[Chord:C:add9]", "[Chord:G:maj]", "[Chord:D:sus4]",
            "[Chord:E:min7]", "[Chord:C:maj7]", "[Chord:D:7]", "[Chord:G:add9]",
        ],
    },
    "melancholic_cello_Am": {
        "label": "深情弦乐 6-4-1-5-2-3-6 (《像我这样的人》《演员》大提琴对话 · A小调)",
        "key": "[Key:A_min]",
        "chords": [
            "[Chord:A:min7]", "[Chord:F:maj7]", "[Chord:C:add9]", "[Chord:G:sus4]",
            "[Chord:D:min7]", "[Chord:E:7]", "[Chord:A:min7]", "[Chord:A:min7]",
        ],
    },
    "canon_sweet_G": {
        "label": "浪漫卡农 1-5-6-3-4-1-2-5 (《告白气球》《小幸运》甜美叙事 · G大调)",
        "key": "[Key:G_maj]",
        "chords": [
            "[Chord:G:add9]", "[Chord:D:maj]", "[Chord:E:min7]", "[Chord:B:min7]",
            "[Chord:C:maj7]", "[Chord:G:maj]", "[Chord:A:min7]", "[Chord:D:sus4]",
        ],
    },
}


def parse_natural_language_prompt(prompt_text: str, base_spec: Dict) -> Tuple[Dict, str]:
    """
    Translates a Chinese/English natural-language music description into a structured
    Mandopop-V3 music-theory specification (`style`, `mood`, `tempo`, `key`, `reg`, `arp`, `bars`, `chords`)
    and a poetic default title if none was explicitly set.
    """
    text = (prompt_text or "").strip()
    if not text:
        return base_spec, ""

    low = text.lower()
    out = dict(base_spec)
    inferred_title = ""
    preset_chosen = None

    # 1. Style & Song Reference Detection
    inferred_preset = None
    if any(k in low for k in ["国风", "古风", "五声", "青花瓷", "大鱼", "红尘", "江南", "烟雨", "竹林", "水墨", "古韵", "guofeng", "pentatonic", "chinese"]):
        out["style"] = "[Style:Guofeng_Lyrical]"
        out["mood"] = "[Mood:Poetic_Breeze]"
        out["reg"] = "[Reg_High:G4_E5]"
        out["arp"] = "[Arp:1_5_8_10_Fluid]"
        out["tempo_bpm"] = 76
        preset_chosen = "guofeng_pentatonic_C"
        inferred_title = "《烟雨江南》"
        inferred_preset = "guofeng_silk"
    elif any(k in low for k in ["摇滚", "疾风", "起风了", "乐队", "燃", "奔跑", "破晓", "青春", "昂扬", "rock", "driving", "band"]):
        out["style"] = "[Style:PopRock_Driving]"
        out["mood"] = "[Mood:Dramatic_Swell]"
        out["reg"] = "[Reg_High:G4_E5]"
        out["arp"] = "[Arp:Broken_9th_Ballad]"
        out["tempo_bpm"] = 96
        preset_chosen = "poprock_wind_G"
        inferred_title = "《迎风奔跑》"
        inferred_preset = "poprock_band"
    elif any(k in low for k in ["都市", "r&b", "rnb", "水星记", "霓虹", "微醺", "夜晚", "星空", "浪漫", "爵士", "交换余生", "urban", "neon", "city"]):
        out["style"] = "[Style:Urban_Romantic]"
        out["mood"] = "[Mood:Nostalgic_Warm]"
        out["reg"] = "[Reg_Mid:E4_C5]"
        out["arp"] = "[Arp:Broken_9th_Ballad]"
        out["tempo_bpm"] = 84
        preset_chosen = "urban_mercury_F"
        inferred_title = "《霓虹微光》"
        inferred_preset = "urban_rhodes"
    elif any(k in low for k in ["悲伤", "忧郁", "大提琴", "催泪", "遗憾", "演员", "像我这样的人", "孤独", "深夜", "眼泪", "sad", "melancholic", "cello", "tear"]):
        out["style"] = "[Style:Ballad_Slow]"
        out["mood"] = "[Mood:Melancholic_Tearful]"
        out["reg"] = "[Reg_High:G4_E5]"
        out["arp"] = "[Arp:1_5_8_10_Fluid]"
        out["tempo_bpm"] = 68
        preset_chosen = "melancholic_cello_Am"
        inferred_title = "《深夜独白》"
        inferred_preset = "cello_piano"
    elif any(k in low for k in ["卡农", "甜", "告白气球", "小幸运", "初恋", "校园", "晴朗", "canon", "sweet"]):
        out["style"] = "[Style:Ballad_Slow]"
        out["mood"] = "[Mood:Nostalgic_Warm]"
        out["reg"] = "[Reg_Mid:E4_C5]"
        out["arp"] = "[Arp:1_5_8_10_Fluid]"
        out["tempo_bpm"] = 76
        preset_chosen = "canon_sweet_G"
        inferred_title = "《晴日心动》"
    elif any(k in low for k in ["周杰伦", "晴天", "七里香", "林俊杰", "抒情", "慢歌", "温暖", "怀旧", "ballad", "warm"]):
        out["style"] = "[Style:Ballad_Slow]"
        out["mood"] = "[Mood:Nostalgic_Warm]"
        out["reg"] = "[Reg_Mid:E4_C5]"
        out["arp"] = "[Arp:1_5_8_10_Fluid]"
        out["tempo_bpm"] = 72
        preset_chosen = "royal_4536251_C"
        inferred_title = "《晴空叙事》"

    if inferred_preset and base_spec.get("preset_style", "intro_chamber") == "intro_chamber":
        out["preset_style"] = inferred_preset

    if preset_chosen:
        out["key"] = CHORD_PRESETS[preset_chosen]["key"]
        out["chords"] = list(CHORD_PRESETS[preset_chosen]["chords"])
        out["preset_key"] = preset_chosen

    # 2. Explicit Mood Overrides
    if any(k in low for k in ["悲", "伤感", "哭", "忧郁", "痛", "遗憾", "小调"]):
        out["mood"] = "[Mood:Melancholic_Tearful]"
        if not preset_chosen:
            out["key"] = "[Key:A_min]"
            out["chords"] = list(CHORD_PRESETS["melancholic_cello_Am"]["chords"])
            out["preset_key"] = "melancholic_cello_Am"
    elif any(k in low for k in ["暖", "治愈", "温柔", "怀旧"]):
        out["mood"] = "[Mood:Nostalgic_Warm]"
    elif any(k in low for k in ["诗意", "清风", "唯美", "空灵"]):
        out["mood"] = "[Mood:Poetic_Breeze]"
    elif any(k in low for k in ["大气", "激昂", "高潮", "史诗"]):
        out["mood"] = "[Mood:Dramatic_Swell]"

    # 3. Explicit Key Overrides
    key_patterns = [
        (r"(a\s*小调|a\s*minor|\bam\b)", "[Key:A_min]", "melancholic_cello_Am"),
        (r"(c\s*大调|c\s*major|\bc\s*maj)", "[Key:C_maj]", "royal_4536251_C"),
        (r"(f\s*大调|f\s*major|\bf\s*maj)", "[Key:F_maj]", "urban_mercury_F"),
        (r"(g\s*大调|g\s*major|\bg\s*maj)", "[Key:G_maj]", "canon_sweet_G"),
    ]
    for pat, k_tok, fallback_p in key_patterns:
        if re.search(pat, low):
            out["key"] = k_tok
            if CHORD_PRESETS.get(out.get("preset_key", ""), {}).get("key") != k_tok:
                out["chords"] = list(CHORD_PRESETS[fallback_p]["chords"])
                out["preset_key"] = fallback_p
            break

    # 4. Explicit Tempo / BPM Overrides
    bpm_match = re.search(r"(\d{2,3})\s*bpm", low)
    if bpm_match:
        raw_bpm = int(bpm_match.group(1))
        valid_bpms = [68, 72, 76, 84, 96]
        out["tempo_bpm"] = min(valid_bpms, key=lambda b: abs(b - raw_bpm))
    elif any(k in low for k in ["极慢", "很慢", "缓"]):
        out["tempo_bpm"] = 68
    elif any(k in low for k in ["轻快", "偏快", "律动", "快一点"]):
        out["tempo_bpm"] = 96

    # 5. Explicit Bar Length Overrides
    if any(k in low for k in ["8小节", "8 小节", "八小节", "短一点", "8 bars"]):
        out["bars"] = 8
    elif any(k in low for k in ["16小节", "16 小节", "十六小节", "长一点", "完整", "16 bars"]):
        out["bars"] = 16

    # 6. Pure Piano + Strings Chamber Override
    if any(k in low for k in ["纯钢琴", "不要鼓", "无鼓", "室内乐", "只要钢琴和弦乐"]):
        out["preset_style"] = "pure_piano_strings"

    # 7. Extract《Title》if user wrote one inside the prompt
    title_match = re.search(r"《([^》]{1,16})》", text)
    if title_match:
        inferred_title = f"《{title_match.group(1)}》"

    return out, inferred_title


STAGE_CHECKPOINTS: Dict[str, str] = {
    "phase4_dpo": "checkpoints/v3_dpo_final/final_model",
    "phase3_grpo": "checkpoints/v3_grpo/final_model",
    "phase2_sft": "checkpoints/v3_sft/final_model",
    "phase1_pretrain": "checkpoints/v3_pretrain/final_model",
}


def extract_piano_roll_from_midi(midi_path: Path) -> Dict:
    """
    Extracts the EXACT post-pedal, post-legato, post-clash-pruned 4-stem note events
    and total audio duration (`max_note_end + 2.8s`) used by `studio_renderer.py`,
    guaranteeing 1:1 visual-acoustic synchronization on the Canvas Piano Roll!
    """
    if not midi_path.exists():
        return {"tracks": [], "duration_sec": 0.0, "bpm": 72.0}

    score = symusic.Score(str(midi_path))
    bpm = float(score.tempos[0].qpm) if score.tempos else 72.0
    score_sec = score.to(symusic.TimeUnit.second)

    lead_notes: List[Tuple[float, float, int, int]] = []
    piano_notes: List[Tuple[float, float, int, int]] = []
    strings_notes: List[Tuple[float, float, int, int]] = []
    bass_notes: List[Tuple[float, float, int, int]] = []
    drum_notes: List[Tuple[float, float, int, int]] = []

    for idx, tr in enumerate(score_sec.tracks):
        name_up = (tr.name or "").upper()
        extracted = [(float(n.time), float(n.time + n.duration), int(n.pitch), int(n.velocity)) for n in tr.notes]
        if not extracted:
            continue
        if tr.is_drum or "DRUM" in name_up or "PERC" in name_up:
            drum_notes.extend(extracted)
        elif tr.program in (32, 33, 34, 35, 36, 37) or "BASS" in name_up:
            bass_notes.extend(extracted)
        elif "STRING" in name_up or "PAD" in name_up or tr.program in (48, 49):
            strings_notes.extend(extracted)
        elif "BRIDGE" in name_up or "LEAD" in name_up or "HOOK" in name_up or "MELODY" in name_up or "VOCAL" in name_up:
            lead_notes.extend(extracted)
        elif "PIANO" in name_up or "ACCOMP" in name_up or tr.program in (0, 1, 2, 4):
            piano_notes.extend(extracted)
        else:
            if idx == 0:
                lead_notes.extend(extracted)
            else:
                piano_notes.extend(extracted)

    all_ends = [e for grp in (lead_notes, piano_notes, strings_notes, bass_notes, drum_notes) for _, e, _, _ in grp]
    max_note_end = max(all_ends) if all_ends else 4.0
    total_sec = min(240.0, max_note_end + 2.8)

    downbeats = list(score_sec.get_downbeats())
    if len(downbeats) >= 2:
        chord_change_times = []
        for i in range(len(downbeats) - 1):
            b_start = float(downbeats[i])
            b_end = float(downbeats[i + 1])
            chord_change_times.append(b_start)
            chord_change_times.append(0.5 * (b_start + b_end))
    else:
        chord_change_times = infer_exact_chord_boundaries(score, max_note_end)

    clean_lead = enforce_monophonic_legato_lead(lead_notes)
    pedaled_piano = apply_pedal_hold_physics(
        piano_notes,
        chord_change_times,
        reference_melody_notes=clean_lead,
        pedal_gap_sec=0.032,
        max_hold_sec=2.4,
    )
    combined_fg = clean_lead + pedaled_piano
    clean_strings = prune_sustained_clashes_against_harmony(strings_notes, chord_change_times, combined_fg)
    clean_bass = prune_sustained_clashes_against_harmony(bass_notes, chord_change_times, combined_fg)

    # Match exact rendered pitches and durations in Stem 1..4:
    # Stem 1: Lead Hook (trimmed right before next lead onset so dry release never overlaps)
    lead_onsets = [s for s, _, _, _ in clean_lead]
    rendered_lead = []
    for i, (s, e, p, v) in enumerate(clean_lead):
        actual_e = e
        if i + 1 < len(lead_onsets):
            actual_e = min(actual_e, max(s + 0.065, lead_onsets[i + 1] - 0.015))
        rendered_lead.append({"start": round(s, 3), "end": round(actual_e, 3), "pitch": int(p), "vel": int(v)})

    # Stem 2: Pedaled Piano Accompaniment (trimmed before same-pitch retrigger)
    rendered_piano = []
    for s, e, p, v in pedaled_piano:
        rendered_piano.append({"start": round(s, 3), "end": round(e, 3), "pitch": int(p), "vel": int(v)})

    # Stem 3: Velvet Strings (with +0.28s bow release)
    rendered_strings = []
    for s, e, p, v in clean_strings:
        dur_s = min(6.0, max(0.45, float(e - s) + 0.28))
        rendered_strings.append({"start": round(s, 3), "end": round(s + dur_s, 3), "pitch": int(p), "vel": int(v)})

    # Stem 4: Acoustic Bass (folded into C2..E3 register) + Soft Kick
    rendered_bass = []
    for s, e, p, v in clean_bass:
        bp = p if p < 52 else (p % 12) + 36
        rendered_bass.append({"start": round(s, 3), "end": round(max(e, s + 0.42), 3), "pitch": int(bp), "vel": int(v)})
    for s, e, p, v in drum_notes:
        if p in (35, 36):
            rendered_bass.append({"start": round(s, 3), "end": round(s + 0.22, 3), "pitch": 36, "vel": 48})

    tracks_out = [
        {"index": 0, "name": "LEAD_HOOK", "label": "主奏钢琴", "color": "#e8c17d", "is_drum": False, "note_count": len(rendered_lead), "notes": rendered_lead},
        {"index": 1, "name": "PIANO_ARP", "label": "延音琶音", "color": "#7cc4fa", "is_drum": False, "note_count": len(rendered_piano), "notes": rendered_piano},
        {"index": 2, "name": "STRINGS_PAD", "label": "室内弦乐", "color": "#b794f4", "is_drum": False, "note_count": len(rendered_strings), "notes": rendered_strings},
        {"index": 3, "name": "BASS_PULSE", "label": "贝斯与律动", "color": "#68d391", "is_drum": False, "note_count": len(rendered_bass), "notes": rendered_bass},
    ]

    return {
        "bpm": round(bpm, 1),
        "duration_sec": round(total_sec, 2),
        "bar_sec": round(4.0 * 60.0 / max(bpm, 30.0), 3),
        "tracks": tracks_out,
    }


STYLE_SEARCH_HINTS: Dict[str, str] = {
    "[Style:Ballad_Slow]": "抒情慢歌 温暖 ballad slow piano",
    "[Style:Guofeng_Lyrical]": "国风五声 古风 烟雨 东方 guofeng lyrical",
    "[Style:Urban_Romantic]": "都市微光 霓虹 夜晚 r&b rnb urban romantic",
    "[Style:PopRock_Driving]": "破晓疾风 流行摇滚 乐队 青春 poprock driving",
}

STYLE_TO_DEFAULT_PRESET: Dict[str, str] = {
    "[Style:Ballad_Slow]": "intro_chamber",
    "[Style:Guofeng_Lyrical]": "guofeng_silk",
    "[Style:Urban_Romantic]": "urban_rhodes",
    "[Style:PopRock_Driving]": "poprock_band",
}

STAGE_SEARCH_HINTS: Dict[str, str] = {
    "phase4_dpo": "phase4 dpo simpo opd 最终版",
    "phase3_grpo": "phase3 grpo rlvr 强化学习",
    "phase2_sft": "phase2 sft cot 监督微调",
    "phase1_pretrain": "phase1 pretrain 预训练",
}


def _atomic_write_json(path: Path, data: Dict) -> None:
    """Atomically saves JSON via a temp file + rename to prevent torn reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}.{threading.get_ident()}")
    tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def _parse_slug_timestamp(slug: str, fallback_mtime: float = 0.0) -> int:
    m = re.match(r"^custom_(\d+)_", slug or "")
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return int(fallback_mtime) if fallback_mtime else 0


def _infer_preset_style(data: Dict, fallback: str = "intro_chamber") -> str:
    preset = (
        data.get("preset_style")
        or (data.get("audio_info") or {}).get("preset_style")
        or STYLE_TO_DEFAULT_PRESET.get(str(data.get("style", "")), "")
    )
    return str(preset or fallback)


def _build_searchable_text(record: Dict) -> str:
    """Builds a lowercase multi-field search string covering CN/EN aliases and metadata."""
    style = str(record.get("style") or "")
    key = str(record.get("key") or "")
    stage = str(record.get("model_stage") or "")
    key_cn = key.replace("[Key:", "").replace("]", "").replace("_maj", "大调").replace("_min", "小调")
    key_en = key.replace("[Key:", "").replace("]", "").replace("_maj", " major").replace("_min", " minor")
    parts = [
        str(record.get("title") or ""),
        str(record.get("notes") or ""),
        str(record.get("nl_prompt") or ""),
        str(record.get("slug") or ""),
        style,
        STYLE_SEARCH_HINTS.get(style, ""),
        key,
        key_cn,
        key_en,
        str(record.get("mood") or ""),
        str(record.get("tempo") or record.get("tempo_bpm") or ""),
        f"{record.get('bars') or ''}小节",
        stage,
        STAGE_SEARCH_HINTS.get(stage, ""),
        str(record.get("preset_style") or ""),
        " ".join(record.get("chords") or []),
        str(record.get("think_blueprint") or ""),
    ]
    return " ".join(parts).lower()


class StudioEngine:
    """Thread-safe model cache and generation/rendering manager."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = MandopopV3Tokenizer()
        self.verifier = MandopopIntroVerifier(self.tokenizer)
        self.models: Dict[str, MandopopV3ForCausalLM] = {}
        self.lock = threading.Lock()
        self.meta_lock = threading.RLock()
        self.showcase_dir = PROJECT_ROOT / "outputs/v3_showcase"
        self.custom_dir = self.showcase_dir / "custom"
        self.custom_dir.mkdir(parents=True, exist_ok=True)
        self.library_meta_path = self.showcase_dir / "library_meta.json"
        self._piano_roll_cache: Dict[str, Tuple[int, Dict]] = {}
        self._summary_cache: Dict[str, Tuple[Tuple[int, int, int], Dict]] = {}

    def _get_cached_piano_roll(self, mid_path: Path) -> Dict:
        if not mid_path.exists():
            return {"tracks": [], "duration_sec": 0.0, "bpm": 72.0}
        cache_key = str(mid_path)
        mtime_ns = mid_path.stat().st_mtime_ns
        with self.meta_lock:
            cached = self._piano_roll_cache.get(cache_key)
            if cached and cached[0] == mtime_ns:
                return cached[1]
        proll = extract_piano_roll_from_midi(mid_path)
        with self.meta_lock:
            self._piano_roll_cache[cache_key] = (mtime_ns, proll)
        return proll

    def _load_library_meta(self) -> Dict[str, Dict]:
        with self.meta_lock:
            if not self.library_meta_path.exists():
                return {}
            try:
                data = json.loads(self.library_meta_path.read_text(encoding="utf-8"))
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}

    def _save_library_meta(self, meta_map: Dict[str, Dict]) -> None:
        with self.meta_lock:
            _atomic_write_json(self.library_meta_path, meta_map)

    def _load_eval_report(self) -> Dict:
        report_path = self.showcase_dir / "v3_evaluation_report.json"
        if not report_path.exists():
            return {}
        try:
            return json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _normalize_summary(
        self,
        raw: Dict,
        *,
        source: str,
        mid_path: Path,
        wav_path: Path,
        override: Optional[Dict] = None,
    ) -> Dict:
        ovr = override or {}
        slug = str(raw.get("slug") or "")
        wav_mtime = int(wav_path.stat().st_mtime) if wav_path.exists() else 0
        mid_mtime = int(mid_path.stat().st_mtime) if mid_path.exists() else 0
        rel_dir = "v3_showcase/custom" if source == "custom" else "v3_showcase"

        tempo_val = raw.get("tempo_bpm")
        if tempo_val is None:
            m_t = re.search(r"(\d+)", str(raw.get("tempo") or ""))
            tempo_val = int(m_t.group(1)) if m_t else 72
        tempo_tok = str(raw.get("tempo") or f"[Tempo:{tempo_val}]")

        duration_sec = raw.get("duration_sec")
        if not duration_sec and mid_path.exists():
            duration_sec = self._get_cached_piano_roll(mid_path).get("duration_sec", 0.0)

        verifier_metrics = raw.get("verifier_metrics") or {}
        created_at = int(
            raw.get("created_at")
            or (_parse_slug_timestamp(slug, mid_mtime or wav_mtime) if source == "custom" else mid_mtime or 0)
        )
        summary = {
            "slug": slug,
            "source": source,
            "title": str(ovr.get("title") or raw.get("title") or slug),
            "notes": str(ovr.get("notes") if "notes" in ovr else (raw.get("notes") or "")),
            "starred": bool(ovr.get("starred") if "starred" in ovr else raw.get("starred", False)),
            "style": str(raw.get("style") or ""),
            "mood": str(raw.get("mood") or ""),
            "tempo": tempo_tok,
            "tempo_bpm": int(round(float(tempo_val))),
            "key": str(raw.get("key") or ""),
            "reg": str(raw.get("reg") or ""),
            "arp": str(raw.get("arp") or ""),
            "bars": int(raw.get("bars") or 16),
            "chords": list(raw.get("chords") or []),
            "model_stage": str(raw.get("model_stage") or "phase4_dpo"),
            "preset_style": _infer_preset_style(raw),
            "nl_prompt": str(raw.get("nl_prompt") or ""),
            "think_blueprint": str(raw.get("think_blueprint") or ""),
            "verifier_metrics": verifier_metrics,
            "r_total": float(verifier_metrics.get("r_total") or 0.0),
            "duration_sec": round(float(duration_sec or 0.0), 2),
            "created_at": created_at,
            "updated_at": int(ovr.get("updated_at") or raw.get("updated_at") or created_at),
            "wav_url": f"/audio/{rel_dir}/{slug}.wav?t={wav_mtime}",
            "mid_url": f"/audio/{rel_dir}/{slug}.mid",
        }
        return summary

    def _load_all_track_records(self) -> List[Dict]:
        report_data = self._load_eval_report()
        report_tracks = {t["slug"]: t for t in report_data.get("showcase_tracks", []) if "slug" in t}
        lib_meta = self._load_library_meta()
        records: List[Dict] = []

        for meta_file in self.custom_dir.glob("*.json"):
            try:
                j_mtime = meta_file.stat().st_mtime_ns
                stem = meta_file.stem
                mid_p = self.custom_dir / f"{stem}.mid"
                wav_p = self.custom_dir / f"{stem}.wav"
                w_mtime = wav_p.stat().st_mtime_ns if wav_p.exists() else 0
                m_mtime = mid_p.stat().st_mtime_ns if mid_p.exists() else 0
                sig = (j_mtime, w_mtime, m_mtime)
                cache_key = str(meta_file)
                with self.meta_lock:
                    cached = self._summary_cache.get(cache_key)
                if cached and cached[0] == sig:
                    records.append(cached[1])
                    continue

                raw = json.loads(meta_file.read_text(encoding="utf-8"))
                slug = str(raw.get("slug") or stem)
                mid_p = self.custom_dir / f"{slug}.mid"
                wav_p = self.custom_dir / f"{slug}.wav"
                summary = self._normalize_summary(
                    raw, source="custom", mid_path=mid_p, wav_path=wav_p
                )
                with self.meta_lock:
                    self._summary_cache[cache_key] = (sig, summary)
                records.append(summary)
            except Exception:
                pass

        for item in SHOWCASE_TRACKS:
            slug = item["slug"]
            mid_p = self.showcase_dir / f"{slug}.mid"
            wav_p = self.showcase_dir / f"{slug}.wav"
            rep = report_tracks.get(slug, {})
            merged = dict(item)
            if rep.get("duration_sec"):
                merged["duration_sec"] = rep["duration_sec"]
            if rep.get("verifier_metrics"):
                merged["verifier_metrics"] = rep["verifier_metrics"]
            records.append(
                self._normalize_summary(
                    merged,
                    source="showcase",
                    mid_path=mid_p,
                    wav_path=wav_p,
                    override=lib_meta.get(slug),
                )
            )

        return records

    def query_tracks(self, params: Optional[Dict[str, str]] = None) -> Dict:
        p = params or {}
        records = self._load_all_track_records()

        counts = {
            "all": len(records),
            "custom": sum(1 for r in records if r["source"] == "custom"),
            "showcase": sum(1 for r in records if r["source"] == "showcase"),
            "starred": sum(1 for r in records if r.get("starred")),
        }
        facets = {
            "styles": sorted({r["style"] for r in records if r.get("style")}),
            "keys": sorted({r["key"] for r in records if r.get("key")}),
            "stages": sorted({r["model_stage"] for r in records if r.get("model_stage")}),
        }

        source_filter = str(p.get("source") or "all").strip().lower()
        style_filter = str(p.get("style") or "").strip()
        key_filter = str(p.get("key") or "").strip()
        stage_filter = str(p.get("stage") or "").strip()
        starred_raw = str(p.get("starred") or "").strip().lower()
        starred_only = starred_raw in ("1", "true", "yes")
        q = str(p.get("q") or "").strip().lower()
        q_tokens = [tok for tok in q.split() if tok]
        sort_by = str(p.get("sort") or "newest").strip().lower()

        try:
            offset = max(0, int(p.get("offset") or 0))
        except ValueError:
            offset = 0
        try:
            limit = max(1, min(200, int(p.get("limit") or 12)))
        except ValueError:
            limit = 12

        filtered: List[Dict] = []
        for rec in records:
            if source_filter in ("custom", "showcase") and rec["source"] != source_filter:
                continue
            if starred_only and not rec.get("starred"):
                continue
            if style_filter and rec.get("style") != style_filter:
                continue
            if key_filter and rec.get("key") != key_filter:
                continue
            if stage_filter and rec.get("model_stage") != stage_filter:
                continue
            if q_tokens:
                hay = _build_searchable_text(rec)
                if not all(tok in hay for tok in q_tokens):
                    continue
            filtered.append(rec)

        if sort_by == "oldest":
            filtered.sort(key=lambda r: (r["created_at"], r["slug"]))
        elif sort_by == "score":
            filtered.sort(key=lambda r: (r["r_total"], r["created_at"], r["slug"]), reverse=True)
        elif sort_by == "duration":
            filtered.sort(key=lambda r: (r["duration_sec"], r["created_at"], r["slug"]), reverse=True)
        elif sort_by == "title":
            filtered.sort(key=lambda r: (r["title"].lower(), -r["created_at"]))
        else:  # "newest": newest custom compositions first, curated showcase after unless explicitly mixed by timestamp
            sort_by = "newest"
            filtered.sort(
                key=lambda r: (
                    1 if r["source"] == "custom" else 0,
                    r["created_at"],
                    r["slug"],
                ),
                reverse=True,
            )

        page_items = [dict(item) for item in filtered[offset : offset + limit]]
        return {
            "items": page_items,
            "total": len(filtered),
            "offset": offset,
            "limit": limit,
            "sort": sort_by,
            "has_more": (offset + len(page_items)) < len(filtered),
            "counts": counts,
            "facets": facets,
        }

    def get_track_detail(self, slug: str) -> Dict:
        clean_slug = str(slug or "").strip()
        if not clean_slug or "/" in clean_slug or "\\" in clean_slug or ".." in clean_slug:
            raise ValueError("Invalid track slug")

        for rec in self._load_all_track_records():
            if rec["slug"] == clean_slug:
                out = dict(rec)
                base_dir = self.custom_dir if rec["source"] == "custom" else self.showcase_dir
                out["piano_roll"] = self._get_cached_piano_roll(base_dir / f"{clean_slug}.mid")
                return out
        raise FileNotFoundError(f"Track not found: {clean_slug}")

    def update_track_metadata(self, payload: Dict) -> Dict:
        slug = str(payload.get("slug") or "").strip()
        if not slug or "/" in slug or "\\" in slug or ".." in slug:
            raise ValueError("Invalid track slug")

        custom_json = self.custom_dir / f"{slug}.json"
        showcase_slugs = {t["slug"] for t in SHOWCASE_TRACKS}
        now_ts = int(time.time())

        if "title" in payload:
            new_title = str(payload.get("title") or "").strip()
            if not new_title:
                raise ValueError("曲名不能为空")
            new_title = new_title[:80]
        else:
            new_title = None

        notes_val = str(payload["notes"]).strip()[:500] if "notes" in payload else None
        starred_val = bool(payload["starred"]) if "starred" in payload else None

        with self.meta_lock:
            if custom_json.exists():
                raw = json.loads(custom_json.read_text(encoding="utf-8"))
                if new_title is not None:
                    raw["title"] = new_title
                if notes_val is not None:
                    raw["notes"] = notes_val
                if starred_val is not None:
                    raw["starred"] = starred_val
                raw["updated_at"] = now_ts
                _atomic_write_json(custom_json, raw)
                self._summary_cache.pop(str(custom_json), None)
                return self._normalize_summary(
                    raw,
                    source="custom",
                    mid_path=self.custom_dir / f"{slug}.mid",
                    wav_path=self.custom_dir / f"{slug}.wav",
                )

            if slug in showcase_slugs:
                lib_meta = self._load_library_meta()
                entry = dict(lib_meta.get(slug) or {})
                if new_title is not None:
                    entry["title"] = new_title
                if notes_val is not None:
                    entry["notes"] = notes_val
                if starred_val is not None:
                    entry["starred"] = starred_val
                entry["updated_at"] = now_ts
                lib_meta[slug] = entry
                self._save_library_meta(lib_meta)
                return self.get_track_detail(slug)

        raise FileNotFoundError(f"Track not found: {slug}")

    def get_model(self, stage_key: str = "phase4_dpo") -> MandopopV3ForCausalLM:
        rel_path = STAGE_CHECKPOINTS.get(stage_key, STAGE_CHECKPOINTS["phase4_dpo"])
        ckpt_path = PROJECT_ROOT / rel_path
        if stage_key not in self.models:
            print(f"🚀 Loading Mandopop-V3 model [{stage_key}] from {ckpt_path} on {self.device} ...", flush=True)
            dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
            model = MandopopV3ForCausalLM.from_pretrained(str(ckpt_path), torch_dtype=dtype).to(self.device)
            model.eval()
            self.models[stage_key] = model
        return self.models[stage_key]

    def get_showcase_catalog(self, params: Optional[Dict[str, str]] = None) -> Dict:
        report_data = self._load_eval_report()
        library = self.query_tracks(params)
        # Attach piano_roll for the first track in the initial page so the player renders immediately
        # without embedding thousands of note events inside every list item.
        if library["items"]:
            first_slug = library["items"][0]["slug"]
            try:
                library["items"][0] = self.get_track_detail(first_slug)
            except Exception:
                pass

        return {
            "device": str(self.device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "chord_presets": CHORD_PRESETS,
            "stage_progression": report_data.get("stage_progression", {}),
            "library": library,
            "showcase_tracks": [t for t in self._load_all_track_records() if t["source"] == "showcase"],
            "custom_tracks": [t for t in library["items"] if t["source"] == "custom"],
        }

    def compose_and_render(self, payload: Dict) -> Dict:
        with self.lock:
            t0 = time.time()
            title = str(payload.get("title") or "自定义华语流行前奏").strip()
            style = str(payload.get("style") or "[Style:Ballad_Slow]")
            mood = str(payload.get("mood") or "[Mood:Nostalgic_Warm]")
            tempo_bpm = int(payload.get("tempo_bpm") or 72)
            tempo_tok = f"[Tempo:{tempo_bpm}]"
            key_tok = str(payload.get("key") or "[Key:C_maj]")
            reg_tok = str(payload.get("reg") or "[Reg_Mid:E4_C5]")
            arp_tok = str(payload.get("arp") or "[Arp:1_5_8_10_Fluid]")
            target_bars = int(payload.get("bars") or 16)
            if target_bars not in (8, 12, 16):
                target_bars = 16

            chords_raw = payload.get("chords")
            if isinstance(chords_raw, str):
                chords = [c.strip() for c in chords_raw.split(",") if c.strip()]
            elif isinstance(chords_raw, list) and len(chords_raw) > 0:
                chords = [str(c).strip() for c in chords_raw]
            else:
                chords = CHORD_PRESETS["royal_4536251_C"]["chords"]

            # Normalize chord tokens
            norm_chords = []
            for c in chords:
                if not c.startswith("[Chord:"):
                    c = f"[Chord:{c}]"
                if c not in self.tokenizer.token_to_id:
                    c = "[Chord:C:maj7]"
                norm_chords.append(c)
            while len(norm_chords) < 8:
                norm_chords.extend(norm_chords)
            norm_chords = norm_chords[:8]

            stage_key = str(payload.get("model_stage") or "phase4_dpo")
            num_candidates = max(1, min(8, int(payload.get("num_candidates") or 4)))
            temperature = float(payload.get("temperature") or 0.86)
            preset_style = str(payload.get("preset_style") or "intro_chamber")
            lead_attack_ms = float(payload.get("lead_attack_ms", 22.0))
            piano_attack_ms = float(payload.get("piano_attack_ms", 25.0))
            bass_attack_ms = float(payload.get("bass_attack_ms", 20.0))

            base_spec = {
                "style": style,
                "mood": mood,
                "tempo_bpm": tempo_bpm,
                "key": key_tok,
                "reg": reg_tok,
                "arp": arp_tok,
                "bars": target_bars,
                "chords": norm_chords,
                "preset_style": preset_style,
            }
            nl_prompt = str(payload.get("nl_prompt") or "").strip()
            if nl_prompt:
                parsed_spec, inferred_title = parse_natural_language_prompt(nl_prompt, base_spec)
                style = parsed_spec["style"]
                mood = parsed_spec["mood"]
                tempo_bpm = int(parsed_spec["tempo_bpm"])
                tempo_tok = f"[Tempo:{tempo_bpm}]"
                key_tok = parsed_spec["key"]
                reg_tok = parsed_spec["reg"]
                arp_tok = parsed_spec["arp"]
                target_bars = int(parsed_spec["bars"])
                norm_chords = parsed_spec["chords"]
                preset_style = parsed_spec.get("preset_style", preset_style)
                if inferred_title and (not payload.get("title") or payload.get("title") == "《微光重逢》"):
                    title = inferred_title

            spec = {
                "style": style,
                "mood": mood,
                "tempo": tempo_tok,
                "key": key_tok,
                "reg": reg_tok,
                "arp": arp_tok,
                "bars": target_bars,
                "chords": norm_chords,
            }

            model = self.get_model(stage_key)
            prompt_ids_list = build_prompt_ids(self.tokenizer, spec, target_bars=target_bars)
            prompt_len = len(prompt_ids_list)
            prompt_tensor = torch.tensor(
                [prompt_ids_list] * num_candidates, dtype=torch.long, device=self.device
            )

            max_new_tokens = int(target_bars * 52)
            fsm = MandopopGrammarLogitsProcessor(self.tokenizer, target_bars=target_bars)

            with torch.no_grad():
                if self.device.type == "cuda":
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        gen_batch = model.generate(
                            input_ids=prompt_tensor,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=temperature,
                            top_p=0.92,
                            top_k=40,
                            pad_token_id=self.tokenizer.pad_id,
                            eos_token_id=self.tokenizer.eos_id,
                            logits_processor=LogitsProcessorList([fsm]),
                            use_cache=True,
                        )
                else:
                    gen_batch = model.generate(
                        input_ids=prompt_tensor,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=0.92,
                        top_k=40,
                        pad_token_id=self.tokenizer.pad_id,
                        eos_token_id=self.tokenizer.eos_id,
                        logits_processor=LogitsProcessorList([fsm]),
                        use_cache=True,
                    )

            candidates = []
            for i in range(num_candidates):
                seq = gen_batch[i].tolist()
                if self.tokenizer.eos_id in seq[prompt_len:]:
                    seq = seq[: seq.index(self.tokenizer.eos_id, prompt_len) + 1]
                m = self.verifier.evaluate_ids(seq, target_bars=target_bars)
                candidates.append((m["r_total"], seq, m))

            candidates.sort(key=lambda x: x[0], reverse=True)
            best_r, best_seq, best_metrics = candidates[0]
            all_candidate_scores = [round(c[0], 4) for c in candidates]

            # Extract <think> blueprint tokens
            tokens_str = self.tokenizer.decode_ids_to_tokens(best_seq)
            think_tokens = []
            in_think = False
            for tok in tokens_str:
                if tok == "<think>":
                    in_think = True
                    continue
                if tok == "</think>":
                    break
                if in_think:
                    think_tokens.append(tok)

            score, chord_times, decode_meta = self.tokenizer.decode_to_symusic_score(
                best_seq, default_bpm=float(tempo_bpm)
            )

            ts = int(time.time())
            safe_slug = re.sub(r"[^a-zA-Z0-9_\u4e00-\u9fa5]+", "_", title)[:24].strip("_") or "custom"
            file_stem = f"custom_{ts}_{safe_slug}"
            mid_path = self.custom_dir / f"{file_stem}.mid"
            wav_path = self.custom_dir / f"{file_stem}.wav"
            meta_path = self.custom_dir / f"{file_stem}.json"

            score.dump_midi(str(mid_path))
            audio_info = render_v3_score_to_stereo_wav(
                score,
                wav_path,
                chord_change_times=chord_times,
                preset_style=preset_style,
                piano_attack_sec=piano_attack_ms / 1000.0,
                lead_piano_attack_sec=lead_attack_ms / 1000.0,
                bass_attack_sec=bass_attack_ms / 1000.0,
            )

            proll = extract_piano_roll_from_midi(mid_path)
            elapsed = round(time.time() - t0, 2)

            result = {
                "slug": file_stem,
                "source": "custom",
                "title": title,
                "notes": "",
                "starred": False,
                "style": style,
                "mood": mood,
                "tempo": tempo_tok,
                "tempo_bpm": tempo_bpm,
                "key": key_tok,
                "reg": reg_tok,
                "arp": arp_tok,
                "bars": decode_meta["num_bars"],
                "chords": norm_chords,
                "model_stage": stage_key,
                "preset_style": preset_style,
                "nl_prompt": nl_prompt,
                "num_candidates": num_candidates,
                "candidate_scores": all_candidate_scores,
                "think_blueprint": " ".join(think_tokens),
                "wav_url": f"/audio/v3_showcase/custom/{file_stem}.wav?t={ts}",
                "mid_url": f"/audio/v3_showcase/custom/{file_stem}.mid",
                "duration_sec": audio_info["duration_sec"],
                "elapsed_sec": elapsed,
                "created_at": ts,
                "updated_at": ts,
                "verifier_metrics": best_metrics,
                "r_total": float(best_metrics.get("r_total") or 0.0),
                "audio_info": audio_info,
                "piano_roll": proll,
            }
            _atomic_write_json(meta_path, result)
            return result

    def rerender_existing_track(self, payload: Dict) -> Dict:
        with self.lock:
            slug = str(payload.get("slug") or "01_ballad_intro_qingtian_warmth")
            lead_attack_ms = float(payload.get("lead_attack_ms", 22.0))
            piano_attack_ms = float(payload.get("piano_attack_ms", 25.0))
            bass_attack_ms = float(payload.get("bass_attack_ms", 20.0))
            preset_style = str(payload.get("preset_style") or "intro_chamber")

            mid_path = PROJECT_ROOT / "outputs/v3_showcase" / f"{slug}.mid"
            if not mid_path.exists():
                mid_path = self.custom_dir / f"{slug}.mid"
            if not mid_path.exists():
                raise FileNotFoundError(f"MIDI file not found for slug: {slug}")

            wav_path = mid_path.with_suffix(".wav")
            score = symusic.Score(str(mid_path))
            t0 = time.time()
            audio_info = render_v3_score_to_stereo_wav(
                score,
                wav_path,
                preset_style=preset_style,
                piano_attack_sec=piano_attack_ms / 1000.0,
                lead_piano_attack_sec=lead_attack_ms / 1000.0,
                bass_attack_sec=bass_attack_ms / 1000.0,
            )
            rel_wav = wav_path.relative_to(PROJECT_ROOT / "outputs")
            now_ts = int(time.time())
            wav_url = f"/audio/{rel_wav.as_posix()}?t={now_ts}"

            custom_meta = self.custom_dir / f"{slug}.json"
            if custom_meta.exists():
                with self.meta_lock:
                    raw = json.loads(custom_meta.read_text(encoding="utf-8"))
                    raw["wav_url"] = wav_url
                    raw["preset_style"] = preset_style
                    raw["audio_info"] = audio_info
                    raw["duration_sec"] = audio_info["duration_sec"]
                    raw["updated_at"] = now_ts
                    _atomic_write_json(custom_meta, raw)
                    self._summary_cache.pop(str(custom_meta), None)

            return {
                "slug": slug,
                "wav_url": wav_url,
                "preset_style": preset_style,
                "elapsed_sec": round(time.time() - t0, 2),
                "audio_info": audio_info,
            }


ENGINE = StudioEngine()
UI_HTML_PATH = Path(__file__).resolve().parent / "ui" / "index.html"


class StudioHTTPRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        print(f"[Studio-UI] {self.address_string()} - {fmt % args}", flush=True)

    def _send_json(self, data: Dict, status: int = 200) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _serve_file(self, file_path: Path) -> None:
        if not file_path.exists() or not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return

        ctype, _ = mimetypes.guess_type(str(file_path))
        if not ctype:
            ctype = "application/octet-stream"
        file_size = file_path.stat().st_size

        # Support HTTP Range requests for HTML5 <audio> scrubbing
        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            try:
                byte_range = range_header.split("=")[1].split("-")
                start = int(byte_range[0]) if byte_range[0] else 0
                end = int(byte_range[1]) if (len(byte_range) > 1 and byte_range[1]) else file_size - 1
                end = min(end, file_size - 1)
                length = max(0, end - start + 1)

                with file_path.open("rb") as f:
                    f.seek(start)
                    chunk = f.read(length)

                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Type", ctype)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(len(chunk)))
                self.end_headers()
                self.wfile.write(chunk)
                return
            except Exception:
                pass

        data = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(data)))
        if file_path == UI_HTML_PATH:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _parse_query(self, parsed_url: urllib.parse.ParseResult) -> Dict[str, str]:
        raw_qs = urllib.parse.parse_qs(parsed_url.query, keep_blank_values=False)
        return {k: vals[-1] for k, vals in raw_qs.items() if vals}

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(parsed.path)
        params = self._parse_query(parsed)

        if path in ("/", "/index.html"):
            self._serve_file(UI_HTML_PATH)
            return

        if path == "/api/catalog":
            self._send_json(ENGINE.get_showcase_catalog(params))
            return

        if path == "/api/tracks":
            self._send_json(ENGINE.query_tracks(params))
            return

        if path == "/api/track":
            slug = params.get("slug", "")
            try:
                self._send_json(ENGINE.get_track_detail(slug))
            except ValueError as e:
                self._send_json({"error": str(e)}, status=400)
            except FileNotFoundError as e:
                self._send_json({"error": str(e)}, status=404)
            return

        if path.startswith("/audio/"):
            rel = path[len("/audio/") :]
            safe_path = (PROJECT_ROOT / "outputs" / rel).resolve()
            outputs_root = (PROJECT_ROOT / "outputs").resolve()
            if not str(safe_path).startswith(str(outputs_root)):
                self.send_error(HTTPStatus.FORBIDDEN, "Access denied")
                return
            self._serve_file(safe_path)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_len = int(self.headers.get("Content-Length", "0"))
        body_bytes = self.rfile.read(content_len) if content_len > 0 else b"{}"
        try:
            payload = json.loads(body_bytes.decode("utf-8"))
        except Exception as e:
            self._send_json({"error": f"Invalid JSON: {e}"}, status=400)
            return

        if path == "/api/compose":
            try:
                res = ENGINE.compose_and_render(payload)
                self._send_json(res)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send_json({"error": str(e)}, status=500)
            return

        if path == "/api/rerender":
            try:
                res = ENGINE.rerender_existing_track(payload)
                self._send_json(res)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send_json({"error": str(e)}, status=500)
            return

        if path == "/api/track/update":
            try:
                res = ENGINE.update_track_metadata(payload)
                self._send_json({"track": res})
            except ValueError as e:
                self._send_json({"error": str(e)}, status=400)
            except FileNotFoundError as e:
                self._send_json({"error": str(e)}, status=404)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send_json({"error": str(e)}, status=500)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mandopop-Studio V3 Interactive Web DAW Server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), StudioHTTPRequestHandler)
    print(f"\n🎹 =====================================================================", flush=True)
    print(f"✨ Mandopop-Studio V3 Web DAW is live at: http://{args.host}:{args.port}", flush=True)
    print(f"🎹 =====================================================================\n", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Mandopop-Studio V3 Web Server...")
        server.server_close()


if __name__ == "__main__":
    main()
