"""
Mandopop-Studio V3 Unified Tokenizer & 3D Coordinate Engine (`v3/tokenizer/v3_unified_tokenizer.py`)
====================================================================================================
Implements the deterministic Bar-Chord-Track-Grid + Fused-Note representation (`vocab_size = 4096`):
- Native tokenization of structured `<think> ... </think>` music-theory planning tokens
- Hierarchical `<Bar_b> [Sec:*] [Chord:*] [Dyn:*]` headers
- 4 Dedicated Tracks per bar (`<Trk:Lead_Hook>`, `<Trk:Piano_Arp>`, `<Trk:Strings_Pad>`, `<Trk:Bass_Perc>`)
- 16th-note intra-bar positions `<Pos_0>` .. `<Pos_15>`
- Atomic Fused Note tokens `<Note_Pitch_Dur_Vel>` (impossible to emit incomplete notes, 3x shorter sequences)
- On-the-fly extraction of 3D Musical Coordinates (`bar_ids`, `grid_ids`, `track_ids`, `chord_ids`)
  for `Mandopop-V3-115M`'s 3D Axial & Active-Chord Residual Embeddings!
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import symusic
except ImportError:
    symusic = None  # Only needed when decoding to MIDI locally; not needed during GPU training!

# Constants for 3D Coordinate Embeddings
MAX_BARS = 32
NUM_GRID_POS = 17       # 0..15 for 16th-note grid, 16 for non-grid/header tokens
NUM_TRACK_ROLES = 6     # 0: Header/CoT, 1: Lead_Hook, 2: Piano_Arp, 3: Strings_Pad, 4: Bass_Perc, 5: Structural
NUM_CHORD_TYPES = 146   # 0: None/N, 1..144: 12 roots x 12 qualities, 145: Other

PITCH_MIN = 24          # C1
PITCH_MAX = 96          # C7 (73 pitches total)
NUM_PITCHES = PITCH_MAX - PITCH_MIN + 1

# Quantized durations in 16th-note grid steps (1 step = 1 sixteenth note = 0.25 beat)
DURATION_BINS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
NUM_DURATIONS = len(DURATION_BINS)

# 4 Expressive Velocity Tiers: p (54), mp (72), mf (90), f (108)
VELOCITY_BINS = [54, 72, 90, 108]
VELOCITY_LABELS = ["p", "mp", "mf", "f"]
NUM_VELOCITIES = len(VELOCITY_BINS)

PERC_TYPES = [
    ("SoftKick", 36),
    ("Rimshot", 37),
    ("BrushSnare", 38),
    ("ClosedHat", 42),
    ("PedalHat", 44),
    ("OpenHat", 46),
    ("WarmCrash", 49),
    ("ShakerRide", 51),
]
NUM_PERC = len(PERC_TYPES)

PITCH_CLASS_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
CHORD_QUALITIES = [
    "maj", "min", "maj7", "min7", "7", "sus4",
    "add9", "m7b5", "dim7", "aug", "6", "maj9",
]

# Chord templates (intervals in semitones from root) for music-theory analysis & synthesis
CHORD_INTERVALS: Dict[str, List[int]] = {
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "7": [0, 4, 7, 10],
    "sus4": [0, 5, 7],
    "add9": [0, 2, 4, 7],
    "m7b5": [0, 3, 6, 10],
    "dim7": [0, 3, 6, 9],
    "aug": [0, 4, 8],
    "6": [0, 4, 7, 9],
    "maj9": [0, 2, 4, 7, 11],
}


class MandopopV3Tokenizer:
    """
    Deterministic Unified Tokenizer for Mandopop-Studio V3 (`vocab_size = 4096`).
    """

    def __init__(self, vocab_size: int = 4096):
        self.vocab_size = vocab_size
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        # Fast lookup tables for note/percussion decoding and 3D coordinate extraction
        self.note_id_to_tuple: Dict[int, Tuple[int, int, int]] = {}
        self.tuple_to_note_id: Dict[Tuple[int, int, int], int] = {}
        self.perc_id_to_tuple: Dict[int, Tuple[int, int]] = {}
        self.tuple_to_perc_id: Dict[Tuple[int, int], int] = {}
        self.chord_token_to_idx: Dict[int, int] = {}
        self.bar_token_to_idx: Dict[int, int] = {}
        self.pos_token_to_idx: Dict[int, int] = {}
        self.trk_token_to_idx: Dict[int, int] = {}

        self._build_vocabulary()

    def _register(self, token: str) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]
        idx = len(self.token_to_id)
        self.token_to_id[token] = idx
        self.id_to_token[idx] = token
        return idx

    def _build_vocabulary(self) -> None:
        # 1. Special Framing Tokens (0..7)
        for t in ["<|pad|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>", "<score>", "</score>", "<|sep|>"]:
            self._register(t)

        # 2. Style, Mood, Tempo, Key & Structure Prompt Tokens
        for style in ["[Style:Ballad_Slow]", "[Style:Guofeng_Lyrical]", "[Style:PopRock_Driving]", "[Style:Urban_Romantic]"]:
            self._register(style)
        for mood in ["[Mood:Nostalgic_Warm]", "[Mood:Melancholic_Tearful]", "[Mood:Poetic_Breeze]", "[Mood:Dramatic_Swell]"]:
            self._register(mood)
        for bpm in [64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 120]:
            self._register(f"[Tempo:{bpm}]")
        for key_str in [
            "[Key:C_maj]", "[Key:A_min]", "[Key:G_maj]", "[Key:E_min]",
            "[Key:F_maj]", "[Key:D_min]", "[Key:D_maj]", "[Key:B_min]",
            "[Key:Bb_maj]", "[Key:G_min]", "[Key:Eb_maj]", "[Key:C_min]",
        ]:
            self._register(key_str)
        for bars in [8, 12, 16, 20, 24, 32]:
            self._register(f"[Bars:{bars}]")

        # 3. CoT Reasoning & Music Theory Planning Tokens
        cot_tokens = [
            "[Plan_Arc]",
            "[Act1:Opening_Hook]", "[Act2:Lyrical_Dialogue]", "[Act3:Emotional_Climax]", "[Act4:Lingering_Outro]",
            "[Plan_Scale]",
            "[Scale:Pentatonic_Core]", "[Scale:Diatonic_Lyrical]", "[Scale:Harmonic_Minor_Color]",
            "[Plan_Register]",
            "[Reg_Low:C4_G4]", "[Reg_Mid:E4_C5]", "[Reg_High:G4_E5]", "[Reg_Climax:A4_G5]",
            "[Motion:Stepwise_Flow]", "[Motion:Third_Sixth_Lyrical]", "[Motion:Octave_Climax_Leap]",
            "[Plan_Arpeggio]",
            "[Arp:1_5_8_10_Fluid]", "[Arp:Broken_9th_Ballad]", "[Arp:Rhythmic_Chord_Pulse]",
            "[Plan_Counterpoint]",
            "[Cpt:Bridge_Fills_Breaths]", "[Cpt:Celli_Warm_Sustain]", "[Cpt:Violin_High_Descant]",
            "[Plan_Harmony]",
            "[Prog:Canon_15634125]", "[Prog:Royal_4536251]", "[Prog:Minor_6415]", "[Prog:Emotional_4563451]",
            "[Plan_Motif]",
            "[Motif:AA_BA_Hook]", "[Motif:ABAC_Symmetric]", "[Motif:Rising_Sequence]",
        ]
        for ct in cot_tokens:
            self._register(ct)

        # 4. Bar Markers (<Bar_1> .. <Bar_32>)
        for b in range(1, MAX_BARS + 1):
            tid = self._register(f"<Bar_{b}>")
            self.bar_token_to_idx[tid] = b - 1

        # 5. Section & Dynamic Tags
        for sec in ["[Sec:Opening_Hook]", "[Sec:Lyrical_Unfolding]", "[Sec:Emotional_Climax]", "[Sec:Lingering_Outro]"]:
            self._register(sec)
        for dyn in ["[Dyn:pp]", "[Dyn:p]", "[Dyn:mp]", "[Dyn:mf]", "[Dyn:f]", "[Dyn:ff]"]:
            self._register(dyn)

        # 6. Track Headers
        track_defs = [
            ("<Trk:Lead_Hook>", 1),
            ("<Trk:Piano_Arp>", 2),
            ("<Trk:Strings_Pad>", 3),
            ("<Trk:Bass_Perc>", 4),
            ("<Track_Empty>", 5),
        ]
        for t_str, role_idx in track_defs:
            tid = self._register(t_str)
            self.trk_token_to_idx[tid] = role_idx

        # 7. Chord Vocabulary (145 Chords)
        tid_n = self._register("[Chord:N]")
        self.chord_token_to_idx[tid_n] = 0
        chord_counter = 1
        for root_name in PITCH_CLASS_NAMES:
            for qual in CHORD_QUALITIES:
                tid = self._register(f"[Chord:{root_name}:{qual}]")
                self.chord_token_to_idx[tid] = chord_counter
                chord_counter += 1

        # 8. Intra-Bar 16th-Note Positions (<Pos_0> .. <Pos_15>)
        for p in range(16):
            tid = self._register(f"<Pos_{p}>")
            self.pos_token_to_idx[tid] = p

        # 9. Fused Note Tokens (<Note_p_d_v>: 73 pitches x 10 durations x 4 velocities = 2,920 tokens)
        for pitch in range(PITCH_MIN, PITCH_MAX + 1):
            for d_idx, dur_step in enumerate(DURATION_BINS):
                for v_idx, vel_label in enumerate(VELOCITY_LABELS):
                    tok_str = f"<Note_{pitch}_d{dur_step}_{vel_label}>"
                    tid = self._register(tok_str)
                    vel_val = VELOCITY_BINS[v_idx]
                    self.note_id_to_tuple[tid] = (pitch, dur_step, vel_val)
                    self.tuple_to_note_id[(pitch, dur_step, v_idx)] = tid

        # 10. Fused Percussion Tokens (<Perc_Name_v>: 8 types x 4 velocities = 32 tokens)
        for perc_name, drum_pitch in PERC_TYPES:
            for v_idx, vel_label in enumerate(VELOCITY_LABELS):
                tok_str = f"<Perc_{perc_name}_{vel_label}>"
                tid = self._register(tok_str)
                vel_val = VELOCITY_BINS[v_idx]
                self.perc_id_to_tuple[tid] = (drum_pitch, vel_val)
                self.tuple_to_perc_id[(drum_pitch, v_idx)] = tid

        # 11. Reserve remaining slots up to vocab_size (4096)
        while len(self.token_to_id) < self.vocab_size:
            self._register(f"<|reserved_{len(self.token_to_id)}|>")

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<|pad|>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<|im_start|>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<|im_end|>"]

    @property
    def think_start_id(self) -> int:
        return self.token_to_id["<think>"]

    @property
    def think_end_id(self) -> int:
        return self.token_to_id["</think>"]

    @property
    def score_start_id(self) -> int:
        return self.token_to_id["<score>"]

    @property
    def score_end_id(self) -> int:
        return self.token_to_id["</score>"]

    def quantize_duration(self, step_16th: int) -> int:
        step_16th = max(1, int(round(step_16th)))
        return min(DURATION_BINS, key=lambda b: abs(b - step_16th))

    def quantize_velocity_idx(self, vel: int) -> int:
        vel = int(np.clip(vel, 20, 127))
        return int(np.argmin([abs(vel - b) for b in VELOCITY_BINS]))

    def encode_note(self, pitch: int, dur_16th: int, velocity: int) -> int:
        p = int(np.clip(pitch, PITCH_MIN, PITCH_MAX))
        d = self.quantize_duration(dur_16th)
        v_idx = self.quantize_velocity_idx(velocity)
        return self.tuple_to_note_id[(p, d, v_idx)]

    def encode_perc(self, drum_pitch: int, velocity: int) -> int:
        valid_pitches = [p for _, p in PERC_TYPES]
        nearest_p = min(valid_pitches, key=lambda vp: abs(vp - drum_pitch))
        v_idx = self.quantize_velocity_idx(velocity)
        return self.tuple_to_perc_id[(nearest_p, v_idx)]

    def encode_tokens(self, tokens: List[str]) -> List[int]:
        return [self.token_to_id.get(t, self.pad_id) for t in tokens]

    def decode_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.id_to_token.get(int(i), "<|pad|>") for i in ids]

    def compute_3d_musical_coordinates(
        self, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given a 2D tensor `input_ids` of shape `(B, L)`, computes 4 coordinate tensors of shape `(B, L)`:
          - `bar_ids`  : 0 .. MAX_BARS-1 (current bar clock index)
          - `grid_ids` : 0 .. 16 (current 16th-note intra-bar position; 16 for headers/CoT)
          - `track_ids`: 0 .. 5  (0=CoT/Header, 1=Lead, 2=Piano, 3=Strings, 4=BassPerc, 5=Struct)
          - `chord_ids`: 0 .. 145 (active chord index for every token in the bar!)
        Runs in fast vectorized/compiled fashion.
        """
        device = input_ids.device
        ids_np = input_ids.detach().cpu().numpy()
        B, L = ids_np.shape

        bar_arr = np.zeros((B, L), dtype=np.int64)
        grid_arr = np.full((B, L), 16, dtype=np.int64)
        trk_arr = np.zeros((B, L), dtype=np.int64)
        chd_arr = np.zeros((B, L), dtype=np.int64)

        bar_map = self.bar_token_to_idx
        pos_map = self.pos_token_to_idx
        trk_map = self.trk_token_to_idx
        chd_map = self.chord_token_to_idx
        score_end = self.score_end_id

        for b_i in range(B):
            cur_bar = 0
            cur_grid = 16
            cur_trk = 0
            cur_chd = 0
            row = ids_np[b_i]
            for t_i in range(L):
                tid = int(row[t_i])
                if tid in bar_map:
                    cur_bar = min(MAX_BARS - 1, bar_map[tid])
                    cur_grid = 16
                    cur_trk = 5
                elif tid in chd_map:
                    cur_chd = chd_map[tid]
                elif tid in trk_map:
                    cur_trk = trk_map[tid]
                    cur_grid = 16
                elif tid in pos_map:
                    cur_grid = pos_map[tid]
                elif tid == score_end:
                    cur_trk = 0
                    cur_grid = 16

                bar_arr[b_i, t_i] = cur_bar
                grid_arr[b_i, t_i] = cur_grid
                trk_arr[b_i, t_i] = cur_trk
                chd_arr[b_i, t_i] = cur_chd

        return (
            torch.from_numpy(bar_arr).to(device),
            torch.from_numpy(grid_arr).to(device),
            torch.from_numpy(trk_arr).to(device),
            torch.from_numpy(chd_arr).to(device),
        )

    def decode_to_symusic_score(
        self,
        token_ids: List[int],
        default_bpm: float = 76.0,
        ticks_per_quarter: int = 480,
    ) -> Tuple[symusic.Score, List[float], Dict]:
        """
        Decodes a V3 token sequence into a 4-track `symusic.Score`, returning:
          - `score`: symusic.Score with LEAD_HOOK, PIANO_ARP, STRINGS_PAD, BASS, DRUMS tracks
          - `chord_change_times`: list of chord boundary timestamps in seconds (for Pedal-Hold physics!)
          - `metadata`: parsed style, tempo, chord progression, and bar count
        """
        # Extract tempo if present in prompt
        bpm = default_bpm
        for tid in token_ids[:40]:
            tok = self.id_to_token.get(int(tid), "")
            if tok.startswith("[Tempo:") and tok.endswith("]"):
                try:
                    bpm = float(tok[7:-1])
                except ValueError:
                    pass

        score = symusic.Score(ticks_per_quarter)
        score.tempos.append(symusic.Tempo(0, bpm))
        score.time_signatures.append(symusic.TimeSignature(0, 4, 4))

        trk_lead = symusic.Track(name="LEAD_HOOK", program=0, is_drum=False)
        trk_piano = symusic.Track(name="PIANO_ARP", program=0, is_drum=False)
        trk_strings = symusic.Track(name="STRINGS_PAD", program=48, is_drum=False)
        trk_bass = symusic.Track(name="BASS", program=35, is_drum=False)
        trk_drums = symusic.Track(name="DRUMS", program=0, is_drum=True)

        ticks_per_16th = ticks_per_quarter // 4  # 120 ticks per 16th note
        ticks_per_bar = ticks_per_quarter * 4    # 1920 ticks per 4/4 bar
        sec_per_bar = (60.0 / bpm) * 4.0

        in_score = False
        cur_bar_idx = -1
        cur_trk_role = 0
        cur_pos_16th = 0
        bar_chords: List[Tuple[int, str]] = []

        for raw_tid in token_ids:
            tid = int(raw_tid)
            tok = self.id_to_token.get(tid, "")
            if tok == "<score>":
                in_score = True
                continue
            elif tok == "</score>" or tok == "<|im_end|>":
                if in_score:
                    break
                continue

            if not in_score:
                if tid in self.bar_token_to_idx:
                    in_score = True
                else:
                    continue

            if tid in self.bar_token_to_idx:
                cur_bar_idx = self.bar_token_to_idx[tid]
                cur_pos_16th = 0
            elif tid in self.chord_token_to_idx and cur_bar_idx >= 0:
                bar_chords.append((cur_bar_idx, tok))
            elif tid in self.trk_token_to_idx:
                cur_trk_role = self.trk_token_to_idx[tid]
                cur_pos_16th = 0
            elif tid in self.pos_token_to_idx:
                cur_pos_16th = self.pos_token_to_idx[tid]
            elif tid in self.note_id_to_tuple and cur_bar_idx >= 0:
                pitch, dur_step, vel = self.note_id_to_tuple[tid]
                start_tick = cur_bar_idx * ticks_per_bar + cur_pos_16th * ticks_per_16th
                dur_tick = dur_step * ticks_per_16th
                note_obj = symusic.Note(start_tick, dur_tick, pitch, vel)

                if cur_trk_role == 1:
                    trk_lead.notes.append(note_obj)
                elif cur_trk_role == 2:
                    trk_piano.notes.append(note_obj)
                elif cur_trk_role == 3:
                    trk_strings.notes.append(note_obj)
                elif cur_trk_role == 4:
                    trk_bass.notes.append(note_obj)
                else:
                    trk_piano.notes.append(note_obj)
            elif tid in self.perc_id_to_tuple and cur_bar_idx >= 0:
                drum_pitch, vel = self.perc_id_to_tuple[tid]
                start_tick = cur_bar_idx * ticks_per_bar + cur_pos_16th * ticks_per_16th
                dur_tick = 2 * ticks_per_16th
                trk_drums.notes.append(symusic.Note(start_tick, dur_tick, drum_pitch, vel))

        for t in (trk_lead, trk_piano, trk_strings, trk_bass, trk_drums):
            if len(t.notes) > 0:
                score.tracks.append(t)

        # Build chord change timestamps in seconds for Pedal-Hold Engine
        num_bars_decoded = max(1, cur_bar_idx + 1)
        chord_change_times: List[float] = []
        for b in range(num_bars_decoded + 1):
            chord_change_times.append(b * sec_per_bar)
            # Also add half-bar pedal refresh if bar is slow (<= 76 BPM) and dense
            if bpm <= 76:
                chord_change_times.append((b + 0.5) * sec_per_bar)

        meta = {
            "bpm": bpm,
            "num_bars": num_bars_decoded,
            "bar_chords": bar_chords,
            "track_note_counts": {
                "lead_hook": len(trk_lead.notes),
                "piano_arp": len(trk_piano.notes),
                "strings_pad": len(trk_strings.notes),
                "bass": len(trk_bass.notes),
                "drums": len(trk_drums.notes),
            },
        }
        return score, sorted(set(chord_change_times)), meta

    def save_vocab(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"vocab_size": self.vocab_size, "token_to_id": self.token_to_id}, f, indent=2)
