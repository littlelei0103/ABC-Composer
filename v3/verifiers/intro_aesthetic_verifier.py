"""
Mandopop-Studio V3 6-Dimensional Intro & Instrumental Aesthetic Verifier (`v3/verifiers/intro_aesthetic_verifier.py`)
====================================================================================================================
Evaluates symbolic V3 token sequences across 6 music-theoretically grounded dimensions:
1. `r_no_stutter`   : Anti-KTV syllable stutter; rewards lyrical intervals (2nds, 3rds, 6ths, octaves) & expressive holds
2. `r_arp_fluidity` : Left-hand `1-5-8-10` flowing piano arpeggio wave motion across the 16th-note bar grid
3. `r_chord_color`  : Strong-beat chord-tone alignment, root bass lock, and `maj7/add9/sus4` Mandopop harmonic color
4. `r_counterpoint` : Complementary breathing (Piano/Strings fill Lead Hook phrase endings) + clean register separation
5. `r_hook_motif`   : Opening Hook (Bars 1-2) recurrence / octave lift in Unfolding (Bars 5-6) & Climax (Bars 9-10)
6. `r_dynamic_arc`  : 4-Act arrangement progression (Opening intimacy -> Strings dialogue -> Full climax -> Outro taper)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from v3.model.modeling_mandopop_v3 import KEY_DIATONIC_PCS, KEY_PENTATONIC_PCS
from v3.tokenizer.v3_unified_tokenizer import (
    CHORD_INTERVALS,
    PITCH_CLASS_NAMES,
    MandopopV3Tokenizer,
)


def parse_chord_token(chord_tok: str) -> Tuple[Optional[int], str, List[int]]:
    """
    Parses `[Chord:F:maj7]` into `(root_pc, quality, chord_pitch_classes)`.
    """
    if not chord_tok.startswith("[Chord:") or chord_tok == "[Chord:N]":
        return None, "N", [0, 2, 4, 5, 7, 9, 11]
    body = chord_tok[7:-1]
    parts = body.split(":")
    if len(parts) != 2:
        return None, "N", [0, 2, 4, 5, 7, 9, 11]
    root_str, qual = parts[0], parts[1]
    if root_str not in PITCH_CLASS_NAMES:
        return None, "N", [0, 2, 4, 5, 7, 9, 11]
    root_pc = PITCH_CLASS_NAMES.index(root_str)
    intervals = CHORD_INTERVALS.get(qual, [0, 4, 7])
    pcs = [int((root_pc + iv) % 12) for iv in intervals]
    return root_pc, qual, pcs


class MandopopIntroVerifier:
    def __init__(self, tokenizer: Optional[MandopopV3Tokenizer] = None):
        self.tok = tokenizer or MandopopV3Tokenizer()

    def _parse_bars(self, token_ids: List[int]) -> List[Dict]:
        """
        Parses a V3 token sequence into a list of bar dictionaries:
        Each bar has:
          - `bar_num`: int (1-based)
          - `section`: str
          - `chord_tok`: str
          - `dyn_tok`: str
          - `tracks`: {1: [(pos, pitch, dur, vel), ...], 2: [...], 3: [...], 4: [...], 'perc': [(pos, dpitch, vel)]}
        """
        bars: List[Dict] = []
        cur_bar: Optional[Dict] = None
        cur_trk = 0
        cur_pos = 0
        in_score = False

        for raw_tid in token_ids:
            tid = int(raw_tid)
            tok_str = self.tok.id_to_token.get(tid, "")
            if tok_str == "<score>":
                in_score = True
                continue
            elif tok_str in ("</score>", "<|im_end|>"):
                if in_score:
                    break
                continue

            if tid in self.tok.bar_token_to_idx:
                in_score = True
                if cur_bar is not None:
                    bars.append(cur_bar)
                cur_bar = {
                    "bar_num": self.tok.bar_token_to_idx[tid] + 1,
                    "section": "[Sec:Opening_Hook]",
                    "chord_tok": "[Chord:C:maj]",
                    "dyn_tok": "[Dyn:mp]",
                    "tracks": {1: [], 2: [], 3: [], 4: [], "perc": []},
                }
                cur_trk = 0
                cur_pos = 0
            elif cur_bar is not None:
                if tok_str.startswith("[Sec:"):
                    cur_bar["section"] = tok_str
                elif tid in self.tok.chord_token_to_idx:
                    cur_bar["chord_tok"] = tok_str
                elif tok_str.startswith("[Dyn:"):
                    cur_bar["dyn_tok"] = tok_str
                elif tid in self.tok.trk_token_to_idx:
                    cur_trk = self.tok.trk_token_to_idx[tid]
                    cur_pos = 0
                elif tid in self.tok.pos_token_to_idx:
                    cur_pos = self.tok.pos_token_to_idx[tid]
                elif tid in self.tok.note_id_to_tuple and cur_trk in (1, 2, 3, 4):
                    pitch, dur, vel = self.tok.note_id_to_tuple[tid]
                    cur_bar["tracks"][cur_trk].append((cur_pos, pitch, dur, vel))
                elif tid in self.tok.perc_id_to_tuple:
                    dpitch, vel = self.tok.perc_id_to_tuple[tid]
                    cur_bar["tracks"]["perc"].append((cur_pos, dpitch, vel))

        if cur_bar is not None:
            bars.append(cur_bar)
        return bars

    def evaluate_ids(self, token_ids: List[int], target_bars: int = 16) -> Dict[str, float]:
        bars = self._parse_bars(token_ids)
        if not bars:
            return {
                "r_total": 0.0,
                "r_syntax": 0.0,
                "r_no_stutter": 0.0,
                "r_arp_fluidity": 0.0,
                "r_chord_color": 0.0,
                "r_counterpoint": 0.0,
                "r_hook_motif": 0.0,
                "r_dynamic_arc": 0.0,
            }

        num_bars = len(bars)
        bar_ratio = min(1.0, num_bars / max(4.0, float(target_bars)))

        # 1. Syntax & Track Presence Gate
        lead_bars = sum(1 for b in bars if len(b["tracks"][1]) >= 1)
        piano_bars = sum(1 for b in bars if len(b["tracks"][2]) >= 2)
        r_syntax = float(
            0.40 * bar_ratio
            + 0.30 * (lead_bars / max(1, num_bars))
            + 0.30 * (piano_bars / max(1, num_bars))
        )

        # 2. R_no_stutter: Singing Melodic Phrasing, Stepwise-to-Leap Balance & Phrase Arch
        all_lead = [(b["bar_num"], pos, p, d, v) for b in bars for (pos, p, d, v) in b["tracks"][1]]
        if len(all_lead) >= 4:
            pitches = [x[2] for x in all_lead]
            durs = [x[3] for x in all_lead]
            intervals = [abs(pitches[i + 1] - pitches[i]) for i in range(len(pitches) - 1)]

            # Count consecutive same-pitch repeats (stutter >= 2 intervals of 0 in a row = 3 identical pitches)
            stutter_count = 0
            run_zero = 0
            for iv in intervals:
                if iv == 0:
                    run_zero += 1
                    if run_zero >= 2:
                        stutter_count += 1
                else:
                    run_zero = 0
            stutter_penalty = min(0.75, stutter_count / max(1.0, len(intervals) * 0.25))

            # Singing Melodic Interval Balance (Stepwise vs. Arpeggio-Leaps):
            # A real singing Mandopop melody has ~40%..80% stepwise motion (1..2 semitones) and ~15%..45% sweet leaps (3..9).
            # Broken-chord arpeggio outlines (>55% leaps of 3..7 semitones and <30% stepwise) sound like accompaniment, not melody!
            nonzero_ivs = [iv for iv in intervals if iv > 0]
            if nonzero_ivs:
                stepwise_ratio = sum(1 for iv in nonzero_ivs if iv in (1, 2)) / len(nonzero_ivs)
                leap_ratio = sum(1 for iv in nonzero_ivs if 3 <= iv <= 9) / len(nonzero_ivs)
                wide_jump_ratio = sum(1 for iv in nonzero_ivs if iv > 9) / len(nonzero_ivs)
                if 0.38 <= stepwise_ratio <= 0.82:
                    step_score = 1.0
                else:
                    step_score = max(0.15, 1.0 - abs(stepwise_ratio - 0.56) * 2.2)
                # Penalize broken-chord arpeggio chains (3+ consecutive leaps of 3..7 semitones in same or alternating direction)
                arp_chain = 0
                cur_leaps = 0
                for iv in nonzero_ivs:
                    if 3 <= iv <= 7:
                        cur_leaps += 1
                        if cur_leaps >= 3:
                            arp_chain += 1
                    else:
                        cur_leaps = 0
                arp_outline_pen = min(0.45, arp_chain / max(1.0, len(nonzero_ivs) * 0.28)) + wide_jump_ratio * 0.8
                iv_score = float(np.clip(0.75 * step_score + 0.25 * min(1.0, leap_ratio / 0.25) - arp_outline_pen, 0.0, 1.0))
            else:
                iv_score = 0.1

            # Reward expressive sustained notes (dur >= 4 sixteenth steps) & Cadential Phrase Endings (Bars 4, 8, 12, 16)
            sustained_ratio = sum(1 for d in durs if d >= 4) / len(durs)
            sustained_score = max(0.0, 1.0 - abs(sustained_ratio - 0.38) * 1.4)

            cadence_bars = [b for b in bars if b["bar_num"] in (4, 8, 12, 16) and b["tracks"][1]]
            if cadence_bars:
                cadence_holds = sum(1.0 for b in cadence_bars if max(d for _, _, d, _ in b["tracks"][1]) >= 4)
                cadence_score = cadence_holds / len(cadence_bars)
            else:
                cadence_score = 0.5

            r_no_stutter = float(
                np.clip(
                    0.48 * iv_score + 0.26 * sustained_score + 0.16 * cadence_score + 0.10 - stutter_penalty,
                    0.0,
                    1.0,
                )
            )
        else:
            r_no_stutter = 0.0

        # 3. R_arp_fluidity: Flowing Left-Hand `1-5-8-10` Broken Chord Arpeggios
        arp_bar_scores: List[float] = []
        for b in bars:
            p_notes = b["tracks"][2]
            if len(p_notes) < 3:
                arp_bar_scores.append(0.1)
                continue
            unique_positions = sorted(set(pos for pos, _, _, _ in p_notes))
            pos_spread = min(1.0, len(unique_positions) / 6.0)
            p_pitches = [p for _, p, _, _ in p_notes]
            p_ivs = [abs(p_pitches[i + 1] - p_pitches[i]) for i in range(len(p_pitches) - 1)]
            # Broken chord arpeggios use 3rds/4ths/5ths/octaves (intervals 3..12)
            broken_chord_ivs = sum(1 for iv in p_ivs if 2 <= iv <= 12) / max(1, len(p_ivs))
            pitch_span = max(p_pitches) - min(p_pitches)
            span_score = 1.0 if 10 <= pitch_span <= 28 else max(0.2, 1.0 - abs(pitch_span - 16) / 20.0)
            arp_bar_scores.append(0.45 * pos_spread + 0.35 * broken_chord_ivs + 0.20 * span_score)
        r_arp_fluidity = float(np.clip(np.mean(arp_bar_scores) if arp_bar_scores else 0.0, 0.0, 1.0))

        # 4. R_chord_color: Strong-Beat Harmonic Lock, Diatonic Purity & Zero Vertical Clashes
        key_tok = "[Key:C_maj]"
        is_penta = False
        for raw_tid in token_ids[:45]:
            ts = self.tok.id_to_token.get(int(raw_tid), "")
            if ts.startswith("[Key:"):
                key_tok = ts
            elif ts in ("[Scale:Pentatonic_Core]", "[Style:Guofeng_Lyrical]"):
                is_penta = True
        diatonic_pcs = set(KEY_PENTATONIC_PCS.get(key_tok, [0, 2, 4, 7, 9]) if is_penta else KEY_DIATONIC_PCS.get(key_tok, [0, 2, 4, 5, 7, 9, 11]))
        full_diatonic_pcs = set(KEY_DIATONIC_PCS.get(key_tok, [0, 2, 4, 5, 7, 9, 11]))

        chord_match_scores: List[float] = []
        color_chord_count = 0
        distinct_chords = set()
        out_of_key_notes = 0
        vertical_clash_count = 0
        total_pitched_notes = 0

        for b in bars:
            root_pc, qual, chord_pcs = parse_chord_token(b["chord_tok"])
            distinct_chords.add(b["chord_tok"])
            if qual in ("maj7", "min7", "add9", "sus4", "maj9", "m7b5", "7"):
                color_chord_count += 1
            if root_pc is None:
                continue

            # Allowed sweet tension extensions (major 2nd/9th, 4th/11th, 5th, and 6th/13th in diatonic scale)
            sweet_pcs = set(chord_pcs) | (
                {(root_pc + 2) % 12, (root_pc + 5) % 12, (root_pc + 7) % 12, (root_pc + 9) % 12} & full_diatonic_pcs
            )
            allowed_bar_pcs = full_diatonic_pcs | set(chord_pcs)

            # Check Lead Hook alignment & out-of-key chromatic slips
            lead_hits = 0.0
            lead_total = 0.0
            for pos, p, d, _ in b["tracks"][1]:
                total_pitched_notes += 1
                is_strong = (pos % 4 == 0 or d >= 6)
                weight = 2.0 if is_strong else 1.0
                lead_total += weight
                pc = p % 12
                if pc not in allowed_bar_pcs or (is_penta and pc not in diatonic_pcs and pc not in chord_pcs):
                    out_of_key_notes += 1
                    lead_hits -= 0.8 * weight
                elif pc in chord_pcs:
                    lead_hits += weight
                elif pc in sweet_pcs:
                    # On weak beats, diatonic passing/neighbor tones are 100% natural singing melody!
                    lead_hits += (0.88 if is_strong else 1.0) * weight
                else:
                    lead_hits += (0.55 if is_strong else 0.90) * weight
            lead_align = max(0.0, lead_hits / lead_total) if lead_total > 0 else 0.5

            # Check Piano Arpeggio alignment + Lowest bass note matching root/3rd/5th
            p_notes = b["tracks"][2]
            if p_notes:
                p_hits = 0.0
                for _, p, _, _ in p_notes:
                    total_pitched_notes += 1
                    pc = p % 12
                    if pc not in allowed_bar_pcs:
                        out_of_key_notes += 1
                        p_hits -= 0.5
                    elif pc in chord_pcs:
                        p_hits += 1.0
                    elif pc in sweet_pcs:
                        p_hits += 0.65
                p_align = max(0.0, p_hits / len(p_notes))
                lowest_p = min(p for _, p, _, _ in p_notes)
                root_lock = 1.0 if (lowest_p % 12) == root_pc else (0.7 if (lowest_p % 12) in chord_pcs else 0.1)
            else:
                p_align = 0.0
                root_lock = 0.0

            # Check vertical clashes (iv in (1, 6, 11) = minor 2nd / minor 9th / tritone / major 7th clash)
            # between Lead Hook and Accompaniment (Piano Arp, Strings, Bass) within the bar
            acc_notes = b["tracks"][2] + b["tracks"][3] + b["tracks"][4]
            for l_pos, l_p, l_d, _ in b["tracks"][1]:
                l_end = l_pos + max(1, l_d)
                for a_pos, a_p, a_d, _ in acc_notes:
                    a_end = a_pos + max(1, a_d)
                    if min(l_end, a_end) > max(l_pos, a_pos):
                        iv = abs(l_p - a_p) % 12
                        if iv in (1, 6) or (iv == 11 and (l_p % 12) not in chord_pcs):
                            vertical_clash_count += 1

            chord_match_scores.append(0.45 * lead_align + 0.35 * p_align + 0.20 * root_lock)

        base_harm = float(np.mean(chord_match_scores)) if chord_match_scores else 0.0
        color_bonus = min(0.18, 0.04 * color_chord_count) + min(0.12, 0.03 * len(distinct_chords))
        chromatic_pen = min(0.55, (out_of_key_notes / max(1.0, float(total_pitched_notes))) * 3.5)
        clash_pen = min(0.45, vertical_clash_count * 0.035)
        r_chord_color = float(np.clip(base_harm * 0.78 + color_bonus - chromatic_pen - clash_pen, 0.0, 1.0))

        # 5. R_counterpoint: Lead Hook & Accompaniment Complementary Breathing + Register Separation
        cpt_scores: List[float] = []
        for b in bars:
            l_notes = b["tracks"][1]
            p_notes = b["tracks"][2]
            s_notes = b["tracks"][3]
            if not l_notes or not p_notes:
                cpt_scores.append(0.2)
                continue

            # Register separation: Lead Hook mean pitch should sit clearly above Piano Arp mean pitch
            l_mean = float(np.mean([p for _, p, _, _ in l_notes]))
            p_mean = float(np.mean([p for _, p, _, _ in p_notes]))
            sep = l_mean - p_mean
            sep_score = 1.0 if 9.0 <= sep <= 26.0 else max(0.1, 1.0 - abs(sep - 15.0) / 18.0)

            # Complementary breathing: in the second half of the bar (pos >= 8), if Lead has <= 2 onsets,
            # Piano + Strings should fill the space with >= 3 flowing onsets!
            l_late = sum(1 for pos, _, _, _ in l_notes if pos >= 8)
            acc_late = sum(1 for pos, _, _, _ in (p_notes + s_notes) if pos >= 8)
            if l_late <= 2 and acc_late >= 2:
                breath_score = 1.0
            elif acc_late >= 2:
                breath_score = 0.75
            else:
                breath_score = 0.35

            cpt_scores.append(0.55 * sep_score + 0.45 * breath_score)
        r_counterpoint = float(np.clip(np.mean(cpt_scores) if cpt_scores else 0.0, 0.0, 1.0))

        # 6. R_hook_motif: Motivic Recurrence (Bar 1-2 Opening Hook vs Bar 5-6 / Bar 9-10)
        def bar_interval_signature(bar_idx: int) -> List[int]:
            if bar_idx >= len(bars):
                return []
            ln = sorted(bars[bar_idx]["tracks"][1], key=lambda x: x[0])
            if len(ln) < 2:
                return []
            return [int(np.sign(ln[i + 1][1] - ln[i][1])) for i in range(len(ln) - 1)]

        def contour_similarity(sig_a: List[int], sig_b: List[int]) -> float:
            if not sig_a or not sig_b:
                return 0.0
            m = min(len(sig_a), len(sig_b))
            matches = sum(1.0 for i in range(m) if sig_a[i] == sig_b[i]) / float(m)
            len_sim = min(len(sig_a), len(sig_b)) / float(max(len(sig_a), len(sig_b)))
            return 0.65 * matches + 0.35 * len_sim

        sig_0 = bar_interval_signature(0)
        sig_1 = bar_interval_signature(1)
        recall_scores: List[float] = []
        for target_b in (4, 8, 12):
            if target_b < len(bars):
                sim_a = contour_similarity(sig_0, bar_interval_signature(target_b))
                sim_b = contour_similarity(sig_1, bar_interval_signature(target_b + 1)) if (target_b + 1) < len(bars) else sim_a
                recall_scores.append(max(sim_a, 0.5 * (sim_a + sim_b)))

        # Penalize trivial adjacent bar cloning (Bar 0 == Bar 1 == Bar 2 exact pitch clone)
        adj_clones = 0
        for b_i in range(len(bars) - 1):
            p_a = [p for _, p, _, _ in bars[b_i]["tracks"][1]]
            p_b = [p for _, p, _, _ in bars[b_i + 1]["tracks"][1]]
            if p_a and p_a == p_b:
                adj_clones += 1
        clone_pen = min(0.6, 0.25 * adj_clones)

        base_motif = float(np.max(recall_scores)) if recall_scores else 0.55
        r_hook_motif = float(np.clip(base_motif - clone_pen, 0.0, 1.0))

        # 7. R_dynamic_arc: 4-Act Narrative & Texture Progression
        act1_bars = [b for b in bars if b["bar_num"] <= 4]
        act2_bars = [b for b in bars if 5 <= b["bar_num"] <= 8]
        act3_bars = [b for b in bars if 9 <= b["bar_num"] <= max(12, num_bars - 4)]

        # Act 1 should be intimate (no heavy drums/bass), Act 3 should have Strings + Bass/Perc + higher Lead register
        act1_quiet = 1.0 if sum(len(b["tracks"][4]) + len(b["tracks"]["perc"]) for b in act1_bars) == 0 else 0.4
        act2_strings = min(1.0, sum(1 for b in act2_bars if len(b["tracks"][3]) > 0) / max(1.0, len(act2_bars))) if act2_bars else 0.6
        if act1_bars and act3_bars:
            p_act1 = [p for b in act1_bars for _, p, _, _ in b["tracks"][1]]
            p_act3 = [p for b in act3_bars for _, p, _, _ in b["tracks"][1]]
            lift = (float(np.mean(p_act3)) - float(np.mean(p_act1))) if (p_act1 and p_act3) else 0.0
            lift_score = 1.0 if 2.0 <= lift <= 14.0 else max(0.2, 1.0 - abs(lift - 6.0) / 12.0)
            act3_full = min(1.0, sum(1 for b in act3_bars if (len(b["tracks"][3]) + len(b["tracks"][4]) + len(b["tracks"]["perc"])) >= 2) / max(1.0, len(act3_bars)))
        else:
            lift_score = 0.6
            act3_full = 0.6

        r_dynamic_arc = float(np.clip(0.25 * act1_quiet + 0.25 * act2_strings + 0.25 * lift_score + 0.25 * act3_full, 0.0, 1.0))

        # Composite Total Reward (scaled 0.0 .. 2.0)
        aesthetic_mean = (
            0.22 * r_no_stutter
            + 0.20 * r_arp_fluidity
            + 0.22 * r_chord_color
            + 0.14 * r_counterpoint
            + 0.12 * r_hook_motif
            + 0.10 * r_dynamic_arc
        )
        r_total = float(r_syntax * (1.0 + aesthetic_mean))

        return {
            "r_total": round(r_total, 4),
            "r_syntax": round(r_syntax, 4),
            "r_no_stutter": round(r_no_stutter, 4),
            "r_arp_fluidity": round(r_arp_fluidity, 4),
            "r_chord_color": round(r_chord_color, 4),
            "r_counterpoint": round(r_counterpoint, 4),
            "r_hook_motif": round(r_hook_motif, 4),
            "r_dynamic_arc": round(r_dynamic_arc, 4),
        }
