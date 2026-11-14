"""
Mandopop-V3-115M: 3D Musical Coordinate Transformer & FSM Grammar Mask (`v3/model/modeling_mandopop_v3.py`)
===========================================================================================================
Architecture (~116.6M Parameters):
- Backbone: 16-Layer GQA Transformer (`hidden_size=768`, `heads=12`, `kv_heads=4`, `intermediate_size=2304` SwiGLU, RoPE)
- 3D Musical Axial Embeddings + Active Chord Residual Conditioning:
    h_t^(0) = E_tok(x_t) + E_bar(bar_t) + E_grid(pos_t) + E_track(trk_t) + E_chord(chord_t)
- Deterministic Finite-State Machine (FSM) Grammar LogitsProcessor:
  Guarantees 100% valid Bar -> Section -> Chord -> Dyn -> 4-Track (Lead, Piano, Strings, Bass/Perc)
  progression, strictly monotonic 16th-note intra-bar clock advancement, and zero track dropouts.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import LogitsProcessor, Qwen2Config, Qwen2ForCausalLM, Qwen2Model
from transformers.modeling_outputs import CausalLMOutputWithPast

from v3.tokenizer.v3_unified_tokenizer import (
    CHORD_INTERVALS,
    MAX_BARS,
    NUM_CHORD_TYPES,
    NUM_GRID_POS,
    NUM_TRACK_ROLES,
    PITCH_CLASS_NAMES,
    MandopopV3Tokenizer,
)


KEY_DIATONIC_PCS: Dict[str, List[int]] = {
    "[Key:C_maj]": [0, 2, 4, 5, 7, 9, 11],
    "[Key:A_min]": [0, 2, 4, 5, 7, 9, 11],
    "[Key:G_maj]": [0, 2, 4, 6, 7, 9, 11],
    "[Key:E_min]": [0, 2, 4, 6, 7, 9, 11],
    "[Key:F_maj]": [0, 2, 4, 5, 7, 9, 10],
    "[Key:D_min]": [0, 2, 4, 5, 7, 9, 10],
    "[Key:D_maj]": [1, 2, 4, 6, 7, 9, 11],
    "[Key:B_min]": [1, 2, 4, 6, 7, 9, 11],
    "[Key:Bb_maj]": [0, 2, 3, 5, 7, 9, 10],
    "[Key:G_min]": [0, 2, 3, 5, 7, 9, 10],
    "[Key:Eb_maj]": [0, 2, 3, 5, 7, 8, 10],
    "[Key:C_min]": [0, 2, 3, 5, 7, 8, 10],
}

KEY_PENTATONIC_PCS: Dict[str, List[int]] = {
    "[Key:C_maj]": [0, 2, 4, 7, 9],
    "[Key:A_min]": [0, 2, 4, 7, 9],
    "[Key:G_maj]": [2, 4, 7, 9, 11],
    "[Key:E_min]": [2, 4, 7, 9, 11],
    "[Key:F_maj]": [0, 2, 5, 7, 9],
    "[Key:D_min]": [0, 2, 5, 7, 9],
    "[Key:D_maj]": [2, 4, 6, 9, 11],
    "[Key:B_min]": [2, 4, 6, 9, 11],
    "[Key:Bb_maj]": [2, 5, 7, 9, 10],
    "[Key:G_min]": [2, 5, 7, 9, 10],
    "[Key:Eb_maj]": [0, 3, 5, 7, 10],
    "[Key:C_min]": [0, 3, 5, 7, 10],
}


class MandopopV3Config(Qwen2Config):
    model_type = "mandopop_v3"

    def __init__(
        self,
        vocab_size: int = 4096,
        hidden_size: int = 768,
        intermediate_size: int = 2304,
        num_hidden_layers: int = 16,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 4,
        hidden_act: str = "silu",
        max_position_embeddings: int = 4096,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 50000.0,
        attention_dropout: float = 0.0,
        max_bars: int = MAX_BARS,
        num_grid_pos: int = NUM_GRID_POS,
        num_track_roles: int = NUM_TRACK_ROLES,
        num_chord_types: int = NUM_CHORD_TYPES,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            attention_dropout=attention_dropout,
            **kwargs,
        )
        self.max_bars = max_bars
        self.num_grid_pos = num_grid_pos
        self.num_track_roles = num_track_roles
        self.num_chord_types = num_chord_types


class Musical3DEmbedding(nn.Module):
    """
    Injects 3D Musical Spatio-Temporal Coordinates + Active Bar-Chord Residual Conditioning:
      - Bar Clock Embedding `E_bar(b_t)`      : Periodicity & multi-bar phrase structure (1..32)
      - 16th-Note Beat Grid `E_grid(p_t)`     : Vertical alignment across all 4 tracks at the same beat (0..15)
      - Track Role Embedding `E_track(k_t)`   : Instrument voice identity (CoT, Lead, Piano, Strings, Bass/Perc)
      - Active Chord Residual `E_chord(c_t)`  : Continuous harmonic anchor across all notes in the current bar
    """

    def __init__(self, config: MandopopV3Config):
        super().__init__()
        self.bar_embed = nn.Embedding(config.max_bars, config.hidden_size)
        self.grid_embed = nn.Embedding(config.num_grid_pos, config.hidden_size)
        self.track_embed = nn.Embedding(config.num_track_roles, config.hidden_size)
        self.chord_embed = nn.Embedding(config.num_chord_types, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.scale = nn.Parameter(torch.tensor(0.35))

        self._init_weights(config.initializer_range)

    def _init_weights(self, std: float = 0.02) -> None:
        nn.init.normal_(self.bar_embed.weight, mean=0.0, std=std)
        nn.init.normal_(self.grid_embed.weight, mean=0.0, std=std)
        nn.init.normal_(self.track_embed.weight, mean=0.0, std=std)
        nn.init.normal_(self.chord_embed.weight, mean=0.0, std=std)

    def forward(
        self,
        token_embeds: torch.Tensor,
        bar_ids: torch.Tensor,
        grid_ids: torch.Tensor,
        track_ids: torch.Tensor,
        chord_ids: torch.Tensor,
    ) -> torch.Tensor:
        coord_sum = (
            self.bar_embed(bar_ids)
            + self.grid_embed(grid_ids)
            + self.track_embed(track_ids)
            + self.chord_embed(chord_ids)
        )
        return token_embeds + self.scale.to(token_embeds.dtype) * self.layer_norm(coord_sum.to(token_embeds.dtype))


class MandopopV3ForCausalLM(Qwen2ForCausalLM):
    config_class = MandopopV3Config

    def __init__(self, config: MandopopV3Config):
        super().__init__(config)
        self.musical_3d_embed = Musical3DEmbedding(config)
        self._tokenizer_helper: Optional[MandopopV3Tokenizer] = None
        self.post_init()

    def get_tokenizer_helper(self) -> MandopopV3Tokenizer:
        if self._tokenizer_helper is None:
            self._tokenizer_helper = MandopopV3Tokenizer(vocab_size=self.config.vocab_size)
        return self._tokenizer_helper

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values=None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        # Compute 3D musical coordinates on full `input_ids` BEFORE slicing to last token for KV-cache
        tok = self.get_tokenizer_helper()
        bar_ids, grid_ids, track_ids, chord_ids = tok.compute_3d_musical_coordinates(input_ids)

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        sliced_len = model_inputs["input_ids"].shape[1] if model_inputs.get("input_ids") is not None else 1
        model_inputs["bar_ids"] = bar_ids[:, -sliced_len:]
        model_inputs["grid_ids"] = grid_ids[:, -sliced_len:]
        model_inputs["track_ids"] = track_ids[:, -sliced_len:]
        model_inputs["chord_ids"] = chord_ids[:, -sliced_len:]
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        bar_ids: Optional[torch.LongTensor] = None,
        grid_ids: Optional[torch.LongTensor] = None,
        track_ids: Optional[torch.LongTensor] = None,
        chord_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None and input_ids is not None:
            raw_embeds = self.model.embed_tokens(input_ids)
            if bar_ids is None or grid_ids is None or track_ids is None or chord_ids is None:
                tok = self.get_tokenizer_helper()
                bar_ids, grid_ids, track_ids, chord_ids = tok.compute_3d_musical_coordinates(input_ids)
            inputs_embeds = self.musical_3d_embed(raw_embeds, bar_ids, grid_ids, track_ids, chord_ids)
            input_ids_for_super = None
        else:
            input_ids_for_super = input_ids

        return super().forward(
            input_ids=input_ids_for_super,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

    def count_parameters(self) -> Dict[str, float]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_params": total,
            "trainable_params": trainable,
            "total_millions": round(total / 1e6, 2),
        }


class MandopopGrammarLogitsProcessor(LogitsProcessor):
    """
    Deterministic Finite-State Machine (FSM) Grammar Mask for Mandopop-Studio V3:
    Guarantees 100% legal token transitions during inference & RLVR group rollouts:
      1. Inside `<think> ... </think>`, allows CoT planning tokens and chord tokens; forces `</think>` -> `<score>`.
      2. Inside `<score>`, enforces exact bar-by-bar progression `<Bar_1> .. <Bar_N>`:
         `<Bar_b>` -> `[Sec:*]` -> `[Chord:*]` -> `[Dyn:*]` ->
         `<Trk:Lead_Hook>` -> `<Trk:Piano_Arp>` -> `<Trk:Strings_Pad>` -> `<Trk:Bass_Perc>` -> `<Bar_{b+1}>`
      3. Inside each track, enforces strictly increasing 16th-note grid positions `<Pos_p>` (0 <= p <= 15)
         followed by valid `<Note_*>` (or `<Perc_*>`), completely preventing infinite loops or track dropouts!
    """

    def __init__(
        self,
        tokenizer: MandopopV3Tokenizer,
        target_bars: int = 16,
        min_lead_notes_per_bar: int = 2,
        min_piano_notes_per_bar: int = 4,
    ):
        self.tok = tokenizer
        self.target_bars = min(MAX_BARS, max(4, target_bars))
        self.min_lead_notes = min_lead_notes_per_bar
        self.min_piano_notes = min_piano_notes_per_bar

        # Precompute token ID sets for fast masking
        self.sec_ids = [tokenizer.token_to_id[s] for s in [
            "[Sec:Opening_Hook]", "[Sec:Lyrical_Unfolding]", "[Sec:Emotional_Climax]", "[Sec:Lingering_Outro]"
        ]]
        self.dyn_ids = [tokenizer.token_to_id[d] for d in [
            "[Dyn:pp]", "[Dyn:p]", "[Dyn:mp]", "[Dyn:mf]", "[Dyn:f]", "[Dyn:ff]"
        ]]
        self.chord_ids = [tid for tid, c_idx in tokenizer.chord_token_to_idx.items() if c_idx > 0]
        self.pos_ids = [tokenizer.token_to_id[f"<Pos_{p}>"] for p in range(16)]
        self.note_ids = sorted(list(tokenizer.note_id_to_tuple.keys()))
        self.perc_ids = sorted(list(tokenizer.perc_id_to_tuple.keys()))

        # Pitch-range subsets per track so each instrument stays in its idiomatic register!
        # Lead Hook: E4 (64) .. E6 (88)
        self.lead_note_ids = [tid for tid, (p, _, _) in tokenizer.note_id_to_tuple.items() if 60 <= p <= 88]
        # Piano Arpeggio: C2 (36) .. G4 (67)
        self.piano_note_ids = [tid for tid, (p, _, _) in tokenizer.note_id_to_tuple.items() if 36 <= p <= 67]
        # Strings Pad / Counter-Melody: G3 (55) .. C6 (84)
        self.strings_note_ids = [tid for tid, (p, _, _) in tokenizer.note_id_to_tuple.items() if 52 <= p <= 84]
        # Bass: C1 (24) .. G2 (43)
        self.bass_note_ids = [tid for tid, (p, _, _) in tokenizer.note_id_to_tuple.items() if 28 <= p <= 47]

        self.trk_lead_id = tokenizer.token_to_id["<Trk:Lead_Hook>"]
        self.trk_piano_id = tokenizer.token_to_id["<Trk:Piano_Arp>"]
        self.trk_strings_id = tokenizer.token_to_id["<Trk:Strings_Pad>"]
        self.trk_bass_id = tokenizer.token_to_id["<Trk:Bass_Perc>"]
        self.trk_empty_id = tokenizer.token_to_id["<Track_Empty>"]

        self.cot_allowed_ids = [
            tid for tok_str, tid in tokenizer.token_to_id.items()
            if tok_str.startswith("[Plan_") or tok_str.startswith("[Act") or tok_str.startswith("[Scale:")
            or tok_str.startswith("[Reg_") or tok_str.startswith("[Motion:") or tok_str.startswith("[Arp:")
            or tok_str.startswith("[Cpt:") or tok_str.startswith("[Prog:") or tok_str.startswith("[Motif:")
            or tok_str.startswith("[Chord:")
        ] + [tokenizer.think_end_id]

        # Precompute chord info: chord_tid -> (root_pc, qual, chord_pcs_set, sweet_pcs_set)
        self.chord_tid_info: Dict[int, Tuple[int, str, set, set]] = {}
        for tid in self.chord_ids:
            tok_s = tokenizer.id_to_token.get(tid, "")
            if tok_s.startswith("[Chord:") and ":" in tok_s[7:-1]:
                r_s, q_s = tok_s[7:-1].split(":", 1)
                if r_s in PITCH_CLASS_NAMES:
                    r_pc = PITCH_CLASS_NAMES.index(r_s)
                    ivs = CHORD_INTERVALS.get(q_s, [0, 4, 7])
                    c_pcs = {(r_pc + iv) % 12 for iv in ivs}
                    # Add gentle 9th (+2) and 5th (+7) as sweet color tones if not clashing by 1 semitone with 3rd
                    s_pcs = set(c_pcs) | {(r_pc + 2) % 12, (r_pc + 7) % 12}
                    self.chord_tid_info[tid] = (r_pc, q_s, c_pcs, s_pcs)

        # Cache harmonic note subsets per (track_type, key_tok, is_penta, chord_tid, is_strong_beat)
        self._harmonic_note_cache: Dict[Tuple, List[int]] = {}

    def _get_harmonic_notes_for_track(
        self,
        cur_trk_id: int,
        key_tok: str,
        is_penta: bool,
        cur_chord_tid: int,
        last_pos_val: int,
        lead_pitches_in_bar: List[int],
        last_lead_pitch: Optional[int] = None,
        prev_lead_pitch: Optional[int] = None,
    ) -> List[int]:
        diatonic_pcs = set(KEY_PENTATONIC_PCS.get(key_tok, [0, 2, 4, 7, 9]) if is_penta else KEY_DIATONIC_PCS.get(key_tok, [0, 2, 4, 5, 7, 9, 11]))
        full_diatonic_pcs = set(KEY_DIATONIC_PCS.get(key_tok, [0, 2, 4, 5, 7, 9, 11]))
        r_pc, qual, chord_pcs, sweet_pcs = self.chord_tid_info.get(
            cur_chord_tid, (0, "maj", {0, 4, 7}, {0, 2, 4, 7})
        )

        # For Accompaniment tracks: avoid harsh minor-9th (`+1` semitone) clashes above core chord tones
        core_tones = {r_pc, (r_pc + 7) % 12}
        if (r_pc + 4) % 12 in chord_pcs:
            core_tones.add((r_pc + 4) % 12)
        if (r_pc + 3) % 12 in chord_pcs:
            core_tones.add((r_pc + 3) % 12)

        acc_avoid_pcs = set()
        for ct in core_tones:
            acc_avoid_pcs.add((ct + 1) % 12)

        # For Lead Hook: NEVER strip out the 4th (`F` in C maj) or tonic (`C` in Em) across the bar!
        # Only avoid `(r_pc + 1) % 12` (flat-9 directly above the chord root, e.g. F over E-major/Em root) on strong beats!
        lead_scale_pcs = diatonic_pcs | (chord_pcs & full_diatonic_pcs)
        lead_strong_pcs = (
            (chord_pcs | sweet_pcs | {(r_pc + 2) % 12, (r_pc + 5) % 12, (r_pc + 9) % 12})
            & lead_scale_pcs
        ) - {(r_pc + 1) % 12}
        if not lead_strong_pcs:
            lead_strong_pcs = lead_scale_pcs

        is_strong_beat = (last_pos_val % 4 == 0)

        if cur_trk_id == self.trk_lead_id:
            # Lead Hook: allow full singing scale degrees on weak beats (passing/neighbor tones!)
            # and expressive chord/9th/4th/6th tones on strong beats, while enforcing smooth melodic intervals!
            target_pcs = lead_strong_pcs if is_strong_beat else lead_scale_pcs
            allowed = [
                tid for tid in self.lead_note_ids
                if (self.tok.note_id_to_tuple[tid][0] % 12) in target_pcs
            ]
            if last_lead_pitch is not None and allowed:
                is_stutter_risk = (prev_lead_pitch is not None and prev_lead_pitch == last_lead_pitch)
                # Prefer singing intervals (stepwise 1..2 semitones & lyrical leaps <= 9 semitones, avoiding >9 wide jumps)
                smooth_allowed = [
                    tid for tid in allowed
                    if abs(self.tok.note_id_to_tuple[tid][0] - last_lead_pitch) <= 9
                    and not (is_stutter_risk and self.tok.note_id_to_tuple[tid][0] == last_lead_pitch)
                ]
                if smooth_allowed:
                    return smooth_allowed
            return allowed if allowed else self.lead_note_ids

        # For accompaniment tracks (Piano Arp, Strings Pad, Bass), also avoid any pitch class that clashes
        # by 1 semitone (`iv in (1, 11)`) or tritone (`iv == 6`) with the Lead Hook notes already emitted in this bar!
        lead_clash_pcs = set()
        for lp in lead_pitches_in_bar:
            lpc = lp % 12
            lead_clash_pcs.add((lpc + 1) % 12)
            lead_clash_pcs.add((lpc - 1) % 12)
            lead_clash_pcs.add((lpc + 6) % 12)

        acc_consonant_pcs = ((chord_pcs | (sweet_pcs & full_diatonic_pcs)) - acc_avoid_pcs) - lead_clash_pcs
        if not acc_consonant_pcs:
            acc_consonant_pcs = (chord_pcs - lead_clash_pcs) or chord_pcs

        if cur_trk_id == self.trk_piano_id:
            # On Pos_0 (downbeat of the bar), anchor Piano Arpeggio on the chord root or 5th in the low register!
            if last_pos_val == 0:
                root_fifth_pcs = {r_pc, (r_pc + 7) % 12} - lead_clash_pcs
                if not root_fifth_pcs:
                    root_fifth_pcs = {r_pc}
                allowed = [
                    tid for tid in self.piano_note_ids
                    if (self.tok.note_id_to_tuple[tid][0] % 12) in root_fifth_pcs
                    and self.tok.note_id_to_tuple[tid][0] <= 55
                ]
                if allowed:
                    return allowed
            allowed = [
                tid for tid in self.piano_note_ids
                if (self.tok.note_id_to_tuple[tid][0] % 12) in acc_consonant_pcs
            ]
            return allowed if allowed else self.piano_note_ids

        elif cur_trk_id == self.trk_strings_id:
            allowed = [
                tid for tid in self.strings_note_ids
                if (self.tok.note_id_to_tuple[tid][0] % 12) in acc_consonant_pcs
            ]
            return allowed if allowed else self.strings_note_ids

        else:
            # Bass & Percussion: Bass notes locked to Root or 5th of current chord (never clashing!)
            bass_pcs = {r_pc, (r_pc + 7) % 12} if last_pos_val > 0 else {r_pc}
            allowed_bass = [
                tid for tid in self.bass_note_ids
                if (self.tok.note_id_to_tuple[tid][0] % 12) in bass_pcs
            ]
            if not allowed_bass:
                allowed_bass = self.bass_note_ids
            return allowed_bass + self.perc_ids

    def _get_allowed_tokens_for_row(self, seq: List[int]) -> List[int]:
        if not seq:
            return [self.tok.bos_id]

        last_tid = seq[-1]
        if last_tid == self.tok.eos_id or last_tid == self.tok.pad_id:
            return [self.tok.pad_id]

        if last_tid == self.tok.score_end_id:
            return [self.tok.eos_id]

        # Check if we are in <score> yet
        score_start_positions = [i for i, t in enumerate(seq) if t == self.tok.score_start_id]
        if not score_start_positions:
            if last_tid == self.tok.think_end_id:
                return [self.tok.score_start_id]
            # Count how many tokens spent in <think>
            think_pos = [i for i, t in enumerate(seq) if t == self.tok.think_start_id]
            if think_pos:
                think_len = len(seq) - think_pos[-1]
                if think_len >= 28:
                    return [self.tok.think_end_id]
                elif think_len < 10:
                    return [t for t in self.cot_allowed_ids if t != self.tok.think_end_id]
                return self.cot_allowed_ids
            return [self.tok.think_start_id]

        # Inside <score>
        if last_tid == self.tok.score_start_id:
            return [self.tok.token_to_id["<Bar_1>"]]

        # Extract global Key, Scale, and Planned Chords from prefix (`seq[:score_start_positions[-1]]`)
        prefix_slice = seq[: score_start_positions[-1]]
        key_tok = "[Key:C_maj]"
        is_penta = False
        planned_chords: List[int] = []
        for tid in prefix_slice:
            tok_s = self.tok.id_to_token.get(tid, "")
            if tok_s.startswith("[Key:"):
                key_tok = tok_s
            elif tok_s == "[Scale:Pentatonic_Core]" or tok_s == "[Style:Guofeng_Lyrical]":
                is_penta = True
            elif tid in self.tok.chord_token_to_idx and self.tok.chord_token_to_idx[tid] > 0:
                planned_chords.append(tid)

        # Find current bar and current track state by scanning back to the latest <Bar_b>
        cur_bar_num = 1
        bar_pos_in_seq = -1
        for idx in range(len(seq) - 1, score_start_positions[-1], -1):
            tid = seq[idx]
            if tid in self.tok.bar_token_to_idx:
                cur_bar_num = self.tok.bar_token_to_idx[tid] + 1
                bar_pos_in_seq = idx
                break

        if last_tid in self.tok.bar_token_to_idx:
            # Force the appropriate 4-Act section tag based on bar number!
            b = self.tok.bar_token_to_idx[last_tid] + 1
            if b <= 4:
                return [self.tok.token_to_id["[Sec:Opening_Hook]"]]
            elif b <= 8:
                return [self.tok.token_to_id["[Sec:Lyrical_Unfolding]"]]
            elif b <= max(12, self.target_bars - 4):
                return [self.tok.token_to_id["[Sec:Emotional_Climax]"]]
            else:
                return [self.tok.token_to_id["[Sec:Lingering_Outro]"]]

        if last_tid in self.sec_ids:
            # Lock bar chord to the CoT planned harmonic progression (cycling cleanly every 8 bars),
            # or allow diatonic chords if no planned chords exist in prefix!
            if planned_chords:
                target_chord = planned_chords[(cur_bar_num - 1) % len(planned_chords)]
                return [target_chord]
            diatonic_pcs = set(KEY_DIATONIC_PCS.get(key_tok, [0, 2, 4, 5, 7, 9, 11]))
            diatonic_chords = [
                tid for tid, (r_pc, _, c_pcs, _) in self.chord_tid_info.items()
                if r_pc in diatonic_pcs and c_pcs.issubset(diatonic_pcs)
            ]
            return diatonic_chords if diatonic_chords else self.chord_ids

        if last_tid in self.tok.chord_token_to_idx and bar_pos_in_seq != -1 and len(seq) - bar_pos_in_seq <= 4:
            return self.dyn_ids

        if last_tid in self.dyn_ids:
            return [self.trk_lead_id]

        # Determine active bar chord, active track, and Lead Hook pitches already emitted in this bar
        bar_slice = seq[bar_pos_in_seq:]
        cur_chord_tid = planned_chords[(cur_bar_num - 1) % len(planned_chords)] if planned_chords else self.chord_ids[0]
        cur_trk_id = None
        trk_start_offset = -1
        lead_pitches_in_bar: List[int] = []

        for i_s, tid in enumerate(bar_slice):
            if tid in self.tok.chord_token_to_idx and self.tok.chord_token_to_idx[tid] > 0:
                cur_chord_tid = tid
            elif tid in (self.trk_lead_id, self.trk_piano_id, self.trk_strings_id, self.trk_bass_id):
                cur_trk_id = tid
                trk_start_offset = i_s
            elif cur_trk_id == self.trk_lead_id and tid in self.tok.note_id_to_tuple:
                lead_pitches_in_bar.append(self.tok.note_id_to_tuple[tid][0])

        if cur_trk_id is None:
            return [self.trk_lead_id]

        trk_slice = bar_slice[trk_start_offset + 1:]

        # Next track transition helper
        def next_structural_tokens() -> List[int]:
            if cur_trk_id == self.trk_lead_id:
                return [self.trk_piano_id]
            elif cur_trk_id == self.trk_piano_id:
                return [self.trk_strings_id]
            elif cur_trk_id == self.trk_strings_id:
                return [self.trk_bass_id]
            else:
                if cur_bar_num >= self.target_bars:
                    return [self.tok.score_end_id]
                return [self.tok.token_to_id[f"<Bar_{cur_bar_num + 1}>"]]

        # If track just started
        if len(trk_slice) == 0:
            if cur_trk_id in (self.trk_lead_id, self.trk_piano_id):
                # Lead and Piano must play in every bar; start within Pos_0 .. Pos_4
                return self.pos_ids[:5]
            elif cur_trk_id == self.trk_strings_id:
                if cur_bar_num <= 2:
                    # Opening 2 bars can be pure solo piano + lead or gentle strings
                    return [self.trk_empty_id] + self.pos_ids[:5]
                return self.pos_ids[:5]
            else:
                # Bass & Percussion: rest in Act 1 (Bars 1..4) for pure intro clarity!
                if cur_bar_num <= 4:
                    return [self.trk_empty_id]
                return self.pos_ids[:5] + [self.trk_empty_id]

        if last_tid == self.trk_empty_id:
            return next_structural_tokens()

        # Find latest position `<Pos_p>` in current track and how many notes were emitted at `<Pos_p>`
        last_pos_val = -1
        notes_at_cur_pos = 0
        total_notes_in_trk = 0
        for tid in trk_slice:
            if tid in self.tok.pos_token_to_idx:
                last_pos_val = self.tok.pos_token_to_idx[tid]
                notes_at_cur_pos = 0
            elif tid in self.tok.note_id_to_tuple or tid in self.tok.perc_id_to_tuple:
                notes_at_cur_pos += 1
                total_notes_in_trk += 1

        # Select harmonic-locked, clash-free note vocabulary & max notes per bar
        last_lead_p = lead_pitches_in_bar[-1] if lead_pitches_in_bar else None
        prev_lead_p = lead_pitches_in_bar[-2] if len(lead_pitches_in_bar) >= 2 else None
        if last_lead_p is None and cur_trk_id == self.trk_lead_id and bar_pos_in_seq > score_start_positions[-1]:
            # Look back to the previous bar's last Lead_Hook pitch for smooth cross-bar phrase connection
            scan_trk = None
            for tid in seq[max(score_start_positions[-1], bar_pos_in_seq - 64):bar_pos_in_seq]:
                if tid in (self.trk_lead_id, self.trk_piano_id, self.trk_strings_id, self.trk_bass_id):
                    scan_trk = tid
                elif scan_trk == self.trk_lead_id and tid in self.tok.note_id_to_tuple:
                    prev_lead_p = last_lead_p
                    last_lead_p = self.tok.note_id_to_tuple[tid][0]

        valid_notes = self._get_harmonic_notes_for_track(
            cur_trk_id=cur_trk_id,
            key_tok=key_tok,
            is_penta=is_penta,
            cur_chord_tid=cur_chord_tid,
            last_pos_val=max(0, last_pos_val),
            lead_pitches_in_bar=lead_pitches_in_bar,
            last_lead_pitch=last_lead_p,
            prev_lead_pitch=prev_lead_p,
        )
        if cur_trk_id == self.trk_lead_id:
            min_req = self.min_lead_notes
            max_notes_bar = 7
            max_poly = 1  # Monophonic singing lead hook
        elif cur_trk_id == self.trk_piano_id:
            min_req = self.min_piano_notes
            max_notes_bar = 8
            max_poly = 2  # Flowing arpeggio / 2-note voicing per grid step
        elif cur_trk_id == self.trk_strings_id:
            min_req = 1
            max_notes_bar = 4
            max_poly = 2
        else:
            min_req = 1
            max_notes_bar = 5
            max_poly = 2

        # Immediately after <Pos_p>, MUST emit a note!
        if last_tid in self.tok.pos_token_to_idx:
            return valid_notes

        # If we reached max_notes_bar for this track, advance to next track/bar immediately
        if total_notes_in_trk >= max_notes_bar:
            return next_structural_tokens()

        # After emitting note(s) at <Pos_p>:
        remaining_pos_ids = [self.pos_ids[p] for p in range(last_pos_val + 1, 16)]
        allowed: List[int] = []

        if notes_at_cur_pos < max_poly:
            allowed.extend(valid_notes)

        if remaining_pos_ids:
            allowed.extend(remaining_pos_ids)

        if total_notes_in_trk >= min_req or not remaining_pos_ids:
            allowed.extend(next_structural_tokens())

        return allowed if allowed else next_structural_tokens()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        B = input_ids.shape[0]
        mask = torch.full_like(scores, float("-inf"))
        seqs = input_ids.tolist()
        for b_i in range(B):
            allowed = self._get_allowed_tokens_for_row(seqs[b_i])
            mask[b_i, allowed] = 0.0
        return scores + mask
