#!/usr/bin/env python3
"""
Mandopop-Studio V3 — Phase 3: Dr. GRPO RLVR Trainer (6D Aesthetic Verifier)
==========================================================================
Trains `Mandopop-V3-116M` from `checkpoints/v3_sft/final_model` using:
  - Group Relative Policy Optimization (`Dr. GRPO`, Liu et al. 2025)
  - 6-Dimensional Mandopop Intro & Theme Aesthetic Verifier (`MandopopIntroVerifier.evaluate_ids`)
  - Hard FSM Grammar Mask (`MandopopGrammarLogitsProcessor`) via HuggingFace `LogitsProcessorList`
  - Length-unbiased policy gradient + token-level KL divergence anchor to SFT reference
  - Strict local VRAM ceiling (< 3.2 GB on 8GB RTX 5070) via micro-batched logprob evaluation
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import LogitsProcessorList

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v3.model.modeling_mandopop_v3 import (
    MandopopGrammarLogitsProcessor,
    MandopopV3ForCausalLM,
)
from v3.tokenizer.v3_unified_tokenizer import MandopopV3Tokenizer
from v3.verifiers.intro_aesthetic_verifier import MandopopIntroVerifier


CURATED_MANDOPOP_SPECS: List[Dict] = [
    # 1. Slow Lyrical Ballad Intros (Jay Chou / JJ Lin / Eric Chou style)
    {
        "style": "[Style:Ballad_Slow]",
        "mood": "[Mood:Nostalgic_Warm]",
        "tempo": "[Tempo:72]",
        "key": "[Key:C_maj]",
        "reg": "[Reg_Mid:E4_C5]",
        "arp": "[Arp:1_5_8_10_Fluid]",
        "chords": [
            "[Chord:F:maj7]", "[Chord:G:7]", "[Chord:E:min7]", "[Chord:A:min7]",
            "[Chord:D:min7]", "[Chord:G:sus4]", "[Chord:C:add9]", "[Chord:C:maj7]",
        ],
    },
    {
        "style": "[Style:Ballad_Slow]",
        "mood": "[Mood:Melancholic_Tearful]",
        "tempo": "[Tempo:68]",
        "key": "[Key:A_min]",
        "reg": "[Reg_High:G4_E5]",
        "arp": "[Arp:1_5_8_10_Fluid]",
        "chords": [
            "[Chord:A:min7]", "[Chord:F:maj7]", "[Chord:C:add9]", "[Chord:G:sus4]",
            "[Chord:D:min7]", "[Chord:E:7]", "[Chord:A:min7]", "[Chord:A:min7]",
        ],
    },
    {
        "style": "[Style:Ballad_Slow]",
        "mood": "[Mood:Nostalgic_Warm]",
        "tempo": "[Tempo:76]",
        "key": "[Key:G_maj]",
        "reg": "[Reg_Mid:E4_C5]",
        "arp": "[Arp:1_5_8_10_Fluid]",
        "chords": [
            "[Chord:G:add9]", "[Chord:D:maj]", "[Chord:E:min7]", "[Chord:B:min7]",
            "[Chord:C:maj7]", "[Chord:G:maj]", "[Chord:A:min7]", "[Chord:D:sus4]",
        ],
    },
    {
        "style": "[Style:Ballad_Slow]",
        "mood": "[Mood:Melancholic_Tearful]",
        "tempo": "[Tempo:72]",
        "key": "[Key:F_maj]",
        "reg": "[Reg_High:G4_E5]",
        "arp": "[Arp:1_5_8_10_Fluid]",
        "chords": [
            "[Chord:Bb:maj7]", "[Chord:C:7]", "[Chord:A:min7]", "[Chord:D:min7]",
            "[Chord:G:min7]", "[Chord:C:sus4]", "[Chord:F:add9]", "[Chord:F:maj7]",
        ],
    },
    # 2. Guofeng Lyrical Intros (青花瓷 / 大鱼 / 红尘客栈 / 雪落下的声音 style)
    {
        "style": "[Style:Guofeng_Lyrical]",
        "mood": "[Mood:Poetic_Breeze]",
        "tempo": "[Tempo:76]",
        "key": "[Key:C_maj]",
        "reg": "[Reg_High:G4_E5]",
        "arp": "[Arp:1_5_8_10_Fluid]",
        "chords": [
            "[Chord:F:maj7]", "[Chord:G:sus4]", "[Chord:E:min7]", "[Chord:A:min7]",
            "[Chord:D:min7]", "[Chord:G:7]", "[Chord:A:min7]", "[Chord:A:min7]",
        ],
    },
    {
        "style": "[Style:Guofeng_Lyrical]",
        "mood": "[Mood:Poetic_Breeze]",
        "tempo": "[Tempo:80]",
        "key": "[Key:G_maj]",
        "reg": "[Reg_Mid:E4_C5]",
        "arp": "[Arp:1_5_8_10_Fluid]",
        "chords": [
            "[Chord:C:maj7]", "[Chord:D:7]", "[Chord:B:min7]", "[Chord:E:min7]",
            "[Chord:A:min7]", "[Chord:D:sus4]", "[Chord:G:add9]", "[Chord:E:min7]",
        ],
    },
    {
        "style": "[Style:Guofeng_Lyrical]",
        "mood": "[Mood:Melancholic_Tearful]",
        "tempo": "[Tempo:68]",
        "key": "[Key:D_min]",
        "reg": "[Reg_High:G4_E5]",
        "arp": "[Arp:1_5_8_10_Fluid]",
        "chords": [
            "[Chord:Bb:maj7]", "[Chord:C:7]", "[Chord:A:min7]", "[Chord:D:min7]",
            "[Chord:G:min7]", "[Chord:A:7]", "[Chord:D:min7]", "[Chord:D:min7]",
        ],
    },
    # 3. Urban Romantic / R&B Intros (水星记 / 告白气球 / 交换余生 style)
    {
        "style": "[Style:Urban_Romantic]",
        "mood": "[Mood:Nostalgic_Warm]",
        "tempo": "[Tempo:84]",
        "key": "[Key:C_maj]",
        "reg": "[Reg_Mid:E4_C5]",
        "arp": "[Arp:Broken_9th_Ballad]",
        "chords": [
            "[Chord:D:min7]", "[Chord:G:7]", "[Chord:E:min7]", "[Chord:A:min7]",
            "[Chord:D:min7]", "[Chord:G:sus4]", "[Chord:C:maj9]", "[Chord:A:min7]",
        ],
    },
    {
        "style": "[Style:Urban_Romantic]",
        "mood": "[Mood:Dramatic_Swell]",
        "tempo": "[Tempo:88]",
        "key": "[Key:F_maj]",
        "reg": "[Reg_High:G4_E5]",
        "arp": "[Arp:Broken_9th_Ballad]",
        "chords": [
            "[Chord:Bb:maj7]", "[Chord:C:sus4]", "[Chord:A:min7]", "[Chord:D:min7]",
            "[Chord:G:min7]", "[Chord:C:7]", "[Chord:F:maj9]", "[Chord:D:min7]",
        ],
    },
    # 4. Pop-Rock Driving Intros (晴天 / 起风了 / 倔强 style)
    {
        "style": "[Style:PopRock_Driving]",
        "mood": "[Mood:Dramatic_Swell]",
        "tempo": "[Tempo:96]",
        "key": "[Key:G_maj]",
        "reg": "[Reg_High:G4_E5]",
        "arp": "[Arp:Broken_9th_Ballad]",
        "chords": [
            "[Chord:E:min7]", "[Chord:C:add9]", "[Chord:G:maj]", "[Chord:D:sus4]",
            "[Chord:E:min7]", "[Chord:C:maj7]", "[Chord:D:7]", "[Chord:G:add9]",
        ],
    },
]


def build_prompt_ids(tokenizer: MandopopV3Tokenizer, spec: Dict, target_bars: int = 8) -> List[int]:
    """
    Constructs the exact V3 prompt + `<think> ... </think> <score>` prefix matching `build_v3_dataset.py`
    so RL rollouts focus 100% of exploration capacity on the 4-track polyphonic score!
    """
    scale_tok = "[Scale:Pentatonic_Core]" if spec["style"] == "[Style:Guofeng_Lyrical]" else "[Scale:Diatonic_Lyrical]"
    tokens = [
        "<|im_start|>",
        spec["style"],
        spec["mood"],
        spec["tempo"],
        spec["key"],
        f"[Bars:{target_bars}]",
        "<think>",
        "[Plan_Arc]", "[Act1:Opening_Hook]", "[Act2:Lyrical_Dialogue]", "[Act3:Emotional_Climax]", "[Act4:Lingering_Outro]",
        "[Plan_Scale]", scale_tok,
        "[Plan_Register]", spec["reg"], "[Motion:Third_Sixth_Lyrical]",
        "[Plan_Arpeggio]", spec["arp"],
        "[Plan_Counterpoint]", "[Cpt:Bridge_Fills_Breaths]",
        "[Plan_Motif]", "[Motif:AA_BA_Hook]",
        "[Plan_Harmony]",
    ] + spec["chords"] + [
        "</think>",
        "<score>",
    ]
    return tokenizer.encode_tokens(tokens)


def compute_sequence_token_logprobs(
    model: MandopopV3ForCausalLM,
    tokenizer: MandopopV3Tokenizer,
    seq_ids: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """
    Computes per-token log-probabilities for generated tokens `seq_ids[:, prompt_len:]`.
    `seq_ids`: `(B, L)`
    Returns: `(B, L - prompt_len)` logprobs.
    """
    bar_ids, grid_ids, track_ids, chord_ids = tokenizer.compute_3d_musical_coordinates(seq_ids)
    out = model(
        input_ids=seq_ids[:, :-1],
        bar_ids=bar_ids[:, :-1],
        grid_ids=grid_ids[:, :-1],
        track_ids=track_ids[:, :-1],
        chord_ids=chord_ids[:, :-1],
        use_cache=False,
    )
    logits = out.logits  # (B, L-1, V)
    gen_logits = logits[:, prompt_len - 1 :, :]  # (B, L - prompt_len, V)
    target_ids = seq_ids[:, prompt_len:]  # (B, L - prompt_len)

    log_probs = F.log_softmax(gen_logits.float(), dim=-1)
    token_log_probs = torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    return token_log_probs


def compute_lead_phrase_craft_score(bars: List[Dict], target_bars: int) -> float:
    """Dense melodic craft reward for Lead_Hook (track 1): arch, range, phrasing, motif continuity."""
    if not bars:
        return 0.0
    bar_notes = [sorted(b["tracks"][1], key=lambda x: x[0]) for b in bars]
    active_bars = sum(1 for bn in bar_notes if len(bn) >= 1)
    coverage = min(1.0, active_bars / max(1.0, target_bars * 0.85))

    # Phrase-end breathing on cadence bars (4, 8, 12, 16 -> indices 3, 7, 11, 15)
    cadence_hits, cadence_total = 0.0, 0
    for idx in (3, 7, 11, 15):
        if idx < len(bar_notes):
            cadence_total += 1
            if bar_notes[idx]:
                last_dur = bar_notes[idx][-1][2]
                cadence_hits += 1.0 if last_dur >= 6 else (0.65 if last_dur >= 4 else 0.15)
    r_cadence = (cadence_hits / cadence_total) if cadence_total > 0 else 0.5

    # Per-bar density balance (ideal Mandopop bar has 2-8 lead notes)
    bar_dens = [1.0 if 2 <= len(bn) <= 8 else (0.5 if len(bn) in (1, 9, 10) else 0.1) for bn in bar_notes]
    r_density = 0.40 * coverage + 0.35 * (sum(bar_dens) / max(1, len(bar_dens))) + 0.25 * r_cadence

    pitches = [p for bn in bar_notes for _, p, _, _ in bn]
    durs = [d for bn in bar_notes for _, _, d, _ in bn]
    if len(pitches) < 6:
        return 0.10

    span = max(pitches) - min(pitches)
    if 7 <= span <= 21:
        r_span = 1.0
    elif span < 7:
        r_span = max(0.2, span / 7.0)
    else:
        r_span = max(0.3, 1.0 - (span - 21) / 14.0)

    diffs = [pitches[i + 1] - pitches[i] for i in range(len(pitches) - 1)]
    rep_ratio = sum(1 for d in diffs if d == 0) / max(1, len(diffs))
    r_rep = 1.0 if rep_ratio <= 0.14 else max(0.15, 1.0 - (rep_ratio - 0.14) * 3.0)

    nz = [abs(d) for d in diffs if d != 0]
    if not nz:
        return 0.15
    step_ratio = sum(1 for d in nz if d <= 2) / len(nz)
    if 0.42 <= step_ratio <= 0.76:
        r_step = 1.0
    elif step_ratio < 0.42:
        r_step = max(0.2, step_ratio / 0.42)
    else:
        r_step = max(0.35, 1.0 - (step_ratio - 0.76) * 2.0)

    consec_leaps = sum(1 for i in range(len(diffs) - 1) if abs(diffs[i]) >= 5 and abs(diffs[i + 1]) >= 5)
    extreme_leaps = sum(1 for d in diffs if abs(d) >= 10)
    leap_pen = min(0.65, consec_leaps * 0.12 + extreme_leaps * 0.10)
    r_contour = max(0.0, 0.25 * r_span + 0.30 * r_rep + 0.45 * r_step - leap_pen)

    long_ratio = sum(1 for d in durs if d >= 4) / max(1, len(durs))
    dur_div = min(1.0, len(set(durs)) / 4.0)
    r_long = 1.0 if 0.14 <= long_ratio <= 0.52 else max(0.2, 1.0 - abs(long_ratio - 0.30) * 2.2)
    r_rhythm = 0.60 * r_long + 0.40 * dur_div

    # Call-and-response contour linkage across 4-bar anchors (Bar 1->5, Bar 9->13) without blind cloning
    def _bar_sign_sig(bn):
        ps = [p for _, p, _, _ in bn[:5]]
        return tuple((1 if ps[i + 1] > ps[i] else (-1 if ps[i + 1] < ps[i] else 0)) for i in range(len(ps) - 1))

    motif_links = []
    for a_idx, b_idx in ((0, 4), (1, 5), (8, 12), (9, 13)):
        if b_idx < len(bar_notes) and len(bar_notes[a_idx]) >= 2 and len(bar_notes[b_idx]) >= 2:
            sa, sb = _bar_sign_sig(bar_notes[a_idx]), _bar_sign_sig(bar_notes[b_idx])
            m_len = min(len(sa), len(sb))
            sim = sum(1 for k in range(m_len) if sa[k] == sb[k]) / max(1, m_len)
            pa = [p for _, p, _, _ in bar_notes[a_idx]]
            pb = [p for _, p, _, _ in bar_notes[b_idx]]
            # Reward strong contour/motif resemblance unless it is a flat carbon copy
            motif_links.append(0.85 if pa == pb else sim)
    r_motif_link = (sum(motif_links) / len(motif_links)) if motif_links else 0.45

    return float(max(0.0, min(1.0, 0.25 * r_density + 0.35 * r_contour + 0.20 * r_rhythm + 0.20 * r_motif_link)))


def build_musical_content_id_set(tokenizer: MandopopV3Tokenizer) -> set:
    return (
        set(tokenizer.note_id_to_tuple.keys())
        | set(tokenizer.pos_token_to_idx.keys())
        | set(tokenizer.perc_id_to_tuple.keys())
    )


def main():
    parser = argparse.ArgumentParser(description="Mandopop-Studio V3.5 Phase 3: Track-Aware Dr. GRPO RLVR Trainer")
    parser.add_argument("--sft_ckpt", type=str, default=str(PROJECT_ROOT / "checkpoints/v3_sft/final_model"))
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "checkpoints/v3_grpo"))
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=850)
    parser.add_argument("--target_bars", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2.0e-5)
    parser.add_argument("--kl_beta", type=float, default=0.02)
    args = parser.parse_args()
    args.max_new_tokens = max(args.max_new_tokens, args.target_bars * 52 + 20)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = MandopopV3Tokenizer()
    verifier = MandopopIntroVerifier(tokenizer)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"==================================================================")
    print(f"🎯 Starting Mandopop-Studio V3 [Phase 3: Dr. GRPO RLVR]")
    print(f"   Init SFT Checkpoint : {args.sft_ckpt}")
    print(f"   Group Size (G)      : {args.group_size} rollouts/prompt")
    print(f"   RL Steps            : {args.steps} | Max New Tokens: {args.max_new_tokens}")
    print(f"   LR                  : {args.lr} | KL Beta: {args.kl_beta}")
    print(f"==================================================================", flush=True)

    policy_model = MandopopV3ForCausalLM.from_pretrained(args.sft_ckpt, torch_dtype=torch.bfloat16).to(device)
    ref_model = MandopopV3ForCausalLM.from_pretrained(args.sft_ckpt, torch_dtype=torch.bfloat16).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    history = []
    rollout_archive: List[Dict] = []
    best_mean_reward = -1.0
    start_time = time.time()

    content_id_set = build_musical_content_id_set(tokenizer)
    for step in range(1, args.steps + 1):
        spec = CURATED_MANDOPOP_SPECS[(step - 1) % len(CURATED_MANDOPOP_SPECS)]
        prompt_ids_list = build_prompt_ids(tokenizer, spec, target_bars=args.target_bars)
        prompt_len = len(prompt_ids_list)
        prompt_tensor = torch.tensor(
            [prompt_ids_list] * args.group_size, dtype=torch.long, device=device
        )

        # 1. Batched Group Rollout Generation with FSM Grammar Processor
        policy_model.eval()
        fsm_processor = MandopopGrammarLogitsProcessor(tokenizer, target_bars=args.target_bars)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                generated_batch = policy_model.generate(
                    input_ids=prompt_tensor,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.94,
                    top_k=56,
                    pad_token_id=tokenizer.pad_id,
                    eos_token_id=tokenizer.eos_id,
                    logits_processor=LogitsProcessorList([fsm_processor]),
                    use_cache=True,
                )

        # 2. Score each rollout with MandopopIntroVerifier + dense Track-Decomposed Rewards
        rewards_list: List[float] = []
        lead_rewards_list: List[float] = []
        piano_rewards_list: List[float] = []
        acc_rewards_list: List[float] = []
        sub_metrics_list: List[Dict[str, float]] = []
        trimmed_sequences: List[torch.Tensor] = []

        for g_idx in range(args.group_size):
            seq = generated_batch[g_idx].tolist()
            if tokenizer.eos_id in seq[prompt_len:]:
                eos_pos = seq.index(tokenizer.eos_id, prompt_len) + 1
                seq = seq[:eos_pos]
            trimmed_sequences.append(torch.tensor(seq, dtype=torch.long, device=device))

            metrics = verifier.evaluate_ids(seq, target_bars=args.target_bars)
            bars = verifier._parse_bars(seq)
            completion = min(1.0, len(bars) / max(1, args.target_bars)) ** 2
            eos_gate = 1.0 if (len(seq) > prompt_len and seq[-1] == tokenizer.eos_id) else 0.92
            r_craft = compute_lead_phrase_craft_score(bars, target_bars=args.target_bars)
            metrics["r_lead_craft"] = r_craft
            metrics["num_bars"] = float(len(bars))

            r_lead = completion * eos_gate * (
                0.40 * metrics["r_no_stutter"] + 0.32 * metrics["r_hook_motif"] + 0.28 * r_craft
            )
            r_piano = completion * eos_gate * (
                0.65 * metrics["r_arp_fluidity"] + 0.35 * metrics["r_counterpoint"]
            )
            r_acc = completion * eos_gate * (
                0.55 * metrics["r_dynamic_arc"] + 0.45 * metrics["r_counterpoint"]
            )
            r_rl = completion * eos_gate * (
                0.34 * metrics["r_no_stutter"]
                + 0.28 * metrics["r_hook_motif"]
                + 0.16 * r_craft
                + 0.12 * metrics["r_arp_fluidity"]
                + 0.10 * metrics["r_dynamic_arc"]
            )
            metrics["r_rl"] = r_rl
            lead_rewards_list.append(r_lead)
            piano_rewards_list.append(r_piano)
            acc_rewards_list.append(r_acc)
            rewards_list.append(r_rl)
            sub_metrics_list.append(metrics)

        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)
        mean_r = rewards.mean().item()
        std_r = rewards.std().item()
        max_r = rewards.max().item()
        min_r = rewards.min().item()

        # Archive best and worst rollouts from this step for Phase 4 SimPO/DPO + OPD
        best_idx = int(torch.argmax(rewards).item())
        worst_idx = int(torch.argmin(rewards).item())
        if rewards_list[best_idx] - rewards_list[worst_idx] >= 0.03:
            rollout_archive.append(
                {
                    "step": step,
                    "spec": spec,
                    "prompt_len": prompt_len,
                    "chosen_ids": trimmed_sequences[best_idx].cpu().tolist(),
                    "rejected_ids": trimmed_sequences[worst_idx].cpu().tolist(),
                    "chosen_reward": round(rewards_list[best_idx], 4),
                    "rejected_reward": round(rewards_list[worst_idx], 4),
                    "chosen_metrics": sub_metrics_list[best_idx],
                    "rejected_metrics": sub_metrics_list[worst_idx],
                }
            )

        def _dr_grpo_adv(vals: List[float]) -> torch.Tensor:
            t = torch.tensor(vals, dtype=torch.float32, device=device)
            return torch.clamp((t - t.mean()) / t.std().clamp(min=0.05), -3.0, 3.0)

        lead_t = torch.tensor(lead_rewards_list, dtype=torch.float32, device=device)
        if std_r < 1e-4 and lead_t.std().item() < 1e-4:
            continue

        advantages = _dr_grpo_adv(rewards_list)
        lead_advs = _dr_grpo_adv(lead_rewards_list)
        piano_advs = _dr_grpo_adv(piano_rewards_list)
        acc_advs = _dr_grpo_adv(acc_rewards_list)

        # 3. Track-Decomposed Policy Gradient + Reference KL Update
        policy_model.train()
        optimizer.zero_grad(set_to_none=True)

        total_pg_loss = 0.0
        total_kl_val = 0.0

        for g_idx in range(args.group_size):
            seq_1d = trimmed_sequences[g_idx].unsqueeze(0)  # (1, L)
            gen_len = seq_1d.shape[1] - prompt_len
            if gen_len < 16:
                continue

            with torch.no_grad():
                _, _, track_ids, _ = tokenizer.compute_3d_musical_coordinates(seq_1d)
                gen_trk = track_ids[:, prompt_len:]
                gen_toks = seq_1d[0, prompt_len:].tolist()
                is_content = torch.tensor(
                    [[tid in content_id_set for tid in gen_toks]],
                    dtype=torch.bool,
                    device=device,
                )
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    ref_logprobs = compute_sequence_token_logprobs(ref_model, tokenizer, seq_1d, prompt_len)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pol_logprobs = compute_sequence_token_logprobs(policy_model, tokenizer, seq_1d, prompt_len)

            log_ratio = ref_logprobs.detach() - pol_logprobs
            token_kl = torch.exp(torch.clamp(log_ratio, max=5.0)) - log_ratio - 1.0

            g_global = advantages[g_idx].detach()
            g_lead = (0.80 * lead_advs[g_idx] + 0.20 * advantages[g_idx]).detach()
            g_piano = (0.70 * piano_advs[g_idx] + 0.30 * advantages[g_idx]).detach()
            g_acc = (0.70 * acc_advs[g_idx] + 0.30 * advantages[g_idx]).detach()

            lead_mask = (gen_trk == 1) & is_content
            piano_mask = (gen_trk == 2) & is_content
            acc_mask = ((gen_trk == 3) | (gen_trk == 4)) & is_content

            token_adv = torch.full_like(pol_logprobs, g_global)
            token_adv = torch.where(lead_mask, g_lead, token_adv)
            token_adv = torch.where(piano_mask, g_piano, token_adv)
            token_adv = torch.where(acc_mask, g_acc, token_adv)

            token_weights = torch.full_like(pol_logprobs, 0.10)
            token_weights = torch.where(lead_mask, 2.00, token_weights)
            token_weights = torch.where(piano_mask, 0.90, token_weights)
            token_weights = torch.where(acc_mask, 0.50, token_weights)

            adv = g_lead
            weighted_pg = (-token_adv * pol_logprobs + args.kl_beta * token_kl) * token_weights
            rollout_loss = (weighted_pg.sum() / token_weights.sum().clamp(min=1.0)) / args.group_size
            rollout_loss.backward()

            total_pg_loss += rollout_loss.item()
            total_kl_val += token_kl.mean().item() / args.group_size

        grad_norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0).item()
        optimizer.step()

        avg_sub = {
            k: round(sum(m[k] for m in sub_metrics_list) / len(sub_metrics_list), 4)
            for k in [
                "r_no_stutter",
                "r_lead_craft",
                "r_hook_motif",
                "r_arp_fluidity",
                "r_chord_color",
                "r_counterpoint",
                "r_dynamic_arc",
                "num_bars",
            ]
        }

        vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3) if torch.cuda.is_available() else 0.0
        elapsed = time.time() - start_time

        if step % 5 == 0 or step == 1 or step == args.steps:
            print(
                f"   [Dr.GRPO] Step {step:3d}/{args.steps} | Bars: {avg_sub['num_bars']:.1f} | R_rl: {mean_r:.4f} (std {std_r:.3f}) | "
                f"Mel: {avg_sub['r_no_stutter']:.3f} | Craft: {avg_sub['r_lead_craft']:.3f} | "
                f"Hook: {avg_sub['r_hook_motif']:.3f} | Arp: {avg_sub['r_arp_fluidity']:.3f} | "
                f"KL: {total_kl_val:.4f} | VRAM: {vram_gb:.2f}GB | {elapsed:.0f}s",
                flush=True,
            )

        history.append(
            {
                "step": step,
                "style": spec["style"],
                "key": spec["key"],
                "r_mean": round(mean_r, 4),
                "r_max": round(max_r, 4),
                "r_min": round(min_r, 4),
                "r_std": round(std_r, 4),
                "kl": round(total_kl_val, 5),
                "grad_norm": round(grad_norm, 4),
                "vram_gb": round(vram_gb, 2),
                **avg_sub,
            }
        )

        cycle_len = len(CURATED_MANDOPOP_SPECS)
        if step >= cycle_len:
            rolling_r = sum(h["r_mean"] for h in history[-cycle_len:]) / cycle_len
            if rolling_r > best_mean_reward:
                best_mean_reward = rolling_r
                policy_model.save_pretrained(str(output_dir / "best_model"))
                tokenizer.save_vocab(str(output_dir / "best_model" / "v3_vocab_4096.json"))

    final_dir = output_dir / "final_model"
    policy_model.save_pretrained(str(final_dir))
    tokenizer.save_vocab(str(final_dir / "v3_vocab_4096.json"))

    with open(output_dir / "grpo_history.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "steps": args.steps,
                "group_size": args.group_size,
                "best_mean_reward": round(best_mean_reward, 4),
                "history": history,
            },
            f,
            indent=2,
        )

    torch.save(rollout_archive, output_dir / "grpo_preference_pairs.pt")
    print(
        f"✅ [Phase 3: Dr. GRPO] Complete! Saved {len(rollout_archive)} preference pairs and final model to {final_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
