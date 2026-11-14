#!/usr/bin/env python3
"""
Mandopop-Studio V3 — Phase 4: Hybrid SimPO / DPO + On-Policy Self-Distillation (OPD)
====================================================================================
Sharpens the final aesthetic preference boundary of `Mandopop-V3-116M` from `checkpoints/v3_grpo/final_model`:
  1. Preference Pair Construction:
     - Combines on-policy preference pairs `(chosen_ids, rejected_ids)` mined during Phase 3 Dr. GRPO
     - Supplements with Golden Intro / GTSinger 2024 sequences (`r_total >= 1.85`) paired against
       synthetically corrupted "KTV syllable-stutter / flat-triad / arpeggio-broken" negative twins (`y_l`)
       so the model explicitly learns to reject repeated-note vocal habits and stiff block chords!
  2. Hybrid Loss Objective:
     - Length-Normalized SimPO / DPO Margin Loss (`L_SimPO`): pushes length-normalized log-probability
       of `y_w` above `y_l` by target margin `gamma`.
     - On-Policy Self-Distillation (`L_OPD`): anchors token-level distribution on `y_w` to the GRPO/SFT
       teacher distribution + NLL anchor so generation fluency and polyphonic lock remain rock-solid.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v3.model.modeling_mandopop_v3 import MandopopV3ForCausalLM
from v3.tokenizer.v3_unified_tokenizer import MandopopV3Tokenizer
from v3.verifiers.intro_aesthetic_verifier import MandopopIntroVerifier


def create_ktv_stutter_negative_twin(
    chosen_ids: List[int],
    tokenizer: MandopopV3Tokenizer,
) -> List[int]:
    """
    Takes a high-reward Golden Mandopop sequence (`chosen_ids`) and synthesizes a structurally valid
    but aesthetically degraded negative twin (`rejected_ids`) exhibiting the exact failure modes we
    want to eliminate:
      1. Out-of-Key Chromatic Slips & Minor-9th / Tritone Vertical Clashes on `<Trk:Lead_Hook>`
      2. Broken-chord arpeggio leaps & KTV Syllable Stutter on `<Trk:Lead_Hook>`
      3. Motif destruction & truncated staccato cadences on Bars >= 5 (`<Trk:Lead_Hook>`)
      4. Stiff / Clashing Piano accompaniment on `<Trk:Piano_Arp>`
    """
    neg = list(chosen_ids)
    cur_trk = 0
    cur_bar = 0
    last_lead_note_id = None
    first_piano_note_id = None
    chromatic_black_pcs = [1, 3, 6, 8, 10]

    for i, tid in enumerate(neg):
        if tid in tokenizer.bar_token_to_idx:
            cur_bar = tokenizer.bar_token_to_idx[tid]
        elif tid in tokenizer.trk_token_to_idx:
            cur_trk = tokenizer.trk_token_to_idx[tid]
            last_lead_note_id = None
            first_piano_note_id = None
        elif tid in tokenizer.note_id_to_tuple:
            pitch, dur, vel = tokenizer.note_id_to_tuple[tid]
            if cur_trk == 1:
                r_draw = random.random()
                if cur_bar in (4, 8, 12, 16) and dur >= 4 and r_draw < 0.70:
                    # Truncate sustained phrase-ending cadence holds into abrupt staccato jumps
                    bad_pitch = min(86, max(60, pitch + random.choice([5, -5, 7, -4])))
                    neg[i] = tokenizer.encode_note(bad_pitch, 1, 72)
                    last_lead_note_id = neg[i]
                elif cur_bar >= 5 and r_draw < 0.35:
                    # Break Bar 5/9/13 motif continuity with wide random registral jumps
                    jump_pitch = min(86, max(60, pitch + random.choice([-7, 7, -5, 6, -8])))
                    neg[i] = tokenizer.encode_note(jump_pitch, 2, 72)
                    last_lead_note_id = neg[i]
                elif r_draw < 0.30:
                    # Inject an out-of-key chromatic slip or semitone clash
                    clash_pitch = min(86, max(60, (pitch // 12) * 12 + random.choice(chromatic_black_pcs)))
                    neg[i] = tokenizer.encode_note(clash_pitch, dur, vel)
                    last_lead_note_id = neg[i]
                elif r_draw < 0.65:
                    # Force broken-chord arpeggio leaps instead of stepwise singing melody
                    arp_jump = random.choice([4, 7, -5, -4, 5])
                    arp_pitch = min(86, max(60, pitch + arp_jump))
                    neg[i] = tokenizer.encode_note(arp_pitch, 2, 72)
                    last_lead_note_id = neg[i]
                elif last_lead_note_id is not None and r_draw < 0.90:
                    # Force repeated-pitch syllable stutter
                    neg[i] = last_lead_note_id
                else:
                    short_id = tokenizer.encode_note(pitch, 2, 72)
                    neg[i] = short_id
                    last_lead_note_id = short_id
            elif cur_trk == 2:
                # Collapse wide flowing arpeggio into repetitive single-pitch notes
                if first_piano_note_id is None:
                    first_piano_note_id = tid
                elif random.random() < 0.60:
                    neg[i] = first_piano_note_id

    return neg


def build_preference_dataset(
    grpo_pairs_path: Path,
    sft_data_path: Path,
    tokenizer: MandopopV3Tokenizer,
    verifier: MandopopIntroVerifier,
    max_pairs: int = 240,
    max_seq_len: int = 960,
) -> List[Dict]:
    pairs: List[Dict] = []

    # 1. Load clean, non-truncated on-policy GRPO preference pairs where chosen genuinely beats rejected on Melody & Hook
    if grpo_pairs_path.exists():
        raw_grpo = torch.load(grpo_pairs_path, weights_only=False)
        for item in raw_grpo:
            if item["chosen_reward"] - item["rejected_reward"] < 0.05:
                continue
            cm = item.get("chosen_metrics", {})
            rm = item.get("rejected_metrics", {})
            if cm.get("r_syntax", 0.0) < 0.99:
                continue
            if cm.get("r_no_stutter", 0.0) < rm.get("r_no_stutter", 0.0):
                continue
            if cm.get("r_hook_motif", 0.0) < rm.get("r_hook_motif", 0.0) - 0.03:
                continue

            c_ids = item["chosen_ids"][:max_seq_len]
            r_ids = item["rejected_ids"][:max_seq_len]
            p_len = min(item.get("prompt_len", 30), len(c_ids) - 16, len(r_ids) - 16)
            if p_len > 0 and len(c_ids) > p_len + 16 and len(r_ids) > p_len + 16:
                pairs.append(
                    {
                        "chosen_ids": c_ids,
                        "rejected_ids": r_ids,
                        "prompt_len": p_len,
                        "chosen_reward": item["chosen_reward"],
                        "rejected_reward": item["rejected_reward"],
                        "source": "on_policy_grpo",
                    }
                )

    # 2. Supplement with Top-Tier singing Golden Intro sequences vs. stutter/arpeggio/motif-broken negative twins
    if sft_data_path.exists():
        sft_samples = torch.load(sft_data_path, weights_only=False)
        rng = random.Random(42)
        rng.shuffle(sft_samples)

        score_start_id = tokenizer.score_start_id
        for item in sft_samples:
            if len(pairs) >= max_pairs:
                break
            if item.get("r_total", 0.0) < 1.82:
                continue
            c_ids = item["input_ids"][:max_seq_len].tolist()
            if score_start_id not in c_ids:
                continue
            p_len = c_ids.index(score_start_id) + 1
            if len(c_ids) <= p_len + 32:
                continue

            c_metrics = verifier.evaluate_ids(c_ids, target_bars=16)
            if c_metrics["r_no_stutter"] < 0.72 or c_metrics["r_hook_motif"] < 0.65:
                continue

            r_ids = create_ktv_stutter_negative_twin(c_ids, tokenizer)
            r_metrics = verifier.evaluate_ids(r_ids, target_bars=16)
            c_reward = float(c_metrics["r_total"])
            r_reward = float(r_metrics["r_total"])

            if (c_reward - r_reward >= 0.18) and (c_metrics["r_no_stutter"] - r_metrics["r_no_stutter"] >= 0.14):
                pairs.append(
                    {
                        "chosen_ids": c_ids,
                        "rejected_ids": r_ids,
                        "prompt_len": p_len,
                        "chosen_reward": round(c_reward, 4),
                        "rejected_reward": round(r_reward, 4),
                        "source": f"golden_{item.get('source', 'intro')}_vs_bad_melody",
                    }
                )

    random.Random(108).shuffle(pairs)
    return pairs[:max_pairs]


_DPO_CONTENT_IDS_CACHE = None


def _get_dpo_content_id_set(tokenizer: MandopopV3Tokenizer) -> set:
    global _DPO_CONTENT_IDS_CACHE
    if _DPO_CONTENT_IDS_CACHE is None:
        _DPO_CONTENT_IDS_CACHE = (
            set(tokenizer.note_id_to_tuple.keys())
            | set(tokenizer.pos_token_to_idx.keys())
            | set(tokenizer.perc_id_to_tuple.keys())
        )
    return _DPO_CONTENT_IDS_CACHE


def compute_response_logprobs_and_opd(
    policy_model: MandopopV3ForCausalLM,
    ref_model: MandopopV3ForCausalLM,
    tokenizer: MandopopV3Tokenizer,
    seq_ids: torch.Tensor,
    prompt_len: int,
    compute_opd_kl: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns Melody-First track-weighted averages (2.5x on Lead_Hook musical tokens):
      - `pol_mean_logprob`: weighted policy logprob over response tokens
      - `ref_mean_logprob`: weighted reference logprob over response tokens
      - `opd_kl_loss`     : weighted KL distillation loss `KL(ref || policy)` on response tokens
    """
    bar_ids, grid_ids, track_ids, chord_ids = tokenizer.compute_3d_musical_coordinates(seq_ids)

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            ref_out = ref_model(
                input_ids=seq_ids[:, :-1],
                bar_ids=bar_ids[:, :-1],
                grid_ids=grid_ids[:, :-1],
                track_ids=track_ids[:, :-1],
                chord_ids=chord_ids[:, :-1],
                use_cache=False,
            )
            ref_logits = ref_out.logits[:, prompt_len - 1 :, :].float()
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pol_out = policy_model(
            input_ids=seq_ids[:, :-1],
            bar_ids=bar_ids[:, :-1],
            grid_ids=grid_ids[:, :-1],
            track_ids=track_ids[:, :-1],
            chord_ids=chord_ids[:, :-1],
            use_cache=False,
        )
        pol_logits = pol_out.logits[:, prompt_len - 1 :, :].float()
        pol_log_probs = F.log_softmax(pol_logits, dim=-1)

    target_ids = seq_ids[:, prompt_len:].unsqueeze(-1)
    pol_tok_lp = torch.gather(pol_log_probs, dim=-1, index=target_ids).squeeze(-1)
    ref_tok_lp = torch.gather(ref_log_probs, dim=-1, index=target_ids).squeeze(-1)

    content_ids = _get_dpo_content_id_set(tokenizer)
    resp_toks = seq_ids[0, prompt_len:].tolist()
    is_content = torch.tensor(
        [[tid in content_ids for tid in resp_toks]],
        dtype=torch.bool,
        device=seq_ids.device,
    )
    resp_trk = track_ids[:, prompt_len:]
    tok_weights = torch.full_like(pol_tok_lp, 0.10)
    tok_weights = torch.where((resp_trk == 1) & is_content, 2.50, tok_weights)
    tok_weights = torch.where((resp_trk == 2) & is_content, 0.80, tok_weights)
    tok_weights = torch.where(((resp_trk == 3) | (resp_trk == 4)) & is_content, 0.50, tok_weights)
    w_sum = tok_weights.sum().clamp(min=1.0)

    pol_mean_lp = (pol_tok_lp * tok_weights).sum() / w_sum
    ref_mean_lp = (ref_tok_lp * tok_weights).sum() / w_sum

    if compute_opd_kl:
        # Per-token weighted KL(p_ref || p_policy) over top-32 teacher tokens
        top_ref_lp, top_ref_idx = torch.topk(ref_log_probs, k=32, dim=-1)
        top_ref_probs = F.softmax(top_ref_lp, dim=-1)
        top_pol_lp = torch.gather(pol_log_probs, dim=-1, index=top_ref_idx)
        token_kl = (top_ref_probs * (top_ref_lp - top_pol_lp)).sum(dim=-1)
        opd_kl = (token_kl * tok_weights).sum() / w_sum
    else:
        opd_kl = torch.tensor(0.0, device=seq_ids.device)

    return pol_mean_lp, ref_mean_lp, opd_kl


def main():
    parser = argparse.ArgumentParser(description="Mandopop-Studio V3.5 Phase 4: Melody-Weighted SimPO/DPO + OPD Trainer")
    parser.add_argument("--grpo_ckpt", type=str, default=str(PROJECT_ROOT / "checkpoints/v3_grpo/final_model"))
    parser.add_argument("--grpo_pairs", type=str, default=str(PROJECT_ROOT / "checkpoints/v3_grpo/grpo_preference_pairs.pt"))
    parser.add_argument("--sft_data", type=str, default=str(PROJECT_ROOT / "data/processed/v3_dataset/sft_samples.pt"))
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "checkpoints/v3_dpo_final"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_pairs", type=int, default=240)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--gamma_margin", type=float, default=0.30)
    parser.add_argument("--opd_weight", type=float, default=0.15)
    parser.add_argument("--nll_weight", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=8e-6)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = MandopopV3Tokenizer()
    verifier = MandopopIntroVerifier(tokenizer)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"==================================================================")
    print(f"💎 Starting Mandopop-Studio V3.5 [Phase 4: Melody-Weighted SimPO/DPO + OPD]")
    print(f"   Init GRPO Checkpoint: {args.grpo_ckpt}")
    print(f"   Beta: {args.beta} | Margin Gamma: {args.gamma_margin} | OPD Weight: {args.opd_weight}")
    print(f"==================================================================", flush=True)

    pairs = build_preference_dataset(
        grpo_pairs_path=Path(args.grpo_pairs),
        sft_data_path=Path(args.sft_data),
        tokenizer=tokenizer,
        verifier=verifier,
        max_pairs=args.max_pairs,
    )
    grpo_cnt = sum(1 for p in pairs if p.get("source") == "on_policy_grpo")
    print(f"   Constructed {len(pairs)} high-contrast pairs ({grpo_cnt} on-policy GRPO, {len(pairs)-grpo_cnt} Golden contrasts) | Chosen Mean R: "
          f"{sum(p['chosen_reward'] for p in pairs)/max(1,len(pairs)):.4f} vs Rejected Mean R: "
          f"{sum(p['rejected_reward'] for p in pairs)/max(1,len(pairs)):.4f})", flush=True)

    policy_model = MandopopV3ForCausalLM.from_pretrained(args.grpo_ckpt, torch_dtype=torch.bfloat16).to(device)
    ref_model = MandopopV3ForCausalLM.from_pretrained(args.grpo_ckpt, torch_dtype=torch.bfloat16).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    grad_accum = 4
    total_steps = (len(pairs) * args.epochs) // grad_accum

    history = []
    global_step = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        random.Random(epoch * 100).shuffle(pairs)
        policy_model.train()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        running_margin = 0.0
        running_acc = 0.0
        running_opd = 0.0
        accum_cnt = 0

        for idx, pair in enumerate(pairs):
            c_tensor = torch.tensor([pair["chosen_ids"]], dtype=torch.long, device=device)
            r_tensor = torch.tensor([pair["rejected_ids"]], dtype=torch.long, device=device)
            p_len = pair["prompt_len"]

            pol_c_lp, ref_c_lp, opd_kl = compute_response_logprobs_and_opd(
                policy_model, ref_model, tokenizer, c_tensor, p_len, compute_opd_kl=True
            )
            pol_r_lp, ref_r_lp, _ = compute_response_logprobs_and_opd(
                policy_model, ref_model, tokenizer, r_tensor, p_len, compute_opd_kl=False
            )

            # Hybrid Reference-Anchored SimPO/DPO Log-Ratio Margin
            c_diff = pol_c_lp - ref_c_lp.detach()
            r_diff = pol_r_lp - ref_r_lp.detach()
            implicit_margin = args.beta * (c_diff - r_diff)

            dpo_loss = -F.logsigmoid(implicit_margin - args.gamma_margin)
            nll_anchor = -pol_c_lp
            total_loss = (dpo_loss + args.opd_weight * opd_kl + args.nll_weight * nll_anchor) / grad_accum
            total_loss.backward()

            running_loss += total_loss.item() * grad_accum
            running_margin += implicit_margin.detach().item()
            running_acc += 1.0 if implicit_margin.detach().item() > 0.0 else 0.0
            running_opd += opd_kl.detach().item()
            accum_cnt += 1

            if accum_cnt == grad_accum or (idx + 1) == len(pairs):
                grad_norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0).item()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                avg_loss = running_loss / accum_cnt
                avg_margin = running_margin / accum_cnt
                avg_acc = running_acc / accum_cnt
                avg_opd = running_opd / accum_cnt
                running_loss = 0.0
                running_margin = 0.0
                running_acc = 0.0
                running_opd = 0.0
                accum_cnt = 0

                if global_step % 10 == 0 or global_step == 1 or global_step == total_steps:
                    vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3) if torch.cuda.is_available() else 0.0
                    elapsed = time.time() - start_time
                    print(
                        f"   [SimPO+OPD] Ep {epoch}/{args.epochs} | Step {global_step:3d}/{total_steps} | "
                        f"Loss: {avg_loss:.4f} | Margin: {avg_margin:+.4f} | PrefAcc: {avg_acc*100:.1f}% | "
                        f"OPD_KL: {avg_opd:.4f} | VRAM: {vram_gb:.2f}GB | {elapsed:.0f}s",
                        flush=True,
                    )
                    history.append(
                        {
                            "step": global_step,
                            "epoch": epoch,
                            "loss": round(avg_loss, 4),
                            "margin": round(avg_margin, 4),
                            "pref_acc": round(avg_acc, 4),
                            "opd_kl": round(avg_opd, 4),
                            "grad_norm": round(grad_norm, 4),
                            "vram_gb": round(vram_gb, 2),
                        }
                    )

    final_dir = output_dir / "final_model"
    policy_model.save_pretrained(str(final_dir))
    tokenizer.save_vocab(str(final_dir / "v3_vocab_4096.json"))

    with open(output_dir / "dpo_opd_history.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_steps": global_step,
                "num_pairs": len(pairs),
                "beta": args.beta,
                "gamma_margin": args.gamma_margin,
                "opd_weight": args.opd_weight,
                "history": history,
            },
            f,
            indent=2,
        )
    print(f"✅ [Phase 4: SimPO/DPO + OPD] Finished! Saved final flagship model to {final_dir}", flush=True)


if __name__ == "__main__":
    main()
