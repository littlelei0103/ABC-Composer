#!/usr/bin/env python3
"""
Mandopop-Studio V3 — End-to-End Inference, Stage Progression Benchmark & Studio Audio Renderer
==============================================================================================
1. Evaluates all 4 V3 training stages (`Phase 1 Pretrain -> Phase 2 Golden CoT SFT -> Phase 3 Dr. GRPO -> Phase 4 SimPO/DPO+OPD`)
   on the 6-Dimensional Mandopop Intro & Theme Aesthetic Verifier (`MandopopIntroVerifier`).
2. Composes 5 flagship Mandopop Intro & Instrumental Theme pieces using the final V3 model
   (`checkpoints/v3_dpo_final/final_model`) with Best-of-N Verifier Reranking:
   - Track 01: 《晴空前的温柔》 (Slow Ballad Golden Intro & Theme — Jay Chou / JJ Lin style)
   - Track 02: 《烟雨青瓷》     (Guofeng Pentatonic Lyrical Intro & Theme — 《青花瓷》《大鱼》 style)
   - Track 03: 《水星微光》     (Urban Romantic / R&B Synth-Piano Intro & Theme — 《水星记》《交换余生》 style)
   - Track 04: 《破晓疾风》     (Pop-Rock Driving Band Intro & Theme — 《晴天》《起风了》 style)
   - Track 05: 《星河回响》     (Melancholic Tearful Cello-Piano Dialogue — 《像我这样的人》《演员》 style)
3. Renders every piece to 44.1kHz stereo `.wav` via `v3/audio/studio_renderer.py` (True Pedal-Hold Physics +
   MuseScore_General Bank 8 Mellow Grand Piano / Slow Violin / Orchestral Pad + Spotify Pedalboard Mastering).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
from transformers import LogitsProcessorList

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v3.audio.studio_renderer import render_v3_score_to_stereo_wav as render_v3_score_to_wav
from v3.model.modeling_mandopop_v3 import (
    MandopopGrammarLogitsProcessor,
    MandopopV3ForCausalLM,
)
from v3.tokenizer.v3_unified_tokenizer import MandopopV3Tokenizer
from v3.train.train_v3_grpo import build_prompt_ids
from v3.verifiers.intro_aesthetic_verifier import MandopopIntroVerifier


SHOWCASE_TRACKS: List[Dict] = [
    {
        "slug": "01_ballad_intro_qingtian_warmth",
        "title": "《晴空前的温柔》 (Slow Ballad Golden Intro & Theme)",
        "style": "[Style:Ballad_Slow]",
        "preset_style": "intro_chamber",
        "mood": "[Mood:Nostalgic_Warm]",
        "tempo": "[Tempo:72]",
        "key": "[Key:C_maj]",
        "reg": "[Reg_Mid:E4_C5]",
        "arp": "[Arp:1_5_8_10_Fluid]",
        "bars": 16,
        "chords": [
            "[Chord:F:maj7]", "[Chord:G:7]", "[Chord:E:min7]", "[Chord:A:min7]",
            "[Chord:D:min7]", "[Chord:G:sus4]", "[Chord:C:add9]", "[Chord:C:maj7]",
        ],
    },
    {
        "slug": "02_guofeng_intro_porcelain_mist",
        "title": "《烟雨青瓷》 (Guofeng Pentatonic Lyrical Intro & Theme)",
        "style": "[Style:Guofeng_Lyrical]",
        "preset_style": "guofeng_silk",
        "mood": "[Mood:Poetic_Breeze]",
        "tempo": "[Tempo:76]",
        "key": "[Key:C_maj]",
        "reg": "[Reg_High:G4_E5]",
        "arp": "[Arp:1_5_8_10_Fluid]",
        "bars": 16,
        "chords": [
            "[Chord:F:maj7]", "[Chord:G:sus4]", "[Chord:E:min7]", "[Chord:A:min7]",
            "[Chord:D:min7]", "[Chord:G:7]", "[Chord:A:min7]", "[Chord:A:min7]",
        ],
    },
    {
        "slug": "03_urban_romantic_mercury_glow",
        "title": "《水星微光》 (Urban Romantic R&B Intro & Theme)",
        "style": "[Style:Urban_Romantic]",
        "preset_style": "urban_rhodes",
        "mood": "[Mood:Nostalgic_Warm]",
        "tempo": "[Tempo:84]",
        "key": "[Key:F_maj]",
        "reg": "[Reg_Mid:E4_C5]",
        "arp": "[Arp:Broken_9th_Ballad]",
        "bars": 16,
        "chords": [
            "[Chord:Bb:maj7]", "[Chord:C:7]", "[Chord:A:min7]", "[Chord:D:min7]",
            "[Chord:G:min7]", "[Chord:C:sus4]", "[Chord:F:maj9]", "[Chord:D:min7]",
        ],
    },
    {
        "slug": "04_poprock_driving_wind_rises",
        "title": "《破晓疾风》 (Pop-Rock Driving Intro & Theme)",
        "style": "[Style:PopRock_Driving]",
        "preset_style": "poprock_band",
        "mood": "[Mood:Dramatic_Swell]",
        "tempo": "[Tempo:96]",
        "key": "[Key:G_maj]",
        "reg": "[Reg_High:G4_E5]",
        "arp": "[Arp:Broken_9th_Ballad]",
        "bars": 16,
        "chords": [
            "[Chord:E:min7]", "[Chord:C:add9]", "[Chord:G:maj]", "[Chord:D:sus4]",
            "[Chord:E:min7]", "[Chord:C:maj7]", "[Chord:D:7]", "[Chord:G:add9]",
        ],
    },
    {
        "slug": "05_melancholic_cello_piano_dialogue",
        "title": "《星河回响》 (Melancholic Cello-Piano Dialogue)",
        "style": "[Style:Ballad_Slow]",
        "preset_style": "cello_piano",
        "mood": "[Mood:Melancholic_Tearful]",
        "tempo": "[Tempo:68]",
        "key": "[Key:A_min]",
        "reg": "[Reg_High:G4_E5]",
        "arp": "[Arp:1_5_8_10_Fluid]",
        "bars": 16,
        "chords": [
            "[Chord:A:min7]", "[Chord:F:maj7]", "[Chord:C:add9]", "[Chord:G:sus4]",
            "[Chord:D:min7]", "[Chord:E:7]", "[Chord:A:min7]", "[Chord:A:min7]",
        ],
    },
]


def evaluate_stage_checkpoint(
    stage_name: str,
    ckpt_dir: Path,
    tokenizer: MandopopV3Tokenizer,
    verifier: MandopopIntroVerifier,
    device: torch.device,
    num_candidates: int = 4,
    seed_base: int = 2026,
) -> Dict:
    print(f"\n🔎 Evaluating Stage [{stage_name}] from {ckpt_dir} ...", flush=True)
    model = MandopopV3ForCausalLM.from_pretrained(str(ckpt_dir), torch_dtype=torch.bfloat16).to(device)
    model.eval()

    all_metrics: List[Dict[str, float]] = []
    best_per_spec: List[Dict] = []

    for spec_idx, spec in enumerate(SHOWCASE_TRACKS[:4]):
        torch.manual_seed(seed_base + spec_idx * 17)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_base + spec_idx * 17)
        target_bars = 12
        prompt_ids_list = build_prompt_ids(tokenizer, spec, target_bars=target_bars)
        prompt_len = len(prompt_ids_list)
        prompt_tensor = torch.tensor([prompt_ids_list] * num_candidates, dtype=torch.long, device=device)

        fsm = MandopopGrammarLogitsProcessor(tokenizer, target_bars=target_bars)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                gen_batch = model.generate(
                    input_ids=prompt_tensor,
                    max_new_tokens=620,
                    do_sample=True,
                    temperature=0.88,
                    top_p=0.92,
                    top_k=40,
                    pad_token_id=tokenizer.pad_id,
                    eos_token_id=tokenizer.eos_id,
                    logits_processor=LogitsProcessorList([fsm]),
                    use_cache=True,
                )

        spec_candidates = []
        for i in range(num_candidates):
            seq = gen_batch[i].tolist()
            if tokenizer.eos_id in seq[prompt_len:]:
                seq = seq[: seq.index(tokenizer.eos_id, prompt_len) + 1]
            m = verifier.evaluate_ids(seq, target_bars=target_bars)
            all_metrics.append(m)
            spec_candidates.append((m["r_total"], seq, m))

        spec_candidates.sort(key=lambda x: x[0], reverse=True)
        best_per_spec.append(
            {
                "slug": spec["slug"],
                "best_metrics": spec_candidates[0][2],
                "best_seq": spec_candidates[0][1],
            }
        )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    keys = ["r_total", "r_syntax", "r_no_stutter", "r_arp_fluidity", "r_chord_color", "r_counterpoint", "r_hook_motif", "r_dynamic_arc"]
    mean_metrics = {k: round(sum(m[k] for m in all_metrics) / len(all_metrics), 4) for k in keys}
    best_metrics = {k: round(sum(b["best_metrics"][k] for b in best_per_spec) / len(best_per_spec), 4) for k in keys}

    print(
        f"   [{stage_name}] Mean R_total: {mean_metrics['r_total']:.4f} | Best-of-{num_candidates} R_total: {best_metrics['r_total']:.4f} "
        f"(Arp: {best_metrics['r_arp_fluidity']:.3f}, Color: {best_metrics['r_chord_color']:.3f}, Hook: {best_metrics['r_hook_motif']:.3f})",
        flush=True,
    )
    return {
        "stage": stage_name,
        "mean_metrics": mean_metrics,
        "best_of_n_metrics": best_metrics,
        "best_per_spec": best_per_spec,
    }


def main():
    parser = argparse.ArgumentParser(description="Mandopop-Studio V3 Showcase Composer & Benchmark")
    parser.add_argument("--ckpt_root", type=str, default=str(PROJECT_ROOT / "checkpoints"))
    parser.add_argument("--out_dir", type=str, default=str(PROJECT_ROOT / "outputs/v3_showcase"))
    parser.add_argument("--num_candidates", type=int, default=6)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = MandopopV3Tokenizer()
    verifier = MandopopIntroVerifier(tokenizer)
    ckpt_root = Path(args.ckpt_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Evaluate Progression Across All 4 Post-Training Stages
    stages = [
        ("Phase1_Pretrain", ckpt_root / "v3_pretrain/final_model"),
        ("Phase2_Golden_CoT_SFT", ckpt_root / "v3_sft/final_model"),
        ("Phase3_Dr_GRPO_RLVR", ckpt_root / "v3_grpo/final_model"),
        ("Phase4_SimPO_DPO_OPD", ckpt_root / "v3_dpo_final/final_model"),
    ]
    stage_results = {}
    for s_name, s_path in stages:
        if s_path.exists():
            stage_results[s_name] = evaluate_stage_checkpoint(
                s_name, s_path, tokenizer, verifier, device, num_candidates=4
            )

    #Also render Stage-by-Stage comparison audio on Track 01 (`01_ballad_intro_qingtian_warmth`)
    stage_audio_dir = out_dir / "stage_evolution"
    stage_audio_dir.mkdir(parents=True, exist_ok=True)
    for s_name, s_data in stage_results.items():
        first_spec_best = s_data["best_per_spec"][0]
        seq_ids = first_spec_best["best_seq"]
        score, chord_times, meta = tokenizer.decode_to_symusic_score(seq_ids, default_bpm=72.0)
        mid_path = stage_audio_dir / f"{s_name}_ballad_intro.mid"
        wav_path = stage_audio_dir / f"{s_name}_ballad_intro.wav"
        score.dump_midi(str(mid_path))
        render_v3_score_to_wav(score, wav_path, chord_change_times=chord_times)

    # 2. Compose 5 Flagship 16-Bar Showcase Pieces with Final Phase 4 Model
    final_ckpt = ckpt_root / "v3_dpo_final/final_model"
    print(f"\n🎹 Composing 5 Flagship 16-Bar Mandopop Intros & Themes using {final_ckpt} ...", flush=True)
    model = MandopopV3ForCausalLM.from_pretrained(str(final_ckpt), torch_dtype=torch.bfloat16).to(device)
    model.eval()

    showcase_manifest = []
    for item in SHOWCASE_TRACKS:
        target_bars = item["bars"]
        prompt_ids_list = build_prompt_ids(tokenizer, item, target_bars=target_bars)
        prompt_len = len(prompt_ids_list)
        prompt_tensor = torch.tensor([prompt_ids_list] * args.num_candidates, dtype=torch.long, device=device)

        fsm = MandopopGrammarLogitsProcessor(tokenizer, target_bars=target_bars)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                gen_batch = model.generate(
                    input_ids=prompt_tensor,
                    max_new_tokens=820,
                    do_sample=True,
                    temperature=0.86,
                    top_p=0.92,
                    top_k=40,
                    pad_token_id=tokenizer.pad_id,
                    eos_token_id=tokenizer.eos_id,
                    logits_processor=LogitsProcessorList([fsm]),
                    use_cache=True,
                )

        candidates = []
        for i in range(args.num_candidates):
            seq = gen_batch[i].tolist()
            if tokenizer.eos_id in seq[prompt_len:]:
                seq = seq[: seq.index(tokenizer.eos_id, prompt_len) + 1]
            m = verifier.evaluate_ids(seq, target_bars=target_bars)
            candidates.append((m["r_total"], seq, m))

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_r, best_seq, best_metrics = candidates[0]

        bpm_val = float(item["tempo"][7:-1])
        score, chord_times, decode_meta = tokenizer.decode_to_symusic_score(best_seq, default_bpm=bpm_val)

        mid_path = out_dir / f"{item['slug']}.mid"
        wav_path = out_dir / f"{item['slug']}.wav"
        score.dump_midi(str(mid_path))
        audio_info = render_v3_score_to_wav(
            score,
            wav_path,
            chord_change_times=chord_times,
            preset_style=item.get("preset_style", "intro_chamber"),
        )

        print(
            f"   ✨ {item['title']} -> R_total: {best_r:.4f} | Duration: {audio_info['duration_sec']:.1f}s | "
            f"Notes: {decode_meta['track_note_counts']} | Saved to {wav_path.name}",
            flush=True,
        )
        showcase_manifest.append(
            {
                "slug": item["slug"],
                "title": item["title"],
                "style": item["style"],
                "mood": item["mood"],
                "tempo_bpm": bpm_val,
                "key": item["key"],
                "bars": decode_meta["num_bars"],
                "duration_sec": round(audio_info["duration_sec"], 2),
                "verifier_metrics": best_metrics,
                "track_note_counts": decode_meta["track_note_counts"],
                "midi_path": str(mid_path),
                "wav_path": str(wav_path),
            }
        )

    summary_report = {
        "stage_progression": {
            k: {
                "mean_metrics": v["mean_metrics"],
                "best_of_n_metrics": v["best_of_n_metrics"],
            }
            for k, v in stage_results.items()
        },
        "showcase_tracks": showcase_manifest,
    }
    report_path = out_dir / "v3_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)

    print(f"\n🎉 All V3 Showcase Tracks & Stage Evolution Benchmarks saved to {out_dir}!", flush=True)


if __name__ == "__main__":
    main()
