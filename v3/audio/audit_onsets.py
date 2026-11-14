"""Render V3 MIDI files and inspect every isolated voice's actual envelopes.

Run from the project root:
    .venv/bin/python -m v3.audio.audit_onsets --source-dir outputs/v3_showcase
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import wave
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import symusic

from v3.audio import studio_renderer as renderer


def inspect_score(midi: Path, wav: Path, report_dir: Path, sample_rate: int = 44100) -> dict:
    """Observe the renderer's shaped voices, without substituting synthesized data."""
    rows = []
    stages = []
    context = None
    call_index = 0
    shape = renderer._shape_note_segment
    polyphonic = renderer.render_polyphonic_notes_zero_click
    strings = renderer.render_velvet_harmonic_strings
    master = renderer._master_stereo_mix
    polyphonic_roles = ("lead", "piano_double", "accompaniment", "bass", "percussion")

    def observed_shape(raw, length, attack_samples, release_samples):
        seg = shape(raw, length, attack_samples, release_samples)
        if context is None:
            raise RuntimeError("An isolated voice was rendered without an audit context")
        index = context["index"]
        note = context["notes"][index]
        context["index"] += 1
        start, end, pitch, velocity = note
        role = context["role"]
        if role == "percussion":
            role = "hihat" if pitch == 42 else "kick"
        pk = float(np.max(np.abs(seg)))
        scale = max(pk, 1e-20)
        prefix = seg[:, :min(seg.shape[1], round(0.020 * sample_rate))].astype(np.float64)
        onset_step = float(np.max(np.abs(seg[:, 0])))
        end_step = float(np.max(np.abs(seg[:, -1])))
        first_difference = float(np.max(np.abs(seg[:, 1] - seg[:, 0])))
        rows.append({
            "instrument": role, "start_sec": float(start), "note_off_sec": float(end),
            "pitch": int(pitch), "velocity": int(velocity),
            "voice_duration_sec": seg.shape[1] / sample_rate,
            "effective_attack_ms": min(max(0, attack_samples), length // 2) * 1000 / sample_rate,
            "peak": pk, "finite": bool(np.isfinite(seg).all()),
            "onset_step": onset_step, "end_step": end_step,
            "first_difference_relative": first_difference / scale,
            "first20ms_rms_relative": float(np.sqrt(np.mean(prefix * prefix))) / scale,
            "first20ms_max_difference_relative": float(np.max(np.abs(np.diff(prefix, axis=1)))) / scale,
            "boundary_flag": onset_step > 1e-7 or end_step > 1e-7 or first_difference / scale > 1e-3,
        })
        return seg

    def in_context(role, notes, function, *args, **kwargs):
        nonlocal context
        previous = context
        context = {"role": role, "notes": notes, "index": 0}
        try:
            audio = function(notes, *args, **kwargs)
            if context["index"] != len(notes):
                raise RuntimeError("Audit expects in-range V3 notes; a voice was skipped or truncated before rendering")
            return audio
        finally:
            context = previous

    def observed_polyphonic(notes, *args, **kwargs):
        nonlocal call_index
        role = polyphonic_roles[call_index]
        call_index += 1
        return in_context(role, notes, polyphonic, *args, **kwargs)

    def observed_strings(notes, *args, **kwargs):
        return in_context("strings", notes, strings, *args, **kwargs)

    def observed_master(mix, sr):
        result = master(mix, sr)
        mask = np.abs(mix) > max(float(np.max(np.abs(mix))) * 1e-3, np.finfo(np.float32).tiny)
        gains = result[mask].astype(np.float64) / mix[mask]
        gain_spread = float(np.ptp(gains)) if gains.size else 0.0
        mean_gain = float(np.mean(gains)) if gains.size else 1.0
        relative_gain_spread = gain_spread / max(abs(mean_gain), 1e-20)
        stages.append({"stage": "master", "finite": bool(np.isfinite(result).all()),
                       "peak": float(np.max(np.abs(result))), "linear_gain_spread": gain_spread,
                       "relative_gain_spread": relative_gain_spread})
        if relative_gain_spread > 1e-6:
            raise RuntimeError("Mastering changed waveform shape instead of applying one gain")
        return result

    with ExitStack() as stack:
        stack.enter_context(patch.object(renderer, "_shape_note_segment", observed_shape))
        stack.enter_context(patch.object(renderer, "render_polyphonic_notes_zero_click", observed_polyphonic))
        stack.enter_context(patch.object(renderer, "render_velvet_harmonic_strings", observed_strings))
        stack.enter_context(patch.object(renderer, "_master_stereo_mix", observed_master))
        if renderer.HAS_PEDALBOARD:
            board = renderer.Pedalboard
            board_roles = iter(("lead", "accompaniment", "strings", "rhythm"))

            def observed_board(plugins):
                effect = board(plugins)
                role = next(board_roles)

                def process(audio, sr):
                    output = effect(audio, sr)
                    stages.append({"stage": role + "_effects", "finite": bool(np.isfinite(output).all()),
                                   "peak": float(np.max(np.abs(output)))})
                    return output

                return process

            stack.enter_context(patch.object(renderer, "Pedalboard", observed_board))
        info = renderer.render_v3_score_to_stereo_wav(symusic.Score(str(midi)), wav, sample_rate=sample_rate)

    with wave.open(str(wav)) as handle:
        pcm = np.frombuffer(handle.readframes(handle.getnframes()), np.int16).astype(np.int32)
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / f"{midi.stem}_onsets.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["instrument", "start_sec", "pitch"])
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(row["instrument"] for row in rows)
    result = {
        "source_midi": str(midi), "source_sha256": hashlib.sha256(midi.read_bytes()).hexdigest(),
        "render_info": info, "voices_checked": len(rows), "by_instrument": dict(counts),
        "boundary_flags": sum(row["boundary_flag"] for row in rows),
        "nonfinite_voices": sum(not row["finite"] for row in rows),
        "clipped_pcm_samples": int(np.count_nonzero(np.abs(pcm) >= 32767)),
        "peak_pcm16": int(np.max(np.abs(pcm))), "stages": stages, "onsets_csv": str(csv_path),
        "strongest_openings": sorted(rows, key=lambda row: row["first20ms_rms_relative"], reverse=True)[:8],
    }
    (report_dir / f"{midi.stem}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("outputs/v3_showcase"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v3_all_onsets"))
    args = parser.parse_args()
    sources = sorted(args.source_dir.glob("*.mid"))
    if not sources:
        parser.error(f"No MIDI files in {args.source_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for source in sources:
        result = inspect_score(source, args.output_dir / source.with_suffix(".wav").name,
                               args.output_dir / "reports")
        results.append(result)
        print(f"{source.stem}: {result['voices_checked']} voices, "
              f"{result['boundary_flags']} boundary flags, "
              f"{result['clipped_pcm_samples']} clipped PCM samples", flush=True)
    summary = {
        "tracks": results, "voices_checked": sum(item["voices_checked"] for item in results),
        "boundary_flags": sum(item["boundary_flags"] for item in results),
        "nonfinite_voices": sum(item["nonfinite_voices"] for item in results),
        "clipped_pcm_samples": sum(item["clipped_pcm_samples"] for item in results),
        "interpretation": "Boundary continuity and clipping checks do not establish perceptual absence of pops. "
                          "Percussive instruments legitimately have strong transient energy; opening-energy rankings are not failures.",
    }
    (args.output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary["boundary_flags"] or summary["nonfinite_voices"] or summary["clipped_pcm_samples"]:
        raise SystemExit("Audit found an envelope or output issue; inspect reports before using these files")


if __name__ == "__main__":
    main()
