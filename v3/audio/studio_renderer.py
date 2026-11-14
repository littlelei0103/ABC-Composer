"""
Studio-Mandopop Renderer V3.2 (`v3/audio/studio_renderer.py`)
=============================================================
Shapes note envelopes, manages sustain, and mixes four instrument stems:

1. SoundFont notes are cached with `noteon`-only rendering, then shaped with
   raised-cosine onset and release envelopes before polyphonic mixing.

2. Additive strings use float64 oscillator phases and integrated vibrato so long
   notes do not accumulate float32 phase noise or increasing vibrato depth near
   release. The rhythm stem retains soft kick and closed-hat notes.

3. Exact Tempo Bar-Boundary Pedal Physics & Anti-Clash Sustain Pruning:
   Previously, when rendering standalone `.mid` files without `chord_change_times`,
   `score_sec.get_downbeats()` returned `[]` (when no TimeSignature event was present), falling
   back to `[0.0, 1.0, 2.0, ...]` (`1.0s` intervals instead of the true bar boundaries at
   `3.333s` for 72 BPM or `3.529s` for 68 BPM!). That caused the sustain pedal to hold the
   previous bar's chord right across the barline into the next bar's chord.
   Now we compute exact bar and half-bar boundaries directly from `score.tempos[0].qpm` AND
   automatically release any sustained accompaniment/string note 25ms early if an incoming
   Lead Hook or Piano note would form a harsh minor-2nd (`iv == 1`) or tritone (`iv == 6`) clash!

4. Mastering uses a single linear peak adjustment. Limiting and compression
   introduced waveform corners in the final mix; listening comparisons confirmed
   that bypassing the master compressor removed two remaining popping transients.
"""

from __future__ import annotations

import math
import threading
import wave
from bisect import bisect_right
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import symusic
import tinysoundfont

try:
    from pedalboard import (
        Chorus,
        HighpassFilter,
        HighShelfFilter,
        LowpassFilter,
        LowShelfFilter,
        PeakFilter,
        Pedalboard,
        Reverb,
    )
    HAS_PEDALBOARD = True
except ImportError:
    HAS_PEDALBOARD = False


DEFAULT_SF2 = "eval/soundfonts/MuseScore_General.sf2"
FALLBACK_SF2 = "eval/soundfonts/TimGM6mb.sf2"

# Global in-memory cache for zero-click single-note waveforms:
# key: (sf2_path, sample_rate, bank, prog, pitch, vel, max_sec_int) -> np.ndarray shape (2, N)
_ZERO_CLICK_NOTE_CACHE: Dict[Tuple[str, int, int, int, int, int, int], np.ndarray] = {}
_SYNTH_POOL: Dict[Tuple[str, int], Tuple[tinysoundfont.Synth, int]] = {}
_SYNTH_POOL_LOCK = threading.Lock()


def _get_pooled_synth(sf2_path: str, sample_rate: int) -> Tuple[tinysoundfont.Synth, int]:
    """Reuses a loaded SoundFont instance per (sf2_path, sample_rate) to eliminate per-note 0.6s disk loads."""
    pool_key = (sf2_path, int(sample_rate))
    entry = _SYNTH_POOL.get(pool_key)
    if entry is None:
        synth = tinysoundfont.Synth(samplerate=int(sample_rate), gain=0.90)
        sfid = synth.sfload(sf2_path)
        entry = (synth, sfid)
        _SYNTH_POOL[pool_key] = entry
    return entry


def compute_expressive_velocity_gain(vel: int, exponent: float = 1.25, floor_gain: float = 0.28) -> float:
    """Restores natural 12-15dB MIDI dynamic contrast after SoundFont dead-zone remapping."""
    norm = max(0.0, min(1.0, (float(vel) - 16.0) / 111.0))
    return float(floor_gain + (1.05 - floor_gain) * (norm ** exponent))


def humanize_arpeggio_chord_rolls(
    notes: List[Tuple[float, float, int, int]],
    roll_step_sec: float = 0.0045,
    max_roll_sec: float = 0.016,
) -> List[Tuple[float, float, int, int]]:
    """
    Applies a subtle bottom-to-top pianist chord roll across near-simultaneous accompaniment notes
    instead of mechanical modulo offsets.
    """
    if not notes:
        return []
    ordered = sorted(notes, key=lambda x: (round(x[0] / 0.012), x[2]))
    out: List[Tuple[float, float, int, int]] = []
    cluster_start = ordered[0][0]
    cluster_idx = 0
    for s, e, p, v in ordered:
        if abs(s - cluster_start) > 0.012:
            cluster_start = s
            cluster_idx = 0
        offset = min(max_roll_sec, cluster_idx * roll_step_sec)
        cluster_idx += 1
        out.append((s + offset, max(s + offset + 0.085, e), p, v))
    out.sort(key=lambda x: (x[0], x[2]))
    return out


def resolve_sf2_path(sf2_path: Optional[str] = None) -> str:
    if sf2_path and Path(sf2_path).exists():
        return str(sf2_path)
    if Path(DEFAULT_SF2).exists():
        return DEFAULT_SF2
    return FALLBACK_SF2


def remap_piano_velocity_safe(vel: int) -> int:
    """
    `MuseScore_General.sf2` Grand Piano (`0:0`) and Mellow Grand Piano (`8:0`) have a defective
    sample zone across velocities `75..107` where amplitude drops by 90% (`RMS 0.010 -> 0.0007`).
    Velocities `36..74` are warm, resonant, and strictly monotonic in loudness.
    We smoothly map input MIDI velocities `[20..127]` -> `[44..72]` (quantized by 2 for fast caching),
    completely eliminating dropped notes!
    """
    v = int(np.clip(vel, 20, 127))
    mapped = int(round(44.0 + (v - 20.0) * (28.0 / 107.0)))
    return int(np.clip((mapped // 2) * 2, 44, 72))


def deduplicate_same_pitch_overlaps(
    notes: List[Tuple[float, float, int, int]],
    min_gap_sec: float = 0.020,
    min_dur_sec: float = 0.085,
) -> List[Tuple[float, float, int, int]]:
    """
    Merges near-simultaneous duplicate onsets (< 25ms apart) and trims same-pitch overlaps.
    """
    if not notes:
        return []

    sorted_n = sorted(notes, key=lambda x: (x[0], x[2], -x[3]))
    by_pitch: Dict[int, List[List[float]]] = {}
    for s, e, p, v in sorted_n:
        p_list = by_pitch.setdefault(int(p), [])
        if p_list and abs(s - p_list[-1][0]) < 0.025:
            p_list[-1][1] = max(p_list[-1][1], e)
            p_list[-1][3] = max(p_list[-1][3], v)
            continue
        if p_list and p_list[-1][1] > s - min_gap_sec:
            p_list[-1][1] = max(p_list[-1][0] + min_dur_sec, s - min_gap_sec)
        p_list.append([float(s), max(float(s) + min_dur_sec, float(e)), int(p), int(v)])

    cleaned: List[Tuple[float, float, int, int]] = []
    for p_list in by_pitch.values():
        for s, e, p, v in p_list:
            cleaned.append((s, e, int(p), int(v)))

    cleaned.sort(key=lambda x: (x[0], x[2]))
    return cleaned


def enforce_monophonic_legato_lead(
    notes: List[Tuple[float, float, int, int]],
    release_overlap_sec: float = 0.010,
) -> List[Tuple[float, float, int, int]]:
    """
    Ensures the singing Lead Hook line flows without muddy semitone collisions
    while maintaining a natural piano legato touch.
    """
    if not notes:
        return []

    sorted_n = sorted(notes, key=lambda x: (x[0], -x[2]))
    dedup: List[List[float]] = []
    for s, e, p, v in sorted_n:
        if dedup and abs(s - dedup[-1][0]) < 0.022:
            continue
        dedup.append([float(s), float(e), int(p), int(v)])

    out: List[Tuple[float, float, int, int]] = []
    for i, (s, e, p, v) in enumerate(dedup):
        target_e = max(e, min(s + 0.72, e + 0.10))
        if i + 1 < len(dedup):
            next_s = dedup[i + 1][0]
            next_p = int(dedup[i + 1][2])
            if next_p == p:
                target_e = min(target_e, max(s + 0.08, next_s - 0.022))
            elif abs(next_p - p) <= 2 or (abs(next_p - p) % 12) in (1, 6, 11):
                # Stepwise / semitone / tritone motion: finish cleanly before next note to prevent clash!
                target_e = min(target_e, max(s + 0.09, next_s + release_overlap_sec))
            else:
                # Consonant interval leap (3rd/4th/5th/6th): warm half-pedal ring
                target_e = min(target_e, max(s + 0.12, next_s + 0.14))
        out.append((s, max(s + 0.085, target_e), int(p), int(v)))

    return deduplicate_same_pitch_overlaps(out)


def infer_exact_chord_boundaries(score: symusic.Score, max_note_end: float) -> List[float]:
    """
    Computes exact bar and half-bar chord change timestamps directly from `score.tempos`.
    Fixes the bug where `score_sec.get_downbeats()` returned `[]` on standalone `.mid` files
    without a TimeSignature event (which previously fell back to `1.0s` intervals and caused
    pedal bleed across barlines!).
    """
    bpm = 72.0
    if score.tempos:
        qpm = float(score.tempos[0].qpm)
        if 30.0 <= qpm <= 240.0:
            bpm = qpm

    beat_sec = 60.0 / bpm
    bar_sec = beat_sec * 4.0
    half_bar_sec = beat_sec * 2.0

    boundaries: List[float] = []
    t = 0.0
    while t <= max_note_end + bar_sec:
        boundaries.append(round(t, 4))
        t += half_bar_sec
    return boundaries


def apply_pedal_hold_physics(
    notes: List[Tuple[float, float, int, int]],
    chord_change_times: List[float],
    reference_melody_notes: Optional[List[Tuple[float, float, int, int]]] = None,
    pedal_gap_sec: float = 0.032,
    max_hold_sec: float = 2.4,
) -> List[Tuple[float, float, int, int]]:
    """
    Simulates real damper pedal (CC64) string resonance for Piano Accompaniment while
    preventing two sources of harmonic mud:
      1. Cross-barline / half-bar chord changes (`chord_change_times`).
      2. Harsh interval clashes (`minor 2nd / major 7th / tritone`) against incoming accompaniment
         or Lead Hook melody notes: if a newly struck note forms `iv in (1, 2, 6, 11)` with a
         currently sustained accompaniment note, that sustained note releases softly 20ms early!
    """
    if not notes:
        return []

    sorted_notes = sorted(notes, key=lambda x: (x[0], x[2]))
    boundaries = sorted(set(chord_change_times))

    # Combine accompaniment and melody onsets to check for incoming harmonic clashes
    all_incoming: List[Tuple[float, int]] = [(s_t, int(p)) for s_t, _, p, _ in sorted_notes]
    if reference_melody_notes:
        all_incoming.extend((s_t, int(p)) for s_t, _, p, _ in reference_melody_notes)
    all_incoming.sort(key=lambda x: x[0])

    pedaled: List[Tuple[float, float, int, int]] = []
    for s_t, e_t, p, vel in sorted_notes:
        target_end = e_t
        for b_t in boundaries:
            if b_t > s_t + 0.08:
                target_end = max(e_t, b_t - pedal_gap_sec)
                break
        else:
            target_end = max(e_t, s_t + 1.4)

        target_end = min(target_end, s_t + max_hold_sec)

        # Release early if the same pitch is re-struck OR if an incoming note forms a harsh clash
        for next_on, next_p in all_incoming:
            if next_on <= s_t + 0.045:
                continue
            if next_on >= target_end:
                break
            iv = abs(next_p - int(p)) % 12
            semitone_dist = abs(next_p - int(p))
            if next_p == int(p):
                # Same pitch re-strike
                target_end = min(target_end, next_on - 0.022)
                break
            elif iv in (1, 6) or semitone_dist == 1:
                # Minor 2nd (iv=1), tritone (iv=6), or tight semitone grind:
                # release the old sustained note cleanly while keeping consonant add9/sus2 shimmer!
                target_end = min(target_end, max(e_t, next_on - 0.015))
                break

        target_end = max(s_t + 0.085, target_end)
        pedaled.append((s_t, target_end, int(p), int(vel)))

    return deduplicate_same_pitch_overlaps(pedaled)


def prune_sustained_clashes_against_harmony(
    sustained_notes: List[Tuple[float, float, int, int]],
    chord_change_times: List[float],
    reference_notes: List[Tuple[float, float, int, int]],
) -> List[Tuple[float, float, int, int]]:
    """
    Trims sustained string/bass notes at chord boundaries or when an incoming piano/lead note
    forms a harsh minor-2nd (`iv == 1`) or tritone (`iv == 6`) against the sustained pitch.
    """
    if not sustained_notes:
        return []

    boundaries = sorted(set(chord_change_times))
    ref_sorted = sorted(reference_notes, key=lambda x: x[0])
    cleaned: List[Tuple[float, float, int, int]] = []

    for s_t, e_t, p, vel in sorted(sustained_notes, key=lambda x: (x[0], x[2])):
        target_end = e_t
        # Trim at bar boundaries (every 2nd half-bar boundary) if the note started in an earlier bar
        for b_t in boundaries[::2]:
            if b_t > s_t + 0.25 and b_t < target_end:
                target_end = min(target_end, b_t - 0.020)
                break

        for ref_s, ref_e, ref_p, _ in ref_sorted:
            if ref_e <= s_t + 0.04:
                continue
            if ref_s >= target_end:
                break
            iv = abs(int(ref_p) - int(p)) % 12
            if iv in (1, 6):
                if ref_s > s_t + 0.12:
                    target_end = min(target_end, ref_s - 0.015)
                    break

        cleaned.append((s_t, max(s_t + 0.12, target_end), int(p), int(vel)))

    return deduplicate_same_pitch_overlaps(cleaned)


def _zero_phase_lowpass_wave(wave_arr: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    """Applies linear-convolution zero-phase FFT lowpass to strip SF2 loop snaps and HF grit."""
    n = wave_arr.shape[1]
    if n < 64 or float(np.max(np.abs(wave_arr))) < 1e-7:
        return wave_arr.astype(np.float32, copy=False)
    pad = min(n, max(256, sample_rate // 4))
    padded = np.pad(wave_arr.astype(np.float64, copy=False), ((0, 0), (0, pad)), mode="constant")
    freqs = np.fft.rfftfreq(padded.shape[1], d=1.0 / float(sample_rate))
    hi = cutoff_hz * 1.55
    resp = np.ones_like(freqs, dtype=np.float64)
    trans = (freqs >= cutoff_hz) & (freqs <= hi)
    resp[trans] = 0.5 * (1.0 + np.cos(np.pi * (freqs[trans] - cutoff_hz) / (hi - cutoff_hz)))
    resp[freqs > hi] = 0.0
    filtered = np.fft.irfft(np.fft.rfft(padded, axis=1) * resp[None, :], n=padded.shape[1], axis=1)[:, :n]
    return filtered.astype(np.float32)


def _synthesize_clean_solo_wave(
    prog: int,
    pitch: int,
    vel: int,
    sample_rate: int,
    n_samples: int,
) -> np.ndarray:
    """
    Synthesizes loop-click-free float64 solo Cello (42) or Bamboo/Chamber Flute (73),
    replacing GM SoundFont loop discontinuities that produce cyclical buzzing clicks.
    """
    t = np.arange(n_samples, dtype=np.float64) / float(sample_rate)
    freq = 440.0 * (2.0 ** ((float(np.clip(pitch, 12, 108)) - 69.0) / 12.0))
    vel_scale = float(np.clip(vel, 20, 127)) / 127.0

    if prog == 42:  # Solo Tenor Cello
        vib_rate = 2.0 * np.pi * 4.9
        vib_int = (0.0020 / vib_rate) * (1.0 - np.cos(vib_rate * t))
        ph_l = 2.0 * np.pi * freq * (0.9993 * t + vib_int)
        ph_r = 2.0 * np.pi * freq * (1.0007 * t - vib_int)
        env = np.exp(-t * 0.22) * (1.0 - np.exp(-t / 0.042))
        w3 = 0.13 if freq * 3.0 < 2200.0 else 0.06
        w4 = 0.05 if freq * 4.0 < 2400.0 else 0.015
        sig_l = (0.56 * np.sin(ph_l) + 0.26 * np.sin(2.0 * ph_l + 0.25) + w3 * np.sin(3.0 * ph_l + 0.7) + w4 * np.sin(4.0 * ph_l + 1.1)) * env
        sig_r = (0.56 * np.sin(ph_r + 0.35) + 0.26 * np.sin(2.0 * ph_r + 0.65) + w3 * np.sin(3.0 * ph_r + 1.05) + w4 * np.sin(4.0 * ph_r + 1.45)) * env
        stereo = np.stack([sig_l, sig_r], axis=0) * (0.14 * vel_scale)
    else:  # Warm Chamber Flute (prog == 73)
        vib_rate = 2.0 * np.pi * 5.2
        vib_int = (0.0017 / vib_rate) * (1.0 - np.cos(vib_rate * t))
        ph_l = 2.0 * np.pi * freq * (0.9995 * t + vib_int)
        ph_r = 2.0 * np.pi * freq * (1.0005 * t - vib_int)
        env = np.exp(-t * 0.18) * (1.0 - np.exp(-t / 0.036))
        sig_l = (0.78 * np.sin(ph_l) + 0.17 * np.sin(2.0 * ph_l + 0.3) + 0.045 * np.sin(3.0 * ph_l + 0.8)) * env
        sig_r = (0.78 * np.sin(ph_r + 0.3) + 0.17 * np.sin(2.0 * ph_r + 0.7) + 0.045 * np.sin(3.0 * ph_r + 1.2)) * env
        stereo = np.stack([sig_l, sig_r], axis=0) * (0.11 * vel_scale)
    return stereo.astype(np.float32)


def _calibrate_sf2_piano_note(
    stereo: np.ndarray,
    pitch: int,
    vel: int,
    sample_rate: int,
) -> np.ndarray:
    """
    Smooths abrupt SoundFont keygroup loudness jumps and extreme per-note L/R panning
    skew in piano presets while keeping natural MIDI velocity dynamics and stereo image.
    """
    win = min(stereo.shape[1], int(0.60 * sample_rate))
    if win < 64:
        return stereo

    rms_l = float(np.sqrt(np.mean(stereo[0, :win] ** 2)))
    rms_r = float(np.sqrt(np.mean(stereo[1, :win] ** 2)))
    rms_total = float(np.sqrt(0.5 * (rms_l * rms_l + rms_r * rms_r)))
    if rms_total < 1e-6:
        return stereo

    out = stereo.copy()
    if rms_l > 1e-6 and rms_r > 1e-6:
        g_lr = math.sqrt(rms_l * rms_r)
        bal_l = float(np.clip((g_lr / rms_l) ** 0.80, 0.58, 1.72))
        bal_r = float(np.clip((g_lr / rms_r) ** 0.80, 0.58, 1.72))
        out[0] *= bal_l
        out[1] *= bal_r

    rms_post = float(np.sqrt(np.mean(out[:, :win] ** 2)))
    pitch_tilt = float(np.clip(1.0 - 0.008 * (float(pitch) - 60.0), 0.70, 1.18))
    vel_curve = (max(float(vel), 16.0) / 60.0) ** 0.78
    target_rms = 0.0034 * pitch_tilt * vel_curve
    rms_gain = float(np.clip((target_rms / max(rms_post, 1e-6)) ** 0.90, 0.36, 3.80))
    return (out * rms_gain).astype(np.float32)


def _get_zero_click_single_note_wave(
    sf2_path: str,
    sample_rate: int,
    bank: int,
    prog: int,
    pitch: int,
    vel: int,
    max_sec: float = 4.2,
) -> np.ndarray:
    """
    Renders and caches a held note waveform (`noteon`-only for SoundFont presets, or click-free
    float64 additive voices for bowed Cello / Flute), applying anti-crackle spectral cleanup.
    """
    max_sec_int = int(round(max_sec * 10))
    key = (sf2_path, sample_rate, int(bank), int(prog), int(pitch), int(vel), max_sec_int)
    cached = _ZERO_CLICK_NOTE_CACHE.get(key)
    if cached is not None:
        return cached

    n_samples = int(max_sec * sample_rate)
    if int(bank) == 0 and int(prog) in (42, 73):
        stereo = _synthesize_clean_solo_wave(int(prog), int(pitch), int(vel), int(sample_rate), n_samples)
    else:
        with _SYNTH_POOL_LOCK:
            cached = _ZERO_CLICK_NOTE_CACHE.get(key)
            if cached is not None:
                return cached
            synth, sfid = _get_pooled_synth(sf2_path, sample_rate)
            synth.sounds_off(0)
            _ = synth.generate(8192)
            preset_name = synth.sfpreset_name(sfid, int(bank), int(prog))
            use_bank = int(bank) if (preset_name or int(bank) == 128) else 0
            synth.program_select(0, sfid, use_bank, int(prog))
            synth.control_change(0, 10, 64)
            synth.control_change(0, 7, 118)
            synth.control_change(0, 11, 118)

            synth.noteon(0, int(np.clip(pitch, 12, 108)), int(np.clip(vel, 1, 127)))
            buf = synth.generate(n_samples)
            synth.sounds_off(0)
            _ = synth.generate(8192)

        interleaved = np.frombuffer(buf, dtype=np.float32)
        stereo = np.stack([interleaved[0::2].copy(), interleaved[1::2].copy()], axis=0)

        # Strip GM SoundFont pluck/aliasing/hat electrical spikes cleanly before envelope onset
        if int(bank) == 128:
            stereo = _zero_phase_lowpass_wave(stereo, sample_rate, cutoff_hz=3400.0)
        elif int(bank) == 0 and int(prog) == 24:
            stereo = _zero_phase_lowpass_wave(stereo, sample_rate, cutoff_hz=2300.0)
        elif int(bank) == 0 and int(prog) == 4:
            stereo = _zero_phase_lowpass_wave(stereo, sample_rate, cutoff_hz=3200.0)
        elif int(bank) in (0, 8) and int(prog) in (0, 1, 2):
            stereo = _calibrate_sf2_piano_note(stereo, int(pitch), int(vel), sample_rate)

    # Apply a 2.5ms ultra-smooth raised-cosine onset ramp to guarantee zero initial DC step
    fade_in = min(int(0.0025 * sample_rate), stereo.shape[1] // 8)
    if fade_in > 1:
        t_in = np.arange(fade_in, dtype=np.float32) / float(fade_in)
        in_env = 0.5 * (1.0 - np.cos(np.pi * t_in))
        stereo[:, :fade_in] *= in_env

    _ZERO_CLICK_NOTE_CACHE[key] = stereo
    return stereo


def _shape_note_segment(
    raw_wave: np.ndarray,
    seg_len: int,
    attack_samples: int,
    release_samples: int,
) -> np.ndarray:
    """Apply endpoint-inclusive fades to a copy of one isolated voice."""
    seg = raw_wave[:, :seg_len].copy()
    attack_len = min(max(0, attack_samples), seg_len // 2)
    release_len = min(max(0, release_samples), seg_len, max(32, seg_len // 2))
    if attack_len > 1:
        t = np.linspace(0.0, 1.0, attack_len, dtype=np.float64)
        seg[:, :attack_len] *= 0.5 * (1.0 - np.cos(np.pi * t))
    if release_len > 1:
        t = np.linspace(0.0, 1.0, release_len, dtype=np.float64)
        seg[:, -release_len:] *= 0.5 * (1.0 + np.cos(np.pi * t))
    return seg


def render_polyphonic_notes_zero_click(
    notes: List[Tuple[float, float, int, int]],
    bank: int,
    prog: int,
    total_sec: float,
    sf2_path: str,
    sample_rate: int = 44100,
    release_sec: float = 0.16,
    pan: float = 0.0,
    max_note_sec: float = 4.2,
    release_before_onsets: Optional[List[float]] = None,
    attack_sec: float = 0.0,
    attack_sec_by_pitch: Optional[Dict[int, float]] = None,
    release_before_same_pitch: bool = False,
    note_gains: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Synthesizes a polyphonic note list `(start_s, end_s, pitch, vel)` with ZERO `synth.noteoff`
    calls, shaping every note's release with a smooth raised-cosine envelope:
      `env(t) = 0.5 * (1 + cos(pi * t))`
    For monophonic parts, `release_before_onsets` makes the entire dry release
    finish before the next melodic attack, rather than only trimming note-off.
    `attack_sec` optionally softens the sample's attack without moving its onset.
    Percussion may override it per pitch. Same-pitch retrigger limits avoid
    piling up cached, independently phased copies while preserving other pitches.
    """
    num_samples = max(int(total_sec * sample_rate), sample_rate)
    out = np.zeros((2, num_samples), dtype=np.float32)
    if not notes:
        return out

    # Constant-power stereo pan multipliers (`pan` in [-1.0, +1.0])
    angle = (float(np.clip(pan, -1.0, 1.0)) + 1.0) * (math.pi / 4.0)
    pan_l = float(math.cos(angle))
    pan_r = float(math.sin(angle))

    rel_samples_default = max(64, int(release_sec * sample_rate))
    next_onsets = sorted(release_before_onsets or [])
    pitch_onsets: Dict[int, List[float]] = {}
    if release_before_same_pitch:
        for s, _, pitch, _ in notes:
            pitch_onsets.setdefault(int(pitch), []).append(float(s))
        for onsets in pitch_onsets.values():
            onsets.sort()

    for note_idx, (s_t, e_t, pitch, vel) in enumerate(notes):
        on_idx = max(0, int(round(s_t * sample_rate)))
        if on_idx >= num_samples:
            continue

        hold_dur = max(0.065, float(e_t - s_t))
        hold_samples = int(round(hold_dur * sample_rate))
        rel_samples = rel_samples_default
        total_note_samples = min(
            int(max_note_sec * sample_rate),
            hold_samples + rel_samples,
            num_samples - on_idx,
        )
        if next_onsets:
            # Ignore this note's own onset (including the 6 ms piano double).
            next_i = bisect_right(next_onsets, s_t + 0.010)
            if next_i < len(next_onsets):
                end_idx = int(round((next_onsets[next_i] - 0.005) * sample_rate))
                total_note_samples = min(total_note_samples, max(0, end_idx - on_idx))
        same_pitch_onsets = pitch_onsets.get(int(pitch), [])
        next_i = bisect_right(same_pitch_onsets, s_t + 0.010)
        if next_i < len(same_pitch_onsets):
            end_idx = int(round((same_pitch_onsets[next_i] - 0.005) * sample_rate))
            total_note_samples = min(total_note_samples, max(0, end_idx - on_idx))
        if total_note_samples <= 32:
            continue

        raw_wave = _get_zero_click_single_note_wave(
            sf2_path=sf2_path,
            sample_rate=sample_rate,
            bank=bank,
            prog=prog,
            pitch=int(pitch),
            vel=int(vel),
            max_sec=max_note_sec,
        )
        seg_len = min(total_note_samples, raw_wave.shape[1])
        note_attack_sec = (attack_sec_by_pitch or {}).get(int(pitch), attack_sec)
        seg = _shape_note_segment(
            raw_wave, seg_len, max(0, int(round(note_attack_sec * sample_rate))), rel_samples,
        )

        gain = float(note_gains[note_idx]) if (note_gains is not None and note_idx < len(note_gains)) else 1.0
        out[0, on_idx : on_idx + seg_len] += seg[0] * (gain * pan_l)
        out[1, on_idx : on_idx + seg_len] += seg[1] * (gain * pan_r)

    return out


def render_velvet_harmonic_strings(
    notes: List[Tuple[float, float, int, int]],
    total_sec: float,
    sample_rate: int = 44100,
) -> np.ndarray:
    """
    Renders additive strings with high-precision phases and bounded stereo vibrato.
    The slow raised-cosine bow envelope avoids abrupt note starts and ends.
    """
    num_samples = max(int(total_sec * sample_rate), sample_rate)
    out = np.zeros((2, num_samples), dtype=np.float32)
    if not notes:
        return out

    for idx_n, (s_t, e_t, pitch, vel) in enumerate(notes):
        on_idx = max(0, int(round(s_t * sample_rate)))
        if on_idx >= num_samples:
            continue

        dur_s = min(6.0, max(0.45, float(e_t - s_t) + 0.28))
        seg_len = min(int(round(dur_s * sample_rate)), num_samples - on_idx)
        if seg_len <= 128:
            continue

        # Float32 time/phase rounding adds broadband numerical error in long
        # notes. Keep oscillator arithmetic in float64; only the mix is float32.
        t = np.arange(seg_len, dtype=np.float64) / float(sample_rate)
        freq = 440.0 * (2.0 ** ((float(pitch) - 69.0) / 12.0))
        # Taper high-register string partials and gain so C6+ pads stay silky
        # without feeding harsh 2-3kHz screech/buzz into the stereo chorus.
        treble_att = float(np.clip((523.25 / max(freq, 523.25)) ** 0.60, 0.45, 1.0))
        amp = (float(np.clip(vel, 20, 127)) / 127.0) * 0.12 * treble_att

        # Warm ensemble detuning (tapered at high frequencies) + gentle 4.8Hz vibrato
        detune_ratio = 0.0018 * float(np.clip(550.0 / max(freq, 260.0), 0.35, 1.0))
        vibrato_depth = 0.0012 * float(np.clip(650.0 / max(freq, 260.0), 0.45, 1.0))
        vibrato_rate = 2.0 * np.pi * 4.8
        vibrato_phase = idx_n * 0.7
        # Phase is the integral of frequency. Using f(t) * t adds a t*f'(t)
        # term, making the pitch swing grow throughout the note and its tail.
        vibrato_integral = (vibrato_depth / vibrato_rate) * (
            np.cos(vibrato_phase) - np.cos(vibrato_rate * t + vibrato_phase)
        )
        phase_l = 2.0 * np.pi * freq * ((1.0 - detune_ratio) * t + vibrato_integral)
        phase_r = 2.0 * np.pi * freq * ((1.0 + detune_ratio) * t - vibrato_integral)

        h2_amp = 0.22 * float(np.clip((700.0 / max(2.0 * freq, 700.0)) ** 1.10, 0.12, 1.0))
        h3_amp = 0.08 * float(np.clip((900.0 / max(3.0 * freq, 900.0)) ** 1.40, 0.06, 1.0))

        # Focused fundamental + pitch-tapered 2nd/3rd partial + controlled sub-octave body
        wave_l = (
            0.74 * np.sin(phase_l)
            + h2_amp * np.sin(2.0 * phase_l + 0.3)
            + h3_amp * np.sin(3.0 * phase_l + 1.1)
            + 0.12 * np.sin(0.5 * phase_l + 0.5)
        )
        wave_r = (
            0.74 * np.sin(phase_r + 0.4)
            + h2_amp * np.sin(2.0 * phase_r + 0.9)
            + h3_amp * np.sin(3.0 * phase_r + 1.7)
            + 0.12 * np.sin(0.5 * phase_r + 1.2)
        )

        seg = _shape_note_segment(
            np.stack((wave_l, wave_r)) * amp,
            seg_len,
            min(int(0.16 * sample_rate), seg_len // 3),
            int(0.30 * sample_rate),
        )
        out[:, on_idx : on_idx + seg_len] += seg

    return out


def _normalize_stem_peak(stem: np.ndarray, target_peak: float) -> np.ndarray:
    """Normalizes a stereo stem to `target_peak` only if it contains active audio."""
    pk = float(np.max(np.abs(stem)))
    if pk < 1e-6:
        return stem
    return stem * (target_peak / pk)


def _normalize_stem_loudness(
    stem: np.ndarray,
    target_rms: float,
    soft_ceiling: float,
) -> np.ndarray:
    """
    Balances a stereo stem using strictly linear RMS gain capped by peak headroom (`soft_ceiling`).
    Avoids nonlinear waveshaping (`tanh`/limiting) that introduces waveform corners and electric fuzz.
    """
    pk = float(np.max(np.abs(stem)))
    if pk < 1e-6:
        return stem
    active = np.abs(stem) > (pk * 0.025)
    active_rms = float(np.sqrt(np.mean(stem[active] ** 2))) if np.any(active) else pk * 0.35
    gain = min(target_rms / max(active_rms, 1e-7), soft_ceiling / pk)
    return stem * gain


def _lowpass_bass_stem(stem: np.ndarray, sample_rate: int, cutoff_hz: float = 1550.0) -> np.ndarray:
    """Smooth zero-padded linear-phase frequency-domain lowpass for Bass without wraparound clicks."""
    if float(np.max(np.abs(stem))) < 1e-6:
        return stem
    return _zero_phase_lowpass_wave(stem, sample_rate, cutoff_hz=cutoff_hz)


def _master_stereo_mix(mix: np.ndarray, sample_rate: int) -> np.ndarray:
    """Set the offline mix peak with linear gain, preserving its waveform shape."""
    mix = np.ascontiguousarray(mix, dtype=np.float32)
    peak = float(np.max(np.abs(mix)))
    if peak <= 1e-5:
        return mix

    # The whole buffer is available offline: one gain keeps headroom without
    # introducing threshold-dependent corners or changing attack envelopes.
    return mix * (0.86 / peak)


PRESET_ORCHESTRATIONS: Dict[str, Dict] = {
    "intro_chamber": {
        "lead": (0, 0),
        "double": (8, 0),
        "double_min_dur": 0.30,
        "double_transpose": 0,
        "double_rms": 0.018,
        "double_attack": 0.080,
        "arp": (8, 0),
        "bass": (0, 32),
        "lead_room": (0.64, 0.64, 0.16, 0.80),
        "arp_room": (0.62, 0.65, 0.16, 0.82),
        "include_rhythm": True,
    },
    "pure_piano_strings": {
        "lead": (0, 0),
        "double": (8, 0),
        "double_min_dur": 0.30,
        "double_transpose": 0,
        "double_rms": 0.020,
        "double_attack": 0.080,
        "arp": (8, 0),
        "bass": (0, 32),
        "lead_room": (0.70, 0.62, 0.18, 0.79),
        "arp_room": (0.68, 0.63, 0.17, 0.82),
        "include_rhythm": False,
    },
    "guofeng_silk": {
        "lead": (0, 0),
        "double": (0, 73),
        "double_min_dur": 0.45,
        "double_transpose": 0,
        "double_rms": 0.026,
        "double_attack": 0.095,
        "arp": (0, 24),
        "bass": (0, 32),
        "lead_room": (0.72, 0.61, 0.19, 0.79),
        "arp_room": (0.68, 0.63, 0.17, 0.82),
        "include_rhythm": True,
    },
    "urban_rhodes": {
        "lead": (0, 0),
        "double": (0, 4),
        "double_min_dur": 0.20,
        "double_transpose": 0,
        "double_rms": 0.028,
        "double_attack": 0.048,
        "arp": (0, 4),
        "bass": (0, 33),
        "lead_room": (0.55, 0.65, 0.14, 0.82),
        "arp_room": (0.55, 0.66, 0.14, 0.82),
        "include_rhythm": True,
    },
    "poprock_band": {
        "lead": (0, 1),
        "double": (0, 0),
        "double_min_dur": 0.28,
        "double_transpose": 0,
        "double_rms": 0.018,
        "double_attack": 0.075,
        "arp": (0, 0),
        "bass": (0, 33),
        "lead_room": (0.50, 0.66, 0.13, 0.84),
        "arp_room": (0.50, 0.67, 0.13, 0.84),
        "include_rhythm": True,
    },
    "cello_piano": {
        "lead": (0, 0),
        "double": (0, 42),
        "double_min_dur": 0.40,
        "double_transpose": -12,
        "double_rms": 0.042,
        "double_attack": 0.085,
        "arp": (8, 0),
        "bass": (0, 32),
        "lead_room": (0.68, 0.62, 0.18, 0.80),
        "arp_room": (0.66, 0.64, 0.17, 0.82),
        "include_rhythm": True,
    },
}


def render_v3_score_to_stereo_wav(
    score: symusic.Score,
    wav_path: str | Path,
    chord_change_times: Optional[List[float]] = None,
    preset_style: str = "intro_chamber",
    sf2_path: Optional[str] = None,
    sample_rate: int = 44100,
    piano_attack_sec: float = 0.025,
    lead_piano_attack_sec: Optional[float] = 0.022,
    bass_attack_sec: float = 0.020,
    kick_attack_sec: float = 0.008,
    hihat_attack_sec: float = 0.005,
) -> Dict:
    """
    Renders a 4-track V3 symusic.Score (or POP909 multi-track score) into a mastered,
    studio stereo `.wav` using the V3.6 pooled `noteoff`-free synthesis engine:
      1. Pooled SoundFont Synth + Velocity Dead-Zone Bypass with Restored 14dB Dynamic Gain
      2. Exact Tempo Bar-Boundary Pedal Physics & add9-Preserving Anti-Clash Sustain Pruning
      3. `noteoff`-Free Zero-Click Polyphonic Engine + Phase-Continuous Velvet Strings + Style Orchestration
      4. Bass/Hi-Hat Frequency Separation, RMS Stem Leveling & Linear Peak Mastering
    `piano_attack_sec` controls accompaniment attacks. Lead piano and its double
    use `lead_piano_attack_sec`; pass None to share the accompaniment setting.
    Set both to zero to disable the additional attack shaping.
    """
    sf2_resolved = resolve_sf2_path(sf2_path)
    orch = PRESET_ORCHESTRATIONS.get(preset_style, PRESET_ORCHESTRATIONS["intro_chamber"])
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
            if "BRIDGE" in name_up and lead_notes:
                strings_notes.extend(extracted)
            else:
                lead_notes.extend(extracted)
        elif "PIANO" in name_up or "ACCOMP" in name_up or tr.program in (0, 1, 2, 4):
            if not lead_notes and len(score_sec.tracks) == 1:
                for n_tup in extracted:
                    if n_tup[2] >= 64:
                        lead_notes.append(n_tup)
                    else:
                        piano_notes.append(n_tup)
            else:
                piano_notes.extend(extracted)
        else:
            if idx == 0:
                lead_notes.extend(extracted)
            else:
                piano_notes.extend(extracted)

    all_ends = [e for grp in (lead_notes, piano_notes, strings_notes, bass_notes, drum_notes) for _, e, _, _ in grp]
    max_note_end = max(all_ends) if all_ends else 4.0
    total_sec = min(240.0, max_note_end + 2.8)

    if not chord_change_times:
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

    # 1. Clean Lead Hook first, then apply exact-bar-boundary + add9-preserving pedal physics
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

    # 2. Render Stem 1: Singing Lead Hook + Style-Aware Color Double with Restored Velocity Contour
    lead_onsets = [s for s, _, _, _ in clean_lead]
    lead_attack_sec = piano_attack_sec if lead_piano_attack_sec is None else lead_piano_attack_sec
    if preset_style == "poprock_band":
        lead_attack_sec = min(lead_attack_sec, 0.014)
    double_attack_sec = lead_attack_sec if orch["double_attack"] is None else max(lead_attack_sec, float(orch["double_attack"]))

    stem1_piano_notes: List[Tuple[float, float, int, int]] = []
    stem1_piano_gains: List[float] = []
    stem1_shimmer_notes: List[Tuple[float, float, int, int]] = []
    stem1_shimmer_gains: List[float] = []

    lead_pitches_arr = [p for _, _, p, _ in clean_lead]
    min_lp = min(lead_pitches_arr) if lead_pitches_arr else 60
    max_lp = max(lead_pitches_arr) if lead_pitches_arr else 76
    lp_span = max(6.0, float(max_lp - min_lp))
    lead_is_piano = orch["lead"][1] in (0, 1, 2)
    double_is_piano = orch["double"][1] in (0, 1, 2)

    for s, e, p, v in clean_lead:
        contour_boost = 0.92 + 0.18 * ((float(p - min_lp)) / lp_span)
        shaped_v = int(np.clip(round(v * contour_boost * 1.08), 32, 127))
        sf_v = remap_piano_velocity_safe(shaped_v) if lead_is_piano else int(np.clip(shaped_v, 48, 88))
        stem1_piano_notes.append((s, e, p, sf_v))
        stem1_piano_gains.append(compute_expressive_velocity_gain(shaped_v, exponent=1.05, floor_gain=0.42))

        if (e - s) >= orch["double_min_dur"]:
            d_pitch = int(p) + int(orch["double_transpose"])
            if orch["double"] == (0, 42):  # Tenor cello sweet register
                d_pitch = max(40, min(67, d_pitch))
            d_sf_v = remap_piano_velocity_safe(int(shaped_v * 0.72)) if double_is_piano else int(np.clip(round(shaped_v * 0.75), 48, 84))
            stem1_shimmer_notes.append((s + 0.008, e, d_pitch, d_sf_v))
            stem1_shimmer_gains.append(compute_expressive_velocity_gain(int(shaped_v * 0.80), exponent=1.05, floor_gain=0.42))

    lead_bank, lead_prog = orch["lead"]
    dbl_bank, dbl_prog = orch["double"]
    stem1_piano_audio = render_polyphonic_notes_zero_click(
        stem1_piano_notes, bank=lead_bank, prog=lead_prog, total_sec=total_sec, sf2_path=sf2_resolved,
        sample_rate=sample_rate, release_sec=0.18, pan=-0.08,
        release_before_onsets=lead_onsets,
        attack_sec=lead_attack_sec,
        note_gains=stem1_piano_gains,
    )
    stem1_shimmer_audio = render_polyphonic_notes_zero_click(
        stem1_shimmer_notes, bank=dbl_bank, prog=dbl_prog, total_sec=total_sec, sf2_path=sf2_resolved,
        sample_rate=sample_rate, release_sec=0.22, pan=0.14,
        release_before_onsets=lead_onsets,
        attack_sec=double_attack_sec,
        note_gains=stem1_shimmer_gains,
    )
    unison_piano_double = double_is_piano and int(orch["double_transpose"]) == 0
    if unison_piano_double:
        stem1_shimmer_audio = _zero_phase_lowpass_wave(
            stem1_shimmer_audio, sample_rate=sample_rate, cutoff_hz=1800.0
        )

    stem1_audio = (
        _normalize_stem_loudness(stem1_piano_audio, target_rms=0.105, soft_ceiling=0.45)
        + _normalize_stem_loudness(
            stem1_shimmer_audio,
            target_rms=orch["double_rms"],
            soft_ceiling=0.11 if unison_piano_double else 0.20,
        )
    )

    # 3. Render Stem 2: Damper-Pedaled Accompaniment with Pianist Bottom-to-Top Chord Roll & Dynamics
    rolled_piano = humanize_arpeggio_chord_rolls(pedaled_piano)
    arp_bank, arp_prog = orch["arp"]
    arp_is_piano = arp_prog in (0, 1, 2)
    stem2_notes: List[Tuple[float, float, int, int]] = []
    stem2_gains: List[float] = []
    for s, e, p, v in rolled_piano:
        acc_v = int(np.clip(round(v * 0.88), 24, 118))
        sf_acc_v = remap_piano_velocity_safe(acc_v) if arp_is_piano else int(np.clip(acc_v, 44, 80))
        stem2_notes.append((s, e, p, sf_acc_v))
        stem2_gains.append(compute_expressive_velocity_gain(acc_v, exponent=1.30, floor_gain=0.26))

    stem2_raw = render_polyphonic_notes_zero_click(
        stem2_notes, bank=arp_bank, prog=arp_prog, total_sec=total_sec, sf2_path=sf2_resolved,
        sample_rate=sample_rate, release_sec=0.22, pan=0.0,
        attack_sec=piano_attack_sec,
        release_before_same_pitch=True,
        note_gains=stem2_gains,
    )
    stem2_audio = _normalize_stem_loudness(stem2_raw, target_rms=0.070, soft_ceiling=0.35)

    # 4. Render Stem 3: Phase-Continuous Velvet Chamber Strings (`render_velvet_harmonic_strings`)
    stem3_raw = render_velvet_harmonic_strings(clean_strings, total_sec=total_sec, sample_rate=sample_rate)
    stem3_audio = _normalize_stem_loudness(stem3_raw, target_rms=0.036, soft_ceiling=0.18)

    # 5. Render Stem 4: Lowpass-Isolated Bass (`0:32`/`0:33`) + Crisp Airy Kick/HiHat Percussion
    stem4_bass_notes: List[Tuple[float, float, int, int]] = []
    stem4_bass_gains: List[float] = []
    stem4_drum_notes: List[Tuple[float, float, int, int]] = []
    stem4_drum_gains: List[float] = []
    if orch["include_rhythm"]:
        for s, e, p, v in clean_bass:
            bp = p if p < 52 else (p % 12) + 36
            bv = int(np.clip(((int(v * 0.78)) // 4) * 4, 44, 84))
            stem4_bass_notes.append((s, max(e, s + 0.42), bp, bv))
            stem4_bass_gains.append(compute_expressive_velocity_gain(v, exponent=1.10, floor_gain=0.40))
        for s, e, p, v in drum_notes:
            if p in (35, 36):
                stem4_drum_notes.append((s, s + 0.22, 36, 52))
                stem4_drum_gains.append(compute_expressive_velocity_gain(v, exponent=0.90, floor_gain=0.55))
            elif p == 42:
                stem4_drum_notes.append((s, s + 0.14, 42, 38))
                stem4_drum_gains.append(compute_expressive_velocity_gain(v, exponent=0.90, floor_gain=0.45))

    bass_bank, bass_prog = orch["bass"]
    stem4_bass_raw = render_polyphonic_notes_zero_click(
        stem4_bass_notes, bank=bass_bank, prog=bass_prog, total_sec=total_sec, sf2_path=sf2_resolved,
        sample_rate=sample_rate, release_sec=0.18, pan=0.0, max_note_sec=3.2,
        attack_sec=bass_attack_sec, release_before_same_pitch=True,
        note_gains=stem4_bass_gains,
    )
    stem4_drum_raw = render_polyphonic_notes_zero_click(
        stem4_drum_notes, bank=128, prog=0, total_sec=total_sec, sf2_path=sf2_resolved,
        sample_rate=sample_rate, release_sec=0.14, pan=0.06, max_note_sec=0.8,
        attack_sec=kick_attack_sec, attack_sec_by_pitch={42: hihat_attack_sec},
        release_before_same_pitch=True,
        note_gains=stem4_drum_gains,
    )
    stem4_bass_warm = _lowpass_bass_stem(
        _normalize_stem_loudness(stem4_bass_raw, target_rms=0.058, soft_ceiling=0.28),
        sample_rate=sample_rate,
        cutoff_hz=1550.0,
    )
    stem4_drum_air = _normalize_stem_loudness(stem4_drum_raw, target_rms=0.011, soft_ceiling=0.055)
    stem4_audio = stem4_bass_warm + stem4_drum_air

    # 6. Multi-Stem Studio DAW Mixing & Mastering via Spotify `pedalboard` (exact 4-bus topology)
    if HAS_PEDALBOARD:
        l_rm, l_dp, l_wet, l_dry = orch["lead_room"]
        a_rm, a_dp, a_wet, a_dry = orch["arp_room"]
        pb_lead = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=95.0),
            PeakFilter(cutoff_frequency_hz=2200.0, gain_db=0.6, q=0.9),
            HighShelfFilter(cutoff_frequency_hz=5500.0, gain_db=-1.6),
            Reverb(room_size=l_rm, damping=l_dp, wet_level=l_wet, dry_level=l_dry, width=0.95),
        ])
        pb_piano = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=60.0),
            LowShelfFilter(cutoff_frequency_hz=180.0, gain_db=1.0),
            HighShelfFilter(cutoff_frequency_hz=4500.0, gain_db=-2.2),
            Reverb(room_size=a_rm, damping=a_dp, wet_level=a_wet, dry_level=a_dry, width=1.0),
        ])
        pb_strings = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=135.0),
            LowpassFilter(cutoff_frequency_hz=1750.0),
            Chorus(rate_hz=0.35, depth=0.04, mix=0.10),
            Reverb(room_size=0.76, damping=0.68, wet_level=0.25, dry_level=0.72, width=0.90),
        ])
        pb_rhythm = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=36.0),
            PeakFilter(cutoff_frequency_hz=95.0, gain_db=1.0, q=0.9),
            LowpassFilter(cutoff_frequency_hz=4200.0),
            Reverb(room_size=0.40, damping=0.70, wet_level=0.08, dry_level=0.94, width=0.80),
        ])
        s1 = pb_lead(np.ascontiguousarray(stem1_audio, dtype=np.float32), sample_rate)
        s2 = pb_piano(np.ascontiguousarray(stem2_audio, dtype=np.float32), sample_rate)
        s3 = pb_strings(np.ascontiguousarray(stem3_audio, dtype=np.float32), sample_rate)
        s4 = pb_rhythm(np.ascontiguousarray(stem4_audio, dtype=np.float32), sample_rate)

        mix = s1 * 1.06 + s2 * 0.85 + s3 * 0.66 + s4 * 0.64
    else:
        mix = stem1_audio * 1.06 + stem2_audio * 0.85 + stem3_audio * 0.66 + stem4_audio * 0.64

    master = _master_stereo_mix(mix, sample_rate)

    # Smooth 60ms fade-out at end
    fade_len = min(int(0.06 * sample_rate), master.shape[1] // 4)
    if fade_len > 1:
        curve = 0.5 * (1.0 + np.cos(np.pi * np.arange(fade_len, dtype=np.float32) / fade_len))
        master[:, -fade_len:] *= curve

    stereo_interleaved = np.clip(master.T * 32767.0, -32768, 32767).astype(np.int16)
    wav_out = Path(wav_path)
    wav_out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(wav_out), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(stereo_interleaved.tobytes())

    return {
        "wav_path": str(wav_out),
        "duration_sec": round(float(master.shape[1]) / sample_rate, 2),
        "size_mb": round(wav_out.stat().st_size / (1024 * 1024), 2),
        "preset_style": preset_style,
        "pedalboard_mastered": HAS_PEDALBOARD,
        "mastering_mode": "linear_peak_normalization",
        "piano_attack_sec": piano_attack_sec,
        "lead_piano_attack_sec": lead_attack_sec,
        "attack_profile_ms": {
            "lead": lead_attack_sec * 1000.0,
            "piano_double": double_attack_sec * 1000.0,
            "accompaniment": piano_attack_sec * 1000.0,
            "strings": 160.0,
            "bass": bass_attack_sec * 1000.0,
            "kick": kick_attack_sec * 1000.0,
            "hihat": hihat_attack_sec * 1000.0,
        },
        "note_counts": {
            "lead": len(clean_lead),
            "piano_pedaled": len(pedaled_piano),
            "strings": len(clean_strings),
            "bass": len(clean_bass),
            "drums": len(stem4_drum_notes),
        },
    }


# Alias for backward compatibility
render_v3_score_to_wav = render_v3_score_to_stereo_wav
