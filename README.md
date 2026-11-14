# 🎼 Mandopop-Studio V3.0 (116.59M)

<p align="center">
  <b>English</b> | <a href="README_zh.md">简体中文</a>
</p>

<p align="center">
  <b>From-Scratch 116.59M 3D Musical Coordinate Transformer · Multi-Source Mandopop Intro & Instrumental Theme Corpus (POP909 + GTSinger 2024 + EMOPIA, 10.72M Tokens) · 4-Phase Post-Training Pipeline (Pretrain → Golden CoT SFT → Dr. GRPO RLVR → SimPO/DPO + OPD) · Anti-Crackle Damper-Pedal SoundFont + Pedalboard Mastering · Interactive Web DAW</b>
</p>

---

## ✨ Key Highlights

- **🧠 3D Musical Coordinate Transformer (`116,593,665` params)**: Builds on a 16-layer grouped-query attention backbone (`hidden_size=768`, `12Q / 4KV`, SwiGLU `2304`, `vocab_size=4096`) with residual axial embeddings for **Bar (`1..32`)**, **16th-Note Grid (`0..15`)**, **Track (`Lead_Hook`, `Piano_Arp`, `Strings_Pad`, `Bass_Perc`)**, and **Active Harmony (`145` Mandopop color chords)**.
- **📐 Unified Bar-Chord-Track-Grid Tokenization + Hard FSM Mask**: Fused atomic note tokens + `<think>` music-theory blueprint planning + deterministic `MandopopGrammarLogitsProcessor`, achieving **100.0% syntax legality (`r_syntax = 1.0000`)** with zero track dropouts and strictly monotonic bar clocks.
- **🎯 4-Phase Post-Training + 6D Aesthetic Verifier**: Progresses from Base Pretraining (`5x A100`) and Golden CoT SFT to local 8GB RTX 5070 **Dr. GRPO RLVR** and **Hybrid SimPO/DPO + OPD**, reaching `Best-of-N R_total = 1.8842` (Peak `1.9360`) under `2.28 GB` VRAM.
- **🎧 Anti-Crackle Damper-Pedal SoundFont + Pedalboard Studio Engine**: Implements exact tempo bar-boundary damper-pedal hold (`apply_pedal_hold_physics()`), anti-clash sustain pruning, `noteon`-only SoundFont caching with raised-cosine envelope shaping, zero-phase FFT lowpass de-gritting, float64 additive strings, and clean linear mastering.
- **🎛️ Interactive Web DAW Studio (`v3/web_studio.py`)**: Real-time composition across styles/moods/keys/chords/model stages, 4-track piano roll visualization, `<think>` blueprint inspection, 6D verifier radar breakdown, and searchable/filterable/pinnable persistent track library.

---

## 💎 Architecture, Training & Audio Engineering

### 1. Multi-Source De-Syllabified Mandopop Corpus (`v3/data/build_v3_dataset.py`)
Vocal melody MIDIs extracted directly from lyrics often repeat identical pitches (`E4-E4-E4-E4`) to fit Mandarin syllables, producing mechanical "karaoke guide" phrasing when played instrumentally. V3 builds a **10.72M-token (`16,071` sequences)** 4-track corpus curated specifically for instrumental lyricism:
- **`POP909` Authentic Intros & Themes (`9,947` sequences)**: Extracts 863 instrumental intros/interludes (`BRIDGE` + `PIANO`) and de-syllabified chorus themes aligned to human-annotated `beat_midi.txt` and `chord_midi.txt`.
- **`GTSinger` (NeurIPS 2024) Chinese Pop Hits (`1,876` sequences)**: Polyphonic 4-track arrangements from 118 iconic Chinese pop tracks paired with `1-5-8-10` piano arpeggios and strings.
- **`EMOPIA`, `MulTTiPop` & `PopSegments` (`4,248` sequences)**: Expressive Asian pop/OST piano and multi-track pop arrangements across emotional quadrants (`Q1–Q4`).

### 2. `MandopopV3ForCausalLM` & FSM Grammar Constraint (`v3/model/modeling_mandopop_v3.py`)
Every token $x_t$ across the 4 interleaved tracks receives explicit spatio-temporal and harmonic coordinates:

$$\mathbf{h}_t^{(0)} = \mathbf{E}_{\text{tok}}(x_t) + \alpha \cdot \text{LN}\Big(\mathbf{E}_{\text{bar}}(\text{bar}_t) + \mathbf{E}_{\text{grid}}(\text{pos}_t) + \mathbf{E}_{\text{track}}(\text{trk}_t) + \mathbf{E}_{\text{chord}}(\text{chord}_t)\Big)$$

Paired with `MandopopGrammarLogitsProcessor`, generation enforces `<think>...</think>` blueprint completion before `<score>`, preserves bar/track structural invariants, and blocks intra-bar timing regressions.

### 3. 4-Phase Post-Training Progression & 6D Aesthetic Verifier (`v3/verifiers/intro_aesthetic_verifier.py`)
`MandopopIntroVerifier` scores sequences along 6 music-theory dimensions: `r_no_stutter`, `r_arp_fluidity`, `r_chord_color`, `r_counterpoint`, `r_hook_motif`, and `r_dynamic_arc`.

| Training Phase | Checkpoint Directory | Platform | Training Summary | Peak VRAM | Syntax ($r_{\text{syn}}$) | Arp Fluidity ($r_{\text{arp}}$) | Chord Color ($r_{\text{col}}$) | Counterpoint ($r_{\text{cpt}}$) | Dynamic Arc ($r_{\text{arc}}$) | **Best-of-N $R_{\text{total}}$** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 1: Base Pretraining** | `checkpoints/v3_pretrain/final_model` | 5x A100 DDP | 4 Epochs (`1,044` steps, Loss `8.59 -> 1.13`) | 4.8 GB/GPU | **1.0000** | 0.9308 | 1.0000 | 0.9153 | 0.8956 | **1.8759** |
| **Phase 2: Golden CoT SFT** | `checkpoints/v3_sft/final_model` | 5x A100 DDP | 4 Epochs (`1,040` steps, Loss `1.09 -> 0.75`) | 4.8 GB/GPU | **1.0000** | 0.9021 | 1.0000 | 0.9174 | 0.8739 | **1.8578** (`Hook: 0.7631`) |
| **Phase 3: Dr. GRPO RLVR** | `checkpoints/v3_grpo/final_model` | Local RTX 5070 | 45 Steps ($G=8$, max reward `1.9420`) | **1.44 GB** | **1.0000** | 0.9681 | 1.0000 | **0.9379** | 0.9356 | **1.8784** |
| **Phase 4: SimPO/DPO + OPD** | `checkpoints/v3_dpo_final/final_model` | Local RTX 5070 | 80 Steps (`160` pairs, PrefAcc **100.0%**) | **2.28 GB** | **1.0000** | **0.9759** | **1.0000** | 0.9339 | **0.9494** | **1.8842** (Peak **1.9360**) |

### 4. Damper-Pedal Physics & Anti-Crackle Audio Renderer (`v3/audio/studio_renderer.py`)
- **True Damper-Pedal Physics + Anti-Clash Pruning**: Defers piano accompaniment `note_off` timestamps to exact tempo-derived chord/bar boundaries with a breathing gap, while releasing sustained notes early before incoming minor-2nd or tritone melody clashes.
- **Click-Free Synthesis & Mastering**: Synthesizes isolated SoundFont notes via `noteon`-only caching with raised-cosine onset/release shaping, applies linear-convolution zero-phase FFT lowpass filtering for loop-noise/grit removal, uses float64 oscillator phase + integrated vibrato on additive strings, mixes 4 stems through Spotify `pedalboard` FX + linear peak headroom scaling, and validates voice envelopes with `v3/audio/audit_onsets.py`.

---

## 🎵 Showcase Pieces (`outputs/v3_showcase/`)

| Track | Title & Style | Tempo / Key / Bars | Duration | $R_{\text{total}}$ | Arp Fluidity | Chord Color | Counterpoint | Hook Motif | Showcase MIDI (`.mid`) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **01** | **《晴空前的温柔》** (`[Style:Ballad_Slow]`) | 72 BPM · `C_maj` · 16 Bars | **56.1s** | **1.8795** | 0.9706 | **1.0000** | 0.9086 | **1.0000** | [01_ballad_intro_qingtian_warmth.mid](outputs/v3_showcase/01_ballad_intro_qingtian_warmth.mid) |
| **02** | **《烟雨青瓷》** (`[Style:Guofeng_Lyrical]`) | 76 BPM · `C_maj` · 16 Bars | **53.3s** | **1.8785** | 0.9725 | **1.0000** | 0.9193 | 0.5667 (`NoStutter: 0.808`) | [02_guofeng_intro_porcelain_mist.mid](outputs/v3_showcase/02_guofeng_intro_porcelain_mist.mid) |
| **03** | **《水星微光》** (`[Style:Urban_Romantic]`) | 84 BPM · `F_maj` · 16 Bars | **48.5s** | **1.9052** | 0.9769 | **1.0000** | 0.9400 | 0.7833 (`Arc: 1.000`) | [03_urban_romantic_mercury_glow.mid](outputs/v3_showcase/03_urban_romantic_mercury_glow.mid) |
| **04** | **《破晓疾风》** (`[Style:PopRock_Driving]`) | 96 BPM · `G_maj` · 16 Bars | **43.0s** | **1.8353** | 0.9715 | **1.0000** | 0.8184 | 0.8246 | [04_poprock_driving_wind_rises.mid](outputs/v3_showcase/04_poprock_driving_wind_rises.mid) |
| **05** | **《星河回响》** (`[Style:Ballad_Slow]` Cello-Piano) | 68 BPM · `A_min` · 16 Bars | **59.3s** | **1.9360** | **0.9802** | **1.0000** | **0.9719** | **0.8917** (`NoStutter: 0.859`) | [05_melancholic_cello_piano_dialogue.mid](outputs/v3_showcase/05_melancholic_cello_piano_dialogue.mid) |

*(Evaluation metadata is stored in [v3_evaluation_report.json](outputs/v3_showcase/v3_evaluation_report.json). Mastered `.wav` renders are omitted from Git to keep the repository lightweight and can be synthesized locally via the Web DAW or `v3/compose_v3_studio.py`.)*

---

## 🚀 Quick Start

### 1. Environment & SoundFont Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
- **SoundFont**: The lightweight [TimGM6mb.sf2](eval/soundfonts/TimGM6mb.sf2) fallback (`5.7 MB`) is included in the repository. To unlock the full `MuseScore_General.sf2` Bank `8/40/128` presets (`206 MB`, excluded via `.gitignore`), place it at `eval/soundfonts/MuseScore_General.sf2`.
- **Checkpoints & Datasets**: Model weights (`checkpoints/`) and training datasets (`data/`) exceed GitHub's 100 MB per-file limit and are excluded from Git.

### 2. Launch the Interactive Web DAW Studio
```bash
PYTHONPATH=. .venv/bin/python v3/web_studio.py --host 127.0.0.1 --port 7860
```
Open `http://127.0.0.1:7860` to compose custom Chinese/English prompts, compare checkpoints across training phases, customize renderer attack envelopes, inspect 4-track piano rolls, and search, filter, star, or rename tracks in the persistent track library.

### 3. Showcase Composer, Onset Audit & Unit Tests
```bash
# Evaluate Phase 1–4 checkpoints and synthesize the 5 showcase pieces
PYTHONPATH=. .venv/bin/python v3/compose_v3_studio.py --num_candidates 6

# Audit rendered onset envelopes for transients/clicks
PYTHONPATH=. .venv/bin/python v3/audio/audit_onsets.py

# Run the V3 audio & Web DAW regression test suite
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -q
```

### 4. Reproduce the 4-Phase Training Pipeline
```bash
# 1. Build multi-source 10.72M-token dataset from local raw corpora under data/raw/
PYTHONPATH=. .venv/bin/python v3/data/build_v3_dataset.py

# 2. Phase 1 Base Pretraining + Phase 2 Golden CoT SFT
PYTHONPATH=. .venv/bin/python v3/train/train_v3_pretrain_sft.py --mode both

# 3. Phase 3 Dr. GRPO RLVR (6D Aesthetic Verifier)
PYTHONPATH=. .venv/bin/python v3/train/train_v3_grpo.py --steps 120 --group_size 8

# 4. Phase 4 Hybrid SimPO/DPO + OPD
PYTHONPATH=. .venv/bin/python v3/train/train_v3_dpo_opd.py --epochs 3
```

---

## 🗂️ Repository Structure

```text
ABC-Composer/
├── README.md
├── README_zh.md
├── requirements.txt                           # Python environment dependencies
├── v3/
│   ├── web_studio.py                          # Interactive Web DAW server & track library API
│   ├── compose_v3_studio.py                   # 4-stage benchmark & 5 showcase composer
│   ├── ui/index.html                          # Web DAW frontend (search, filters, piano roll, player)
│   ├── model/modeling_mandopop_v3.py          # 116.59M 3D Coordinate Transformer + FSM LogitsProcessor
│   ├── tokenizer/
│   │   ├── v3_unified_tokenizer.py            # Bar-Chord-Track-Grid + atomic note tokenizer
│   │   └── v3_vocab_4096.json                 # 4096-token vocabulary
│   ├── verifiers/intro_aesthetic_verifier.py  # 6-dimensional Mandopop aesthetic verifier
│   ├── audio/
│   │   ├── studio_renderer.py                 # Damper-pedal SF2 renderer & Pedalboard mastering
│   │   └── audit_onsets.py                    # Per-voice onset transient & crackle auditor
│   ├── data/build_v3_dataset.py               # Multi-source corpus builder & CoT generator
│   └── train/
│       ├── train_v3_pretrain_sft.py           # Phase 1 Pretrain + Phase 2 CoT SFT trainer
│       ├── train_v3_grpo.py                   # Phase 3 Dr. GRPO RLVR trainer
│       └── train_v3_dpo_opd.py                # Phase 4 Hybrid SimPO/DPO + OPD trainer
├── tests/                                     # 25 unit tests covering audio rendering, onsets & Web DAW
├── eval/soundfonts/TimGM6mb.sf2               # Bundled fallback SoundFont (MuseScore_General.sf2 optional)
└── outputs/v3_showcase/                       # Showcase MIDI tracks, stage evolution MIDIs & eval report
```
