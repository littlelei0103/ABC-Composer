# 🎼 Mandopop-Studio V3.0 (116.59M)

<p align="center">
  <a href="README.md">English</a> | <b>简体中文</b>
</p>

<p align="center">
  <b>从零训练 116.59M 三维音乐坐标 Transformer · 多源华语流行前奏/器乐主题语料（POP909 + GTSinger 2024 + EMOPIA，10.72M Tokens）· 四阶段后训练链路（Pretrain → CoT SFT → Dr. GRPO RLVR → SimPO/DPO + OPD）· 真实延音踏板物理 + Pedalboard 4 轨母带渲染 · 交互式 Web DAW</b>
</p>

---

## ✨ 核心特性

- **🧠 116.59M 三维音乐坐标 Transformer（`116,593,665` 参数）**：16 层 GQA 主干网络（`hidden_size=768`、`12Q / 4KV`、SwiGLU `2304`、`vocab_size=4096`），为每个 Token 注入 **小节号（`1..32`）**、**十六分音符网格位置（`0..15`）**、**乐器声部（`Lead_Hook` / `Piano_Arp` / `Strings_Pad` / `Bass_Perc`）** 与 **当前激活和弦（`145` 种华语流行色彩和弦）** 的三维残差条件嵌入。
- **📐 统一多轨词表 + 硬状态机（FSM）文法约束**：采用融合原子音符与 `<think>` 乐理思维链蓝图规划，结合推理期 `MandopopGrammarLogitsProcessor`，保证 **100.0% 语法合法率（`r_syntax = 1.0000`）**、零掉轨与严格单调的小节内时钟推进。
- **🎯 四阶段后训练演化 + 六维美学 Verifier**：从多卡预训练与高质量 CoT SFT 出发，在单张 8GB RTX 5070 上依次完成 **Dr. GRPO RLVR**（`1.44 GB` 显存）与 **Hybrid SimPO/DPO + OPD** 对齐（`2.28 GB` 显存），`Best-of-N R_total` 提升至 **`1.8842`**（峰值 **`1.9360`**）。
- **🎧 零爆音延音踏板 SoundFont + Pedalboard 母带引擎**：实现了按真实 BPM 小节/和弦边界自动切换的钢琴延音踏板物理（`apply_pedal_hold_physics()`）与半音/三全音冲突提前收音机制，配合孤立音符升余弦包络整形、零相位 FFT 低通去噪、float64 精度加性弦乐与线性峰值母带控制，彻底消除采样换音爆破声。
- **🎛️ 交互式 Web DAW 作曲工作室（`v3/web_studio.py`）**：支持中英文提示词实时作曲、四阶段模型对比、包络攻击参数实时调音重渲染、4 轨钢琴卷帘窗、`<think>` 作曲蓝图查看、6 维美学雷达图，以及支持搜索、多维筛选、星标收藏、重命名和分页浏览的持久化作品库。

---

## 💎 技术架构与工程细节

### 1. 多源去唱词化华语流行器乐语料（`v3/data/build_v3_dataset.py`）
直接从流行歌曲人声导唱轨提取的 MIDI 常因中文一字一音而连续重复同一音高（如 `E4-E4-E4-E4`），改用器乐演奏会显得机械单调（“卡拉OK导唱陷阱”）。V3 专门构建了 **`16,071` 条四轨序列（`10.72M Tokens`）** 的器乐向高质量数据集：
- **`POP909` 原版前奏/间奏与主题（`9,947` 条）**：提取 `863` 首真实华语流行前奏/间奏（`BRIDGE` + `PIANO`）与去唱词化的副歌器乐主题，并对齐人工标注的 `beat_midi.txt` 与 `chord_midi.txt`。
- **`GTSinger` (NeurIPS 2024) 近年华语金曲（`1,876` 条）**：基于 118 首代表性中文流行曲目（如《大鱼》《水星记》《云烟成雨》《像我这样的人》《借》《不染》《交换余生》《演员》《告白气球》等）构建的多轨 `1-5-8-10` 流动琶音与弦乐织体改编。
- **`EMOPIA`、`MulTTiPop` 与 `PopSegments`（`4,248` 条）**：覆盖 `Q1–Q4` 情绪象限的流行钢琴及多轨伴奏片段。

### 2. 三维时空和声模型与 FSM 文法约束（`v3/model/modeling_mandopop_v3.py`）
所有 4 个交错声部中的每个音符 Token $x_t$ 均叠加精确的时空与和声位置表征：

$$\mathbf{h}_t^{(0)} = \mathbf{E}_{\text{tok}}(x_t) + \alpha \cdot \text{LN}\Big(\mathbf{E}_{\text{bar}}(\text{bar}_t) + \mathbf{E}_{\text{grid}}(\text{pos}_t) + \mathbf{E}_{\text{track}}(\text{trk}_t) + \mathbf{E}_{\text{chord}}(\text{chord}_t)\Big)$$

推理与 RL 采样阶段全程挂载 `MandopopGrammarLogitsProcessor`，强制在进入 `<score>` 前完成 `<think>...</think>` 结构化乐理规划，确保 4 轨齐全与小节拍号严格守恒。

### 3. 四阶段后训练演化与 6 维 Verifier 评测（`v3/verifiers/intro_aesthetic_verifier.py`）
`MandopopIntroVerifier` 从 6 个可验证乐理维度评价生成序列：`r_no_stutter`（去重复顿音）、`r_arp_fluidity`（左手流动琶音）、`r_chord_color`（色彩和弦贴合度）、`r_counterpoint`（声部呼吸与对位）、`r_hook_motif`（动机再现）与 `r_dynamic_arc`（四幕动态起伏）。

| 训练阶段 | 权重目录 | 训练硬件 | 训练概要 | 峰值显存 | Syntax ($r_{\text{syn}}$) | Arp 流畅度 ($r_{\text{arp}}$) | 和弦色彩 ($r_{\text{col}}$) | 声部对位 ($r_{\text{cpt}}$) | 动态起伏 ($r_{\text{arc}}$) | **Best-of-N $R_{\text{total}}$** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 1: Base Pretraining** | `checkpoints/v3_pretrain/final_model` | 5x A100 DDP | 4 Epochs（`1,044` steps, Loss `8.59 -> 1.13`） | 4.8 GB/GPU | **1.0000** | 0.9308 | 1.0000 | 0.9153 | 0.8956 | **1.8759** |
| **Phase 2: Golden CoT SFT** | `checkpoints/v3_sft/final_model` | 5x A100 DDP | 4 Epochs（`1,040` steps, Loss `1.09 -> 0.75`） | 4.8 GB/GPU | **1.0000** | 0.9021 | 1.0000 | 0.9174 | 0.8739 | **1.8578** (`Hook: 0.7631`) |
| **Phase 3: Dr. GRPO RLVR** | `checkpoints/v3_grpo/final_model` | 本地 RTX 5070 | 45 Steps（$G=8$, 最高奖励 `1.9420`） | **1.44 GB** | **1.0000** | 0.9681 | 1.0000 | **0.9379** | 0.9356 | **1.8784** |
| **Phase 4: SimPO/DPO + OPD** | `checkpoints/v3_dpo_final/final_model` | 本地 RTX 5070 | 80 Steps（`160` pairs, 偏好准确率 **100.0%**） | **2.28 GB** | **1.0000** | **0.9759** | **1.0000** | 0.9339 | **0.9494** | **1.8842**（峰值 **1.9360**） |

### 4. 延音踏板物理与抗爆音音频渲染（`v3/audio/studio_renderer.py`）
- **精确拍速踏板物理与冲突提前释音**：弥补 `tinysoundfont` 忽略 MIDI `CC 64` 踏板信号的缺陷，按真实 BPM 计算和弦/小节边界延音（保留自然换踏板呼吸间隙），并在主旋律即将撞上小二度或三全音时提前收束冲突延音。
- **全链路抗爆音母带**：单音离线仅触发 `noteon` 并缓存整形（升余弦淡入淡出 + 零相位线性卷积 FFT 低通消除循环点毛刺），加性合成弦乐全程使用 `float64` 相位累加与积分颤音防止长音尾部相位漂移噪声，4 轨经 Spotify `pedalboard` 链与纯线性峰值缩放混音，并配套 `v3/audio/audit_onsets.py` 做自动化起音爆破审计。

---

## 🎵 Showcase 示范曲目（`outputs/v3_showcase/`）

| 曲目 | 标题与风格 | 速度 / 调性 / 小节 | 时长 | $R_{\text{total}}$ | Arp 流畅度 | 和弦色彩 | 对位呼吸 | Hook 动机 | MIDI 作品（`.mid`） |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **01** | **《晴空前的温柔》** (`[Style:Ballad_Slow]`) | 72 BPM · `C_maj` · 16 Bars | **56.1s** | **1.8795** | 0.9706 | **1.0000** | 0.9086 | **1.0000** | [01_ballad_intro_qingtian_warmth.mid](outputs/v3_showcase/01_ballad_intro_qingtian_warmth.mid) |
| **02** | **《烟雨青瓷》** (`[Style:Guofeng_Lyrical]`) | 76 BPM · `C_maj` · 16 Bars | **53.3s** | **1.8785** | 0.9725 | **1.0000** | 0.9193 | 0.5667 (`NoStutter: 0.808`) | [02_guofeng_intro_porcelain_mist.mid](outputs/v3_showcase/02_guofeng_intro_porcelain_mist.mid) |
| **03** | **《水星微光》** (`[Style:Urban_Romantic]`) | 84 BPM · `F_maj` · 16 Bars | **48.5s** | **1.9052** | 0.9769 | **1.0000** | 0.9400 | 0.7833 (`Arc: 1.000`) | [03_urban_romantic_mercury_glow.mid](outputs/v3_showcase/03_urban_romantic_mercury_glow.mid) |
| **04** | **《破晓疾风》** (`[Style:PopRock_Driving]`) | 96 BPM · `G_maj` · 16 Bars | **43.0s** | **1.8353** | 0.9715 | **1.0000** | 0.8184 | 0.8246 | [04_poprock_driving_wind_rises.mid](outputs/v3_showcase/04_poprock_driving_wind_rises.mid) |
| **05** | **《星河回响》** (`[Style:Ballad_Slow]` 大提琴钢琴对话) | 68 BPM · `A_min` · 16 Bars | **59.3s** | **1.9360** | **0.9802** | **1.0000** | **0.9719** | **0.8917** (`NoStutter: 0.859`) | [05_melancholic_cello_piano_dialogue.mid](outputs/v3_showcase/05_melancholic_cello_piano_dialogue.mid) |

*(完整评测指标记录于 [v3_evaluation_report.json](outputs/v3_showcase/v3_evaluation_report.json)。由于音频体积较大，`.wav` 文件未直接纳入 Git 跟踪，可通过下方 Web DAW 或脚本在本地一键合成。)*

---

## 🚀 快速上手

### 1. 环境与音色库配置
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
- **SoundFont 音色库**：仓库内置轻量级兜底音色 [TimGM6mb.sf2](eval/soundfonts/TimGM6mb.sf2)（`5.7 MB`）。如需体验完整的 `MuseScore_General.sf2` Bank `8/40/128` 抒情音色预设（`206 MB`，因超出 GitHub 单文件上限已忽略），请将其放在 `eval/soundfonts/MuseScore_General.sf2`。
- **模型权重与数据集**：`checkpoints/` 与 `data/` 因体量较大未提交至 Git，可运行下方复现流程从源码重新构建并训练，或将本地训练权重放到 `checkpoints/` 目录。

### 2. 启动交互式 Web DAW 工作室
```bash
PYTHONPATH=. .venv/bin/python v3/web_studio.py --host 127.0.0.1 --port 7860
```
浏览器打开 `http://127.0.0.1:7860`，即可用自然语言/风格标签即时作曲、切换不同训练阶段模型、自定义重渲染包络参数、查看四轨钢琴卷帘窗，并在持久化作品库中进行搜索、筛选、星标收藏与重命名。

### 3. 批量展示合成、起音审计与单元测试
```bash
# 运行 Phase 1–4 对比评测并重新生成 5 首 Showcase 音频
PYTHONPATH=. .venv/bin/python v3/compose_v3_studio.py --num_candidates 6

# 对渲染波形执行逐音符起音爆破与异常峰值审计
PYTHONPATH=. .venv/bin/python v3/audio/audit_onsets.py

# 运行 V3 音频渲染与 Web DAW 作品库回归测试套件
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -q
```

### 4. 从零复现 V3 四阶段训练流程
```bash
# 1. 基于 data/raw/ 下的原始语料构建 10.72M-token 数据集
PYTHONPATH=. .venv/bin/python v3/data/build_v3_dataset.py

# 2. Phase 1 基础预训练 + Phase 2 Golden CoT SFT
PYTHONPATH=. .venv/bin/python v3/train/train_v3_pretrain_sft.py --mode both

# 3. Phase 3 Dr. GRPO RLVR（6 维美学 Verifier）
PYTHONPATH=. .venv/bin/python v3/train/train_v3_grpo.py --steps 120 --group_size 8

# 4. Phase 4 Hybrid SimPO/DPO + OPD 偏好对齐
PYTHONPATH=. .venv/bin/python v3/train/train_v3_dpo_opd.py --epochs 3
```

---

## 🗂️ 仓库目录结构

```text
ABC-Composer/
├── README.md                                  # 英文说明文档
├── README_zh.md                               # 中文说明文档
├── requirements.txt                           # Python 运行环境依赖
├── v3/
│   ├── web_studio.py                          # 交互式 Web DAW 服务端与作品库 API
│   ├── compose_v3_studio.py                   # 4 阶段对比评测与 Showcase 合成脚本
│   ├── ui/index.html                          # Web DAW 前端界面（检索/筛选/卷帘窗/播放器）
│   ├── model/modeling_mandopop_v3.py          # 116.59M 三维坐标 Transformer 与 FSM 约束器
│   ├── tokenizer/
│   │   ├── v3_unified_tokenizer.py            # Bar-Chord-Track-Grid + 原子音符分词器
│   │   └── v3_vocab_4096.json                 # 4096 词表定义
│   ├── verifiers/intro_aesthetic_verifier.py  # 6 维华语流行乐理美学 Verifier
│   ├── audio/
│   │   ├── studio_renderer.py                 # 延音踏板 SF2 渲染器与 Pedalboard 母带链
│   │   └── audit_onsets.py                    # 逐音符起音包络与爆破声检测脚本
│   ├── data/build_v3_dataset.py               # 多源去唱词化语料与 CoT 构造器
│   └── train/
│       ├── train_v3_pretrain_sft.py           # Phase 1 预训练与 Phase 2 CoT SFT 训练器
│       ├── train_v3_grpo.py                   # Phase 3 Dr. GRPO RLVR 训练器
│       └── train_v3_dpo_opd.py                # Phase 4 Hybrid SimPO/DPO + OPD 训练器
├── tests/                                     # 25 项音频渲染、起音审计与 Web 作品库单元测试
├── eval/soundfonts/TimGM6mb.sf2               # 内置轻量兜底 SoundFont（MuseScore_General.sf2 可选）
└── outputs/v3_showcase/                       # Showcase MIDI、阶段演化 MIDI 与评测报告 JSON
```
