# LoRA for Musical Mode Control in ACE-Step 1.5

> **Bachelor's Thesis (TFG)** — Ignacio Franco Almendárez  
> Grado en Ciencia de Datos e Inteligencia Artificial, UPM ETSISI, 2026  
> Supervisor: Francisco Serradilla

This repository is a fork of [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) that adds explicit **modal control** via LoRA fine-tuning. The goal is to make ACE-Step generate music consistently in a specific diatonic mode (Ionian, Dorian, Phrygian, Lydian, Mixolydian, Aeolian, Locrian) without degrading the base model's musical quality.

---

## Key Results

- **Robust control on major modes** (Ionian, Lydian, Mixolydian): the adapted model reliably generates the target modal color.
- **Partial control on minor modes** (Dorian, Phrygian, Aeolian): functional but the base model shows a systematic bias toward major tonalities that LoRA only partially corrects.
- **Locrian**: remains the hardest mode to control due to its diminished tonic and rarity in training data.
- **Optimal configuration found**: rank=64, alpha=128, lr=1e-4, dropout=0.05 (Sweep over 24 configs).
- **Confirmation experiment** with optimal config specifically reduced Dorian bias, achieving >50% success rate on strict evaluation.

Full results, figures and analysis in `docs/manual_eval_success/` and `memoria/`.

---

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Clone the repo
git clone <repo-url>
cd ACE-Step-1.5

# Install all dependencies (including TFG extras: pandas, matplotlib, seaborn, groq)
uv sync
```

The base ACE-Step model weights are downloaded automatically on first use. See [docs/en/INSTALL.md](docs/en/INSTALL.md) for GPU requirements and setup details.

---

## Quick Start — Using the Trained LoRA

Launch the Gradio UI (same as upstream ACE-Step):

```bash
uv run python -m acestep.webui
```

In the **LoRA** tab, load the adapter weights from `lora_output/` and use one of the evaluation prompts from `prompts_modal_sound.txt`. Example for D Dorian:

```
dorian mode, jazz-fusion instrumental, D dorian, electric piano lead,
fretless bass, brushed drums, 96 BPM, minor with hopeful color,
natural 6th flavor, stable tonic on D, no modulation, no vocals
```

---

## Reproducing the Experiments

### 1. Prepare the modal dataset

> **Note:** The dataset (105 audio clips sourced from existing recordings) is not included in this repository due to copyright restrictions. The script below documents the preparation pipeline and can be adapted to your own audio sources with mode annotations.

```bash
uv run python scripts/prepare_modes_dataset.py --help
```

### 2. Train a LoRA adapter

Use the Gradio UI (Training tab) or the Stage 2 training script:

```bash
bash run_modal_lora_training_round2.sh
```

### 3. Run the hyperparameter sweep

```bash
uv run python scripts/modal_hparam_sweep.py --help
```

Evaluates 24 configurations (rank, alpha, lr, dropout) systematically.

### 4. Evaluate checkpoints

```bash
# Batch inference for all checkpoints
uv run python scripts/modal_batch_infer.py --help

# Compute automatic metrics (auto_score_0_1)
uv run python scripts/modal_eval_pipeline.py --help

# Rank checkpoints per experiment
uv run python scripts/rank_checkpoints_per_experiment.py --help
```

### 5. Select candidates for manual evaluation

```bash
uv run python scripts/prepare_manual_eval_shortlist.py --help
```

### 6. Analyse results

```bash
uv run python scripts/modal_hparam_analysis.py \
  --results-csv hparam_sweep/stage1_results.csv \
  --output-dir hparam_sweep/analysis_output
```

Manual evaluation scores are in `docs/manual_eval_Scoring.csv`. The rubric (tonal center, modal color, non-modulation, musical coherence — each 1–5) is described in `docs/manual_eval_Leyenda.csv` and in the thesis (`memoria/`).

### 7. Interactive demo (with LLM classifier)

```bash
# Requires ACE-Step server running on :7860 and a GROQ_API_KEY env var
uv run python scripts/demo_modal_lora.py --api-url http://127.0.0.1:7860
```

---

## Repository Structure

```
.
├── acestep/                    # Core ACE-Step framework (upstream + TFG patches)
│   ├── core/generation/handler/lora/   # LoRA lifecycle — CUDA sanitisation fix
│   ├── training/               # LoRA injection, LoKr utilities, VAE preprocessing
│   └── ui/gradio/events/training/      # Gradio training events (LoRA + LoKr)
│
├── scripts/                    # TFG pipeline scripts
│   ├── prepare_modes_dataset.py
│   ├── modal_hparam_sweep.py
│   ├── modal_batch_infer.py
│   ├── modal_eval_pipeline.py
│   ├── rank_checkpoints_per_experiment.py
│   ├── prepare_manual_eval_shortlist.py
│   ├── modal_hparam_analysis.py
│   ├── regenerate_top_audios.py
│   ├── demo_modal_lora.py
│   └── _one_shot/              # Single-use auxiliary scripts (kept for traceability)
│
├── docs/
│   ├── manual_eval_Scoring.csv         # Manual evaluation scores
│   ├── manual_eval_Leyenda.csv         # Rubric definition
│   ├── manual_eval_Resumen_ckpt.csv    # Per-checkpoint summary
│   ├── manual_eval_success/            # Success rate analysis + figures
│   └── _process/               # Internal research notes (Spanish)
│
├── hparam_sweep/               # Hyperparameter sweep results
│   └── stage1_results.csv      # 24-config sweep metrics
│
├── memoria/                    # Bachelor's thesis (LaTeX source)
├── prompts_modal_sound.txt     # Fixed evaluation prompts (Blocks A–E)
├── CONTRIBUTION.md             # Detailed diff vs. upstream ACE-Step
└── pyproject.toml              # Dependencies (includes TFG extras)
```

Upstream documentation (installation, API, training guide) is in `docs/en/`.

---

## What Changed vs. Upstream

See [CONTRIBUTION.md](CONTRIBUTION.md) for a detailed file-by-file breakdown.

Summary of TFG-specific changes to upstream code:

| File | Change |
|------|--------|
| `acestep/core/generation/handler/lora/lifecycle.py` | CUDA runtime sanitisation before PEFT injection; LoKr + DoRA support |
| `acestep/training/lora_injection.py` | Defensive hook handling for DiT models |
| `acestep/training/lokr_utils.py` | New LoKr adapter injection utilities |
| `acestep/training/dataset_builder_modules/preprocess_vae.py` | VAE preprocessing improvements for modal dataset |
| `acestep/llm_inference.py` | GPU state sanitisation before LLM inference |
| `acestep/ui/gradio/events/training/` | LoRA and LoKr training UI events |

---

## Citation

If you use this work, please cite both this thesis and the upstream model:

```bibtex
@mastersthesis{2026IgnacioFrancoAlmendarez,
  title   = {LoRA para control de modos musicales en ACE-Step 1.5},
  type    = {Bachelor's Thesis},
  author  = {Ignacio Franco Almendárez},
  school  = {E.T.S. de Ingeniería de Sistemas Informáticos, UPM},
  year    = {2026},
}
```

---

## Upstream

This work builds on **ACE-Step 1.5** by StepFun.  
Repository: https://github.com/ace-step/ACE-Step-1.5  
License: MIT
