# ⚡ CHEAT SHEET: Fase 2 - Quick Reference

Imprime esto o mantenlo abierto durante la ejecución de Stage 1.

---

## Setup (Haz UNA sola vez)

```powershell
# 1. Activar venv
& .\.venv\Scripts\Activate.ps1

# 2. Verifica Python
python --version   # → 3.11 o 3.12

# 3. Verifica GPU
nvidia-smi          # → Must show your GPU with ≥2GB free

# 4. Verifica dataset
ls ./preprocessed_tensors/modal_tfg/  # → 7 carpetas (modes)
```

---

## Fase 2.1: Stage 1 Execution

**Comando exacto:**
```powershell
python scripts/modal_hparam_sweep_stage1.py `
    --dataset-dir ./preprocessed_tensors/modal_tfg `
    --output-dir ./hparam_sweep_stage1_results `
    --seed 1111
```

**Monitor en otra terminal:**
```powershell
tail -f ./hparam_sweep_stage1_results/sweep_stage1.log
```

**Expected output cada experimento:**
```
[✓] [N/24] Experiment: {...}
Training SUCCESS | Duration: 943.2s
Results saved to: ./hparam_sweep_stage1_results/stage1_results.csv
```

**Duración**: ~2-4 horas (24 experimentos × ~15 min cada uno)

---

## Fase 2.2: Post-Stage1 Analysis

**INMEDIATAMENTE después que Stage 1 termine:**

```powershell
python scripts/modal_hparam_analysis.py `
    --results-csv ./hparam_sweep_stage1_results/stage1_results.csv `
    --output-dir ./hparam_analysis_stage1
```

**Outputs:**
```
hparam_analysis_stage1/
├── best_experiments_top10.csv ← ABRE ESTO PRIMERO
├── sensitivity_analysis.json
└── heatmap_*.png (visualizaciones)
```

---

## Fase 2.3: Decision Tree

Abre: `hparam_analysis_stage1/best_experiments_top10.csv`

**Mira la primera fila (top experiment):**

```
exp_id | frozen_mode | rank | alpha | lr | dropout | auto_score_0_1
─────────────────────────────────────────────────────────────────
    7  |      B      |  64  | 128   | 1e-04 | 0.05 |    0.7681
```

**Decision:**

- **IF auto_score_0_1 ≥ 0.770:**  
  → **Stage 2A** (simple): Repeat exp_7 pero con frozen_mode=A y C
  - Tiempo: 1-2 horas
  - Luego: Manual validation

- **IF auto_score_0_1 ∈ [0.750, 0.770):**  
  → **Stage 2B** (refine): Ajusta grid según sensitivity analysis
  - Tiempo: 2-3 horas
  - Luego: Rerun Stage 1.5 + manual validation

- **IF auto_score_0_1 < 0.750:**  
  → **DEBUG**: Verifica 1 sample manual
  - Escucha el audio del checkpoint 140 de exp_7
  - ¿Suena peor que baseline E? → Bug en evaluación
  - ¿Suena igual o mejor que E? → Métrica conservadora, ignore

- **IF top 10 todos ∈ {0.72, 0.73}:**  
  → **STAGE 3** (only if desperate): Full 432 combos
  - Tiempo: 40+ horas
  - Típicamente no necesario para TFG

---

## Stage 2A: Quick Reprise (if auto ≥ 0.770)

Winner fue: `exp_7 | B | r=64 | alpha=128 | lr=1e-4 | dropout=0.05`

**Repite con Frozen A y C:**
```powershell
# Modify only script line 34 STAGE1_GRID frozen_mode:
# OLD: "frozen_mode": ["B"]
# NEW: "frozen_mode": ["A", "C"]

# Also change FIXED_CONFIG epochs to 80 (faster):
# OLD: "epochs": 140
# NEW: "epochs": 80

# Run subset of Stage 1 (solo winner config):
# Manually edit combos[] or create Stage 2 specific version

# For now, manual trigger (simplified):
python train.py fixed `
    --rank 64 ` 
    --alpha 128 `
    --lr 1e-4 `
    --dropout 0.05 `
    --frozen-layer-type all `
    --epochs 80 `
    [other args...]
```

*Nota: Implementar script `modal_hparam_sweep_stage2.py` si tiempo*

---

## Manual Validation (Critical Step!)

**Generar audios del winner:**
```powershell
# Reemplaza exp_NNN con exp_id del ganador
python scripts/modal_batch_infer.py `
    --output-dir ./hparam_sweep_stage1_results/exp_NNN_* `
    --prompts-block E `
    --num-samples 56
```

**Outputs:**
```
./hparam_sweep_stage1_results/exp_NNN_B_r64_a128_lr1e-04_do0.05/epoch_140_inference/
├── ionian_seed1111.wav
├── ionian_seed2222.wav
├── dorian_seed1111.wav
...
└── locrian_seed2222.wav
```

**Escucha & Puntúa:**

Template (copia para cada modo):
```
═══════════════════════════════════════════════════════════
Modo: IONIAN | Seed: 1111 | Checkpoint: 140
═══════════════════════════════════════════════════════════

Harmonías/Voicing (1=pobre, 5=excelente): [__]
Estabilidad/Coherencia (1-5): [__]
Claridad modal (1-5): [__]
Overall impression (1-5): [__]

TOTAL: [__]/20

Notas:
  [Tu análisis libre]

─────────────────────────────────────────────────────────────
```

**Comparación Final:**

| Modo | E Manual | Winner Manual | Δ | Improved? |
|------|----------|---------------|---|-----------|
| Ionian | 16.25 | [__] | +[__] | ☐ |
| Dorian | 12.5 | [__] | +[__] | ☐ |
| ... | ... | ... | ... | ... |
| TOTAL | 85.75 | [__] | +[__] | ☐ |

**Éxito:** Target Δ ≥ +6 puntos (7% improvement)

---

## Redacción Final (1 hora)

Template: `docs/Modal_LoRA_Phase2_Final_Report_ES.md`

Secciones:
1. **Metodología**: 24 combos Stage 1 + decision tree + manual validation
2. **Resultados Cuantitativos**: auto_score_0_1 improved from 0.735 → [___]
3. **Resultados Cualitativos**: Manual scores + delta vs baseline
4. **Validación**: Reproducibilidad + correlación auto vs manual
5. **Conclusión**: "LoRA óptimo demuestra coherencia modal viable"

---

## Emergency Fixes

| Problema | Solución |
|----------|----------|
| OOM (Out of Memory) | Reduce batch_size=2 en train.py; re-run script |
| Timeout (>2h per exp) | Reduce epochs=80 o num_inference_steps=4 |
| GPU not detected | `nvidia-smi` first; check CUDA availability |
| Dataset not found | Verify `ls ./preprocessed_tensors/modal_tfg/` |
| Script not found | Confirm cwd is root of project (`pwd` showsbase dir) |
| CSV is empty | Training status = FAILED; check logs for error |
| Number mismatch in header | CSV corrupted; check sweep_stage1.log for why |

---

## Critical Checks

- ✅ GPU activated and memory free
- ✅ Venv activated (prompt shows `(.venv)`)
- ✅ Dataset exists (7 mode folders)
- ✅ Scripts are in `./scripts/`
- ✅ Docs are in `./docs/`
- ✅ Output dir exists or will be created (OK)

---

## Timeline (Visual)

```
NOW: Setup (10 min) 🚀
     ↓
T+10min: Stage 1 starts (2-4 hours) ⏳
     ↓
T+2.5h: Stage 1 done, Análisis runs (30 min) 📊
     ↓
T+3h: Decision → Stage 2A/2B/Debug (1-3 hours) 🔀
     ↓
T+4-6h: Manual Validation (1-2 hours) 🎧
     ↓
T+6-8h: Redacción Final (1 hour) ✍️
     ↓
T+7-9h: DONE! Ready for profesor! 🎉
```

**Total: 7-9 hours este fin de semana**

---

## Shortcuts

**Copy-paste ready commands:**

```powershell
# Setup
& .\.venv\Scripts\Activate.ps1; python --version; nvidia-smi

# Stage 1
python scripts/modal_hparam_sweep_stage1.py --dataset-dir ./preprocessed_tensors/modal_tfg --output-dir ./hparam_sweep_stage1_results --seed 1111

# Analysis
python scripts/modal_hparam_analysis.py --results-csv ./hparam_sweep_stage1_results/stage1_results.csv --output-dir ./hparam_analysis_stage1 --plot-type all

# Monitor logs
tail -f ./hparam_sweep_stage1_results/sweep_stage1.log

# Check results
ls -la ./hparam_sweep_stage1_results/
cat ./hparam_analysis_stage1/best_experiments_top10.csv
```

---

## Key Files (Ctrl+Click or Hover)

- [Phase 2 Quick Start](./Modal_LoRA_Phase2_QuickStart_ES.md)
- [Execution Tracker](./Modal_LoRA_Phase2_Execution_Tracker_ES.md)
- [Full Matrix Details](./Modal_LoRA_Phase2_Experimental_Matrix_ES.md)
- [Visual Overview](./Modal_LoRA_Phase2_Visual_Overview_ES.md)
- [Final Report Template](./Modal_LoRA_Phase2_Final_Report_ES.md)

---

**¡Buena suerte!** 🚀🎯

Print this or keep open in second monitor during Stage 1 execution.

