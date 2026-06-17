# 📑 Índice Completo: Fase 2 Hiperparámetros LoRA Modal

**Generado**: 2026-04-18  
**Estado**: ✅ COMPLETADO - Listo para ejecutar

---

## 🎯 INICIO RÁPIDO (5 minutos)

**START HERE**: [`README_FASE2_START_HERE.md`](./README_FASE2_START_HERE.md)
→ Resumen ejecutivo + 7 acciones clave

**DURANTE EJECUCIÓN**: [`CHEATSHEET_Fase2.md`](./CHEATSHEET_Fase2.md)
→ Referencia rápida (imprime o abre en segundo monitor)

---

## 📚 DOCUMENTACIÓN COMPLETA

### 1. Análisis Fase 1
**Archivo**: [`Modal_LoRA_Phase1_Manual_Analysis_ES.md`](./Modal_LoRA_Phase1_Manual_Analysis_ES.md)  
**Propósito**: Tus síntomas manuales → hipótesis sobre hiperparámetros  
**Contenido**:
- Tabla de scores manuales (14 muestras epoch_100)
- Interpretación de síntomas (tónica débil, ambigüedad mayor/menor, Locrian caótico)
- Hipótesis específicas (H1-H4) sobre qué hiperparámetros afectan cada síntoma
- Predicciones para Fase 2

**Lectura**: 10 min | **Imprescindible**: ⭐⭐⭐

---

### 2. Matriz Experimental Completa
**Archivo**: [`Modal_LoRA_Phase2_Experimental_Matrix_ES.md`](./Modal_LoRA_Phase2_Experimental_Matrix_ES.md)  
**Propósito**: Diseño grid detallado + hipótesis rigurosas  
**Contenido**:
- Definición de variables (frozen_mode A/B/C, rank, alpha, lr, dropout)
- Total: 432 combos (3 × 4 × 4 × 3 × 3)
- Stage 1: 24 combos (Frozen Mode B, subset params)
- Stage 2: 18 combos (top-3 configs × 3 frozen modes)
- Stage 3: 360+ combos (full exhaustive, opcional)
- 4 hipótesis específicas a validar
- Cronograma estimado

**Lectura**: 15 min | **Imprescindible**: ⭐⭐

---

### 3. Quick Start (Instrucciones Ejecutables)
**Archivo**: [`Modal_LoRA_Phase2_QuickStart_ES.md`](./Modal_LoRA_Phase2_QuickStart_ES.md)  
**Propósito**: Comandos exactos para cada fase  
**Contenido**:
- Setup inicial (verificar env, GPU, dataset)
- **Fase 2.1**: Stage 1 command + expected output
- **Fase 2.2**: Análisis post-Stage1
- **Fase 2.3**: Decision tree (Stage 2A vs 2B vs Debug)
- **Fase 2.4**: Manual validation (generar audios, escuchar)
- Troubleshooting mini-guide

**Lectura**: 5 min | **Imprescindible**: ⭐⭐⭐ **EJECUTA ESTO**

---

### 4. Execution Tracker (Checklist + Decision Tree)
**Archivo**: [`Modal_LoRA_Phase2_Execution_Tracker_ES.md`](./Modal_LoRA_Phase2_Execution_Tracker_ES.md)  
**Propósito**: Tu guía durante la ejecución en tiempo real  
**Contenido**:
- Checklist pre-ejecución (GPU, dataset, scripts)
- Cronograma detallado con duraciones
- Señales de éxito/error a monitorear
- Decision tree: IF score ≥ 0.77 → Stage 2A, ELSE 2B, ELSE Debug
- Template para manual validation (copia-pega el rubric)
- Success criteria recapitulación
- FAQ troubleshooting

**Lectura**: 10 min | **Imprescindible**: ⭐⭐⭐ **CONSULTA DURANTE EJECUCIÓN**

---

### 5. Visual Overview (Diagrama ASCII)
**Archivo**: [`Modal_LoRA_Phase2_Visual_Overview_ES.md`](./Modal_LoRA_Phase2_Visual_Overview_ES.md)  
**Propósito**: Flowchart y tablas de métricas esperadas  
**Contenido**:
- Flowchart de dos fases (ASCII art completo)
- Tabla de progresión esperada (baseline → Stage 1 → Stage 2 → Final)
- Red flags (qué síntomas indican problema)
- Resumen cambios vs Fase 1
- Timeline con % completado

**Lectura**: 5 min | **Imprescindible**: ⭐ (referencia)

---

### 6. Final Report Template
**Archivo**: [`Modal_LoRA_Phase2_Final_Report_ES.md`](./Modal_LoRA_Phase2_Final_Report_ES.md)  
**Propósito**: Template para redactar conclusiones  
**Contenido**:
- Secciones: Metodología, Resultados Cuantitativos, Cualitativos, Validación, Conclusión
- Placeholders para tus datos
- Argumentación defensible para profesor

**Lectura**: 1 min (solo hojear) | **Imprescindible**: ⭐ (después Stage 2)

---

## 🐍 SCRIPTS NUEVOS

### 1. `modal_hparam_sweep_stage1.py` (~450 LOC)
**Propósito**: Automatización de 24 entrenamientos  
**Ubicación**: `./scripts/modal_hparam_sweep_stage1.py`  
**Entrada**: CLI args (dataset_dir, output_dir, seed)  
**Salida**: `stage1_results.csv` (24 filas con metrics para cada combo)  
**Uso**:
```powershell
python scripts/modal_hparam_sweep_stage1.py \
    --dataset-dir ./preprocessed_tensors/modal_tfg \
    --output-dir ./hparam_sweep_stage1_results \
    --seed 1111
```
**Duración**: 2-4 horas  
**Dependencias**: Python 3.11+, PyTorch, loguru

---

### 2. `modal_hparam_analysis.py` (~350 LOC)
**Propósito**: Análisis post-sweep (ranking, heatmaps, sensitivity)  
**Ubicación**: `./scripts/modal_hparam_analysis.py`  
**Entrada**: `stage1_results.csv` (de sweep_stage1.py)  
**Salida**:
- `best_experiments_top10.csv` (ranking)
- `sensitivity_analysis.json` (numérico)
- `heatmap_rank_alpha.png`, `heatmap_lr_dropout.png` (visualización)
- `sensitivity_*.png` (líneas de cada parámetro)  

**Uso**:
```powershell
python scripts/modal_hparam_analysis.py \
    --results-csv ./hparam_sweep_stage1_results/stage1_results.csv \
    --output-dir ./hparam_analysis_stage1 \
    --plot-type all
```
**Duración**: 30 min  
**Dependencias**: pandas, matplotlib (optional)

---

## 📊 FLUJO DE DATOS

```
Tu Manual Eval E (14 muestras)
    ↓
    [Fase 1 Analysis]
    ↓
Síntomas + Hipótesis
    ↓
    [modal_hparam_sweep_stage1.py]
    ↓
stage1_results.csv (24 filas × 15 columnas)
    ↓
    [modal_hparam_analysis.py]
    ↓
best_experiments_top10.csv + heatmaps
    ↓
Decision: Stage 2A/2B/Debug
    ↓
    [Conditional: train.py again o Script Stage2]
    ↓
stage2_results.csv (6-18 filas)
    ↓
    [modal_batch_infer.py]
    ↓
Generated wavs (14 samples: 7 modes × 2 seeds)
    ↓
Tú escuchas + puntúas
    ↓
Manual scores vs auto_score_0_1
    ↓
    [modal_hparam_sweep_stage2_final_report]
    ↓
Conclusiones + Presentación Profesor
```

---

## 📁 ESTRUCTURA ESPERADA TRAS EJECUCIÓN

```
hparam_sweep_stage1_results/          ← Output dir
├── sweep_stage1.log                  ← Full log
├── stage1_results.csv                ← CRÍTICO: 24 rows × 15 cols
├── exp_001_B_r32_a32_lr1e-04_do0.05/ ← Experiment 1 output
│   ├── model.pth
│   ├── checkpoint_140/
│   │   ├── pytorch_model.bin
│   │   └── adapter_config.json
│   ├── epoch_140_inference/          ← Generated audios
│   │   ├── ionian_seed1111.wav
│   │   ├── ionian_seed2222.wav
│   │   └── ...
│   └── ...
├── exp_002_B_.../ ← Experiment 2
│   └── ...
└── ... (up to exp_024)

hparam_analysis_stage1/               ← Analysis output
├── best_experiments_top10.csv        ← IMPORTANTE
├── sensitivity_analysis.json
├── heatmap_rank_alpha.png
├── heatmap_lr_dropout.png
├── sensitivity_rank.png
├── sensitivity_alpha.png
├── sensitivity_lr.png
└── sensitivity_dropout.png
```

---

## 🎯 SUCCESS CRITERIA

| Criterio | Baseline E | Target | Esperado |
|----------|-----------|--------|----------|
| auto_score_0_1 | 0.735 | ≥ 0.765 | 0.770-0.785 |
| modal_accuracy | 86% | ≥ 88% | 88-92% |
| Manual listening | 85.75 pts | ≥ 92 pts | 92-100 pts |
| Modos coherentes | 5/7 | 6-7/7 | 7/7 (ideal) |
| Reproducibilidad | N/A | ±0.02 métric | < 0.01 (bueno) |

---

## 🔄 ORDEN DE LECTURA

1. **AHORA** → [`README_FASE2_START_HERE.md`](./README_FASE2_START_HERE.md) (5 min)
2. **ANTES DE EJECUTAR** → [`CHEATSHEET_Fase2.md`](./CHEATSHEET_Fase2.md) (5 min)
3. **DURANTE SETUP** → [`Modal_LoRA_Phase2_QuickStart_ES.md`](./Modal_LoRA_Phase2_QuickStart_ES.md) (5 min)
4. **DURANTE EJECUCIÓN** → [`Modal_LoRA_Phase2_Execution_Tracker_ES.md`](./Modal_LoRA_Phase2_Execution_Tracker_ES.md) (consultar)
5. **DESPUÉS Stage 1** → CSV results + [`Modal_LoRA_Phase2_Experimental_Matrix_ES.md`](./Modal_LoRA_Phase2_Experimental_Matrix_ES.md) (decision)
6. **DESPUÉS Manual Validation** → [`Modal_LoRA_Phase2_Final_Report_ES.md`](./Modal_LoRA_Phase2_Final_Report_ES.md) (redactación)

---

## ⏰ TIMELINE ESTIMADO

| Hito | Duración | Acumulado |
|------|----------|-----------|
| Setup | 10 min | 10 min |
| Stage 1 | 2-4 h | 2.3-4.3 h |
| Análisis | 30 min | 3-5 h |
| Decision + Manual | 1-2 h | 4-7 h |
| Redacción | 1 h | 5-8 h |
| **TOTAL** | **5-8 h** | **Este fin de semana** |

---

## 🚀 COMIENZA AHORA

→ Abre: [`README_FASE2_START_HERE.md`](./README_FASE2_START_HERE.md)

→ Luego ejecuta el comando de Stage 1 desde [`CHEATSHEET_Fase2.md`](./CHEATSHEET_Fase2.md)

**¡A por ello!** 🎯

