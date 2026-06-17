# Quick Start: Fase 2 - Búsqueda de Hiperparámetros

## Resumen Ejecutivo

Tras analizar tus evaluaciones manuales de prompt E (Fase 1), hemos identificado:
- **Fortalezas**: Coherencia armónica (in_mode_ratio) robusta incluso sin tónica explícita
- **Debilidades**: Tónica débil en modos "fáciles" (Ionian, Lydian), ambigüedad mayor/menor en modos menores
- **Estrategia**: Grid search profundo (144 combinaciones) en 3 modos de congelado (A/B/C)

---

## Fase 2.0: Setup Inicial

### Requisitos

```bash
# 1. Verificar entorno activado
& .\.venv\Scripts\Activate.ps1

# 2. Verificar dataset modal disponible
ls ./preprocessed_tensors/modal_tfg

# 3. Verificar capacidad GPU (~2-4GB VRAM por experimento)
nvidia-smi
```

### Rutas Críticas

```
./train.py                          ← Entrada a training (subcommand "fixed")
./scripts/modal_hparam_sweep_stage1.py  ← Automatización de búsqueda (nuevo)
./scripts/modal_batch_infer.py       ← Inferencia y scoring
./scripts/modal_eval_pipeline.py     ← Métricas automáticas
```

---

## Fase 2.1: Stage 1 (24 Experimentos, ~2-3 horas)

### Comando

```powershell
# Stage 1: Frozen Mode B (solo LoRA) + subset de hiperparámetros
python scripts/modal_hparam_sweep_stage1.py `
    --dataset-dir ./preprocessed_tensors/modal_tfg `
    --output-dir ./hparam_sweep_stage1_results `
    --seed 1111 `
    --parallel 1
```

### Qué Sucederá

1. **Generación de matriz**: 3×2×2×2 = 24 combinaciones de (rank, alpha, lr, dropout)
2. **Training**: ~15 min por experimento en GPU A100; ~360 min total en serie
3. **Evaluación** (opcional): Auto-scoring con `modal_eval_pipeline.py` (~2 min per exp)
4. **Logging**: Resultados → `./hparam_sweep_stage1_results/stage1_results.csv`

### Salida Esperada

```
Archivo: stage1_results.csv
Columnas:
  exp_id, frozen_mode, rank, alpha, lr, dropout,
  epoch_checkpoint, auto_score_0_1, modal_accuracy,
  in_mode_ratio, tonal_stability, tonic_dominance,
  template_similarity, training_status, training_duration_sec, training_error

Ejemplo fila:
  1, B, 32, 32, 1e-04, 0.05, 140, 0.742, 0.857, 0.823, 0.912, 0.091, 0.667, SUCCESS, 943.2,
```

### Interpretación Initial

Busca en CSV:
- **auto_score_0_1**: ¿Superó 0.735 (baseline E)? ¿Cuánto?
- **Modal accuracy**: ¿Mejoró a ~90%?
- **Tonic dominance**: ¿Subió para Ionian/Lydian? (era 0.091-0.004 en E)
- **Volatilidad**: ¿Varían mucho las métricas entre combos?

**Si Stage 1 muestra claro ganador** → Siguiente: Stage 2  
**Si Stage 1 muestra alta varianza** → Refine rangos y re-run

---

## Fase 2.2: Stage 2 (Análisis + Top-3 Combinations)

### Comando de Análisis Post-Stage1

```powershell
python scripts/modal_hparam_analysis.py `
    --results-csv ./hparam_sweep_stage1_results/stage1_results.csv `
    --output-dir ./hparam_analysis_stage1 `
    --plot-type all
```

Genera:
- `best_experiments_top10.csv` ← Top 10 por auto_score_0_1
- `heatmap_rank_alpha.png` ← Heatmap (r, alpha) vs score
- `sensitivity_lr_dropout.png` ← Sensibilidad a (lr, dropout)
- `convergence_analysis.json` ← Métricas por epoch

### Decisión de Stage 2

**Opción A (Simple)**: Si top-3 experimentos todos tienen r ≥ 64 + alpha ≥ 64 + lr=1e-4:
- Repetir winner en Frozen Modes A y C (solo 3 combos × 2 otros modos = 6 experimentos)
- Tiempo: ~2 horas

**Opción B (Exploratoria)**: Si resultados mixtos en Stage 1:
- Refine rangos basado en sensibilidad
- Re-run Stage 1 con nuevos límites
- Tiempo: ~3 horas (repetir Stage 1)

---

## Fase 2.3: Stage 2 (18 Experimentos, ~2-3 horas)

### Comando (After Analysis)

```powershell
# Assumiendo top winner es: Frozen Mode B, r=64, alpha=128, lr=1e-4, dropout=0.05
# Probar en Modos A y C también

python scripts/modal_hparam_sweep_stage1.py `
    --dataset-dir ./preprocessed_tensors/modal_tfg `
    --output-dir ./hparam_sweep_stage2_results `
    --seed 1111 `
    --stage 2 `
    --top-params ./hparam_analysis_stage1/best_experiments_top10.csv
```

*(Script mejorado para soportar `--stage 2` y `--top-params` en próxima iteración)*

---

## Fase 2.4: Stage 3 (Exhaustivo, Opcional)

Si Stage 2 aún muestra mejoras significativas: full 432 experimentos.

**Tiempo**: ~50+ horas en GPU serial; ~5-10 horas en paralelo (8 GPUs).

Típicamente no necesario para TFG; Stage 1+2 suficientes.

---

## Validación Manual Post-Sweep

Después de Stage 1 (o mejor, después de Stage 2):

1. **Generar audios** con top 3 checkpoints usando best hyperparams:
   ```powershell
   python scripts/modal_batch_infer.py `
       --output-dir ./hparam_sweep_stage1_results/exp_XXX `
       --prompts-block E `
       --num-samples 56
   ```

2. **Escuchar** los corrugated audios del checkpoint 140 (o checkpoint óptimo por métrica)
3. **Puntuar** con la misma rúbrica que usaste en Fase 1 (5 criterios × 7 modos × 2 seeds)
4. **Comparar** manual scores (nuevo) vs auto_score_0_1 (viejo):
   - ¿Mejoraron manualmente?
   - ¿Métricas auto correlacionan con juicio audiol?
   - ¿Sigue habiendo Locrian caótico?

---

## Troubleshooting

### Error: "train.py: comando no encontrado"
```powershell
# Asegúrate de estar en raíz del proyecto
cd C:\Users\nacho\Desktop\Universidad\Cuarto\TFG\Acestep-1.5\ACE-Step-1.5
python train.py fixed --help
```

### Error: "Dataset not found"
```powershell
# Verifica ruta exacta y contenido
ls ./preprocessed_tensors/modal_tfg
# Debe tener subdirs: ionian/, dorian/, ... una por modo
```

### Error: "Out of memory"
```
- Reduce batch_size en train.py (cfg: --batch-size 2)
- O reduce num_inference_steps en eval (cfg: 4 pasos)
- O reduce stage 1 a 12 experimentos (solo r ∈ [32, 64])
```

### Stage 1 se detiene/cuelga
```
- Check GPU: nvidia-smi
- Check disco espacio: dir D:\ (si en D:)
- Monitorear logs: tail -f ./hparam_sweep_stage1_results/sweep_stage1.log
```

---

## Timeline Estimado

| Fase | Experimentos | Tiempo | Acción |
|------|-------------|--------|--------|
| **1** (Pre-sweep) | Manual eval + design | 1 hora | ✅ Completado |
| **2.0** (Setup) | Verificación | 10 min | Haz ahora |
| **2.1** (Stage 1) | 24 | 2-3 horas | Hoy/mañana |
| **2.2** (Análisis) | Post-processing | 30 min | Después Stage 1 |
| **2.3** (Stage 2) | 6-18 | 1-3 horas | Sábado/domingo |
| **2.4** (Validación) | Manual listen | 2-3 horas | Después Stage 2 |
| **TOTAL** | ~50 exps | **6-10 horas** | This weekend |

---

## Cuándo Parar

✅ **Stop at Stage 1 si**:
- auto_score_0_1 ≥ 0.76 y modal_accuracy ≥ 88%
- Ganador claro (1-2 combos significativamente mejores)
- Concordancia entre métricas (mismo winner en auto + manual listening)

✅ **Skip to Stage 3 si**:
- Stage 2 muestra mejoras consistentes
- Equipo tiene 48+ horas y múltiples GPUs
- Objetivo es paper (no solo TFG)

---

## Next Steps

1. Ejecuta Stage 1 ahora
2. Comparte `stage1_results.csv` conmigo
3. Analizamos ganadores + decidimos Stage 2
4. Valida manualmente 3-5 muestras del top checkpoint
5. Redacta conclusiones para profesor

