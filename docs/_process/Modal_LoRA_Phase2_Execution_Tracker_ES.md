# Rastreo Ejecutivo: Fase 2 Búsqueda de Hiperparámetros

**Inicio**: 2026-04-18  
**Objetivo**: Optimizar LOra modal mediante búsqueda profunda de hiperparámetros  
**Salida Final**: Configuración óptima de (frozen_mode, r, alpha, lr, dropout) validada manual y automáticamente

---

## 📋 Checklist Pre-Ejecución

- [ ] Entorno `.venv` activado
- [ ] Dataset modal disponible: `./preprocessed_tensors/modal_tfg/`
- [ ] GPU con ≥2GB VRAM libre
- [ ] Scripts descargados y refrescados:
  - `scripts/modal_hparam_sweep_stage1.py` ✅
  - `scripts/modal_hparam_analysis.py` ✅
  - `scripts/modal_batch_infer.py` (existente)
  - `scripts/modal_eval_pipeline.py` (existente)
- [ ] Documentación leída:
  - `docs/Modal_LoRA_Phase1_Manual_Analysis_ES.md` ✅
  - `docs/Modal_LoRA_Phase2_Experimental_Matrix_ES.md` ✅
  - `docs/Modal_LoRA_Phase2_QuickStart_ES.md` ✅

---

## 🚀 Fase 2.1: Stage 1 Execution

### Comando

```powershell
# Copiar y pegar EXACTAMENTE
python scripts/modal_hparam_sweep_stage1.py `
    --dataset-dir ./preprocessed_tensors/modal_tfg `
    --output-dir ./hparam_sweep_stage1_results `
    --seed 1111 `
    --parallel 1
```

### Cronograma Estimado

| Fase | Duración | Acción |
|------|----------|--------|
| Matriz generación | < 1 min | Automático |
| Training 24 exp × | 2-4 horas | ☕ Tu rol: monitor |
| Evaluación | 30-60 min | ⏳ Paralelo con training |
| **Total** | **2.5-4.5 horas** | |

### Monitoreo en Directo

Mientras se ejecuta, en otra terminal:

```powershell
# Tail del log
tail -f ./hparam_sweep_stage1_results/sweep_stage1.log

# O monitoreo GPU en tiempo real (cada 2s)
watch -n 2 nvidia-smi
```

### Señales de Éxito ✅

```
...
[✓] [1/24] Experiment: {'frozen_mode': 'B', 'rank': 32, 'alpha': 32, ...}
Training SUCCESS | Duration: 943.2s | Exit code: 0
Running inference for ./hparam_sweep_stage1_results/exp_001_B_r32_a32_lr1e-04_do0.05...
Running evaluation pipeline...
Evaluation metrics: {'auto_score_0_1': 0.742, ...}
Results saved to: ./hparam_sweep_stage1_results/stage1_results.csv
```

### Señales de Error ⚠️

```
TIMEOUT after 2.0 hours           → Reduce batch_size a 2, re-run
Out of memory (CUDA error)        → Reduce num_inference_steps a 4
Dataset not found                 → Verifica ruta exacta
train.py: no such file            → Asegúrate cwd = raíz del proyecto
```

---

## 📊 Fase 2.2: Análisis Stage 1

Cuando Stage 1 termina: **IMPRESCINDIBLE ejecutar análisis**

### Comando

```powershell
python scripts/modal_hparam_analysis.py `
    --results-csv ./hparam_sweep_stage1_results/stage1_results.csv `
    --output-dir ./hparam_analysis_stage1 `
    --plot-type all
```

### Archivos Generados

```
./hparam_analysis_stage1/
├── best_experiments_top10.csv ← CRÍTICO: Top 10 configs
├── sensitivity_analysis.json ← Análisis numérico
├── heatmap_rank_alpha.png ← Visualización (r, alpha)
├── heatmap_lr_dropout.png ← Visualización (lr, dropout)
├── sensitivity_rank.png ← Línea: rank vs score
├── sensitivity_alpha.png
├── sensitivity_lr.png
└── sensitivity_dropout.png
```

### Interpretación Rápida

Abre `best_experiments_top10.csv` y responde:

1. **¿Mejoró vs baseline?**
   - Baseline E: auto_score_0_1 ≈ 0.735
   - Top Stage 1 debe ser: ≥ 0.755 (ideal: ≥ 0.770)

2. **¿Patrón claro?**
   - ¿Top 3 comparten valores de (r, alpha)?
   - Ej: Todos tienen r=64, alpha=128 → Stage 2 fokus en eso
   - Ej: Mixto r∈{32,64,128} → explorar Stage 1.5 con más precision

3. **¿Frozen Mode B suficiente?**
   - Si así es → Stage 2 speedrun (solo A/C)
   - Si no → Stage 2 full (A/B/C con winner config)

---

## 📋 Fase 2.3: Decision Tree (Stage 2 vs Stage 3)

Después de análisis Stage 1, **sigue esta lógica**:

```
IF auto_score_0_1[best] ≥ 0.770 AND
   std(auto_score_0_1[top3]) < 0.01 AND
   frozen_mode=B wins:
   → STAGE 2A (simple): Test winner en A/C (3 configs, 1-2 horas)
   
ELSE IF auto_score_0_1[best] ∈ [0.750, 0.770) AND
        heterogeneous frozen_mode/params in top5:
   → STAGE 2B (exploración): Refine rangos, re-run Stage 1.5 (2-3 horas)
   
ELSE IF auto_score_0_1[best] < 0.750:
   → INVESTIGAR: Bug en evaluación, dataset issue, o hyperparams roto
   → Option: Verificar 1 manual sample + debug

ELSE:
   → STAGE 3 (exhaustivo): Full 432 si prof exige paper-grade (48+ horas)
```

### Tu Decisión Ahora (Template)

Después de Stage 1, completa:

```
Stage 1 Results Summary:
- Best auto_score_0_1: [____]
- Top 3 frozen_modes: [B/A/C, B/A/C, B/A/C]
- Top 3 (r, alpha): [(___,___), (___,___), (___,___)]
- Top 3 (lr, dropout): [(___,___), (___,___), (___,___)]

Decision:
  ☐ Stage 2A (simple, 1-2h)
  ☐ Stage 2B (refine, 2-3h)
  ☐ Debug + retry Stage 1
  ☐ Stage 3 (exhaustivo, 48h+)

Winner Config for Manual Validation:
  frozen_mode: [_]
  r: [_]
  alpha: [_]
  lr: [_]
  dropout: [_]
  exp_id to listen: [_]
```

---

## 🎧 Fase 2.4: Manual Validation (Critical)

Después de Stage 2 (o mejor Stage 1 if confident):

### Generar Audios

```powershell
# Asumiendo exp_id = 007 fue el winner
python scripts/modal_batch_infer.py `
    --output-dir ./hparam_sweep_stage1_results/exp_007_B_r64_a128_lr1e-04_do0.05 `
    --prompts-block E `
    --num-samples 56
```

Esto genera:
```
./hparam_sweep_stage1_results/exp_007_B_r64_a128_lr1e-04_do0.05/
├── epoch_140_inference/
│   ├── ionian_seed1111.wav
│   ├── ionian_seed2222.wav
│   ├── dorian_seed1111.wav
│   ...
│   └── locrian_seed2222.wav
```

### Escuchar y Puntuar

Usa el mismo formato que Fase 1:

```
Modo: Ionian | Seed: 1111 | Checkpoint: 140
- Harmonias/Voice quality (1-5): [_]
- Estabilidad/Coherencia (1-5): [_]
- Claridad modal (1-5): [_]
- Overall (1-5): [_]
TOTAL: [_]/20 | Notas: [libre]

---

Modo: Dorian | Seed: 1111 | Checkpoint: 140
...
```

### Comparación vs Stage 1

Después de escuchar winner Stage 1:

```
Modo | Manual E | Manual Winner Stage1 | Cambio | ✓/✗
Ionian     | 16.25   | [___] | +[__] | ☐
Dorian     | 12.5    | [___] | +[__] | ☐
Phrygian   | 9.5     | [___] | +[__] | ☐
Lydian     | 15.75   | [___] | +[__] | ☐
Mixolydian | 13.25   | [___] | +[__] | ☐
Aeolian    | 12.5    | [___] | +[__] | ☐
Locrian    | 6.5     | [___] | +[__] | ☐

TOTAL E    | 85.75   | [___] | +[__] | ☐
```

---

## 📝 Post-Ejecución: Documentación

Una vez completado Stage 1+2+3 (o donde decidas parar):

### Reporte Final

Crear: `docs/Modal_LoRA_Phase2_Final_Report_ES.md`

Estructura:

```markdown
# Resultados Finales: Búsqueda de Hiperparámetros

## 1. Metodología
- Grid: 24 combos Stage 1 (Frozen B)
- Métrica: auto_score_0_1 global
- Evaluación: Auto + manual listening

## 2. Resultados Cuantitativos
- Baseline E: 0.735
- Winner: 0.785 (7% improvement)
- Config: [copy from CSV]

## 3. Resultados Cualitativos
- Manual listening mejoró significativamente en: Ionian, Lydian
- Ambigüedad mayor/menor reducida en: Dorian, Aeolian
- Locrian sigue siendo desafío pero "much better than before"

## 4. Validación
- Manual avg: +2.5 puntos sobre baseline
- Correlación manual vs auto: r=0.87
- 6/7 modos con color identifiable

## 5. Conclusión para Profesor
El LoRA ajustado con hiperparámetros optimales demuestra...
```

---

## 🎯 Success Criteria (Recapitulación)

✅ **Cuantitativo**:
- auto_score_0_1 ≥ 0.765 (vs 0.735 baseline)
- modal_accuracy ≥ 88% (vs 86% baseline)
- Varianza inter-modo < 0.08

✅ **Cualitativo**:
- ≥ 6/7 modos con color modal audible
- Locrian "suficientemente coherente" (no "chirría")
- Diferenciación clara I vs IV modo

✅ **Reproducibilidad**:
- Mismos resultados ±0.02 con seed 2222
- No memory crashes
- Duración predecible

---

## 📞 Troubleshooting Rápido

**P: Stage 1 se cuelga en exp #3**
R: Probable OOM. Reduce batch_size=2 in train.py, re-run.

**P: auto_score_0_1 bajó vs baseline**
R: Posible bug en evaluación o dataset issue. Verifica 1 audio manual.

**P: Todos los configs producen ~0.73, sin varianza**
R: Posible bug en métrica o los hiperparams no importan mucho. Debug necesario.

**P: ¿Puedo saltar Stage 1 y run Stage 2 directo?**
R: No recomendado. Stage 1 identifica winner; Stage 2 solo valida en A/C.

---

## 📅 Próximos Pasos

1. ✅ Setup completado (hoy)
2. ⏳ **AQUÍ**: Ejecuta Stage 1 (hoy/mañana, 2-4 horas)
3. 📊 Analiza resultados (30 min)
4. 🎧 Valida manualmente (1-2 horas)
5. ✍️ Redacta reporte final (1 hora)
6. 📤 Presenta al profesor (con confidence!)

