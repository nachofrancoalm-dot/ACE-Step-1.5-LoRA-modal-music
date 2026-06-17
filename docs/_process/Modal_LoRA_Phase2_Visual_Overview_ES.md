# Resumen Visual: Fase 1 + Fase 2 LoRA Modal

## Flujo Ejecutivo de Dos Fases

```
┌─────────────────────────────────────────────────────────────────────┐
│ FASE 1: Análisis Manual (Completado ✅)                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Tus evaluaciones manuales de 14 muestras (epoch_100, prompt E)   │
│  ↓                                                                  │
│  Hallazgos clave:                                                   │
│    • Tónica aprendida del NAME, no del prompt                      │
│    • in_mode_ratio robusta (~83% modal)                            │
│    • Ambigüedad mayor/menor en modos menores                       │
│  ↓                                                                  │
│  Síntomas → Hipótesis sobre hiperparámetros                        │
│    • Tónica débil → Probar r/alpha altos, dropout bajo             │
│    • Ambigüedad mayor → Probar dropout alto                        │
│    • Locrian frágil → Probar frozen_mode A (todo trainable)        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│ FASE 2: Búsqueda de Hiperparámetros (Comenzar AHORA ⏳)           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  STAGE 1: Frozen Mode B (24 combos) → 2-4 horas                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Grid: r∈{32,64,128} × alpha∈{32,128} ×                      │  │
│  │       lr∈{1e-4,1e-3} × dropout∈{0.05,0.1}                   │  │
│  │                                                               │  │
│  │ Output: stage1_results.csv (24 rows)                         │  │
│  │ Métrica: auto_score_0_1 global                               │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                        ↓                                            │
│  ANÁLISIS + DECISION TREE (30 min)                                 │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Top 10 ranked by auto_score_0_1                              │  │
│  │ Heatmaps: (r,alpha) & (lr,dropout)                           │  │
│  │ Sensitivity: Efecto de cada parámetro                        │  │
│  │                                                               │  │
│  │ Decision:                                                     │  │
│  │   IF score ≥ 0.770 & clear winner → Stage 2A (fast)         │  │
│  │   IF score ∈ [0.750,0.770) & mixed → Stage 2B (refine)      │  │
│  │   IF score < 0.750 → Debug                                  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                        ↓                                            │
│  STAGE 2A: Frozen A/C (simple) ← IF winner claro                  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Top config from Stage 1                                      │  │
│  │ Frozen Mode: A & C (solo LoRA fue B)                         │  │
│  │ Total: 3 combos × 2 modos = 6 entrenamientos                │  │
│  │ Tiempo: ~1-2 horas                                           │  │
│  │                                                               │  │
│  │ Output: stage2_results.csv (6 rows)                          │  │
│  └──────────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────────┤
│  │ O                                                              │
│  └────────────────────────────────────────────────────────────────┤
│  STAGE 2B: Refine Stage 1 ← IF resultados mixtos                  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Adjust grid based on sensitivity analysis:                   │  │
│  │   - Si r=64 domina → ignore r∈{32,128}                       │  │
│  │   - Si dropout=0.1 mejor → focus dropout∈{0.1,0.15}         │  │
│  │   - Si lr=1e-4 mejor → ignore lr=1e-3                        │  │
│  │                                                               │  │
│  │ Re-run Stage 1 con grid refinado (12-18 combos)              │  │
│  │ Tiempo: ~2-3 horas                                           │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                        ↓                                            │
│  MANUAL VALIDATION (1-2 horas)                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Generar audios con best checkpoint (140 o dinámico)          │  │
│  │ Escuchar 7 modos × 2 seeds (14 samples)                      │  │
│  │ Puntuar usando mismo rubric que Fase 1                       │  │
│  │                                                               │  │
│  │ Comparación:                                                 │  │
│  │   Fase 1 (E manual): 85.75 / 140 = 0.613                     │  │
│  │   Fase 2 (Winner manual): [_____] / 140 = [_____]            │  │
│  │   Delta: [Esperado: +5-10 puntos, ~+7%]                      │  │
│  │                                                               │  │
│  │ Validación: ¿Mejora manual correlaciona con auto_score_0_1?  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                        ↓                                            │
│  STAGE 3 (OPCIONAL): Full Search ← IF aún incierto               │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Full 432 combos (3 frozen × 4 r × 4 alpha × 3 lr × 3 dropout)│  │
│  │ Tiempo: 40-50 horas secuencial, 5-10h con paralelo           │  │
│  │ Típicamente NO necesario para TFG                            │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│ FASE 3: Redacción Final (0.5-1 hora)                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Reporte para profesor:                                            │
│    • Metodología: Grid search + manual validation                  │
│    • Resultados: auto_score_0_1 improved from 0.735 → [___]       │
│    • Best config: frozen_mode=[_], r=[_], alpha=[_], ...          │
│    • Manual validation: 6/7 modos coherentes, delta +[_] pts      │
│    • Conclusión: LoRA es viable para generación modal coherente    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Métricas Clave a Seguir

### Progression Expected (Best Case)

| Fase | Métrica | Valor | Delta |
|------|---------|-------|-------|
| **Baseline (E)** | auto_score_0_1 | 0.735 | — |
| | modal_accuracy | 86% | — |
| | Manual (você) | 85.75 pts | — |
| **Stage 1 Best** | auto_score_0_1 | ≥0.765 | +4% |
| | modal_accuracy | ≥88% | +2% |
| **Stage 2 Best** | auto_score_0_1 | ≥0.775 | +5-6% |
| | modal_accuracy | ≥90% | +4% |
| | Manual (target) | ≥92 pts | +6+ pts |

### Red Flags

| Sintoma | Problema | Acción |
|---------|----------|--------|
| auto_score_0_1 < 0.730 | Hiperparams roto o bug eval | Debug 1 sample manual |
| Std(top3) > 0.02 | Alta volatilidad | Stage 2B (refine) |
| Frozen A/B/C todos ∼ iguales | Frozen mode no importa | Simpl a Stage 3 (B only) |
| Manual scores bajan en Stage 2 | Overfitting | Aumentar dropout |
| Locrian sigue < 5.0 | Modos frágil no mejora | Aceptar limitación |

---

## Documentos Clave Creados

```

docs/
├── Modal_LoRA_Phase1_Manual_Analysis_ES.md
│   └─ Tu evaluación manual + hipótesis → hiperparámetros
│
├── Modal_LoRA_Phase2_Experimental_Matrix_ES.md
│   └─ Grid detallado, 3 frozen modes, success criteria
│
├── Modal_LoRA_Phase2_QuickStart_ES.md
│   └─ Instrucciones paso a paso para ejecutar
│
├── Modal_LoRA_Phase2_Execution_Tracker_ES.md
│   └─ Checklist + decision tree + manual validation (AQUÍ)
│
└── Modal_LoRA_Phase2_Final_Report_ES.md
    └─ Para redactar después (template)

scripts/
├── modal_hparam_sweep_stage1.py
│   └─ Automatización de 24 entrenamientos
│
└── modal_hparam_analysis.py
    └─ Parsing + ranking + heatmaps post-Stage1
```

---

## Resumen: Qué Cambió Desde Fase 1

### Antes (Fase 1)

```
✓ Identificaste que prompt E tiene coherencia modal
✓ Capturaste verbalmente los síntomas (tónica débil, ambigüedad)
✓ Entendiste que las métricas auto no explican todo
✗ Pero sin framework => seguir manual listening ad-hoc
```

### Ahora (Fase 2)

```
✓ Síntomas → Hipótesis → Experimentos (ciclo científico real)
✓ 24 entrenamientos automatizados (no manual tuning)
✓ Análisis numérico riguroso (sensitivity, heatmaps)
✓ Manual validation INTEGRADO (no separado)
✓ Decision tree claro (Stage 2A/B/3 según resultados)
✓ Ready para defender resultados al profesor
```

---

## Estimados Finales

| Componente | Tiempo | Status |
|------------|--------|--------|
| Documentación Fase 2 | 2 horas | ✅ Completado |
| Stage 1 Ejecución | 2-4 horas | ⏳ Listo |
| Análisis + Decision | 0.5 horas | ⏳ Listo |
| Stage 2A (si simple) | 1-2 horas | ⏳ Potencial |
| Manual Validation | 1-2 horas | ⏳ Potencial |
| Redacción Final | 1 hora | ⏳ Potencial |
| **TOTAL (sin Stage 2+)** | **4-5.5 horas** | **This weekend** |
| **TOTAL (con Stage 2A)** | **6-7.5 horas** | **This weekend** |

---

## PRÓXIMO PASO: ¡Ejecutar Stage 1!

Ve a: [Modal_LoRA_Phase2_QuickStart_ES.md › Fase 2.1](./Modal_LoRA_Phase2_QuickStart_ES.md)

Comando:
```powershell
python scripts/modal_hparam_sweep_stage1.py `
    --dataset-dir ./preprocessed_tensors/modal_tfg `
    --output-dir ./hparam_sweep_stage1_results `
    --seed 1111 `
    --parallel 1
```

Good luck! 🚀

