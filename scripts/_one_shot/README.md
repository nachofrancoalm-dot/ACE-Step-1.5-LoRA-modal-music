# Scripts de un solo uso

Scripts utilizados puntualmente durante el desarrollo del TFG, conservados por trazabilidad pero **no necesarios para reproducir el pipeline principal**.

| Script | Uso |
|--------|-----|
| `recalc_auto_score_stage1.py` | Recálculo de `auto_score_0_1` sobre resultados ya generados tras un cambio de fórmula. |
| `modal_backfill_stage1_metrics.py` | Relleno retroactivo de métricas faltantes en CSV de Stage 1. |
| `build_modal_scoring_xlsx.py` | Generación puntual de la hoja Excel para la evaluación manual. |
| `run_modal_lora_training_round2.py` | Orquestación específica de la segunda tanda de entrenamiento. |

Pueden eliminarse en una limpieza posterior sin afectar al pipeline reproducible documentado en [CONTRIBUTION.md](../../CONTRIBUTION.md).
