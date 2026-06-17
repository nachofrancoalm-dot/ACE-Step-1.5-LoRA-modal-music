# Bitacora TFG - LoRA modal (plantilla)

Fecha inicio: ____-__-__
Autor: __________________
Dataset congelado version: __________________
Trigger tag fijo: __________________

## 1) Resumen de runs

| run_id | estado | fecha_inicio | fecha_fin | config_clave | mejor_ckpt | score_medio | notas |
|---|---|---|---|---|---|---|---|
| A | en progreso |  |  | r=64 a=128 d=0.05 lr=1e-4 |  |  |  |
| B | pendiente |  |  | r=64 a=128 d=0.00 lr=1e-4 |  |  |  |
| C | pendiente |  |  | r=64 a=128 d=0.05 lr=5e-5 |  |  |  |
| D | pendiente |  |  | r=128 a=256 d=0.05 lr=1e-4 |  |  |  |

## 2) Registro diario (bitacora operativa)

| fecha | run_id | accion | epoch_actual | tiempo_sesion_min | incidencias | decision_tomada | proximo_paso |
|---|---|---|---|---|---|---|---|
|  | A | inicio entrenamiento | 0 |  |  |  |  |
|  | A | evaluacion checkpoint | 60 |  |  |  |  |
|  | A | evaluacion checkpoint | 80 |  |  |  |  |
|  | A | evaluacion checkpoint | 100 |  |  |  |  |
|  | A | evaluacion checkpoint | 120 |  |  |  |  |
|  | A | evaluacion checkpoint | 140 |  |  |  |  |

## 3) Configuracion exacta por run

### Run A
- comando:

```bash
python train.py fixed --checkpoint-dir ./checkpoints --model-variant turbo --dataset-dir ./preprocessed_tensors/modal --output-dir ./lora_output/modal_run_A --adapter-type lora --rank 64 --alpha 128 --dropout 0.05 --lr 1e-4 --batch-size 1 --gradient-accumulation 4 --epochs 140 --save-every 20 --optimizer-type adamw --scheduler-type cosine --cfg-ratio 0.15 --seed 42
```

- cambios respecto run anterior: baseline

### Run B
- cambios respecto A: dropout=0.0

### Run C
- cambios respecto A: lr=5e-5

### Run D
- cambios respecto A: rank=128 alpha=256

## 4) Evaluacion por checkpoint (agregada)

Escala por criterio: 1 a 5.

| run_id | ckpt_epoch | promedio_centro_tonal | promedio_color_modal | promedio_no_modulacion | promedio_coherencia | promedio_contraste_entre_modos | score_total_promedio_25 | decision |
|---|---|---|---|---|---|---|---|---|
| A | 60 |  ionian 5, dorian 2 Phrygian 1 Lydian 1 Mixolydian 2|  ionian 4, dorian 3.5 Phrygian 1, lydian 1, Mixolydian |  ionian 5 dorian 4 Phrygian 1 Lydian 1|  ionian 4 dorian 3.5 Phrygian 1 Lydian 2|  |  |  | 
| A | 80 |  |  |  |  |  |  |  |
| A | 100 |  |  |  |  |  |  |  |
| A | 120 |  |  |  |  |  |  |  |
| A | 140 |  |  |  |  |  |  |  |
| A | 160 | ionian 3.5 | ionian 3 |  |  |  |  |  |

### 4.1 Significado de cada campo (redaccion TFG)

- run_id: identificador del experimento (ejemplo: r64_a128_d005).
- ckpt_epoch: checkpoint concreto evaluado (ejemplo: epoch_160_loss_0.1766).
- promedio_centro_tonal: media de la claridad/estabilidad de la tonica en los clips evaluados.
- promedio_color_modal: media de cuanto se reconoce el color del modo objetivo (dorian, lydian, etc.).
- promedio_no_modulacion: media de la ausencia de cambios tonales no deseados.
- promedio_coherencia: media de continuidad musical global (fraseo, estabilidad, consistencia).
- promedio_contraste_entre_modos: media de diferencia perceptible entre modos al comparar prompts equivalentes.
- score_total_promedio_25: suma de los 5 promedios anteriores (maximo 25).
- decision: estado final del checkpoint (descartar, candidato, finalista).

### 4.2 Escala ancla ultra concreta (1-5)

Usar esta misma escala para todos los criterios y checkpoints.

1. Muy bajo: casi no cumple el criterio; resultado confuso o incorrecto la mayor parte del clip.
2. Bajo: cumple de forma puntual, pero falla con frecuencia; hay dudas claras al escuchar.
3. Aceptable: cumple de forma suficiente; se reconoce el criterio, aunque con inconsistencias.
4. Alto: cumple claramente en casi todo el clip; fallos pequenos y no dominantes.
5. Excelente: cumple de forma muy clara y estable durante todo el clip; sin dudas perceptivas.

Anclas practicas por criterio:

- centro_tonal:
1. tonica inestable o irreconocible.
3. tonica perceptible pero se pierde en algunas partes.
5. tonica clara y estable casi todo el tiempo.

- color_modal:
1. suena mayor/menor generico, no al modo objetivo.
3. hay rasgos del modo, pero mezclados con color ambiguo.
5. modo objetivo claramente identificable.

- no_modulacion:
1. cambios tonales frecuentes o evidentes.
3. algun desvio puntual.
5. se mantiene en el centro/modalidad esperada.

- coherencia:
1. ideas inconexas, transiciones bruscas.
3. estructura aceptable con algunos cortes raros.
5. discurso musical continuo y consistente.

- contraste_entre_modos:
1. distintos modos suenan muy parecidos.
3. diferencias moderadas.
5. diferencias claras y repetibles entre modos.

Regla rapida de decision:

- score_total_promedio_25 < 15: descartar.
- 15-18.99: candidato debil (escuchar solo si faltan opciones).
- 19-21.99: candidato fuerte.
- >= 22: finalista.

## 5) Evaluacion detallada por clip (7 modos x 2 seeds)

Prompts fijos sugeridos:
- ionian
- dorian
- phrygian
- lydian
- mixolydian
- aeolian
- locrian

Seeds fijas sugeridas:
- 1111
- 2222

| run_id | ckpt_epoch | modo | seed | centro_tonal (1-5) | color_modal (1-5) | no_modulacion (1-5) | coherencia (1-5) | contraste (1-5) | total (0-25) | notas |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 60 | ionian | 1111 |  |  |  |  |  |  |  |
| A | 60 | ionian | 2222 |  |  |  |  |  |  |  |
| A | 60 | dorian | 1111 |  |  |  |  |  |  |  |
| A | 60 | dorian | 2222 |  |  |  |  |  |  |  |
| A | 60 | phrygian | 1111 |  |  |  |  |  |  |  |
| A | 60 | phrygian | 2222 |  |  |  |  |  |  |  |
| A | 60 | lydian | 1111 |  |  |  |  |  |  |  |
| A | 60 | lydian | 2222 |  |  |  |  |  |  |  |
| A | 60 | mixolydian | 1111 |  |  |  |  |  |  |  |
| A | 60 | mixolydian | 2222 |  |  |  |  |  |  |  |
| A | 60 | aeolian | 1111 |  |  |  |  |  |  |  |
| A | 60 | aeolian | 2222 |  |  |  |  |  |  |  |
| A | 60 | locrian | 1111 |  |  |  |  |  |  |  |
| A | 60 | locrian | 2222 |  |  |  |  |  |  |  |

## 6) Ranking final de modelos

| posicion | run_id | mejor_ckpt | score_total_promedio_25 | fortalezas | debilidades | usar_para_resultado_final |
|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  | si/no |
| 2 |  |  |  |  |  | si/no |
| 3 |  |  |  |  |  | si/no |
| 4 |  |  |  |  |  | si/no |

## 7) Parada temprana (checklist rapido)

Marcar si aplica:
- [ ] score estancado en 3 checkpoints seguidos
- [ ] degradacion clara tras maximo local
- [ ] artefactos recurrentes sin mejora modal

Decision:
- [ ] detener run
- [ ] continuar run
- [ ] lanzar siguiente run

## 8) Evidencias para memoria del TFG

- ruta audios evaluados:
- ruta checkpoints:
- graficas (score vs epoch):
- figura comparativa por modo:
- resumen de por que gano el run final:

## 9) Automatizacion (batch)

### 9.1 Generacion por checkpoint

```bash
uv run python scripts/modal_batch_infer.py \
	--api-base-url http://127.0.0.1:7860 \
	--checkpoints-root ./lora_output/<run_name> \
	--checkpoint-pattern "checkpoints/epoch_*_loss_*/adapter" \
	--prompts-file ./prompts_modal_sound.txt \
	--prompts-block B \
	--trigger <trigger_tag> \
	--seeds 1111 2222 \
	--duration 30 \
	--inference-steps 8 \
	--copy-audio-dir ./gradio_outputs/modal_eval_<run_name> \
	--output-csv ./gradio_outputs/modal_manifest_<run_name>.csv
```

### 9.2 Evaluacion automatica de audios

```bash
uv run python scripts/modal_eval_pipeline.py \
	--input-dir ./gradio_outputs/modal_eval_<run_name> \
	--recursive \
	--output-csv ./gradio_outputs/modal_metrics_<run_name>.csv \
	--print-summary
```

La columna principal para comparar runs es `auto_score_0_1`. Si necesitas compatibilidad con resultados anteriores, `auto_score_1_5` sigue saliendo como derivada historica.

### 9.3 Trazabilidad minima por run

- manifest CSV:
- metrics CSV:
- hoja de scoring manual:

## 10) Preseleccion de checkpoints (escucha manual)

Objetivo: reducir escucha a checkpoints con mejor rendimiento automatico (media + estabilidad).

Plantilla lista para Excel (filas pre-rellenadas + formulas):
- docs/Modal_LoRA_Manual_Scoring_Excel_Ready.csv
- docs/Modal_LoRA_Manual_Scoring_Visual.xlsx (recomendada, con colores, validaciones y resumen)

| modelo_run | checkpoints_escucha (prioridad) | checkpoint_reserva | estado_escucha | notas |
|---|---|---|---|---|
| r128_a256_d005 | epoch_20_loss_0.1982; epoch_100_loss_0.1788; epoch_140_loss_0.1714 | epoch_180_loss_0.1603 | pendiente |  |
| r64_a128_d0 | epoch_160_loss_0.1725; epoch_180_loss_0.1748; epoch_80_loss_0.1921 | epoch_200_loss_0.1704 | pendiente |  |
| r64_a128_d005 | epoch_160_loss_0.1766; epoch_100_loss_0.1843; epoch_200_loss_0.1701 | epoch_120_loss_0.1808 | pendiente |  |
| r64_a128_d005_lr5e-5 | epoch_100_loss_0.1965; epoch_160_loss_0.1952; epoch_140_loss_0.1960 | epoch_200_loss_0.1936 | pendiente |  |

Regla practica de uso:

1. Escuchar primero solo los checkpoints de prioridad.
2. Si hay empate o duda en un modelo, escuchar el checkpoint de reserva.
3. Marcar estado_escucha como: pendiente, en_proceso, completado.
