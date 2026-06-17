# Plan experimental detallado - LoRA modal para TFG (ACE-Step)

Fecha: 2026-03-14
Objetivo: obtener un LoRA que responda de forma coherente a prompts modales (Ionian, Dorian, Phrygian, Lydian, Mixolydian, Aeolian, Locrian) en un entorno reproducible y defendible para TFG.

## 1) Hipotesis y criterio de exito

Hipotesis principal:
- El problema no es solo numero de epochs. La coherencia modal depende de:
  1. capacidad del adapter,
  2. consistencia del etiquetado textual,
  3. seleccion de checkpoint por audio (no por epoch final).

Criterio de exito minimo (apto para TFG):
- Al menos 5 de 7 modos con caracter modal reconocible en test ciego interno.
- Diferenciacion audible entre al menos 4 modos mayores/menores cercanos (ejemplo: Ionian vs Lydian, Dorian vs Aeolian).
- Reproducibilidad: misma conclusion en 2 seeds de inferencia distintas.

## 2) Preparacion previa obligatoria (dia 0)

### 2.1 Congelar dataset

- No mezclar nuevas canciones durante esta fase.
- Verificar por muestra:
  - caption incluye explicitamente modo y centro tonal (ejemplo: D Dorian, modal vamp on D).
  - bpm y keyscale correctos.
  - letras: usar [Instrumental] si aplica, sin ruido.
- Mantener genre_ratio = 0 (priorizar caption descriptivo).

### 2.2 Trigger tag unico

- Elegir un trigger fijo, por ejemplo: modal_tfg_v1.
- El trigger debe aparecer en captions de entrenamiento segun tu flujo.
- En inferencia, siempre incluir exactamente el mismo trigger.

### 2.3 Split de evaluacion

- Crear un banco fijo de 7 prompts (uno por modo) y no cambiarlo entre runs.
- Guardar resultados por epoch checkpoint para comparacion A/B.

## 3) Configuracion base recomendada (Side-Step corrected)

Usar train.py fixed (modo corregido):
- adapter-type: lora
- rank: 64
- alpha: 128
- dropout: 0.05
- lr: 1e-4
- optimizer-type: adamw
- scheduler-type: cosine
- batch-size: 1
- gradient-accumulation: 4
- epochs: 140
- save-every: 20
- cfg-ratio: 0.15
- gradient-checkpointing: enabled
- seed: 42

## 4) Matriz experimental minima (4 runs)

Regla: cambiar solo una variable por run para poder atribuir mejoras.

Run A (baseline fuerte)
- r=64, alpha=128, dropout=0.05, lr=1e-4

Run B (menos regularizacion)
- Igual que A, pero dropout=0.0

Run C (paso mas conservador)
- Igual que A, pero lr=5e-5

Run D (capacidad alta)
- Igual que A, pero r=128, alpha=256, dropout=0.05

Recomendacion de ejecucion:
- Orden: A -> B -> C -> D.
- Si A ya es claramente bueno, ejecutar solo B o C para validacion y ahorrar tiempo.

## 5) Comandos listos (PowerShell, Windows)

Nota: ajustar rutas segun tu maquina. Mantener mismo dataset-dir en todas las corridas.

Run A
python train.py fixed --checkpoint-dir ./checkpoints --model-variant turbo --dataset-dir ./preprocessed_tensors/modal --output-dir ./lora_output/modal_run_A --adapter-type lora --rank 64 --alpha 128 --dropout 0.05 --lr 1e-4 --batch-size 1 --gradient-accumulation 4 --epochs 140 --save-every 20 --optimizer-type adamw --scheduler-type cosine --cfg-ratio 0.15 --seed 42

Run B
python train.py fixed --checkpoint-dir ./checkpoints --model-variant turbo --dataset-dir ./preprocessed_tensors/modal --output-dir ./lora_output/modal_run_B --adapter-type lora --rank 64 --alpha 128 --dropout 0.0 --lr 1e-4 --batch-size 1 --gradient-accumulation 4 --epochs 140 --save-every 20 --optimizer-type adamw --scheduler-type cosine --cfg-ratio 0.15 --seed 42

Run C
python train.py fixed --checkpoint-dir ./checkpoints --model-variant turbo --dataset-dir ./preprocessed_tensors/modal --output-dir ./lora_output/modal_run_C --adapter-type lora --rank 64 --alpha 128 --dropout 0.05 --lr 5e-5 --batch-size 1 --gradient-accumulation 4 --epochs 140 --save-every 20 --optimizer-type adamw --scheduler-type cosine --cfg-ratio 0.15 --seed 42

Run D
python train.py fixed --checkpoint-dir ./checkpoints --model-variant turbo --dataset-dir ./preprocessed_tensors/modal --output-dir ./lora_output/modal_run_D --adapter-type lora --rank 128 --alpha 256 --dropout 0.05 --lr 1e-4 --batch-size 1 --gradient-accumulation 4 --epochs 140 --save-every 20 --optimizer-type adamw --scheduler-type cosine --cfg-ratio 0.15 --seed 42

## 6) Seleccion de checkpoint (clave para no perder semanas)

No elegir ultimo checkpoint por defecto.

Para cada run:
- Evaluar checkpoints: 60, 80, 100, 120, 140.
- Generar siempre con:
  - mismos 7 prompts,
  - mismo bloque de prompts (A o B),
  - mismas seeds de inferencia (por ejemplo 2 seeds fijas).
- Puntuar cada audio con rubrica y elegir best checkpoint por score medio.

### Automatizacion recomendada (Linux, layout real de checkpoints)

Para runs con estructura tipo:
`lora_output/r128_a256_d005/checkpoints/epoch_100_loss_xxx/adapter`

1) Generar audios por checkpoint y guardar manifiesto:

```bash
uv run python scripts/modal_batch_infer.py \
  --api-base-url http://127.0.0.1:7860 \
  --checkpoints-root ./lora_output/r128_a256_d005 \
  --checkpoint-pattern "checkpoints/epoch_*_loss_*/adapter" \
  --prompts-file ./prompts_modal_sound.txt \
  --prompts-block B \
  --trigger modal_tfg_v1 \
  --seeds 1111 2222 \
  --duration 30 \
  --inference-steps 8 \
  --copy-audio-dir ./gradio_outputs/modal_eval_r128_a256_d005 \
  --output-csv ./gradio_outputs/modal_manifest_r128_a256_d005.csv
```

2) Evaluar metricas automaticas de los audios generados:

```bash
uv run python scripts/modal_eval_pipeline.py \
  --input-dir ./gradio_outputs/modal_eval_r128_a256_d005 \
  --recursive \
  --output-csv ./gradio_outputs/modal_metrics_r128_a256_d005.csv \
  --print-summary
```

## 7) Rubrica de evaluacion (0 a 5 por item)

Por cada prompt generado:
1. Centro tonal estable (drone/tonica perceptible).
2. Color modal correcto (grados caracteristicos del modo).
3. Ausencia de modulacion no deseada.
4. Coherencia estilistica general.
5. Claridad de diferencia frente a otros modos.

Score total por clip: suma 5 criterios (max 25).
Score por checkpoint: media de todos los clips.

Score automatico auxiliar:
- `auto_score_0_1`: media ponderada de las metricas automaticas, ya normalizada entre 0 y 1.
- `auto_score_1_5`: conversion historica del mismo valor a escala 1-5 para comparacion con registros previos.

## 8) Plantilla de registro (copiar en hoja de calculo)

Columnas sugeridas:
- run_id
- epoch_ckpt
- mode_prompt
- seed
- tonal_center_score
- modal_color_score
- no_modulation_score
- style_coherence_score
- inter_mode_contrast_score
- total_score
- notas

## 9) Criterios de parada temprana

Detener un run si ocurre alguna:
- Estancamiento de score en 3 checkpoints consecutivos.
- Deterioro claro de contraste modal tras un maximo local.
- Evidencia de sobreajuste sonoro (artefactos recurrentes) sin mejora modal.

## 10) Entregable para memoria del TFG

Incluir en resultados:
- Tabla de hiperparametros por run.
- Grafica score medio vs epoch por run.
- 1 ejemplo de audio por modo del mejor checkpoint final.
- Discusion: por que el mejor run gano (capacidad, regularizacion o LR).

## 11) Plan de contingencia (si no llega a objetivo)

Escalon 1 (bajo riesgo):
- Mantener run ganador y ampliar hasta 180 epochs solo si seguia subiendo score.

Escalon 2 (datos):
- Reetiquetar captions para hacer mas explicito el rasgo modal por modo.

Escalon 3 (arquitectura):
- Probar LoKR solo despues de cerrar LoRA baseline defendible.

## 12) Checklist diario rapido

- Dataset congelado y validado.
- Trigger fijo aplicado.
- Un solo cambio por run.
- Checkpoints evaluados con prompts y seeds fijas.
- Registro completo de scores y notas.
