# Contribuciones del TFG — LoRA Modal para ACE-Step 1.5

Este repositorio es una bifurcación de [ACE-Step 1.5](https://github.com/ace-step/ACE-Step)
desarrollada como Proyecto Fin de Grado (Grado en Ciencia de Datos e Inteligencia Artificial,
UPM ETSISI, 2026).

**Autor:** Ignacio Franco Almendárez  
**Tutor:** Francisco Serradilla  
**Título:** *LoRA modal para control de modos musicales en ACE-Step 1.5*

---

## Qué se ha añadido respecto al original

### Ficheros nuevos (contribución original)

| Ruta | Descripción |
|------|-------------|
| `scripts/modal_batch_infer.py` | Inferencia por lotes para evaluación sistemática de checkpoints |
| `scripts/modal_eval_pipeline.py` | Pipeline de métricas automáticas (auto_score_0_1) |
| `scripts/modal_hparam_sweep_stage1.py` | Búsqueda sistemática de 24 configuraciones de hiperparámetros |
| `scripts/modal_hparam_analysis.py` | Análisis y visualización de resultados de la búsqueda |
| `scripts/rank_checkpoints_per_experiment.py` | Ranking determinista de checkpoints por experimento |
| `scripts/prepare_manual_eval_shortlist.py` | Selección de candidatos para evaluación manual |
| `scripts/prepare_modes_dataset.py` | Construcción del dataset modal (105 clips, 7 modos) |
| `scripts/regenerate_top_audios.py` | Regeneración de audios de los mejores checkpoints |
| `scripts/_one_shot/` | Scripts auxiliares de un solo uso (recálculo, backfill, generación de xlsx, orquestación de la segunda tanda); conservados por trazabilidad |
| `prompts_modal_sound.txt` | Protocolo de evaluación: prompts fijos por modo (Bloques A–E) |
| `memoria/` | Memoria del TFG en LaTeX |
| `acestep/llm_inference_runtime_sanitize_test.py` | Tests para sanitización de runtime |
| `test_modal_eval_pipeline.py` | Tests del pipeline de evaluación automática |

### Ficheros modificados del original ACE-Step

| Fichero | Cambio principal |
|---------|-----------------|
| `acestep/core/generation/handler/lora/lifecycle.py` | Corrección de sanitización de runtime CUDA antes de inyección PEFT; soporte LoKr con DoRA |
| `acestep/core/generation/handler/lora/lifecycle_test.py` | Tests adicionales para los cambios anteriores |
| `acestep/llm_inference.py` | Sanitización de estado GPU antes de inferencia LLM |
| `acestep/training/lora_injection.py` | Manejo defensivo de hooks para modelos DiT |
| `acestep/training/lokr_utils.py` | Utilidades para inyección de adaptadores LoKr |
| `acestep/training/dataset_builder_modules/preprocess_vae.py` | Mejoras en preprocesado VAE para dataset modal |
| `acestep/ui/gradio/events/training/lora_training.py` | Ajustes en eventos de entrenamiento LoRA |
| `acestep/ui/gradio/events/training/lokr_training.py` | Soporte LoKr en la UI |
| `acestep/ui/gradio/events/wiring/training_run_wiring.py` | Cableado de eventos actualizado |
| `acestep/ui/gradio/events/wiring/training_lokr_wiring.py` | Cableado LoKr en UI |
| `pyproject.toml` | Dependencias añadidas: pandas, matplotlib, seaborn, openpyxl |

---

## Cómo reproducir el trabajo

### 1. Instalar dependencias
```bash
uv sync
```

### 2. Preparar el dataset modal
```bash
uv run python scripts/prepare_modes_dataset.py --help
```

### 3. Entrenar un adaptador LoRA
Usar la interfaz Gradio (pestaña Training) o el script de orquestación.

### 4. Ejecutar la búsqueda de hiperparámetros
```bash
uv run python scripts/modal_hparam_sweep_stage1.py --help
```

### 5. Evaluar checkpoints
```bash
uv run python scripts/modal_batch_infer.py --help
uv run python scripts/modal_eval_pipeline.py --help
```

### 6. Analizar resultados
```bash
uv run python scripts/modal_hparam_analysis.py \
  --results-csv hparam_sweep/stage1_results.csv \
  --output-dir hparam_sweep/analysis_output
```

---

## Entorno experimental

Los entrenamientos e inferencias se ejecutaron en una máquina GPU externa
proporcionada por el tutor (acceso SSH desde Windows mediante PuTTY y transferencia
de ficheros mediante WinSCP). La máquina local se empleó únicamente para
desarrollo de scripts, edición de la memoria y análisis de resultados.
