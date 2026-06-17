# 📊 RESUMEN EJECUTIVO: Fase 2 - Búsqueda de Hiperparámetros LoRA Modal

## Contexto Rápido (Lee en 30 sec)

**Antes**:  
Tu analysis manual mostró que prompt E genera audio modal coherente _incluso sin tónica explícita_. Pero métricas automáticas colapsaron la tónica en ciertos modos. ¿Bug en la métrica? ¿Hiperparámetros del LoRA no óptimos?

**Ahora**:  
Abordamos esto científicamente: **búsqueda sistematizada de hiperparámetros** (24-432 entrenamientos) para encontrar configuración LoRA que maximice coherencia modal objetivo en todas las dimensiones (tónica, color armónico, diferenciación inter-modo).

---

## Qué Se Entregó

### 📝 Documentación (5 guías completas)

| # | Documento | Qué Hacerr | Cuándo |
|----|-----------|---------|--------|
| 1 | **Phase1_Manual_Analysis** | Lee tus síntomas & hipótesis conectadas | Orientación |
| 2 | **Phase2_Experimental_Matrix** | Entender grid completo (3 frozen × 4r × 4α × 3lr × 3do) | Referencia |
| 3 | **Phase2_QuickStart** ⭐ | Comandos exactos para ejecutar Stage 1, 2, 3 | **PRIMERO** |
| 4 | **Phase2_Execution_Tracker** ⭐ | Checklist + decision tree + manual validation template | **DURANTE** |
| 5 | **Phase2_Visual_Overview** | Flowchart ASCII de las dos fases + metrics expected | Orientación |

### 🐍 Scripts Automatizados (2 nuevos)

| Script | Función | Líneas | Cuándo |
|--------|---------|--------|--------|
| `modal_hparam_sweep_stage1.py` | Ejecuta 24 entrenamientos automáticamente | ~450 LOC | Hoy |
| `modal_hparam_analysis.py` | Parsea CSV, ranking top-N, heatmaps, sensitivity | ~350 LOC | Después Stage 1 |

---

## Flujo 3 Fases

```
FASE 1 ✅ (Completada, tu análisis manual)
  ↓
  Síntomas: Tónica débil en Ionian/Lydian; ambigüedad mayor/menor en menores
  Hipótesis: Hiperparámetros (r, alpha, lr, dropout, frozen_mode) inóptimos
  
FASE 2 ⏳ (Listo para ejecutar, este fin de semana)
  ├─ STAGE 1 (2-4 horas): 24 combos Frozen Mode B
  │  Output: stage1_results.csv (auto_score_0_1 por config)
  │
  ├─ ANÁLISIS (30 min): Decision tree + heatmaps
  │  → IF auto_score_0_1 ≥ 0.77 → Stage 2A (simple)
  │  → ELSE → Stage 2B (refine) o DEBUG
  │
  ├─ STAGE 2A/B (~1-3 horas, conditional)
  │  Output: stage2_results.csv (validación en Frozen A/C)
  │
  └─ MANUAL VALIDATION (1-2 horas, critical)
     Generar audios, escuchar, puntuar → Comparar vs baseline
  
FASE 3 (Redacción, ~1 hora)
  Reporte final + presentación al profesor
```

---

## Timeline

| Item | Duración | Cuando |
|------|----------|--------|
| Setup checklist | 10 min | Hoy (ahora) |
| **Stage 1 ejecución** | **2-4 horas** | **Hoy/mañana** |
| Análisis + decision | 30 min | Después Stage 1 |
| Stage 2 (si aplica) | 1-3 horas | Sábado? |
| Manual validation | 1-2 horas | Sábado/domingo |
| Redacción final | 1 hora | Domingo |
| **TOTAL (esperado)** | **6-10 horas** | **This weekend** |

---

## Métrica de Éxito

| Métrica | Baseline E | Target | Status |
|---------|-----------|--------|--------|
| auto_score_0_1 | 0.735 | ≥ 0.765 | Esperado +4% |
| modal_accuracy | 86% | ≥ 88% | Esperado +2-4% |
| Tu manual score | 85.75 pts | ≥ 92 pts | Esperado +6-10 pts |
| Modos coherentes | 5/7 | 6-7/7 | Target: 7/7 |
| Reproducible | N/A | ±0.02 en metrics | Required |

---

## 🚀 ACCIÓN INMEDIATA

### 1️⃣ Verifica Setup (5 min)

Linux (Bash):

```bash
# En terminal (PWD = raíz del proyecto)
source .venv/bin/activate
python --version  # Debe ser 3.11 o 3.12

# Instalar dependencias del proyecto (recomendado)
uv sync

# Si no usas uv, instala al menos lo mínimo para estos scripts
pip install -r requirements.txt

# Verifica dataset
ls ./preprocessed_tensors/modal_tfg/
# Debe listar: aeolian, dorian, ionian, locrian, lydian, mixolydian, phrygian

# Verifica GPU
nvidia-smi
```

Windows PowerShell:

```powershell
# En terminal (PWD = raíz del proyecto)
& .\.venv\Scripts\Activate.ps1
python --version  # Debe ser 3.11 o 3.12

# Verifica dataset
ls ./preprocessed_tensors/modal_tfg/
# Debe listar: aeolian, dorian, ionian, locrian, lydian, mixolydian, phrygian

# Verifica GPU
nvidia-smi
# Debe mostrar GPU disponible con ≥2GB libre
```

### 2️⃣ Lee Guía Rápida (5 min)

Ve a: `docs/Modal_LoRA_Phase2_QuickStart_ES.md`

Lee hasta el título "Fase 2.1: Stage 1"

### 3️⃣ Ejecuta Stage 1 (2-4 horas)

Linux (Bash):

```bash
python scripts/modal_hparam_sweep_stage1.py \
  --dataset-dir ./preprocessed_tensors/modal_tfg \
  --output-dir ./hparam_sweep_stage1_results \
  --seed 1111 \
  --parallel 1
```

Windows PowerShell:

```powershell
# Copiar exacto:
python scripts/modal_hparam_sweep_stage1.py `
    --dataset-dir ./preprocessed_tensors/modal_tfg `
    --output-dir ./hparam_sweep_stage1_results `
    --seed 1111 `
    --parallel 1
```

Se verá así:
```
================================================================================
STAGE 1 HYPERPARAMETER SWEEP START
Output directory: ./hparam_sweep_stage1_results
================================================================================
Generated 24 experiment combinations
[1/24] Experiment: {'frozen_mode': 'B', 'rank': 32, 'alpha': 32, 'lr': 1e-4, 'dropout': 0.05}
Starting: python train.py fixed ...
Training SUCCESS | Duration: 943.2s | Exit code: 0
Running inference for ./hparam_sweep_stage1_results/exp_001_B_r32_a32_lr1e-04_do0.05...
...
[24/24] ...
STAGE 1 SWEEP COMPLETE
Total experiments: 24
Successful: 24
Total duration: 3.5 hours
Results: ./hparam_sweep_stage1_results/stage1_results.csv
```

### 4️⃣ Ejecuta Análisis (30 min)

Cuando Stage 1 termina:

Linux (Bash):

```bash
python scripts/modal_hparam_analysis.py \
  --results-csv ./hparam_sweep_stage1_results/stage1_results.csv \
  --output-dir ./hparam_analysis_stage1 \
  --plot-type all
```

Windows PowerShell:

```powershell
python scripts/modal_hparam_analysis.py `
    --results-csv ./hparam_sweep_stage1_results/stage1_results.csv `
    --output-dir ./hparam_analysis_stage1 `
    --plot-type all
```

Abre CSV generado: `hparam_analysis_stage1/best_experiments_top10.csv`

### 5️⃣ Usa Decision Tree (30 min)

Ve a: `docs/Modal_LoRA_Phase2_Execution_Tracker_ES.md › Fase 2.3: Decision Tree`

Sigue lógica:
- ¿auto_score_0_1[best] ≥ 0.77? → Sí → **Stage 2A** (rápido, solo A/C)
- ¿Entre 0.75-0.77 y resultados mixtos? → Sí → **Stage 2B** (refine grid)
- ¿< 0.75? → Sí → **DEBUG** (verificar 1 sample manual)

### 6️⃣ Manual Validation (1-2 horas)

Si Stage 1 OK, genera audios con mejor config:

```powershell
# Reemplaza exp_NNN con mejor experiment_id de CSV
python scripts/modal_batch_infer.py `
    --output-dir ./hparam_sweep_stage1_results/exp_NNN_* `
    --prompts-block E `
    --num-samples 56
```

Escucha los 14 archivos `.wav` generados (7 modes × 2 seeds) → Puntúa en template (Tracker_ES.md)

### 7️⃣ Redacta Conclusión (1 hora)

Después de manual validation:

Abre: `docs/Modal_LoRA_Phase2_Final_Report_ES.md` (template)

Completa con:
- Configuración ganadora
- Métricas finales
- Comparación manual E vs Winner
- Conclusión para profesor

---

## Red Flags

| Problema | Solución |
|----------|----------|
| Stage 1 cuelga o OOM | Reduce batch_size=2 en train.py, re-run |
| auto_score_0_1 < 0.730 | Escucha 1 sample manual, verifica dataset |
| Top 3 configs muy distintos | Variabilidad alta → Stage 2B (refine) |
| Locrian sigue "chirría" | Limitación del modelo; aceptar + documentar |
| Tónica sigue débil en Ionian | Probar r=128 + alpha=128 en Stage 2 |

---

## Archivos Por Leer (En Orden)

| # | Archivo | Tipo | Tiempo | Imprescindible? |
|----|---------|------|--------|---|
| 1 | **Phase2_QuickStart_ES.md** | Guía | 5 min | ✅ YES |
| 2 | **Phase2_Execution_Tracker_ES.md** | CheckList | 10 min | ✅ YES |
| 3 | Phase1_Manual_Analysis_ES.md | Contexto | 10 min | ⭕ Recommended |
| 4 | Phase2_Experimental_Matrix_ES.md | Detalle | 15 min | ⭕ Recommended |
| 5 | Phase2_Visual_Overview_ES.md | Diagrama | 5 min | ⭕ Optional |

---

## Comandos Clave (Copy-Paste Ready)

### Setup
```powershell
& .\.venv\Scripts\Activate.ps1
```

### Stage 1
```powershell
python scripts/modal_hparam_sweep_stage1.py --dataset-dir ./preprocessed_tensors/modal_tfg --output-dir ./hparam_sweep_stage1_results --seed 1111 --parallel 1
```

### Análisis
```powershell
python scripts/modal_hparam_analysis.py --results-csv ./hparam_sweep_stage1_results/stage1_results.csv --output-dir ./hparam_analysis_stage1 --plot-type all
```

### Manual Validation (After decision)
```powershell
python scripts/modal_batch_infer.py --output-dir ./hparam_sweep_stage1_results/exp_XXX_* --prompts-block E --num-samples 56
```

---

## FAQ Rápido

**P: ¿Puedo correr múltiples Stage 1 en paralelo?**  
R: Script actual es serial. Si tienes 3+ GPUs, copia el script y ajusta output_dir para cada GPU.

**P: ¿Qué pasa si training se cuelga en mitad?**  
R: Mata el proceso (Ctrl+C), reduce batch_size a 2, re-run. CSV se guardó (no se pierde prog).

**P: ¿Debo escuchar TODAS las muestras de todos los checkpoints?**  
R: No. Solo usa final checkpoint (epoch 140) × 7 modes × 2 seeds = 14 samples.

**P: ¿Si Stage 1 no muestra mejora, abandono?**  
R: No. Stage 2B refina rangos. O Stage 3 si prof exige profundidad máxima.

**P: ¿Puedo modificar el grid sin cambiar script?**  
R: Edita líneas 34-40 en `modal_hparam_sweep_stage1.py` (STAGE1_GRID dict).

---

## Contacto / Soporte

Si algo está roto:
1. Verifica GPU (`nvidia-smi`)
2. Verifica dataset (`ls ./preprocessed_tensors/modal_tfg/`)
3. Lee logs: `tail -f ./hparam_sweep_stage1_results/sweep_stage1.log`
4. Verifica 1 manual sample manualmente
5. Escala: Reduce batch_size, num_inference_steps, o número de modos si es prueba

---

## ¡LISSSSTO! 🚀

**Próximo paso**: Lee QuickStart, verifica setup, ¡corre Stage 1 AHORA!

Tiempo total del proyecto: 6-10 horas este fin de semana → **Tesis con resultados sólidos** 📊

¡A por ello!

