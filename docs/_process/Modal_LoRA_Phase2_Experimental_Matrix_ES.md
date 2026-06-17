# Fase 2: Matriz Experimental Profunda de Hiperparámetros

**Fecha**: 2026-04-18  
**Objetivo**: Buscar combinación óptima de hiperparámetros para maximizar coherencia modal a través de 3 estrategias de entrenamiento (frozen modes A/B/C).

---

## 1. Variables Experimentales

### 1.1 Frozen Modes (Estrategia de Congelado)

```
A = Todo trainable:
    - Modelo DiT base: trainable
    - LoRA: trainable
    - Interpretación: Máxima flexibilidad, riesgo de overfitting

B = Solo LoRA (LoRA mode):
    - Modelo DiT base: congelado (frozen_layer_type = 'all')
    - LoRA: trainable
    - Interpretación: Estándar LoRA puro, máxima eficiencia VRAM

C = LoRA + Último Block DiT:
    - Modelo DiT base: congelado excepto 2 últimos bloques
    - LoRA: trainable
    - Interpretación: Trade-off: algo de capacidad sin descontrol
```

### 1.2 Rank (r)

Valores: **[8, 32, 64, 128]**

Lógica:
- r=8: Minimal (baseline), bajo overhead VRAM
- r=32: Conservative stability
- r=64: Recomendado en literatura LoRA
- r=128: Máxima capacidad para modos complejos (Locrian)

### 1.3 Alpha

Valores: **[16, 32, 64, 128]**

Lógica:
- alpha / r = scaling factor
- 16/8 = 2.0; 32/32 = 1.0; 64/64 = 1.0; 128/128 = 1.0
- Valores bajos (16) = menos peso LoRA sobre base
- Valores altos (128) = LoRA domina

### 1.4 Learning Rate (lr)

Valores: **[1e-4, 5e-4, 1e-3]**

Lógica:
- 1e-4: Muy conservador, fine-tuning fino
- 5e-4: Moderado, ajustes medios
- 1e-3: Agresivo, riesgo ruptura pero faster convergence

### 1.5 Dropout

Valores: **[0.05, 0.1, 0.15]**

Lógica:
- 0.05: Minimal (baseline)
- 0.1: Moderate regularization, reduce overfitting
- 0.15: Strong, puede hurt con training reducido

---

## 2. Diseño de Matriz

### 2.1 Total de Experimentos

```
Frozen Modes: 3 (A, B, C)
Rank values: 4 (8, 32, 64, 128)
Alpha values: 4 (16, 32, 64, 128)
LR values: 3 (1e-4, 5e-4, 1e-3)
Dropout values: 3 (0.05, 0.1, 0.15)

Total: 3 × 4 × 4 × 3 × 3 = 432 entrenamientos
```

### 2.2 Iteración Sugerida (Staged Approach)

**Stage 1** (rápido): Solo Frozen Mode B (LoRA puro) con subset de valores
```
r: [32, 64, 128]      (3 valores)
alpha: [32, 128]      (2 valores)
lr: [1e-4, 1e-3]      (2 valores)
dropout: [0.05, 0.1]  (2 valores)

Subtotal Stage 1: 3 × 2 × 2 × 2 = 24 entrenamientos
```

Métrica: auto_score_0_1 global. Si encontramos claro ganador, avanzar a Stage 2.

**Stage 2** (mediano): Frozen Modes B + C + A pero solo con top-3 combinations de (r, alpha, lr, dropout) del Stage 1
```
Top 3 combinations from Stage 1
Frozen Modes: A, B, C (3 modos)
lr: [1e-4] o [1e-4, 5e-4] según results

Subtotal Stage 2: 3 × 3 × 2 ≈ 18 entrenamientos
```

**Stage 3** (exhaustivo): Full 432 si Stage 2 aún muestra variabilidad alta

### 2.3 Configuración de Training Fija

```python
# Fijo para todos los experimentos
optimizer: adamw
scheduler: cosine
epochs: 140
batch_size: 4
save_every: 20  # checkpoints: 20, 40, 60, 80, 100, 120, 140
seed: 1111 (usar fijo para reproducibilidad)
model_variant: turbo (inferencia rápida)
num_inference_steps: 8
guidance_scale: 4.0
```

---

## 3. Matriz Detallada (Stage 1 - LoRA Puro)

| Exp# | Frozen | r | alpha | lr | dropout | Descripción |
|------|--------|---|-------|-----|---------|-------------|
| 1-8 | B | 32 | 32 | 1e-4 | 0.05-0.1 | Conservative baseline |
| 9-16 | B | 32 | 128 | 1e-4 | 0.05-0.1 | High alpha, low lr |
| 17-24 | B | 64 | 64 | 1e-4 | 0.05-0.1 | Classical LoRA (r=alpha) |
| 25-32 | B | 128 | 128 | 1e-4 | 0.05-0.1 | Maximum capacity, low lr |
| 33-40 | B | 32 | 32 | 1e-3 | 0.05-0.1 | High lr, conservative |
| 41-48 | B | 64 | 64 | 1e-3 | 0.05-0.1 | Classical + high lr |
| 49-56 | B | 128 | 128 | 1e-3 | 0.05-0.1 | Max capacity + high lr |

*Expandir con todas las combinaciones en implementación*

---

## 4. Métricas a Registrar (por experimento)

Para CADA experimento, registrar en CSV:

```
exp_id,frozen_mode,rank,alpha,lr,dropout,
epoch_checkpoint,
auto_score_0_1_global,
modal_accuracy,
in_mode_ratio,
tonal_stability,
tonic_dominance,
template_similarity,
duration_hours
```

Esto permite post-analysis de trade-offs: ej "¿r=256 mejora tonic_dominance pero baja in_mode_ratio?"

---

## 5. Hipótesis Específicas a Validar

### H1: Frozen Mode B suficiente para la mayoría de modos

**Predicción**: Mode B (solo LoRA) + r=64, alpha=64, lr=1e-4, dropout=0.05 compite con A y C.  
**Justificación**: LoRA demostró eficiencia en literatura, VRAM más eficiente.

### H2: Modos fuertes (Ionian, Lydian) requieren alpha/lr bajo

**Predicción**: r=64, alpha=64 con lr=1e-4 mejora tónica; lr=1e-3 la daña.  
**Justificación**: Fase 1 mostró tónica presente en manual pero métrica colapsada; hiperparámetro agresivo puede romper.

### H3: Modos débiles (Phrygian, Locrian) requieren dropout > 0.05

**Predicción**: r=128, dropout=0.1 mejora Phrygian/Locrian; dropout=0.05 permite overfitting a acordes mayores.  
**Justificación**: Fase 1 mostró ambigüedad mayor/menor; regularización frenará eso.

### H4: Frozen Mode A mejora modos débiles pero hurt storage

**Predicción**: Mode A + r=64 > Mode B + r=128 para Locrian, pero Mode A requiere 2x checkpoints.  
**Justificación**: Locrian es frágil; capacidad adicional del modelo base podría ayudar.

---

## 6. Cronograma Estimado

Asumiendo:
- ~15 min por entrenamiento (140 epochs, batch=4, num_inference_steps=8)
- ~2 min evaluación post-training
- GPU: NVIDIA A100 o similar

**Stage 1 (24 experimentos)**:
- Secuencial: 24 × 17 min = ~6.8 horas → en paralelo con GPU múltiples: ~2 horas
- **Tiempo total Stage 1**: ~2-3 horas

**Stage 2 (18 experimentos)**:
- ~2.5 × 17 = ~42 min en paralelo

**Stage 3 (360+ experimentos)** (opcional):
- ~40+ horas en paralelo o varios días en secuencial

**Recomendación**: Ejecutar Stage 1 este fin de semana, analizar resultados lunes, decidir Stage 2/3.

---

## 7. Success Criteria

✅ **Criterio Cuantitativo**:
- auto_score_0_1 global ≥ 0.75 (vs current ~0.71-0.735 en E)
- Modal accuracy ≥ 90% (vs current ~86%)
- Per-mode variance < 0.05 (coherencia entre modos)

✅ **Criterio Cualitativo** (manual validation subset):
- Mínimo 5/7 modos con color modal reconocible (target: 7/7)
- Diferenciación clara entre Ionian/Lydian y Dorian/Aeolian
- Locrian con "coherencia suficiente para TFG"

✅ **Criterio de Reproducibilidad**:
- Same results ± 0.01 en auto_score_0_1 con 2 seeds distintos
- No crashes o memory errors

---

## 8. Next Action

→ Ejecutar `modal_hparam_sweep_stage1.py` (script automatizado, ver archivo siguiente)

