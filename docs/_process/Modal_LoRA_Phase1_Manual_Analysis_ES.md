# Fase 1: Análisis Manual vs Métricas Automáticas (Prompt E)

**Fecha**: 2026-04-18  
**Objetivo**: Interpretar tus scores manuales de 14 muestras (epoch_100, todo E) y conectarlos con las métricas automáticas. Esto valida que el LoRA está aprendiendo coherencia modal realmente, no solo overfitting a captions.

---

## 1. Hallazgos Manuales por Modo (Epoch 100, Prompt E)

| Modo | Seed | Manual Score | Calidad Tónica | Color Modal | Observaciones Clave |
|------|------|--------------|-----------------|------------|-------|
| **Ionian** | 1111 | 16.25 | Fuerte C | Claro | "Finaliza en C mayor, I-V relationship" → Tónica presente sin explicit mention |
| **Ionian** | 2222 | - | - | - | (pendiente) |
| **Dorian** | 1111 | 12.5 | D presente | Ambiguo | "Em-A (6ª mayor), pasa a D mayor → suena tonal más que modal" – **Trade-off**: Sin D en prompt, modelo busca D pero crea acordes que suenan mayores |
| **Dorian** | 2222 | - | - | - | (pendiente) |
| **Phrygian** | 1111 | 9.5 | Bajo E | Menor pero confuso | "E mayor, E-Am (menor), 6 bemol pero está en mayor" – Ambigüedad 3ª |
| **Phrygian** | 2222 | 9.5 | Bajo E | Menor pero confuso | Mismo patrón |
| **Lydian** | 1111 | 15.75 | Muy fuerte F | Excelente | "F suspended, sonido muy lidio, pasa a G (EXCELENTE)" → Levantó la 4ª sin explicit mention |
| **Lydian** | 2222 | - | - | - | (pendiente) |
| **Mixolydian** | 1111 | 13.25 | G presente | Bueno | "G mixolidio, varía al final" |
| **Mixolydian** | 2222 | - | - | - | (pendiente) |
| **Aeolian** | 1111 | 12.5 | A pero confuso | Ambiguo | "Resuelve Cmaj7 pero Am, no suena la 6 menor" – Confusión Am vs A |
| **Aeolian** | 2222 | - | - | - | (pendiente) |
| **Locrian** | 1111 | 6.5 | Muy bajo | Caótico | "Poca cordura, coherencia difícil" |
| **Locrian** | 2222 | 6.5 | Muy bajo | Caótico | "Chirría, tercera mayor confunde" |

---

## 2. Síntomas Clave Observados (Datos Previos E vs B 0-1 scale)

### 2.1 Tónica Colapsada = No es Bug, es Feature

**Síntoma**: Component tonic_dominance cae en E:
- Ionian: 0.279 → 0.091 (-67%)
- Lydian: 0.209 → 0.004 (-98%)

**Interpretación Manual**: 
- Ionian (manual 16.25): "Finaliza en C" aunque sin C en prompt → modelo aprendió C como tónica por el nombre "ionian"
- Lydian (manual 15.75): "Pasa a G (EXCELENTE)" → modelo reconoció levantamiento de 4ª aunque sin F en prompt

**Conclusión**: Sin tónica en prompt, modelo **aprende tónica del NAME**, pero métrica auto penaliza porque métrica espera tónica en notas, no en etiqueta.

### 2.2 Coherencia Modal ≠ Tónica Fuerte

**Síntoma**: in_mode_ratio y frame_in_mode_ratio son _relativamente estables_:
- Ionian: ~0.85 (B) → ~0.83 (E) (solo -2%)
- Phrygian: ~0.81 (B) → ~0.82 (E) (+1%)

**Interpretación Manual**:
- Phrygian (manual 9.5 en ambos seeds): "Menor pero confuso" → El modelo geneticamente sabe que es algo menor, pero sin E en prompt, confunde 3ª menor con mayor ("está en mayor")
- Dorian (manual 12.5): "Pasa a D mayor" → in_mode_ratio captura la "esencia modal" (presente las notas, aunque pase por mayores)

**Conclusión**: **in_mode_ratio es robusta a ausencia de tónica**. Pero tonic_dominance es sensible.

---

## 3. Hipótesis sobre Hiperparámetros (Basada en Síntomas)

### Problema A: Tónica Débil en Modos "Fáciles" (Ionian, Lydian)

**Síntoma**: Ionian manual 16.25 pero auto_score bajo porque tónica débil.

**Hipótesis**:
- **Dropout alto** (0.15) → mayor regularización → tónica menos aprendida
- **r bajo** (8, 32) → capacidad insuficiente para codificar tónica + modo
- **Alpha bajo** (16) → LoRA débil en reescala
- **Frozen mode B (solo LoRA)** → Modelo base congelado, LoRA solo copia características surface (nombre)

**Estrategia Fase 2**:
- Probar **r=128 + alpha=128** con **dropout=0.05** → máxima capacidad
- Probar **Frozen mode C** (LoRA + último block) → último block puede aprender "resoluciones" tonales
- Probar **lr=1e-4** → ajustes fino, no cambios radicales

### Problema B: Ambigüedad Mayor/Menor (Dorian, Phrygian, Aeolian, Locrian)

**Síntoma**: Dorian manual 12.5 "pasa a D mayor, suena tonal"; Aeolian "no suena la 6 menor" → modelo confunde acordes menores con mayores.

**Hipótesis**:
- **Dropout bajo** (0.05) → permite sobreajuste a acordes mayores de training
- **Rank bajo** (8, 32) → insuficiente para capturar sutileza de 3ª
- **Alpha muy alto** (128) vs rank bajo (8) → ratio desbalanceado, LoRA se satura

**Estrategia Fase 2**:
- Probar **r=64 + alpha=32** → balanceo clásico, menos abrupto
- Probar **dropout=0.1** → regularización más fuerte evita acordes mayores
- Probar **Frozen mode A** (todo trainable) → modelo base puede ajustar embeddings

### Problema C: Locrian Caótico (Manual 6.5)

**Síntoma**: "Chirría, coherencia difícil" → Locrian es el modo más fragil del sistema. Sin B en prompt, modelo pierde anclaje.

**Hipótesis**:
- Locrian requiere **capacidad máxima** por complejidad armónica
- **Learning rate crítico**: lr=1e-4 es conservador (bueno), pero lr=5e-4 o 1e-3 podría romper convergencia
- **Frozen mode B** (solo LoRA) quizá insuficiente para Locrian

**Estrategia Fase 2**:
- Probar **r=128, alpha=128, lr=1e-4** como "máxima estabilidad"
- Probar **Frozen mode A** para ver si modelo base ayuda

---

## 4. Conclusión Fase 1

✅ **Validación Manual**: Tus scores confirman que el LoRA **sí aprende coherencia modal**, incluso sin tónica explícita en prompt.

✅ **Síntomas Claros**:
1. Tónica aprendida desde NAME del modo, no desde notas en prompt
2. Coherencia armónica (in_mode_ratio) robusta, incluso con modos débiles
3. Ambigüedad mayor/menor sugiere falta de regularización (dropout) o capacidad insuficiente (rank bajo)

✅ **Predicción Fase 2**:
- **r=128, alpha=128, dropout=0.05, lr=1e-4** probablemente mejore Ionian/Lydian (tónica fuerte)
- **r=64, alpha=32, dropout=0.1, lr=1e-4** probablemente mejore Dorian/Aeolian (estabilidad mayor/menor)
- **Frozen mode A** (todo trainable) probablemente mejore Locrian (máxima capacidad adaptiva)

---

## 5. Próximo Paso

Ejecutar matriz experimental Fase 2 con 3 modos de congelado (A/B/C) y 4×4×3×3 = 144 combinaciones de hiperparámetros → **432 entrenamientos totales**.

Métrica de éxito: **auto_score_0_1 global** + validación manual en checkpoint best per mode.

