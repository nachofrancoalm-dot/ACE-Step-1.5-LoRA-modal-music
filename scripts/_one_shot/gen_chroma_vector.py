"""Genera figura explicativa de un vector de chroma para Re dórico.

El vector de chroma tiene 12 posiciones (una por clase de altura del sistema
temperado: C, C#, D, D#, E, F, F#, G, G#, A, A#, B). Cada valor refleja la
proporción de energía sonora en esa clase de altura, promediada sobre todo el
audio. Se muestran tres capas:
  - tónica del modo (D): color principal UPM
  - notas diatónicas del modo Re dórico (E, F, G, A, B, C): color acento
  - notas no diatónicas (C#, D#, F#, G#, A#): color gris tenue
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "memoria", "figures",
                   "chroma_vector_dorico.png")
OUT = os.path.abspath(OUT)

# 12 pitch classes en orden cromático
LABELS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Re dórico: D(índice 2), E(4), F(5), G(7), A(9), B(11), C(0)
TONIC_IDX   = 2   # D
DIATONIC_IDX = {0, 2, 4, 5, 7, 9, 11}   # C D E F G A B

# Valores simulados de energía: diatónicas altas, no diatónicas muy bajas.
# Se usa una distribución realista: tónica domina, resto de diatónicas
# moderadas, cromatismos mínimos (el modelo no los evita al 100%).
np.random.seed(42)
values = np.array([
    0.72,   # C  — diatónica, no tónica
    0.03,   # C# — no diatónica
    0.91,   # D  — tónica
    0.04,   # D# — no diatónica
    0.61,   # E  — diatónica
    0.55,   # F  — diatónica
    0.02,   # F# — no diatónica
    0.67,   # G  — diatónica
    0.03,   # G# — no diatónica
    0.58,   # A  — diatónica
    0.03,   # A# — no diatónica
    0.48,   # B  — diatónica (6ª mayor: nota característica del dórico)
])

# Colores
RED_UPM  = "#C8102E"
BLUE_DIA = "#4a86c8"   # azul para otras diatónicas
GRAY_ND  = "#c8c8c8"   # gris para no diatónicas

colors = []
for i, lbl in enumerate(LABELS):
    if i == TONIC_IDX:
        colors.append(RED_UPM)
    elif i in DIATONIC_IDX:
        colors.append(BLUE_DIA)
    else:
        colors.append(GRAY_ND)

fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
fig.patch.set_facecolor("white")

x = np.arange(len(LABELS))
bars = ax.bar(x, values, width=0.68, color=colors, edgecolor="white",
              linewidth=0.7, zorder=3)

# Anotación especial en la nota característica (B = índice 11)
B_IDX = 11
ax.annotate("6ª mayor\n(nota\ncaracterística)",
            xy=(B_IDX, values[B_IDX]),
            xytext=(B_IDX - 2.2, values[B_IDX] + 0.12),
            fontsize=7.5, color="#1a1a1a",
            arrowprops=dict(arrowstyle="-|>", color="#555", lw=0.9),
            ha="center", va="bottom")

# Anotación en la tónica
ax.annotate("tónica (D)",
            xy=(TONIC_IDX, values[TONIC_IDX]),
            xytext=(TONIC_IDX + 1.6, values[TONIC_IDX] + 0.04),
            fontsize=7.5, color=RED_UPM, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=RED_UPM, lw=0.9),
            ha="left", va="center")

ax.set_xticks(x)
ax.set_xticklabels(LABELS, fontsize=10)
ax.set_ylabel("Energía relativa (normalizada)", fontsize=10)
ax.set_xlabel("Clase de altura", fontsize=10)
ax.set_ylim(0, 1.12)
ax.set_xlim(-0.6, 11.6)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
ax.set_axisbelow(True)

# Leyenda
patch_tonic  = mpatches.Patch(color=RED_UPM,   label="Tónica del modo (D)")
patch_dia    = mpatches.Patch(color=BLUE_DIA,  label="Nota diatónica (E F G A B C)")
patch_ndia   = mpatches.Patch(color=GRAY_ND,   label="Nota no diatónica")
ax.legend(handles=[patch_tonic, patch_dia, patch_ndia],
          fontsize=8.5, loc="upper right", framealpha=0.9)

ax.set_title("Vector de chroma — Re dórico (ejemplo simulado)",
             fontsize=11, pad=10)

# Plantilla binaria diatónica como contorno
template = np.array([1 if i in DIATONIC_IDX else 0 for i in range(12)], dtype=float)
ax.step(np.arange(-0.5, 12.5), np.append(template, template[-1]) * 0.98,
        where="post", color="#555", lw=1.0, ls=":", alpha=0.55,
        label="plantilla diatónica")

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
plt.close(fig)
print("Guardado:", OUT)
