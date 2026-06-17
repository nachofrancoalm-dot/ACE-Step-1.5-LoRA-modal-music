"""
Genera las figuras para la sección de modos diatónicos de la memoria.
Uso: uv run python scripts/generate_mode_figures.py
Salidas en memoria/figures/
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

OUT = Path("memoria/figures")
OUT.mkdir(exist_ok=True)

# ── Paleta ───────────────────────────────────────────────────────────────────
C_WHITE_KEY = "#FFFFFF"   # teclas blancas del piano
C_SCALE_DOT = "#2E6DB8"   # círculo indicador de nota en escala
C_ROOT_DOT  = "#1F3E6E"   # círculo indicador de tónica
C_BLACK_KEY = "#2C2C2C"   # teclas negras
C_T         = "#1A7A3E"   # tono (T) — verde oscuro
C_S         = "#B03020"   # semitono (S) — rojo oscuro
# Cuadrícula de modos
C_ROOT      = "#1F3E6E"   # azul oscuro — tónica
C_HIGHLIGHT = "#6CA0D4"   # azul medio — notas del modo
C_CHARAC    = "#E07B00"   # naranja oscuro — nota característica
C_GRID_BG   = "#F5F5F5"   # fondo de celda neutro


# ════════════════════════════════════════════════════════════════════════════
# FIGURA 1: Teclado con escala de Do mayor + marcas T/S
# ════════════════════════════════════════════════════════════════════════════
def draw_piano():
    fig, ax = plt.subplots(figsize=(13, 4.0))
    ax.set_xlim(-0.3, 7.3)
    ax.set_ylim(-0.5, 1.75)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("#FAFAFA")

    white_notes = ["C", "D", "E", "F", "G", "A", "B"]
    KEY_H = 1.10   # altura de tecla blanca

    # ── Teclas blancas coloreadas según pertenencia a la escala ─────────────
    for i, note in enumerate(white_notes):
        if note == "C":
            fc      = "#1F3E6E"   # azul oscuro — tónica
            tc      = "white"
            border  = "#0D2050"
        else:
            fc      = "#D0E4F7"   # azul muy claro — nota de la escala
            tc      = "#1F3E6E"
            border  = "#7AAAD4"

        rect = FancyBboxPatch(
            (i + 0.03, 0.0), 0.94, KEY_H,
            boxstyle="round,pad=0.03",
            linewidth=1.6, edgecolor=border, facecolor=fc, zorder=1
        )
        ax.add_patch(rect)

        # Nombre de nota en la parte inferior de la tecla
        ax.text(i + 0.50, 0.13, note,
                ha="center", va="bottom",
                fontsize=14, fontweight="bold", color=tc,
                fontfamily="monospace", zorder=3)

    # ── Teclas negras (encima, sin colorear) ─────────────────────────────────
    black_positions = [0.64, 1.64, 3.64, 4.64, 5.64]
    for bx in black_positions:
        rect = FancyBboxPatch(
            (bx, 0.44), 0.72, 0.66,
            boxstyle="round,pad=0.02",
            linewidth=0.8, edgecolor="#111111", facecolor=C_BLACK_KEY, zorder=2
        )
        ax.add_patch(rect)

    # ── Marcas T / S encima del teclado ──────────────────────────────────────
    intervals = ["T", "T", "S", "T", "T", "T", "S"]
    colors_iv = [C_T, C_T, C_S, C_T, C_T, C_T, C_S]
    y_line  = KEY_H + 0.20   # y de la línea horizontal
    y_tick  = KEY_H + 0.12   # y de los pequeños ticks verticales
    y_label = KEY_H + 0.34   # y de la etiqueta T/S

    for i in range(6):
        col   = colors_iv[i]
        x_mid = i + 1.0
        x_l   = i + 0.50   # centro de la tecla izquierda
        x_r   = i + 1.50   # centro de la tecla derecha

        # Línea horizontal entre los dos centros de tecla
        ax.plot([x_l, x_r], [y_line, y_line], color=col, lw=2.0, solid_capstyle="round")
        # Ticks verticales en los extremos
        for xv in (x_l, x_r):
            ax.plot([xv, xv], [y_tick, y_line], color=col, lw=1.6)
        # Etiqueta
        ax.text(x_mid, y_label, intervals[i],
                ha="center", va="bottom", fontsize=13,
                fontweight="bold", color=col)

    # Último S (B→siguiente Do, fuera del rango visible)
    ax.text(7.12, y_label, "(S)", ha="left", va="bottom",
            fontsize=10, color=C_S, fontstyle="italic")

    # ── Leyenda ──────────────────────────────────────────────────────────────
    root_patch  = mpatches.Patch(facecolor="#1F3E6E", edgecolor="#0D2050",
                                 linewidth=1, label="Tónica (Do)")
    scale_patch = mpatches.Patch(facecolor="#D0E4F7", edgecolor="#7AAAD4",
                                 linewidth=1, label="Notas de la escala")
    black_patch = mpatches.Patch(facecolor=C_BLACK_KEY, edgecolor="#111",
                                 linewidth=1, label="Notas fuera de la escala (teclas negras)")
    t_patch     = mpatches.Patch(color=C_T, label="T = tono (2 semitonos)")
    s_patch     = mpatches.Patch(color=C_S, label="S = semitono (1 semitono)")
    ax.legend(handles=[root_patch, scale_patch, black_patch, t_patch, s_patch],
              loc="lower left", fontsize=9.5, framealpha=0.97,
              bbox_to_anchor=(0.0, -0.04), ncol=2)

    ax.set_title("Escala de Do mayor (jónico): C · D · E · F · G · A · B",
                 fontsize=14, fontweight="bold", pad=12, color="#1F3E6E")

    fig.tight_layout()
    path = OUT / "modos_teclado_c_mayor.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close(fig)
    print(f"Guardado: {path}")


# ════════════════════════════════════════════════════════════════════════════
# FIGURA 2: Cuadrícula de los 7 modos — patrón T/S + nota característica
# ════════════════════════════════════════════════════════════════════════════
MODES = [
    # (nombre, notas, patrón T/S, grado_caracteristico 0-based, descripcion)
    ("Jónico",    ["C","D","E","F","G","A","B"], ["T","T","S","T","T","T","S"], None,  "Mayor natural — estable, brillante"),
    ("Dórico",    ["D","E","F","G","A","B","C"], ["T","S","T","T","T","S","T"],    5,  "Menor con 6ª mayor — melancolía jazz"),
    ("Frigio",    ["E","F","G","A","B","C","D"], ["S","T","T","T","S","T","T"],    1,  "Menor con 2ª menor — flamenco, misterio"),
    ("Lidio",     ["F","G","A","B","C","D","E"], ["T","T","T","S","T","T","S"],    3,  "Mayor con 4ª aumentada — etéreo, fantástico"),
    ("Mixolidio", ["G","A","B","C","D","E","F"], ["T","T","S","T","T","S","T"],    6,  "Mayor con 7ª menor — blues, rock"),
    ("Eólico",    ["A","B","C","D","E","F","G"], ["T","S","T","T","S","T","T"], None,  "Menor natural — oscuro, clásico"),
    ("Locrio",    ["B","C","D","E","F","G","A"], ["S","T","T","S","T","T","T"],    4,  "Disminuido — muy inestable"),
]

def draw_modes_grid():
    n_modes = 7
    n_cols  = 7
    fig, ax = plt.subplots(figsize=(14, 6.2))
    ax.set_xlim(-3.1, n_cols + 0.2)
    ax.set_ylim(-0.7, n_modes + 1.0)
    ax.axis("off")
    fig.patch.set_facecolor("#FAFAFA")

    cell_w, cell_h = 0.96, 0.78
    col_xs = np.arange(n_cols) * 1.0

    # Encabezados de grado
    for j in range(n_cols):
        ax.text(col_xs[j] + cell_w / 2, n_modes + 0.60,
                f"Grado {j+1}", ha="center", va="center",
                fontsize=9, color="#444", fontstyle="italic")

    for i, (name, notes, pattern, charac, desc) in enumerate(MODES):
        row_y = n_modes - 1 - i

        # Nombre y descripción del modo
        ax.text(-0.10, row_y + cell_h / 2 + 0.12, name,
                ha="right", va="center", fontsize=12,
                fontweight="bold", color="#1F3E6E")
        ax.text(-0.10, row_y + cell_h / 2 - 0.18, desc,
                ha="right", va="center", fontsize=8,
                color="#555", fontstyle="italic")

        for j, (note, intv) in enumerate(zip(notes, pattern)):
            cx = col_xs[j]

            # Color de fondo de celda
            if j == 0:
                fc        = C_ROOT       # azul oscuro — tónica
                text_col  = "white"
                border    = "#0D2A50"
            elif j == charac:
                fc        = C_CHARAC     # naranja — nota característica
                text_col  = "white"
                border    = "#A05800"
            else:
                fc        = C_GRID_BG    # gris claro — resto
                text_col  = "#222222"
                border    = "#BBBBBB"

            rect = FancyBboxPatch(
                (cx + 0.03, row_y + 0.06), cell_w, cell_h,
                boxstyle="round,pad=0.04",
                linewidth=1.2, edgecolor=border, facecolor=fc, zorder=2
            )
            ax.add_patch(rect)

            # Nombre de nota
            ax.text(cx + cell_w / 2 + 0.03, row_y + cell_h / 2 + 0.14,
                    note, ha="center", va="center",
                    fontsize=12, fontweight="bold", color=text_col, zorder=3)

            # Intervalo (T/S) debajo del nombre
            intv_col = C_T if intv == "T" else C_S
            ax.text(cx + cell_w / 2 + 0.03, row_y + cell_h / 2 - 0.16,
                    intv, ha="center", va="center",
                    fontsize=9, fontweight="bold", color=intv_col, zorder=3)

    # Leyenda
    handles = [
        mpatches.Patch(color=C_ROOT,    label="Tónica del modo"),
        mpatches.Patch(color=C_CHARAC,  label="Nota característica"),
        mpatches.Patch(color=C_GRID_BG, label="Resto de notas",
                       edgecolor="#BBBBBB", linewidth=1),
        mpatches.Patch(color=C_T,       label="T = tono"),
        mpatches.Patch(color=C_S,       label="S = semitono"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9.5,
              framealpha=0.97, ncol=3, bbox_to_anchor=(1.0, -0.06))

    ax.set_title(
        "Los 7 modos diatónicos derivados de la escala de Do mayor\n"
        "Mismas notas, distinto punto de partida → distinto patrón de intervalos",
        fontsize=13, fontweight="bold", color="#1F3E6E", pad=10
    )

    fig.tight_layout()
    path = OUT / "modos_grid_intervalos.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close(fig)
    print(f"Guardado: {path}")


# ════════════════════════════════════════════════════════════════════════════
# FIGURA 3: Comparativa Jónico vs Dórico — mismo Do mayor, distinta raíz
# ════════════════════════════════════════════════════════════════════════════
def draw_ionian_dorian_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.4))
    fig.patch.set_facecolor("#FAFAFA")
    fig.suptitle(
        "Mismas notas, distinto carácter: Do jónico vs Re dórico",
        fontsize=14, fontweight="bold", color="#1F3E6E", y=1.03
    )

    pairs = [
        ("Do jónico  (C mayor)",
         ["C","D","E","F","G","A","B"],
         ["T","T","S","T","T","T","S"],
         None,
         "Brillante · estable · mayor"),
        ("Re dórico",
         ["D","E","F","G","A","B","C"],
         ["T","S","T","T","T","S","T"],
         5,
         "Melancólico · jazz · menor con esperanza"),
    ]

    for ax, (title, notes, pattern, charac, desc) in zip(axes, pairs):
        ax.set_xlim(-0.3, 7.3)
        ax.set_ylim(-0.15, 1.55)
        ax.axis("off")
        ax.set_facecolor("#FAFAFA")
        ax.set_title(title, fontsize=12, fontweight="bold",
                     color="#1F3E6E", pad=8)
        ax.text(3.5, 1.44, desc, ha="center", va="center",
                fontsize=9.5, color="#444", fontstyle="italic")

        for j, (note, intv) in enumerate(zip(notes, pattern)):
            if j == 0:
                fc       = C_ROOT
                tc       = "white"
                border   = "#0D2A50"
            elif j == charac:
                fc       = C_CHARAC
                tc       = "white"
                border   = "#A05800"
            else:
                fc       = C_GRID_BG
                tc       = "#222222"
                border   = "#BBBBBB"

            rect = FancyBboxPatch(
                (j + 0.05, 0.60), 0.90, 0.62,
                boxstyle="round,pad=0.04",
                linewidth=1.4, edgecolor=border, facecolor=fc, zorder=2
            )
            ax.add_patch(rect)
            ax.text(j + 0.50, 0.93, note, ha="center", va="center",
                    fontsize=14, fontweight="bold", color=tc, zorder=3)

            # Intervalo con línea bracket
            if j < 6:
                intv_col = C_T if intv == "T" else C_S
                y_ln = 0.32
                y_tk = 0.24
                ax.plot([j + 0.50, j + 1.50], [y_ln, y_ln],
                        color=intv_col, lw=2.0, solid_capstyle="round", zorder=3)
                for xv in (j + 0.50, j + 1.50):
                    ax.plot([xv, xv], [y_tk, y_ln],
                            color=intv_col, lw=1.5, zorder=3)
                ax.text(j + 1.0, 0.08, intv, ha="center", va="bottom",
                        fontsize=11, fontweight="bold", color=intv_col, zorder=3)

        # Etiqueta nota característica
        if charac is not None:
            ax.annotate(
                "nota característica\ndel modo",
                xy=(charac + 0.50, 1.22),
                xytext=(charac + 0.50, 1.38),
                ha="center", fontsize=8.5, color=C_CHARAC, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=C_CHARAC, lw=1.8)
            )

    # Leyenda compartida
    handles = [
        mpatches.Patch(color=C_ROOT,    label="Tónica"),
        mpatches.Patch(color=C_CHARAC,  label="Nota característica"),
        mpatches.Patch(color=C_GRID_BG, label="Resto de notas",
                       edgecolor="#BBBBBB", linewidth=1),
        mpatches.Patch(color=C_T,       label="T = tono"),
        mpatches.Patch(color=C_S,       label="S = semitono"),
    ]
    fig.legend(handles=handles, loc="lower center", fontsize=9.5,
               ncol=5, framealpha=0.97, bbox_to_anchor=(0.5, -0.10))

    fig.tight_layout()
    path = OUT / "modos_comparativa_ionico_dorico.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close(fig)
    print(f"Guardado: {path}")


if __name__ == "__main__":
    draw_piano()
    draw_modes_grid()
    draw_ionian_dorian_comparison()
    print("Todas las figuras generadas.")
