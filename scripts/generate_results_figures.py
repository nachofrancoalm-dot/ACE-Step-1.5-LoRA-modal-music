"""Genera figuras de resultados para la memoria. Uso: uv run python scripts/generate_results_figures.py"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT = Path("memoria/figures")
OUT.mkdir(exist_ok=True)

AZUL_OSCURO  = "#1F3E6E"
AZUL_CLARO   = "#6CA0D4"
NARANJA      = "#E07B00"
GRIS_FONDO   = "#F5F5F5"


def draw_runD_confirm_comparison():
    modos   = ["Jónico", "Dórico", "Frigio", "Lidio",
               "Mixolidio", "Eólico", "Locrio", "Global"]
    run_d   = [89, 59, 55, 78, 60, 57, 38, 62]
    confirm = [81, 71, 47, 55, 53, 50, 33, 56]

    x      = np.arange(len(modos))
    width  = 0.36

    fig, ax = plt.subplots(figsize=(13, 5.5))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    bars_d = ax.bar(x - width / 2, run_d,   width, label="run_D (referencia)",
                    color=AZUL_OSCURO, edgecolor="white", linewidth=0.6, zorder=3)
    bars_c = ax.bar(x + width / 2, confirm, width, label="Confirmación (r=32, α=32, lr=10⁻⁴)",
                    color=AZUL_CLARO,  edgecolor="white", linewidth=0.6, zorder=3)

    bars_d[1].set_edgecolor(NARANJA)
    bars_d[1].set_linewidth(2.5)
    bars_c[1].set_edgecolor(NARANJA)
    bars_c[1].set_linewidth(2.5)
    bars_c[1].set_color(NARANJA)

    for bar in list(bars_d) + list(bars_c):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                f"{int(h)}%", ha="center", va="bottom",
                fontsize=8.5, color="#333")

    ax.annotate(
        "+12 pp\nen dórico",
        xy=(x[1] + width / 2, 71),
        xytext=(x[1] + width / 2 + 0.55, 77),
        fontsize=9.5, fontweight="bold", color=NARANJA,
        arrowprops=dict(arrowstyle="-|>", color=NARANJA, lw=1.6),
        ha="left"
    )

    ax.axhline(62, color=AZUL_OSCURO, linestyle="--", linewidth=1.2,
               alpha=0.5, zorder=2)
    ax.text(7.55, 63, "62 % (run_D global)", fontsize=8.5,
            color=AZUL_OSCURO, va="bottom", alpha=0.8)

    ax.axvline(len(modos) - 1.5, color="#CCCCCC", linestyle=":", linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(modos, fontsize=11)
    ax.set_ylabel("Puntuación (% del máximo, escala 0-20)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    legend_patch_d = mpatches.Patch(color=AZUL_OSCURO, label="run_D — referencia (r=64, α=128, lr=5·10⁻⁵)")
    legend_patch_c = mpatches.Patch(color=AZUL_CLARO,  label="Confirmación — config óptima de la búsqueda (r=32, α=32, lr=10⁻⁴)")
    legend_patch_h = mpatches.Patch(color=NARANJA,     label="Modo dórico — única mejora significativa (+12 pp)")
    ax.legend(handles=[legend_patch_d, legend_patch_c, legend_patch_h],
              fontsize=9.5, loc="upper right", framealpha=0.95)

    ax.set_title(
        "Evaluación manual por modo: run_D vs experimento de confirmación\n"
        "Puntuación normalizada (% del máximo de 20 puntos)",
        fontsize=12, fontweight="bold", color=AZUL_OSCURO, pad=10
    )

    fig.tight_layout()
    path = OUT / "comparativa_runD_confirm.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close(fig)
    print(f"Guardado: {path}")



def draw_sweep_all_experiments():
    import csv

    rows = []
    with open("hparam_sweep/stage1_results.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "exp_id": int(r["exp_id"]),
                "lr":     float(r["lr"]),
                "score":  float(r["auto_score_0_1"]),
                "rank":   float(r["rank"]),
                "alpha":  float(r["alpha"]),
                "anomaly": float(r["training_duration_sec"]) < 2000,
            })

    rows.sort(key=lambda x: x["score"], reverse=True)

    labels  = [f"Exp_{r['exp_id']:02d}" for r in rows]
    scores  = [r["score"] for r in rows]
    colors  = [AZUL_OSCURO if r["lr"] < 5e-4 else "#C0392B" for r in rows]
    hatches = ["///" if r["anomaly"] else "" for r in rows]

    fig, ax = plt.subplots(figsize=(7, 10))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    y = np.arange(len(rows))
    bars = ax.barh(y, scores, color=colors, edgecolor="white",
                   linewidth=0.5, height=0.72, zorder=3)

    for bar, h in zip(bars, hatches):
        if h:
            bar.set_hatch(h)
            bar.set_edgecolor("#888")

    for bar, score in zip(bars, scores):
        ax.text(score + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{score:.4f}", va="center", fontsize=7.5, color="#333")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, fontfamily="monospace")
    ax.set_xlabel("auto_score_0_1", fontsize=11)
    ax.set_xlim(0.660, 0.733)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()

    cutoff = next(i for i, r in enumerate(rows) if r["lr"] >= 5e-4)
    ax.axhline(cutoff - 0.5, color="#999", linestyle="--", linewidth=1.2)
    ax.text(0.6615, cutoff - 0.5 - 0.25, "← lr = 10⁻⁴",
            fontsize=8, color=AZUL_OSCURO, va="top")
    ax.text(0.6615, cutoff - 0.5 + 0.25, "← lr = 10⁻³",
            fontsize=8, color="#C0392B", va="bottom")

    p1 = mpatches.Patch(color=AZUL_OSCURO, label="lr = 10⁻⁴  (media 0.721, σ = 0.003)")
    p2 = mpatches.Patch(color="#C0392B",   label="lr = 10⁻³  (media 0.693, σ = 0.010)")
    p3 = mpatches.Patch(facecolor="white", edgecolor="#888", hatch="///",
                        label="Exp_19 — duración anómala (800 s)")
    ax.legend(handles=[p1, p2, p3], fontsize=9, loc="lower right",
              framealpha=0.97)

    ax.set_title(
        "Búsqueda de hiperparámetros: 24 experimentos\n"
        "ordenados por auto_score_0_1",
        fontsize=12, fontweight="bold", color=AZUL_OSCURO, pad=10
    )

    fig.tight_layout()
    path = OUT / "sweep_all_experiments.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close(fig)
    print(f"Guardado: {path}")


def draw_auto_vs_manual_top5():
    """Slope chart: cada experimento es una línea de auto_rank → manual_rank."""
    exps        = ["Exp_01", "Exp_02", "Exp_09", "Exp_10", "Exp_18"]
    auto_vals   = [0.7244,   0.7244,   0.7243,   0.7241,   0.7241]
    manual_vals = [8.48,     8.48,     8.66,     8.70,     9.89]
    auto_rank   = [1, 2, 3, 4.2, 3.8]   # Exp_10/Exp_18 en empate real, separados visualmente
    manual_rank = [5, 4, 3, 2, 1]

    styles = {
        "Exp_01": {"color": "#C0392B", "lw": 2.8, "zorder": 5, "alpha": 1.0},
        "Exp_18": {"color": NARANJA,   "lw": 2.8, "zorder": 5, "alpha": 1.0},
        "Exp_02": {"color": "#AAAAAA", "lw": 1.4, "zorder": 3, "alpha": 0.8},
        "Exp_09": {"color": "#AAAAAA", "lw": 1.4, "zorder": 3, "alpha": 0.8},
        "Exp_10": {"color": "#AAAAAA", "lw": 1.4, "zorder": 3, "alpha": 0.8},
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    X_AUTO, X_MAN = 0, 1

    for exp, ar, mr, av, mv in zip(exps, auto_rank, manual_rank,
                                   auto_vals, manual_vals):
        st = styles[exp]
        ax.plot([X_AUTO, X_MAN], [ar, mr],
                color=st["color"], lw=st["lw"],
                zorder=st["zorder"], alpha=st["alpha"],
                solid_capstyle="round")

        for xp, yr in [(X_AUTO, ar), (X_MAN, mr)]:
            ax.scatter(xp, yr, color=st["color"], s=80,
                       zorder=st["zorder"] + 1, alpha=st["alpha"])

        ax.text(X_AUTO - 0.06, ar,
                f"{exp}  {av:.4f}",
                ha="right", va="center", fontsize=9.5,
                color=st["color"], fontweight="bold" if st["lw"] > 2 else "normal")

        ax.text(X_MAN + 0.06, mr,
                f"{mv:.2f}/20  {exp}",
                ha="left", va="center", fontsize=9.5,
                color=st["color"], fontweight="bold" if st["lw"] > 2 else "normal")

    for xp, label in [(X_AUTO, "Ranking automático\n(auto_score_0_1)"),
                      (X_MAN,  "Ranking manual\n(evaluación experta)")]:
        ax.text(xp, 0.55, label, ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=AZUL_OSCURO)

    ax.text(-0.55, 1, "1º (mejor)", ha="center", va="center",
            fontsize=8.5, color="#555", fontstyle="italic")
    ax.text(-0.55, 5, "5º (peor)",  ha="center", va="center",
            fontsize=8.5, color="#555", fontstyle="italic")

    ax.set_xlim(-0.7, 1.7)
    ax.set_ylim(0.1, 5.7)
    ax.invert_yaxis()
    ax.axis("off")

    for xp in (X_AUTO, X_MAN):
        ax.axvline(xp, color="#CCCCCC", lw=1.0, zorder=0)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="#C0392B", lw=2.8, label="Exp_01 — 1º auto, 5º manual"),
        Line2D([0], [0], color=NARANJA,   lw=2.8, label="Exp_18 — 4º auto, 1º manual"),
        Line2D([0], [0], color="#AAAAAA", lw=1.4, label="Resto de experimentos"),
    ]
    ax.legend(handles=handles, fontsize=9, loc="lower center",
              bbox_to_anchor=(0.5, -0.04), framealpha=0.97, ncol=3)

    ax.set_title(
        "Divergencia entre ranking automático y evaluación manual\n"
        "Top-5 experimentos de la búsqueda de hiperparámetros",
        fontsize=12, fontweight="bold", color=AZUL_OSCURO, pad=14
    )

    fig.tight_layout()
    path = OUT / "auto_vs_manual_top5.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close(fig)
    print(f"Guardado: {path}")


if __name__ == "__main__":
    draw_runD_confirm_comparison()
    draw_sweep_all_experiments()
    draw_auto_vs_manual_top5()
