"""Compute success-rate rankings from the manual evaluation rubric.

Reads ``docs/manual_scoring_updated.csv`` and, for each rubric criterion plus
the total score, produces:

* CSV rankings of experiments (``Run + Checkpoint``) and per-mode breakdowns
  (``Run + Checkpoint + Modo``), each with mean score and percentage of audios
  exceeding two thresholds (permissive and strict).
* Bar charts (top-10 experiments) comparing both thresholds per criterion.
* A heatmap of the strict success rate per experiment x criterion.

Outputs are written under ``docs/manual_eval_success/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "docs" / "manual_eval_Scoring.csv"
OUTPUT_DIR = ROOT / "docs" / "manual_eval_success"
FIGURES_DIR = OUTPUT_DIR / "figures"

# (column, human label, permissive threshold, strict threshold)
CRITERIA = [
    ("Centro_tonal_1_5", "Centro tonal", 2.0, 3.0),
    ("Color_modal_1_5", "Color modal", 2.0, 3.0),
    ("No_modulacion_1_5", "No modulacion", 2.0, 3.0),
    ("Coherencia_1_5", "Coherencia", 2.0, 3.0),
    ("Total_0_20", "Total (0-20)", 10.0, 12.0),
]

TOP_N = 10


def load_scores() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV, sep=";", decimal=",")
    df["Experimento"] = df["Run"].astype(str) + " | " + df["Checkpoint"].astype(str)
    return df


def aggregate(df: pd.DataFrame, group_cols: list[str], column: str,
              perm: float, strict: float) -> pd.DataFrame:
    sub = df[group_cols + [column]].dropna(subset=[column]).copy()
    grouped = sub.groupby(group_cols, dropna=False)[column]
    out = grouped.agg(
        n_audios="count",
        media="mean",
    ).reset_index()
    pct_perm = grouped.apply(lambda s: (s >= perm).mean() * 100.0).reset_index(
        name="pct_exito_permisivo"
    )
    pct_strict = grouped.apply(lambda s: (s >= strict).mean() * 100.0).reset_index(
        name="pct_exito_estricto"
    )
    out = out.merge(pct_perm, on=group_cols).merge(pct_strict, on=group_cols)
    out["media"] = out["media"].round(2)
    out["pct_exito_permisivo"] = out["pct_exito_permisivo"].round(1)
    out["pct_exito_estricto"] = out["pct_exito_estricto"].round(1)
    return out.sort_values(
        ["pct_exito_estricto", "pct_exito_permisivo", "media"], ascending=False
    ).reset_index(drop=True)


def safe_slug(label: str) -> str:
    return (
        label.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
        .replace("/", "_")
    )


def plot_top_experiments(ranking: pd.DataFrame, label: str, perm: float,
                         strict: float, output_path: Path) -> None:
    top = ranking.head(TOP_N).iloc[::-1]
    y = np.arange(len(top))
    height = 0.4

    fig, ax = plt.subplots(figsize=(10, 0.55 * len(top) + 1.5))
    ax.barh(y - height / 2, top["pct_exito_permisivo"], height=height,
            label=f"Permisivo (>= {perm:g})", color="#6FA8DC")
    ax.barh(y + height / 2, top["pct_exito_estricto"], height=height,
            label=f"Estricto (>= {strict:g})", color="#1F4E79")

    ax.set_yticks(y)
    ax.set_yticklabels(top["Experimento"], fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% audios sobre umbral")
    ax.set_title(f"Top {len(top)} experimentos - {label}")
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_heatmap(rankings: dict[str, pd.DataFrame], output_path: Path) -> None:
    pivot = None
    for label, ranking in rankings.items():
        col = ranking[["Experimento", "pct_exito_estricto"]].rename(
            columns={"pct_exito_estricto": label}
        )
        pivot = col if pivot is None else pivot.merge(col, on="Experimento", how="outer")

    if pivot is None or pivot.empty:
        return

    pivot = pivot.set_index("Experimento")
    pivot["__order__"] = pivot.mean(axis=1, skipna=True)
    pivot = pivot.sort_values("__order__", ascending=False).drop(columns="__order__")
    pivot = pivot.head(TOP_N)

    fig, ax = plt.subplots(figsize=(1.4 * len(pivot.columns) + 3, 0.45 * len(pivot) + 2))
    data = pivot.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if np.isnan(value):
                continue
            color = "white" if value >= 55 else "black"
            ax.text(j, i, f"{value:.0f}", ha="center", va="center",
                    color=color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("% exito estricto")
    ax.set_title(f"Top {len(pivot)} experimentos - % exito estricto por criterio")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_scores()
    rankings_main: dict[str, pd.DataFrame] = {}

    for column, label, perm, strict in CRITERIA:
        if column not in df.columns:
            print(f"[skip] columna no encontrada: {column}")
            continue

        slug = safe_slug(label)

        main_ranking = aggregate(
            df, ["Run", "Checkpoint", "Experimento"], column, perm, strict
        )
        main_csv = OUTPUT_DIR / f"ranking_{slug}_run_checkpoint.csv"
        main_ranking.drop(columns="Experimento").to_csv(main_csv, index=False)

        mode_ranking = aggregate(
            df, ["Run", "Checkpoint", "Modo"], column, perm, strict
        )
        mode_csv = OUTPUT_DIR / f"ranking_{slug}_run_checkpoint_modo.csv"
        mode_ranking.to_csv(mode_csv, index=False)

        figure_path = FIGURES_DIR / f"top{TOP_N}_{slug}.png"
        plot_top_experiments(main_ranking, label, perm, strict, figure_path)

        rankings_main[label] = main_ranking
        print(f"[ok] {label}: {main_csv.name}, {mode_csv.name}, {figure_path.name}")

    heatmap_path = FIGURES_DIR / f"heatmap_top{TOP_N}_estricto.png"
    plot_heatmap(rankings_main, heatmap_path)
    print(f"[ok] heatmap: {heatmap_path.name}")


if __name__ == "__main__":
    main()
