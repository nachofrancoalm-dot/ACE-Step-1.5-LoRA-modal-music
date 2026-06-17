#!/usr/bin/env python3
"""Post-Stage1 analysis: rankings, sensitivity heatmaps, and Stage 2 recommendations."""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from loguru import logger
except ModuleNotFoundError:
    logger = logging.getLogger("modal_hparam_analysis")
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
        )

def load_results(results_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    logger.info(f"Loaded {len(df)} experiments from {results_csv}")
    return df


def get_top_experiments(
    df: pd.DataFrame, metric: str = "auto_score_0_1", top_n: int = 10
) -> pd.DataFrame:
    successful = df[df["training_status"] == "SUCCESS"].copy()
    if len(successful) == 0:
        logger.warning("No successful experiments found!")
        return pd.DataFrame()

    successful[metric] = pd.to_numeric(successful[metric], errors="coerce")
    top = successful.nlargest(top_n, metric)
    logger.info(f"Top {top_n} experiments by {metric}:")
    for idx, row in top.iterrows():
        logger.info(
            f"  #{idx+1}: {row['frozen_mode']}-r{row['rank']}-a{row['alpha']}-"
            f"lr{row['lr']}-do{row['dropout']} → {metric}={row[metric]:.4f}"
        )
    return top


def generate_sensitivity_summary(df: pd.DataFrame) -> dict[str, Any]:
    successful = df[df["training_status"] == "SUCCESS"].copy()
    successful["auto_score_0_1"] = pd.to_numeric(
        successful["auto_score_0_1"], errors="coerce"
    )

    summary = {}
    for param in ["rank", "alpha", "lr", "dropout"]:
        grouped = successful.groupby(param)["auto_score_0_1"].agg(
            ["mean", "std", "count"]
        )
        logger.info(f"\nSensitivity to {param}:")
        logger.info(grouped)
        summary[f"{param}_effect"] = grouped.to_dict()

    return summary


def generate_heatmap_data(
    df: pd.DataFrame, x_param: str, y_param: str, z_metric: str = "auto_score_0_1"
) -> dict:
    successful = df[df["training_status"] == "SUCCESS"].copy()
    successful[z_metric] = pd.to_numeric(successful[z_metric], errors="coerce")

    pivot = successful.pivot_table(
        index=y_param, columns=x_param, values=z_metric, aggfunc="mean"
    )

    logger.info(f"\nHeatmap: {x_param} × {y_param} → {z_metric}")
    logger.info(pivot)

    return pivot.to_dict()


def export_top_experiments_csv(
    top_df: pd.DataFrame, output_file: Path
) -> None:
    top_df.to_csv(output_file, index=False)
    logger.info(f"Exported top experiments to: {output_file}")


def export_sensitivity_json(summary: dict, output_file: Path) -> None:
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Exported sensitivity analysis to: {output_file}")


def plot_heatmap(
    heatmap_dict: dict,
    title: str,
    output_file: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        pivot = pd.DataFrame(heatmap_dict)
        plt.figure(figsize=(8, 6))
        plt.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
        plt.colorbar()
        plt.xlabel(pivot.columns.name)
        plt.ylabel(pivot.index.name)
        plt.title(title)
        plt.xticks(range(len(pivot.columns)), pivot.columns)
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()
        logger.info(f"Saved heatmap to: {output_file}")
    except ImportError:
        logger.warning("matplotlib not available; skipping heatmap plots")


def plot_sensitivity_lines(
    df: pd.DataFrame,
    param: str,
    metric: str = "auto_score_0_1",
    output_file: Path = None,
) -> None:
    try:
        import matplotlib.pyplot as plt

        successful = df[df["training_status"] == "SUCCESS"].copy()
        successful[metric] = pd.to_numeric(successful[metric], errors="coerce")

        grouped = successful.groupby(param)[metric].agg(["mean", "std"])

        plt.figure(figsize=(10, 6))
        plt.plot(grouped.index, grouped["mean"], "o-", markersize=8, linewidth=2)
        plt.fill_between(
            grouped.index,
            grouped["mean"] - grouped["std"],
            grouped["mean"] + grouped["std"],
            alpha=0.3,
        )
        plt.xlabel(param)
        plt.ylabel(f"{metric} (mean ± std)")
        plt.title(f"Sensitivity to {param}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150)
            logger.info(f"Saved sensitivity plot to: {output_file}")
        else:
            plt.show()
        plt.close()
    except ImportError:
        logger.warning("matplotlib not available; skipping sensitivity plots")


def analyze_results(
    results_csv: Path,
    output_dir: Path,
    metric: str = "auto_score_0_1",
    top_n: int = 10,
    plot_type: str = "all",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")

    df = load_results(results_csv)

    logger.info(f"\n{'='*80}")
    logger.info(f"Total experiments: {len(df)}")
    logger.info(f"Successful: {len(df[df['training_status'] == 'SUCCESS'])}")
    logger.info(f"Failed: {len(df[df['training_status'] == 'FAILED'])}")

    top_df = get_top_experiments(df, metric=metric, top_n=top_n)
    if len(top_df) > 0:
        export_top_experiments_csv(
            top_df, output_dir / f"best_experiments_top{top_n}.csv"
        )

    sensitivity = generate_sensitivity_summary(df)
    export_sensitivity_json(sensitivity, output_dir / "sensitivity_analysis.json")

    if plot_type in ["all", "heatmap"]:
        hm_rank_alpha = generate_heatmap_data(
            df, x_param="rank", y_param="alpha", z_metric=metric
        )
        plot_heatmap(
            hm_rank_alpha,
            f"{metric} by Rank × Alpha",
            output_dir / "heatmap_rank_alpha.png",
        )

        hm_lr_dropout = generate_heatmap_data(
            df, x_param="dropout", y_param="lr", z_metric=metric
        )
        plot_heatmap(
            hm_lr_dropout,
            f"{metric} by LR × Dropout",
            output_dir / "heatmap_lr_dropout.png",
        )

    if plot_type in ["all", "sensitivity"]:
        for param in ["rank", "alpha", "lr", "dropout"]:
            plot_sensitivity_lines(
                df,
                param=param,
                metric=metric,
                output_file=output_dir / f"sensitivity_{param}.png",
            )

    if len(top_df) > 0:
        best_row = top_df.iloc[0]
        logger.info(f"\nWinner Configuration:")
        logger.info(
            f"  Mode: {best_row['frozen_mode']}, "
            f"r={best_row['rank']}, "
            f"alpha={best_row['alpha']}, "
            f"lr={best_row['lr']}, "
            f"dropout={best_row['dropout']}"
        )
        logger.info(f"  Metric: {metric}={best_row[metric]:.4f}")
        logger.info(
            f"  Training Duration: {best_row['training_duration_sec']} seconds"
        )

        logger.info(
            f"\nRecommendation: "
            f"Repeat this config in Frozen Modes A and C, "
            f"or refine range if variance is high."
        )

    logger.info(f"Analysis complete. Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Stage 1 Hyperparameter Sweep Results"
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        required=True,
        help="Path to stage1_results.csv from sweep",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="auto_score_0_1",
        help="Metric to rank by (default: auto_score_0_1)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top experiments to export (default: 10)",
    )
    parser.add_argument(
        "--plot-type",
        type=str,
        choices=["all", "heatmap", "sensitivity", "none"],
        default="all",
        help="Which plots to generate (default: all)",
    )

    args = parser.parse_args()

    analyze_results(
        results_csv=args.results_csv,
        output_dir=args.output_dir,
        metric=args.metric,
        top_n=args.top_n,
        plot_type=args.plot_type,
    )


if __name__ == "__main__":
    main()
