"""Audit and consolidate all manual-evaluation audios into a single tree.

For every row in ``docs/manual_eval_Scoring.csv`` (Run + Checkpoint + Modo + Seed)
this script:

1. Searches the configured source folders for the matching ``.flac`` file.
2. Copies the file into ``audios_unificados/<Run>/<Checkpoint>/<modo>_seed<NNNN>.flac``.
3. Records the outcome (OK / FALTA) in ``docs/audio_audit.csv``.

It also lists source folders that are NOT mapped to any CSV Run so they can be
reviewed/discarded by hand.
"""

from __future__ import annotations

import re
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "docs" / "manual_eval_Scoring.csv"
OUT_DIR = ROOT / "audios_unificados"
AUDIT_CSV = ROOT / "docs" / "audio_audit.csv"

RUN_TO_FOLDERS: dict[str, list[Path]] = {
    "confirm_r32_a32_lr1e-4": [ROOT / "lora_output/confirm_r32_a32_lr1e-4/generated_audio"],
    "r128_a256_d005": [ROOT / "gradio_outputs/modal_eval_r128_a256_d005"],
    "r64_a128_d0": [ROOT / "gradio_outputs/modal_eval_r64_a128_d0"],
    "r64_a128_d005": [ROOT / "gradio_outputs/modal_eval_r64_a128_d005"],
    "r64_a128_d005_lr5e-5": [ROOT / "gradio_outputs/modal_eval_r64_a128_d005_lr5e-5"],
    "RUN_D": [ROOT / "gradio_outputs/eval_D"],
    "Prompt solo modo": [ROOT / "gradio_outputs/eval_E_smoke_fixed"],
    "Prompt solo modo + instrumento": [ROOT / "gradio_outputs/modal_run_D_eval_audio"],
    "Exp_01": [
        ROOT / "hparam_sweep/exp_001_B_r32_a32_lr1e-04_do0.05/generated_audio",
        ROOT / "manual_eval_shortlist/01_exp_001_B_r32_a32_lr1e-04_do0.05",
    ],
    "Exp_02": [
        ROOT / "hparam_sweep/exp_002_B_r32_a32_lr1e-04_do0.1/generated_audio",
        ROOT / "manual_eval_shortlist/02_exp_002_B_r32_a32_lr1e-04_do0.1",
    ],
    "Exp_09": [ROOT / "manual_eval_shortlist/03_exp_009_B_r64_a32_lr1e-04_do0.05"],
    "Exp_10": [ROOT / "manual_eval_shortlist/04_exp_010_B_r64_a32_lr1e-04_do0.1"],
    "Exp_18": [
        ROOT / "manual_eval_shortlist/04_exp_018_B_r128_a32_lr1e-04_do0.1",
        ROOT / "manual_eval_shortlist/05_exp_018_B_r128_a32_lr1e-04_do0.1",
    ],
}

UNMAPPED_FOLDERS = [
    ROOT / "gradio_outputs/eval_B_smoke_fixed",
    ROOT / "gradio_outputs/eval_D_detailed_prompt",
    ROOT / "gradio_outputs/eval_E_mode_only_name",
    ROOT / "lora_output/r32_a_16_drop_01",
    ROOT / "lora_output/r8_a8_drop_01",
    ROOT / "lora_output/modal_run_D",
    ROOT / "lora_output/r128_a256_d005",
    ROOT / "lora_output/r64_a128_d0",
    ROOT / "lora_output/r64_a128_d005",
    ROOT / "lora_output/r64_a128_d005_lr5e-5",
]

FILENAME_RE = re.compile(
    r"^epoch_(?P<epoch>\d+)_loss_(?P<loss>[\d_.]+)_(?P<modo>[a-z]+)_seed(?P<seed>\d+)\.flac$"
)


def normalize_checkpoint(ckpt: str) -> str:
    """Normalize checkpoint names: some files use '_' as decimal separator (e.g. epoch_140_loss_0_1952)."""
    return ckpt.replace(".", "_")


def find_audio(run_folders: list[Path], ckpt: str, modo: str, seed: int) -> Path | None:
    target_norm = normalize_checkpoint(f"{ckpt}_{modo}_seed{seed}.flac")
    for folder in run_folders:
        if not folder.is_dir():
            continue
        for path in folder.glob("*.flac"):
            if normalize_checkpoint(path.name) == target_norm:
                return path
    return None


def main() -> None:
    df = pd.read_csv(CSV_PATH, sep=";", decimal=",")
    df = df[["Run", "Checkpoint", "Modo", "Seed"]].drop_duplicates()

    audit_rows = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    per_run_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"ok": 0, "missing": 0})

    for _, row in df.iterrows():
        run = row["Run"]
        ckpt = str(row["Checkpoint"]).strip()
        modo = str(row["Modo"]).strip()
        seed = int(row["Seed"])

        folders = RUN_TO_FOLDERS.get(run, [])
        src = find_audio(folders, ckpt, modo, seed)

        target_dir = OUT_DIR / run / ckpt
        target_path = target_dir / f"{modo}_seed{seed}.flac"

        if src is None:
            audit_rows.append({
                "Run": run, "Checkpoint": ckpt, "Modo": modo, "Seed": seed,
                "estado": "FALTA", "origen": "", "destino": str(target_path.relative_to(ROOT)),
            })
            per_run_stats[run]["missing"] += 1
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target_path)
        audit_rows.append({
            "Run": run, "Checkpoint": ckpt, "Modo": modo, "Seed": seed,
            "estado": "OK", "origen": str(src.relative_to(ROOT)),
            "destino": str(target_path.relative_to(ROOT)),
        })
        per_run_stats[run]["ok"] += 1

    audit_df = pd.DataFrame(audit_rows)
    AUDIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(AUDIT_CSV, index=False)

    print("=== Resumen por Run ===")
    for run, stats in sorted(per_run_stats.items()):
        total = stats["ok"] + stats["missing"]
        print(f"  {run:40s} OK {stats['ok']:3d} / FALTA {stats['missing']:3d} / total {total}")

    print(f"\nAuditoria escrita en: {AUDIT_CSV.relative_to(ROOT)}")
    print(f"Audios unificados en: {OUT_DIR.relative_to(ROOT)}")

    print("\n=== Carpetas no mapeadas (descartadas) ===")
    for folder in UNMAPPED_FOLDERS:
        if folder.is_dir():
            n = len(list(folder.rglob("*.flac")))
            print(f"  {folder.relative_to(ROOT)}: {n} .flac")


if __name__ == "__main__":
    main()
