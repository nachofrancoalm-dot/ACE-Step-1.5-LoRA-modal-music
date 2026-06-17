#!/usr/bin/env python3
"""Recalculate auto_score_0_1 column in stage1_results.csv from component metrics.

Formula (matches modal_eval_pipeline.py):
  auto_score_0_1 = 0.40*in_mode_ratio + 0.20*modal_accuracy + 0.20*tonal_stability 
                   + 0.10*tonic_dominance + 0.10*template_similarity
"""

import csv
import sys
from pathlib import Path


def main() -> int:
    """Recalculate auto_score_0_1 and write back to CSV."""
    if len(sys.argv) < 2:
        print("Usage: python recalc_auto_score_stage1.py <path/to/stage1_results.csv>")
        return 1

    csv_path = Path(sys.argv[1]).expanduser().resolve()
    if not csv_path.exists():
        print(f"[FAIL] CSV not found: {csv_path}")
        return 1

    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            print("[FAIL] CSV has no header")
            return 1
        for row in reader:
            rows.append(row)

    updated = 0
    for row in rows:
        try:
            in_mode = float(row.get("in_mode_ratio", "-1"))
            modal_acc = float(row.get("modal_accuracy", "-1"))
            tonal_stab = float(row.get("tonal_stability", "-1"))
            tonic_dom = float(row.get("tonic_dominance", "-1"))
            template = float(row.get("template_similarity", "-1"))

            # Skip if any component is missing
            if any(v < 0 for v in [in_mode, modal_acc, tonal_stab, tonic_dom, template]):
                continue

            # Recalculate using same formula as modal_eval_pipeline.py
            auto_score_01 = (
                0.40 * in_mode
                + 0.20 * modal_acc
                + 0.20 * tonal_stab
                + 0.10 * tonic_dom
                + 0.10 * template
            )
            auto_score_01 = max(0.0, min(1.0, auto_score_01))  # Clamp to [0, 1]
            row["auto_score_0_1"] = f"{auto_score_01:.6f}"
            updated += 1
        except Exception as e:
            print(f"[WARN] exp {row.get('exp_id')}: {e}")
            continue

    # Write back
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Updated {updated}/{len(rows)} rows")
    print(f"[OK] CSV written: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
