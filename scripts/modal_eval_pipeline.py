#!/usr/bin/env python3
"""Compute modal coherence metrics for generated audio files and export to CSV."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"}
EPS = 1e-9

MODE_NAMES = [
    "ionian",
    "dorian",
    "phrygian",
    "lydian",
    "mixolydian",
    "aeolian",
    "locrian",
]

MODE_INTERVALS: Dict[str, Sequence[int]] = {
    "ionian": (0, 2, 4, 5, 7, 9, 11),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "aeolian": (0, 2, 3, 5, 7, 8, 10),
    "locrian": (0, 1, 3, 5, 6, 8, 10),
}

# Tonic pitch class per mode, matching the prompt set (C ionian, D dorian, ...).
MODE_TONIC_PC: Dict[str, int] = {
    "ionian": 0,      # C
    "dorian": 2,      # D
    "phrygian": 4,    # E
    "lydian": 5,      # F
    "mixolydian": 7,  # G
    "aeolian": 9,     # A
    "locrian": 11,    # B
}

PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def list_audio_files(input_dir: Path, recursive: bool) -> List[Path]:
    pattern = "**/*" if recursive else "*"
    files = [
        p for p in input_dir.glob(pattern)
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ]
    return sorted(files)


def infer_mode_from_name(path: Path) -> Optional[str]:
    """Infer target mode from path text; prefers "ionian mode" over bare "ionian"."""
    text = path.as_posix().lower().replace("_", " ").replace("-", " ")

    for mode in MODE_NAMES:
        if re.search(rf"\b{re.escape(mode)}\s+mode\b", text):
            return mode

    for mode in MODE_NAMES:
        if re.search(rf"\b{re.escape(mode)}\b", text):
            return mode

    return None


def load_audio_mono(path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=False)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    audio = np.asarray(audio, dtype=np.float32)

    if sr != target_sr:
        gcd = math.gcd(sr, target_sr)
        up = target_sr // gcd
        down = sr // gcd
        audio = resample_poly(audio, up=up, down=down).astype(np.float32)
        sr = target_sr

    if audio.size == 0:
        raise ValueError("empty audio")

    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak

    return audio, sr


def compute_framewise_chroma(
    audio: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    min_freq: float,
    max_freq: float,
) -> np.ndarray:
    """Return a [12, frames] chroma matrix from STFT magnitude, normalized per frame."""
    if audio.size < n_fft:
        pad_width = n_fft - audio.size
        audio = np.pad(audio, (0, pad_width))

    frames = np.lib.stride_tricks.sliding_window_view(audio, n_fft)[::hop_length]
    if frames.size == 0:
        frames = np.expand_dims(np.pad(audio, (0, max(0, n_fft - audio.size)))[:n_fft], axis=0)

    window = np.hanning(n_fft).astype(np.float32)
    spectrum = np.fft.rfft(frames * window[None, :], axis=1)
    magnitude = np.abs(spectrum).astype(np.float32)

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr).astype(np.float32)
    valid = (freqs >= min_freq) & (freqs <= max_freq)
    valid_indices = np.where(valid)[0]

    chroma = np.zeros((12, magnitude.shape[0]), dtype=np.float32)
    if valid_indices.size == 0:
        return chroma

    for idx in valid_indices:
        freq = float(freqs[idx])
        midi = 69.0 + 12.0 * math.log2(freq / 440.0)
        pc = int(round(midi)) % 12
        chroma[pc] += magnitude[:, idx]

    frame_sums = np.sum(chroma, axis=0, keepdims=True)
    chroma /= (frame_sums + EPS)
    return chroma


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    num = float(np.dot(vec_a, vec_b))
    den = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if den <= EPS:
        return 0.0
    return max(0.0, min(1.0, num / den))


def normalize_unit_interval(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def compute_modal_scores(base_score_01: float) -> Dict[str, float]:
    normalized_score = normalize_unit_interval(base_score_01)
    legacy_score_1_5 = 1.0 + 4.0 * normalized_score
    return {
        "auto_score_0_1": normalized_score,
        "auto_score_1_5": legacy_score_1_5,
    }


def evaluate_modal_metrics(chroma: np.ndarray, mode: str) -> Dict[str, float]:
    avg_chroma = np.mean(chroma, axis=1)
    avg_chroma /= (float(np.sum(avg_chroma)) + EPS)

    tonic_pc = MODE_TONIC_PC[mode]
    scale_pcs = [(tonic_pc + i) % 12 for i in MODE_INTERVALS[mode]]

    scale_template = np.zeros(12, dtype=np.float32)
    scale_template[scale_pcs] = 1.0
    scale_template /= float(np.sum(scale_template))

    in_mode_ratio = float(np.sum(avg_chroma[scale_pcs]))
    out_of_mode_ratio = max(0.0, 1.0 - in_mode_ratio)

    dominant_pc = np.argmax(chroma, axis=0)
    if dominant_pc.size <= 1:
        tonal_stability = 0.0
    else:
        transitions = np.mean(dominant_pc[1:] != dominant_pc[:-1])
        tonal_stability = max(0.0, 1.0 - float(transitions))

    tonic_energy_ratio = float(avg_chroma[tonic_pc])
    tonic_dominance_ratio = float(np.mean(dominant_pc == tonic_pc))

    per_frame_in_mode = np.sum(chroma[scale_pcs, :], axis=0)
    frame_in_mode_ratio = float(np.mean(per_frame_in_mode))

    template_similarity = cosine_similarity(avg_chroma, scale_template)

    base_score_01 = (
        0.40 * in_mode_ratio
        + 0.20 * frame_in_mode_ratio
        + 0.20 * tonal_stability
        + 0.10 * tonic_dominance_ratio
        + 0.10 * template_similarity
    )
    scores = compute_modal_scores(base_score_01)

    return {
        "in_mode_ratio": in_mode_ratio,
        "out_of_mode_ratio": out_of_mode_ratio,
        "frame_in_mode_ratio": frame_in_mode_ratio,
        "tonal_stability": tonal_stability,
        "tonic_energy_ratio": tonic_energy_ratio,
        "tonic_dominance_ratio": tonic_dominance_ratio,
        "template_similarity": template_similarity,
        "auto_score_0_1": scores["auto_score_0_1"],
        "auto_score_1_5": scores["auto_score_1_5"],
        "tonic_pitch_class": float(tonic_pc),
    }


def analyze_file(
    path: Path,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    min_freq: float,
    max_freq: float,
) -> Dict[str, str]:
    mode = infer_mode_from_name(path)
    audio, sr = load_audio_mono(path, target_sr=sample_rate)
    duration_s = float(len(audio) / sr)

    chroma = compute_framewise_chroma(
        audio=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        min_freq=min_freq,
        max_freq=max_freq,
    )

    dominant_pitch_class = int(np.argmax(np.mean(chroma, axis=1))) if chroma.size else -1

    row: Dict[str, str] = {
        "file": str(path),
        "mode_inferred": mode or "unknown",
        "duration_s": f"{duration_s:.3f}",
        "dominant_pitch_class": str(dominant_pitch_class),
        "dominant_pitch_name": (
            PITCH_CLASS_NAMES[dominant_pitch_class] if dominant_pitch_class >= 0 else "unknown"
        ),
        "status": "ok",
        "error": "",
    }

    if mode is None:
        row.update({
            "in_mode_ratio": "",
            "out_of_mode_ratio": "",
            "frame_in_mode_ratio": "",
            "tonal_stability": "",
            "tonic_energy_ratio": "",
            "tonic_dominance_ratio": "",
            "template_similarity": "",
            "auto_score_0_1": "",
            "auto_score_1_5": "",
            "expected_tonic_pc": "",
            "expected_tonic_name": "",
        })
        return row

    metrics = evaluate_modal_metrics(chroma, mode=mode)
    tonic_pc = int(metrics["tonic_pitch_class"])

    row.update({
        "in_mode_ratio": f"{metrics['in_mode_ratio']:.4f}",
        "out_of_mode_ratio": f"{metrics['out_of_mode_ratio']:.4f}",
        "frame_in_mode_ratio": f"{metrics['frame_in_mode_ratio']:.4f}",
        "tonal_stability": f"{metrics['tonal_stability']:.4f}",
        "tonic_energy_ratio": f"{metrics['tonic_energy_ratio']:.4f}",
        "tonic_dominance_ratio": f"{metrics['tonic_dominance_ratio']:.4f}",
        "template_similarity": f"{metrics['template_similarity']:.4f}",
        "auto_score_0_1": f"{metrics['auto_score_0_1']:.3f}",
        "auto_score_1_5": f"{metrics['auto_score_1_5']:.3f}",
        "expected_tonic_pc": str(tonic_pc),
        "expected_tonic_name": PITCH_CLASS_NAMES[tonic_pc],
    })
    return row


def summarize(rows: Sequence[Dict[str, str]]) -> str:
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    known_mode_rows = [r for r in ok_rows if r.get("mode_inferred") != "unknown"]

    if not rows:
        return "No rows generated."

    lines = [
        f"Files processed: {len(rows)}",
        f"Successful: {len(ok_rows)}",
        f"With inferred mode: {len(known_mode_rows)}",
    ]

    if known_mode_rows:
        scores = [float(r["auto_score_0_1"]) for r in known_mode_rows if r.get("auto_score_0_1")]
        if scores:
            lines.append(f"Mean auto_score_0_1: {np.mean(scores):.3f}")
            lines.append(f"Std auto_score_0_1: {np.std(scores):.3f}")

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Modal evaluation pipeline for generated audio (CSV output)."
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Folder with generated audio files.")
    parser.add_argument(
        "--output-csv",
        type=str,
        default="./modal_eval_metrics.csv",
        help="Path to write CSV metrics.",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively scan input-dir.")
    parser.add_argument(
        "--include-unknown-modes",
        action="store_true",
        help="Include rows where mode cannot be inferred from filename/path.",
    )
    parser.add_argument("--sample-rate", type=int, default=22050, help="Analysis sample rate.")
    parser.add_argument("--n-fft", type=int, default=4096, help="FFT size for chroma extraction.")
    parser.add_argument("--hop-length", type=int, default=512, help="Hop length for frame analysis.")
    parser.add_argument("--min-freq", type=float, default=40.0, help="Minimum frequency in Hz.")
    parser.add_argument("--max-freq", type=float, default=5000.0, help="Maximum frequency in Hz.")
    parser.add_argument("--print-summary", action="store_true", help="Print summary to stdout.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[FAIL] Invalid input directory: {input_dir}")
        return 1

    files = list_audio_files(input_dir=input_dir, recursive=args.recursive)
    if not files:
        print(f"[FAIL] No audio files found under: {input_dir}")
        return 1

    rows: List[Dict[str, str]] = []

    for path in files:
        try:
            row = analyze_file(
                path=path,
                sample_rate=args.sample_rate,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                min_freq=args.min_freq,
                max_freq=args.max_freq,
            )
        except Exception as exc:
            row = {
                "file": str(path),
                "mode_inferred": "unknown",
                "duration_s": "",
                "dominant_pitch_class": "",
                "dominant_pitch_name": "",
                "in_mode_ratio": "",
                "out_of_mode_ratio": "",
                "frame_in_mode_ratio": "",
                "tonal_stability": "",
                "tonic_energy_ratio": "",
                "tonic_dominance_ratio": "",
                "template_similarity": "",
                "auto_score_0_1": "",
                "auto_score_1_5": "",
                "expected_tonic_pc": "",
                "expected_tonic_name": "",
                "status": "error",
                "error": str(exc),
            }

        if row["mode_inferred"] == "unknown" and not args.include_unknown_modes:
            continue

        rows.append(row)

    if not rows:
        print("[FAIL] No rows to write. Use --include-unknown-modes if needed.")
        return 1

    output_csv = Path(args.output_csv).expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "file",
        "mode_inferred",
        "duration_s",
        "dominant_pitch_class",
        "dominant_pitch_name",
        "in_mode_ratio",
        "out_of_mode_ratio",
        "frame_in_mode_ratio",
        "tonal_stability",
        "tonic_energy_ratio",
        "tonic_dominance_ratio",
        "template_similarity",
        "auto_score_0_1",
        "auto_score_1_5",
        "expected_tonic_pc",
        "expected_tonic_name",
        "status",
        "error",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Wrote {len(rows)} rows to: {output_csv}")
    if args.print_summary:
        print(summarize(rows))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
