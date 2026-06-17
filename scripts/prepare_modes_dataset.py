"""
Prepare LoRA dataset for musical modes from a folder-per-mode structure.

Usage (from project root):
    uv run python scripts/prepare_modes_dataset.py \
        --dataset-dir ./Dataset \
        --output ./datasets/modos_dataset.json

Each subfolder name is used as the caption for all songs inside it.
Expected structure:
    Dataset/
        ionian/      *.wav
        dorian/      *.wav
        phrygian/    *.wav
        lydian/      *.wav
        mixolydian/  *.wav
        aeolian/     *.wav
        locrian/     *.wav
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import datetime

SUPPORTED_AUDIO_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".opus"}

# Minimal captions: only mode name to avoid over-constraining style, tonic, or instrumentation.
MODE_CAPTIONS_SAFE: dict[str, str] = {
    "ionian": "ionian mode",
    "dorian": "dorian mode",
    "phrygian": "phrygian mode",
    "lydian": "lydian mode",
    "mixolydian": "mixolydian mode",
    "aeolian": "aeolian mode",
    "locrian": "locrian mode",
}

MODE_CAPTIONS_MODAL_COLOR: dict[str, str] = {
    "ionian": "ionian mode, bright major color",
    "dorian": "dorian mode, minor color, natural 6th flavor",
    "phrygian": "phrygian mode, dark minor color, flat 2nd flavor",
    "lydian": "lydian mode, bright major color, raised 4th flavor",
    "mixolydian": "mixolydian mode, dominant major color, flat 7th flavor",
    "aeolian": "aeolian mode, natural minor color, flat 6th and flat 7th flavor",
    "locrian": "locrian mode, diminished unstable color, flat 2nd and flat 5th flavor",
}

CAPTION_PROFILES: dict[str, dict[str, str]] = {
    "safe": MODE_CAPTIONS_SAFE,
    "modal_color": MODE_CAPTIONS_MODAL_COLOR,
}


def _get_duration(audio_path: str) -> float:
    try:
        import soundfile as sf
        info = sf.info(audio_path)
        return round(info.duration, 2)
    except Exception:
        pass
    try:
        import wave
        with wave.open(audio_path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return round(frames / float(rate), 2)
    except Exception:
        return 0.0


def build_dataset(dataset_dir: str, caption_profile: str = "safe") -> list[dict]:
    samples: list[dict] = []
    captions = CAPTION_PROFILES[caption_profile]

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    for mode_folder in sorted(os.listdir(dataset_dir)):
        mode_path = os.path.join(dataset_dir, mode_folder)
        if not os.path.isdir(mode_path):
            continue

        mode_name = mode_folder.lower().strip()
        caption = captions.get(mode_name, f"{mode_name} mode")

        audio_files = sorted(
            f for f in os.listdir(mode_path)
            if os.path.splitext(f)[1].lower() in SUPPORTED_AUDIO_FORMATS
        )

        if not audio_files:
            print(f"  [WARN] No audio files found in: {mode_path}")
            continue

        print(f"  [{mode_folder}] {len(audio_files)} files  →  caption: \"{caption}\"")

        for filename in audio_files:
            audio_path = os.path.join(mode_path, filename)
            duration = _get_duration(audio_path)

            sample = {
                "id": str(uuid.uuid4())[:8],
                "audio_path": os.path.abspath(audio_path),
                "filename": filename,
                "caption": caption,
                "genre": "",
                "lyrics": "[Instrumental]",
                "raw_lyrics": "",
                "formatted_lyrics": "",
                "bpm": None,
                "keyscale": "",
                "timesignature": "",
                "duration": duration,
                "language": "unknown",
                "is_instrumental": True,
                "custom_tag": mode_name,
                "labeled": True,
                "prompt_override": None,
            }
            samples.append(sample)

    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare modes LoRA dataset JSON")
    parser.add_argument(
        "--dataset-dir",
        default="./Dataset",
        help="Root folder containing one subfolder per mode (default: ./Dataset)",
    )
    parser.add_argument(
        "--output",
        default="./datasets/modos_dataset.json",
        help="Output JSON path (default: ./datasets/modos_dataset.json)",
    )
    parser.add_argument(
        "--caption-profile",
        choices=["safe", "modal_color"],
        default="safe",
        help="Caption profile to use: safe (default) or modal_color",
    )
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    output_path = os.path.abspath(args.output)

    print(f"\nScanning: {dataset_dir}")
    samples = build_dataset(dataset_dir, caption_profile=args.caption_profile)

    if not samples:
        print("\n[ERROR] No samples found. Check your dataset directory.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dataset = {
        "metadata": {
            "name": "modos_escala_mayor",
            "custom_tag": "",
            "tag_position": "replace",
            "created_at": datetime.now().isoformat(),
            "num_samples": len(samples),
            "all_instrumental": True,
            "genre_ratio": 0,
        },
        "samples": samples,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\n✅  Dataset saved: {output_path}")
    print(f"   Total samples : {len(samples)}")
    print(f"   Caption profile: {args.caption_profile}")


if __name__ == "__main__":
    main()
