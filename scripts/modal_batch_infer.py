#!/usr/bin/env python3
"""Batch inference for modal LoRA checkpoints via ACE-Step API."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import time
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import Request, urlopen


DEFAULT_MODE_PROMPTS: Dict[str, str] = {
    "ionian": "{trigger} ionian mode, C ionian, instrumental, electric piano, fretless bass, brushed drums, modal vamp on C, no vocals",
    "dorian": "{trigger} dorian mode, D dorian, instrumental, electric piano, fretless bass, brushed drums, modal vamp on D, no vocals",
    "phrygian": "{trigger} phrygian mode, E phrygian, instrumental, electric piano, fretless bass, brushed drums, modal vamp on E, no vocals",
    "lydian": "{trigger} lydian mode, F lydian, instrumental, electric piano, fretless bass, brushed drums, modal vamp on F, no vocals",
    "mixolydian": "{trigger} mixolydian mode, G mixolydian, instrumental, electric piano, fretless bass, brushed drums, modal vamp on G, no vocals",
    "aeolian": "{trigger} aeolian mode, A aeolian, instrumental, electric piano, fretless bass, brushed drums, modal vamp on A, no vocals",
    "locrian": "{trigger} locrian mode, B locrian, instrumental, electric piano, fretless bass, brushed drums, modal vamp on B, no vocals",
}

MODE_ORDER = [
    "ionian",
    "dorian",
    "phrygian",
    "lydian",
    "mixolydian",
    "aeolian",
    "locrian",
]


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _build_headers(api_token: Optional[str]) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    return headers


def _post_json(base_url: str, endpoint: str, payload: Dict[str, Any], api_token: Optional[str]) -> Dict[str, Any]:
    url = f"{_normalize_base_url(base_url)}{endpoint}"
    data = dict(payload)
    if api_token:
        data.setdefault("ai_token", api_token)

    req = Request(
        url=url,
        data=json.dumps(data).encode("utf-8"),
        headers=_build_headers(api_token),
        method="POST",
    )

    try:
        with urlopen(req, timeout=600) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code} {endpoint}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error {endpoint}: {exc}") from exc


def _check_api_online(base_url: str, api_token: Optional[str]) -> None:
    url = f"{_normalize_base_url(base_url)}/health"
    req = Request(url=url, headers=_build_headers(api_token), method="GET")
    try:
        with urlopen(req, timeout=10) as resp:
            _ = resp.read()
            if resp.status < 200 or resp.status >= 300:
                raise RuntimeError(f"Health check failed with HTTP {resp.status}")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(
            f"API health check failed at {url}: HTTP {exc.code} {detail}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"Cannot connect to API at {url}. Start ACE-Step API (or Gradio with --enable-api) "
            f"and verify --api-base-url/port. Original error: {exc}"
        ) from exc


def _unwrap_data(response: Dict[str, Any], endpoint: str) -> Any:
    code = response.get("code", 500)
    if code != 200:
        raise RuntimeError(f"API error on {endpoint}: code={code}, error={response.get('error')}")
    return response.get("data")


def _symlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        try:
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        except Exception:
            pass
    try:
        dst.symlink_to(src)
    except Exception:
        # symlink fails on Windows or cross-filesystem mounts
        shutil.copy2(src, dst)


def _ensure_adapter_compat_path(checkpoint_path: Path) -> Path:
    """Return a LoRA adapter path safe for /v1/lora/load.

    ACE-Step API uses the folder name as a module identifier and can fail when
    the name contains dots (e.g. epoch_100_loss_0.8379). This function builds
    a sanitized temp-dir view so the API never sees dots in the adapter name.
    """
    if checkpoint_path.is_file() and checkpoint_path.name == "adapter_model.safetensors":
        return _ensure_adapter_compat_path(checkpoint_path.parent)

    if checkpoint_path.is_dir():
        model_file = checkpoint_path / "adapter_model.safetensors"
        config_file = checkpoint_path / "adapter_config.json"
        readme_file = checkpoint_path / "README.md"

        if model_file.is_file() and config_file.is_file():
            # Hash the full path to avoid collisions between experiments that
            # share a basename like "epoch_80_loss_0.5770".
            resolved = str(checkpoint_path.resolve())
            path_hash = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:12]
            safe_stem = checkpoint_path.name.replace(".", "_")
            safe_name = f"adapter_{safe_stem}__{path_hash}"
            temp_root = Path(tempfile.gettempdir()) / "modal_batch_infer_adapter_view"
            adapter_dir = temp_root / safe_name

            adapter_dir.mkdir(parents=True, exist_ok=True)

            _symlink_or_copy(model_file, adapter_dir / "adapter_model.safetensors")
            _symlink_or_copy(config_file, adapter_dir / "adapter_config.json")
            if readme_file.is_file():
                _symlink_or_copy(readme_file, adapter_dir / "README.md")

            return adapter_dir

    return checkpoint_path


def _load_checkpoint(base_url: str, checkpoint_path: Path, api_token: Optional[str], lora_scale: float) -> None:
    load_path = _ensure_adapter_compat_path(checkpoint_path)

    # Best-effort unload to avoid stale adapter state between checkpoints.
    try:
        _post_json(base_url, "/v1/lora/unload", {}, api_token)
    except Exception:
        pass

    resp = _post_json(
        base_url,
        "/v1/lora/load",
        {"lora_path": str(load_path)},
        api_token,
    )
    _unwrap_data(resp, "/v1/lora/load")

    resp = _post_json(
        base_url,
        "/v1/lora/toggle",
        {"use_lora": True},
        api_token,
    )
    _unwrap_data(resp, "/v1/lora/toggle")

    resp = _post_json(
        base_url,
        "/v1/lora/scale",
        {"scale": lora_scale},
        api_token,
    )
    _unwrap_data(resp, "/v1/lora/scale")


def _release_task(
    base_url: str,
    prompt: str,
    seed: int,
    duration: float,
    inference_steps: int,
    audio_format: str,
    api_token: Optional[str],
) -> str:
    payload = {
        "task_type": "text2music",
        "prompt": prompt,
        "lyrics": "[Instrumental]",
        "vocal_language": "unknown",
        "batch_size": 1,
        "use_random_seed": False,
        "seed": seed,
        "inference_steps": inference_steps,
        "audio_duration": duration,
        "audio_format": audio_format,
        "sample_mode": False,
        "use_format": False,
    }

    resp = _post_json(base_url, "/release_task", payload, api_token)
    data = _unwrap_data(resp, "/release_task")
    task_id = (data or {}).get("task_id")
    if not task_id:
        raise RuntimeError(f"Missing task_id in /release_task response: {resp}")
    return task_id


def _query_result(base_url: str, task_id: str, api_token: Optional[str]) -> List[Dict[str, Any]]:
    payload = {"task_id_list": [task_id]}
    resp = _post_json(base_url, "/query_result", payload, api_token)
    data = _unwrap_data(resp, "/query_result")
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected /query_result payload: {resp}")
    return data


def _wait_for_task(
    base_url: str,
    task_id: str,
    api_token: Optional[str],
    poll_interval: float,
    timeout_s: float,
) -> Dict[str, Any]:
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        items = _query_result(base_url, task_id, api_token)
        if not items:
            time.sleep(poll_interval)
            continue

        item = items[0]
        status = item.get("status")
        # status: 0 = running/queued, 1 = succeeded, 2 = failed
        if status in (1, 2):
            return item

        time.sleep(poll_interval)

    raise TimeoutError(f"Task {task_id} did not finish within {timeout_s:.1f}s")


def _extract_audio_path_from_result_item(result_item: Dict[str, Any]) -> str:
    """Extract audio path from query_result item.

    Two API shapes supported:
    - Standalone api_server: result_item["result"] is a JSON string list
    - Gradio embedded API: result_item["data"] is a list of dicts
    """
    # Standalone API server format
    raw_result = result_item.get("result")
    if isinstance(raw_result, str) and raw_result:
        try:
            parsed = json.loads(raw_result)
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                file_path = parsed[0].get("file")
                if isinstance(file_path, str) and file_path:
                    return file_path
        except Exception:
            pass

    # Gradio embedded API format
    data = result_item.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        file_path = data[0].get("file")
        if isinstance(file_path, str) and file_path:
            return file_path

    return ""


def _resolve_audio_reference_to_local_path(audio_ref: str) -> str:
    """Resolve an API audio reference to a local path.

    Handles plain paths (/home/.../audio.flac) and URL-encoded
    /v1/audio?path=... references from both standalone and Gradio API shapes.
    """
    if not audio_ref:
        return ""

    if audio_ref.startswith("/") and not audio_ref.startswith("/v1/audio"):
        return audio_ref

    parsed = urlparse(audio_ref)
    looks_like_audio_endpoint = (
        parsed.path == "/v1/audio" or audio_ref.startswith("/v1/audio?")
    )
    if not looks_like_audio_endpoint:
        return audio_ref

    qs = parse_qs(parsed.query)
    path_values = qs.get("path") or qs.get("file")
    if not path_values:
        return audio_ref
    return unquote(path_values[0])


def _normalize_checkpoint_path(path: Path) -> Optional[Path]:
    """Return the loadable adapter directory for a given path.

    Supports two layouts:
    - Legacy: epoch_xxx/adapter/
    - New: epoch_xxx/{adapter_config.json, adapter_model.safetensors}
    """
    if not path.exists():
        return None

    if path.is_dir() and path.name == "adapter":
        return path

    if path.is_dir():
        adapter_dir = path / "adapter"
        if adapter_dir.is_dir():
            return adapter_dir

        if (path / "adapter_config.json").is_file() and (path / "adapter_model.safetensors").is_file():
            return path

        return None

    if path.is_file() and path.name == "adapter_model.safetensors":
        parent = path.parent
        if (parent / "adapter_config.json").is_file():
            return parent

    return None


def _iter_checkpoints(root: Path, pattern: str) -> List[Path]:
    raw_items = sorted([p for p in root.glob(pattern) if p.exists()])
    normalized: List[Path] = []
    seen: set[str] = set()

    for item in raw_items:
        candidate = _normalize_checkpoint_path(item)
        if candidate is None:
            continue
        key = str(candidate.resolve())
        if key in seen:
            continue
        seen.add(key)
        normalized.append(candidate)

    return normalized


def _checkpoint_label(path: Path) -> str:
    if path.name == "adapter" and path.parent.name:
        return path.parent.name
    return path.name


def _extract_epoch(label: str) -> int:
    match = re.search(r"epoch_(\d+)", label)
    if not match:
        return -1
    return int(match.group(1))


def _sort_checkpoints(checkpoints: List[Path]) -> List[Path]:
    return sorted(
        checkpoints,
        key=lambda p: (_extract_epoch(_checkpoint_label(p)), _checkpoint_label(p)),
    )


def _parse_mode_prompts_from_block(text: str) -> Dict[str, str]:
    mode_prompts: Dict[str, str] = {}
    lines = text.splitlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.endswith(":"):
            mode = line[:-1].strip().lower()
            if mode in MODE_ORDER:
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines):
                    prompt_line = lines[j].strip()
                    if prompt_line and not prompt_line.endswith(":"):
                        mode_prompts[mode] = prompt_line
                i = j
        i += 1

    return mode_prompts


def _load_prompts_from_file(prompts_file: Path, block: str) -> Dict[str, str]:
    """Load prompts for one block from the prompts file.

    Expected structure: BLOQUE X header, then Mode:\\n[trigger] ... per mode.
    """
    text = prompts_file.read_text(encoding="utf-8")

    block_letter = block.upper().strip()
    if block_letter not in {"A", "B", "D", "E"}:
        raise ValueError("prompts block must be A, B, D, or E")

    marker = f"BLOQUE {block_letter}"
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"Could not find '{marker}' in prompts file: {prompts_file}")

    next_block_match = re.search(r"\n=+\nBLOQUE\s+[A-Z]", text[start + len(marker):])
    if next_block_match:
        end = start + len(marker) + next_block_match.start()
    else:
        notes_idx = text.find("NOTAS PARA LA MEMORIA", start + 1)
        end = notes_idx if notes_idx >= 0 else len(text)

    block_text = text[start:end]
    parsed = _parse_mode_prompts_from_block(block_text)

    missing = [mode for mode in MODE_ORDER if mode not in parsed]
    if missing:
        raise ValueError(
            f"Missing mode prompts in BLOQUE {block_letter}: {', '.join(missing)}"
        )

    return {mode: parsed[mode] for mode in MODE_ORDER}


def _render_prompt(template: str, trigger: str) -> str:
    trig = trigger.strip()
    trig_text = f"{trig} " if trig else ""
    prompt = template.replace("[trigger]", trig_text).strip()
    return prompt.format(trigger=trig_text).strip()


def _copy_audio_if_needed(src_path: str, dst_dir: Optional[Path], dst_name: str) -> str:
    if dst_dir is None:
        return src_path

    resolved = _resolve_audio_reference_to_local_path(src_path)
    src = Path(resolved)
    if not src.exists():
        raise FileNotFoundError(
            f"Audio source path not found after resolving reference: {src_path} -> {resolved}"
        )
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / dst_name
    shutil.copy2(src, dst)
    return str(dst)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch modal inference across LoRA checkpoints")
    parser.add_argument("--api-base-url", type=str, default="http://127.0.0.1:7860", help="ACE-Step API base URL")
    parser.add_argument("--api-token", type=str, default=None, help="Optional API token")
    parser.add_argument("--checkpoints-root", type=str, required=True, help="Root folder containing LoRA checkpoints")
    parser.add_argument(
        "--checkpoint-pattern",
        type=str,
        default="checkpoints/epoch_*_loss_*/adapter",
        help=(
            "Glob pattern under checkpoints-root "
            "(default: checkpoints/epoch_*_loss_*/adapter)"
        ),
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default="./prompts_modal_sound.txt",
        help="Prompts file path (default: ./prompts_modal_sound.txt)",
    )
    parser.add_argument(
        "--prompts-block",
        type=str,
        default="B",
        choices=["A", "B", "D", "E", "a", "b", "d", "e"],
        help="Prompt block to use from prompts file: A (minimal), B (standard), D (mode only), or E (mode name only)",
    )
    parser.add_argument("--trigger", type=str, default="", help="Optional trigger token prepended to prompts")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1111, 2222], help="Fixed seeds")
    parser.add_argument("--duration", type=float, default=30.0, help="Audio duration in seconds")
    parser.add_argument("--inference-steps", type=int, default=8, help="Inference steps")
    parser.add_argument("--lora-scale", type=float, default=1.0, help="LoRA scale via /v1/lora/scale")
    parser.add_argument("--audio-format", type=str, default="flac", help="Output audio format")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Polling interval seconds")
    parser.add_argument("--task-timeout", type=float, default=1200.0, help="Task timeout seconds")
    parser.add_argument("--copy-audio-dir", type=str, default=None, help="Optional folder to copy generated audios")
    parser.add_argument(
        "--output-csv",
        type=str,
        default="./gradio_outputs/modal_batch_manifest.csv",
        help="CSV manifest output path",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    root = Path(args.checkpoints_root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"[FAIL] Invalid checkpoints root: {root}")
        return 1

    checkpoints = _iter_checkpoints(root, args.checkpoint_pattern)
    if not checkpoints:
        fallback_patterns = [
            "checkpoints/epoch_*_loss_*",
            "checkpoints/epoch_*_loss_*/adapter",
            "checkpoints/epoch_*_loss_*/adapter_model.safetensors",
        ]
        for pat in fallback_patterns:
            checkpoints = _iter_checkpoints(root, pat)
            if checkpoints:
                print(f"[INFO] No matches for '{args.checkpoint_pattern}', using fallback pattern '{pat}'")
                break

    if not checkpoints:
        print(f"[FAIL] No checkpoints found in {root} with pattern {args.checkpoint_pattern}")
        return 1

    checkpoints = _sort_checkpoints(checkpoints)

    prompts_file = Path(args.prompts_file).expanduser().resolve()
    if not prompts_file.exists() or not prompts_file.is_file():
        print(f"[FAIL] Prompts file not found: {prompts_file}")
        return 1

    try:
        mode_prompts = _load_prompts_from_file(prompts_file, args.prompts_block)
    except Exception as exc:
        print(f"[FAIL] Could not parse prompts file: {exc}")
        return 1

    copy_dir = Path(args.copy_audio_dir).expanduser().resolve() if args.copy_audio_dir else None
    output_csv = Path(args.output_csv).expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    try:
        _check_api_online(args.api_base_url, args.api_token)
    except Exception as exc:
        print(f"[FAIL] {exc}")
        return 1

    rows: List[Dict[str, str]] = []

    for checkpoint_path in checkpoints:
        ckpt_name = _checkpoint_label(checkpoint_path)
        print(f"[INFO] Loading checkpoint: {checkpoint_path}")

        try:
            _load_checkpoint(
                base_url=args.api_base_url,
                checkpoint_path=checkpoint_path,
                api_token=args.api_token,
                lora_scale=float(args.lora_scale),
            )
        except Exception as exc:
            print(f"[WARN] Failed to load checkpoint {checkpoint_path}: {exc}")
            for mode in mode_prompts:
                for seed in args.seeds:
                    rows.append(
                        {
                            "checkpoint": ckpt_name,
                            "mode": mode,
                            "seed": str(seed),
                            "task_id": "",
                            "status": "load_error",
                            "audio_path": "",
                            "error": str(exc),
                        }
                    )
            continue

        for mode, template in mode_prompts.items():
            prompt = _render_prompt(template, args.trigger)

            for seed in args.seeds:
                task_id = ""
                try:
                    task_id = _release_task(
                        base_url=args.api_base_url,
                        prompt=prompt,
                        seed=int(seed),
                        duration=float(args.duration),
                        inference_steps=int(args.inference_steps),
                        audio_format=args.audio_format,
                        api_token=args.api_token,
                    )

                    result_item = _wait_for_task(
                        base_url=args.api_base_url,
                        task_id=task_id,
                        api_token=args.api_token,
                        poll_interval=float(args.poll_interval),
                        timeout_s=float(args.task_timeout),
                    )

                    status = result_item.get("status")
                    error_msg = result_item.get("error") or ""
                    audio_path = ""

                    if status == 1:
                        audio_path = _extract_audio_path_from_result_item(result_item)
                        if copy_dir and audio_path:
                            ext = Path(audio_path).suffix or f".{args.audio_format}"
                            new_name = f"{ckpt_name}_{mode}_seed{seed}{ext}"
                            audio_path = _copy_audio_if_needed(audio_path, copy_dir, new_name)

                    rows.append(
                        {
                            "checkpoint": ckpt_name,
                            "mode": mode,
                            "seed": str(seed),
                            "task_id": task_id,
                            "status": "succeeded" if status == 1 else "failed",
                            "audio_path": audio_path,
                            "error": str(error_msg),
                        }
                    )
                    print(f"[INFO] {ckpt_name} | {mode} | seed={seed} -> status={status}")

                except Exception as exc:
                    rows.append(
                        {
                            "checkpoint": ckpt_name,
                            "mode": mode,
                            "seed": str(seed),
                            "task_id": task_id,
                            "status": "error",
                            "audio_path": "",
                            "error": str(exc),
                        }
                    )
                    print(f"[WARN] {ckpt_name} | {mode} | seed={seed} failed: {exc}")

    fieldnames = ["checkpoint", "mode", "seed", "task_id", "status", "audio_path", "error"]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Manifest saved: {output_csv}")
    print(f"[OK] Rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
