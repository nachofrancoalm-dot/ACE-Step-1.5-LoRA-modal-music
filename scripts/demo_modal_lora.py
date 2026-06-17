#!/usr/bin/env python3
"""Demo interactivo: clasificación modal con LLM (Groq) + generación con LoRA (ACE-Step).

Uso: uv run python scripts/demo_modal_lora.py [--api-url URL] [--checkpoint PATH] [--port N]
Requiere: ACE-Step corriendo con --enable-api, GROQ_API_KEY en entorno.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

_THIS_DIR = Path(__file__).parent
_spec = importlib.util.spec_from_file_location(
    "modal_batch_infer", _THIS_DIR / "modal_batch_infer.py"
)
_mbi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mbi)

_check_api_online               = _mbi._check_api_online
_load_checkpoint                = _mbi._load_checkpoint
_release_task                   = _mbi._release_task
_wait_for_task                  = _mbi._wait_for_task
_extract_audio_path_from_result = _mbi._extract_audio_path_from_result_item
_resolve_audio_path             = _mbi._resolve_audio_reference_to_local_path
_normalize_base_url             = _mbi._normalize_base_url

# run_D epoch 100: best global result from the TFG study.
DEFAULT_CHECKPOINT = (
    "/home/musica/ACE-Step-1.5/lora_output/modal_run_D/"
    "checkpoints/epoch_100_loss_0.1965"
)

MODAL_PROMPTS: dict[str, str] = {
    "ionian": (
        "ionian mode, jazz-fusion instrumental, C ionian, electric piano, "
        "fretless bass, 96 BPM, bright major character, stable tonic on C, "
        "no modulation, no vocals"
    ),
    "dorian": (
        "dorian mode, jazz-fusion instrumental, D dorian, electric piano, "
        "fretless bass, 96 BPM, minor with hopeful color, natural 6th flavor, "
        "no modulation, no vocals"
    ),
    "phrygian": (
        "phrygian mode, jazz-fusion instrumental, E phrygian, electric piano, "
        "fretless bass, 96 BPM, unstable minor character, flat 2nd flavor, "
        "no modulation, no vocals"
    ),
    "lydian": (
        "lydian mode, jazz-fusion instrumental, F lydian, electric piano, "
        "fretless bass, 96 BPM, bright dreamy character, sharp 4th flavor, "
        "no modulation, no vocals"
    ),
    "mixolydian": (
        "mixolydian mode, jazz-fusion instrumental, G mixolydian, electric piano, "
        "fretless bass, 96 BPM, major with bluesy color, flat 7th flavor, "
        "no modulation, no vocals"
    ),
    "aeolian": (
        "aeolian mode, jazz-fusion instrumental, A aeolian, electric piano, "
        "fretless bass, 96 BPM, natural minor character, melancholic flavor, "
        "no modulation, no vocals"
    ),
    "locrian": (
        "locrian mode, jazz-fusion instrumental, B locrian, electric piano, "
        "fretless bass, 96 BPM, unstable diminished character, "
        "flat 2nd and flat 5th tension, no modulation, no vocals"
    ),
}

MODE_INFO: dict[str, dict[str, str]] = {
    "ionian":     {"emoji": "☀️",  "nombre": "Jónico",     "desc": "Mayor clásico. Brillante, estable, alegre. El modo más familiar."},
    "dorian":     {"emoji": "🎷",  "nombre": "Dórico",     "desc": "Menor con 6ª mayor. Jazz, funk, soul. Oscuro pero esperanzador."},
    "phrygian":   {"emoji": "🌙",  "nombre": "Frigio",     "desc": "Menor con 2ª disminuida. Flamenco, metal, misterio oriental."},
    "lydian":     {"emoji": "✨",  "nombre": "Lidio",      "desc": "Mayor con 4ª aumentada. Onírico, flotante, fantástico."},
    "mixolydian": {"emoji": "🎸",  "nombre": "Mixolidio",  "desc": "Mayor con 7ª menor. Blues, rock, dominante. Potente y relajado."},
    "aeolian":    {"emoji": "🌧️", "nombre": "Eólico",     "desc": "Menor natural. Melancólico, clásico, introspectivo."},
    "locrian":    {"emoji": "⚡",  "nombre": "Locrio",     "desc": "Disminuido. Extremadamente inestable. Experimental, disonante."},
}

MODE_LIST = list(MODAL_PROMPTS.keys())

_SYSTEM_PROMPT = """Eres un experto en teoría musical.
Tu tarea es clasificar una descripción musical en uno de los 7 modos diatónicos.

Modos disponibles:
- ionian: mayor, alegre, estable
- dorian: menor con 6ª mayor, jazz, funk
- phrygian: menor con 2ª disminuida, flamenco, oscuro
- lydian: mayor con 4ª aumentada, onírico
- mixolydian: mayor con 7ª menor, blues, rock
- aeolian: menor natural, melancólico
- locrian: disminuido, muy inestable, experimental

Responde ÚNICAMENTE con el nombre del modo en minúsculas (una sola palabra).
No añadas explicación ni puntuación."""


def _classify_mode_groq(description: str, api_key: str) -> str:
    try:
        from groq import Groq
    except ImportError:
        raise RuntimeError(
            "Groq no está instalado. Ejecuta: uv add groq\n"
            "O selecciona el modo manualmente en el desplegable."
        )

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": description},
        ],
        temperature=0.1,
        max_tokens=15,
    )
    raw = response.choices[0].message.content.strip().lower()
    for mode in MODE_LIST:
        if mode in raw:
            return mode
    return "ionian"  # fallback cuando el modelo no devuelve un modo reconocible


def generate(
    description: str,
    mode_override: str,
    api_url: str,
    groq_key: str,
    checkpoint_path: str,
    duration: float,
    seed: int,
    lora_scale: float,
    inference_steps: int,
) -> tuple[str | None, str, str]:
    """Retorna (audio_path, modo_detectado, markdown_info). Lanza gr.Error en caso de fallo."""
    import gradio as gr

    api_url = _normalize_base_url(api_url or "http://127.0.0.1:7860")

    try:
        _check_api_online(api_url, api_token=None)
    except RuntimeError as e:
        raise gr.Error(f"ACE-Step API no disponible: {e}")

    if mode_override and mode_override != "auto":
        mode = mode_override
        source = "seleccionado manualmente"
    else:
        if not description.strip():
            raise gr.Error("Escribe una descripción o selecciona un modo manualmente.")
        key = groq_key.strip() or os.environ.get("GROQ_API_KEY", "")
        if not key:
            raise gr.Error(
                "Se necesita una Groq API key para la detección automática.\n"
                "Consíguela gratis en console.groq.com o selecciona el modo manualmente."
            )
        try:
            mode = _classify_mode_groq(description, key)
        except RuntimeError as e:
            raise gr.Error(str(e))
        source = "detectado por LLM"

    prompt = MODAL_PROMPTS[mode]

    ckpt = Path(checkpoint_path.strip() or DEFAULT_CHECKPOINT)
    try:
        _load_checkpoint(api_url, ckpt, api_token=None, lora_scale=lora_scale)
    except RuntimeError as e:
        raise gr.Error(f"Error cargando checkpoint LoRA: {e}")

    try:
        task_id = _release_task(
            api_url, prompt, int(seed), float(duration),
            inference_steps=int(inference_steps),
            audio_format="flac",
            api_token=None,
        )
    except RuntimeError as e:
        raise gr.Error(f"Error al lanzar tarea de generación: {e}")

    try:
        result = _wait_for_task(
            api_url, task_id, api_token=None,
            poll_interval=2.0, timeout_s=300.0,
        )
    except TimeoutError:
        raise gr.Error("La generación tardó demasiado (>5 min). Inténtalo de nuevo.")

    if result.get("status") == 2:
        raise gr.Error(f"Generación fallida: {result}")

    audio_ref = _extract_audio_path_from_result(result)
    audio_path = _resolve_audio_path(audio_ref)

    if not audio_path or not Path(audio_path).exists():
        raise gr.Error(f"No se encontró el fichero de audio generado: {audio_ref!r}")

    info = MODE_INFO[mode]
    md = (
        f"## {info['emoji']} {info['nombre']} ({mode})\n\n"
        f"{info['desc']}\n\n"
        f"*Modo {source}*\n\n"
        f"**Prompt usado:**\n> {prompt}"
    )

    return audio_path, mode, md


def build_ui(default_api_url: str, default_checkpoint: str) -> "gr.Blocks":
    import gradio as gr

    with gr.Blocks(
        title="Demo LoRA Modal — ACE-Step 1.5",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# 🎵 Demo LoRA Modal — ACE-Step 1.5\n"
            "Describe la música que quieres y el sistema detectará el modo musical "
            "y generará audio con el mejor checkpoint LoRA del estudio."
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                description = gr.Textbox(
                    label="Descripción musical",
                    placeholder=(
                        "ej: algo oscuro y misterioso con tensión, "
                        "como música de suspense...\n"
                        "ej: algo alegre y brillante, veraniego\n"
                        "ej: jazz melancólico, nostálgico"
                    ),
                    lines=4,
                )
                mode_override = gr.Dropdown(
                    choices=["auto"] + MODE_LIST,
                    value="auto",
                    label="Modo (auto = detectar con LLM | manual = forzar)",
                )

                with gr.Accordion("⚙️ Configuración", open=False):
                    groq_key = gr.Textbox(
                        label="Groq API Key (solo para detección automática)",
                        placeholder="gsk_...",
                        type="password",
                        value=os.environ.get("GROQ_API_KEY", ""),
                    )
                    api_url = gr.Textbox(
                        label="ACE-Step API URL",
                        value=default_api_url,
                    )
                    checkpoint = gr.Textbox(
                        label="Ruta del checkpoint LoRA",
                        value=default_checkpoint,
                    )
                    duration = gr.Slider(10, 60, value=30, step=5, label="Duración (s)")
                    seed = gr.Number(value=1111, label="Seed (-1 = aleatorio)", precision=0)
                    lora_scale = gr.Slider(0.5, 1.5, value=1.0, step=0.1, label="LoRA scale")
                    steps = gr.Slider(4, 32, value=8, step=4, label="Pasos de inferencia")

                btn = gr.Button("🎵 Generar audio", variant="primary", size="lg")

            with gr.Column(scale=1):
                audio_out = gr.Audio(label="Audio generado", type="filepath")
                mode_badge = gr.Textbox(
                    label="Modo detectado",
                    interactive=False,
                    max_lines=1,
                )
                mode_info = gr.Markdown(label="Información del modo")

        gr.Markdown(
            "---\n"
            "**Nota:** Los modos mayores (Jónico, Lidio, Mixolidio) producen "
            "mejores resultados que los menores, dado el sesgo inherente del modelo base "
            "hacia sonoridades mayores. Ver memoria del TFG §4.8."
        )

        btn.click(
            fn=generate,
            inputs=[
                description, mode_override, api_url, groq_key,
                checkpoint, duration, seed, lora_scale, steps,
            ],
            outputs=[audio_out, mode_badge, mode_info],
        )

    return demo


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Demo LoRA Modal con clasificador LLM")
    p.add_argument("--api-url", default="http://127.0.0.1:7860",
                   help="URL de la API de ACE-Step")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                   help="Ruta del adapter LoRA a cargar")
    p.add_argument("--port", type=int, default=7861,
                   help="Puerto del servidor Gradio")
    p.add_argument("--share", action="store_true",
                   help="Generar enlace público de Gradio")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    try:
        import gradio as gr
    except ImportError:
        print("Gradio no está instalado. Ejecuta: uv add gradio", file=sys.stderr)
        sys.exit(1)

    demo = build_ui(
        default_api_url=args.api_url,
        default_checkpoint=args.checkpoint,
    )
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
