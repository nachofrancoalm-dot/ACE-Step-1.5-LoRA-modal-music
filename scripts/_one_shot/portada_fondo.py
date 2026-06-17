"""Genera fondo de portada TFG – paleta UPM + guinos musicales.

Elementos:
  - Degradado oscuro con acento rojo UPM
  - Pentagrama tenue de fondo (modos musicales)
  - Teclas de piano tenues en la base
  - Onda de audio nítida con glow fino
  - Barras de frecuencia suaves
  - Nodos de red neuronal (LoRA) zona derecha
Salida 1920x1080 @200 dpi (alta resolución, sin borrosidad).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter1d

W, H = 1920, 1080
DPI = 200          # alta resolución → sin borrosidad al escalar en Canva
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "presentacion_assets")
OUT_DIR = os.path.abspath(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# Colores UPM
BG_TOP    = "#0d0d10"   # casi negro azulado
BG_BOT    = "#1a0608"   # toque rojizo muy sutil
RED_UPM   = "#C8102E"   # rojo institucional UPM
GOLD      = "#e8a020"   # dorado cálido (acento secundario)
WHITE     = "#f0f0f0"


def hex_to_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)], dtype=float) / 255.


# ── Fondo degradado ───────────────────────────────────────────────────────────
def gradient_bg(ax):
    top, bot = hex_to_rgb(BG_TOP), hex_to_rgb(BG_BOT)
    yy, xx = np.mgrid[0:H, 0:W]
    t = np.clip(0.7 * (yy / H) + 0.3 * (xx / W), 0, 1)[..., None]
    img = bot * t + top * (1 - t)
    # viñeta perimetral sutil
    cx, cy = W * 0.5, H * 0.5
    r = np.sqrt(((xx - cx) / (W * 0.72))**2 + ((yy - cy) / (H * 0.72))**2)
    img *= np.clip(1 - 0.45 * r, 0.45, 1.0)[..., None]
    # mancha de luz roja difusa centro-izquierda (zona título)
    gx = np.exp(-((xx - W * 0.3) / (W * 0.35))**2)
    gy = np.exp(-((yy - H * 0.72) / (H * 0.55))**2)
    glow = (gx * gy)[..., None] * hex_to_rgb(RED_UPM) * 0.18
    img = np.clip(img + glow, 0, 1)
    ax.imshow(img, extent=[0, W, 0, H], origin="lower", zorder=0, aspect="auto")


# ── Pentagrama (modos musicales) ──────────────────────────────────────────────
def staff_lines(ax):
    """5 líneas de pentagrama tenues que cruzan toda la imagen."""
    y_center = H * 0.62
    spacing = H * 0.022
    for i in range(-2, 3):
        y = y_center + i * spacing
        ax.axhline(y, color=(*hex_to_rgb(RED_UPM), 0.09), lw=0.8, zorder=1)
    # notas circulares en posiciones escalares tipo modo dórico
    positions_x = [W*0.05, W*0.09, W*0.13, W*0.17, W*0.21, W*0.25, W*0.29]
    # alturas relativas al pentagrama para un modo dórico (D-E-F-G-A-B-C)
    offsets = [0, 1, -0.5, 0.5, 1.5, 2, -1]
    for xv, ov in zip(positions_x, offsets):
        yv = y_center + ov * spacing * 0.5
        ax.plot(xv, yv, "o", ms=5, color=(*hex_to_rgb(WHITE), 0.18),
                markeredgewidth=0, zorder=2)


# ── Teclas de piano ───────────────────────────────────────────────────────────
def piano_keys(ax, rng):
    """Silueta tenue de teclas en la base, con teclas blancas y negras."""
    key_w = W / 52          # 52 teclas blancas aprox.
    key_h = H * 0.10
    y0 = 0
    red = hex_to_rgb(RED_UPM)
    for i in range(52):
        x0 = i * key_w
        # tecla blanca con borde sutil
        alpha = 0.06 + 0.04 * rng.random()
        ax.add_patch(Rectangle((x0, y0), key_w * 0.92, key_h,
                               facecolor=(*hex_to_rgb(WHITE), alpha),
                               edgecolor=(*red, 0.07), lw=0.4, zorder=1))
    # teclas negras (posiciones estándar)
    black_pattern = [0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0]  # 1 = tecla negra
    bkey_w = key_w * 0.58
    bkey_h = key_h * 0.60
    bi = 0
    for i in range(52):
        if black_pattern[i % 12] == 0:
            x0 = i * key_w + key_w * 0.67
            alpha = 0.10 + 0.05 * rng.random()
            ax.add_patch(Rectangle((x0, y0), bkey_w, bkey_h,
                                   facecolor=(*red, alpha),
                                   edgecolor="none", zorder=2))


# ── Onda de audio nítida ──────────────────────────────────────────────────────
def waveform(ax, rng):
    """Onda suave con glow fino — sin borrosidad (alta resolución + sigma bajo)."""
    n = 3600
    x = np.linspace(0, W, n)
    base = H * 0.35

    # envolvente: más alta en el centro, apagada en los extremos
    env = 0.08 + 0.18 * np.exp(-((x - W * 0.52) / (W * 0.38))**2)

    # señal: mezcla de armónicos (simula forma musical)
    sig = (np.sin(2*np.pi * 4  * x / W)
         + 0.60 * np.sin(2*np.pi * 9  * x / W + 0.7)
         + 0.35 * np.sin(2*np.pi * 17 * x / W + 1.4)
         + 0.20 * np.sin(2*np.pi * 31 * x / W + 2.1)
         + 0.06 * rng.standard_normal(n))
    # suavizado leve → onda limpia, no borrosa
    sig = gaussian_filter1d(sig, sigma=1.5)
    sig /= np.max(np.abs(sig))
    y = base + sig * H * env

    red = hex_to_rgb(RED_UPM)
    gold = hex_to_rgb(GOLD)

    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    t = np.linspace(0, 1, len(segs))
    # color: rojo UPM → dorado
    colors = red[None, :] * (1 - t)[:, None] + gold[None, :] * t[:, None]

    # glow: capas anchas muy transparentes → borde nítido sin mancha
    for lw, al in [(22, 0.03), (12, 0.07), (6, 0.13)]:
        lc = LineCollection(segs, colors=np.c_[colors, np.full(len(segs), al)],
                            linewidths=lw, capstyle="round", zorder=3)
        ax.add_collection(lc)
    # línea principal: fina y opaca → nitidez total
    lc = LineCollection(segs, colors=np.c_[colors, np.ones(len(segs))],
                        linewidths=1.6, capstyle="round", zorder=4)
    ax.add_collection(lc)


# ── Barras de frecuencia ──────────────────────────────────────────────────────
def freq_bars(ax, rng):
    n = 110
    xs = np.linspace(W * 0.02, W * 0.98, n)
    bw = (xs[1] - xs[0]) * 0.50
    h_vals = (np.abs(np.sin(np.linspace(0, 8, n))) * 0.5 + 0.5) * (0.4 + 0.6 * rng.random(n))
    red = hex_to_rgb(RED_UPM)
    gold = hex_to_rgb(GOLD)
    for i, (xv, hh) in enumerate(zip(xs, h_vals)):
        t = i / n
        col = red * (1 - t) + gold * t
        ax.bar(xv, hh * H * 0.11, width=bw, bottom=H * 0.10,
               color=(*col, 0.22), zorder=1, edgecolor="none")


# ── Nodos (LoRA / red neuronal) ───────────────────────────────────────────────
def nodes(ax, rng):
    pts = np.c_[rng.uniform(W * 0.58, W * 0.97, 22),
                rng.uniform(H * 0.50, H * 0.95, 22)]
    red = hex_to_rgb(RED_UPM)
    for i in range(len(pts)):
        d = np.linalg.norm(pts - pts[i], axis=1)
        for j in np.argsort(d)[1:3]:
            ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]],
                    color=(*red, 0.09), lw=0.8, zorder=1)
    sizes = rng.uniform(8, 45, len(pts))
    ax.scatter(pts[:, 0], pts[:, 1], s=sizes, color=(*red, 0.28),
               zorder=2, edgecolor="none")
    # 3 nodos destacados
    for idx in [3, 11, 18]:
        ax.scatter(pts[idx, 0], pts[idx, 1], s=80, color=(*red, 0.65),
                   zorder=3, edgecolor=(*hex_to_rgb(GOLD), 0.5), linewidths=0.8)


# ── Main ──────────────────────────────────────────────────────────────────────
def make():
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    rng = np.random.default_rng(42)

    gradient_bg(ax)
    staff_lines(ax)
    nodes(ax, rng)
    freq_bars(ax, rng)
    piano_keys(ax, rng)
    waveform(ax, rng)

    out = os.path.join(OUT_DIR, "portada_upm.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print("Guardado:", out)


if __name__ == "__main__":
    make()
