from __future__ import annotations

import math
from pathlib import Path

import numpy as np


ROOT = Path("/Users/simranarora/Documents/Simran outreach")
OUT = ROOT / "assets" / "creatives"
FRAMES = OUT / "_frames"

W = 900
H = 900

WHITE = np.array([250, 248, 253], dtype=np.float32)
PLUM = np.array([128, 76, 214], dtype=np.float32)
PLUM_DARK = np.array([78, 45, 143], dtype=np.float32)
LILAC = np.array([230, 217, 255], dtype=np.float32)
YELLOW = np.array([247, 214, 118], dtype=np.float32)
BEIGE = np.array([233, 220, 197], dtype=np.float32)
BROWN = np.array([156, 98, 49], dtype=np.float32)
MINT = np.array([118, 198, 164], dtype=np.float32)
INK = np.array([42, 30, 66], dtype=np.float32)
SILVER = np.array([227, 229, 236], dtype=np.float32)


def ensure_dirs() -> None:
    (OUT / "gifs").mkdir(parents=True, exist_ok=True)
    FRAMES.mkdir(parents=True, exist_ok=True)


def make_canvas() -> np.ndarray:
    img = np.ones((H, W, 3), dtype=np.float32)
    img[:] = WHITE
    yy, xx = np.mgrid[0:H, 0:W]
    glow1 = np.exp(-(((xx - 160) ** 2) / (2 * 220**2) + ((yy - 120) ** 2) / (2 * 170**2)))
    glow2 = np.exp(-(((xx - 760) ** 2) / (2 * 260**2) + ((yy - 760) ** 2) / (2 * 240**2)))
    tint = glow1[..., None] * np.array([0.06, 0.03, 0.12]) + glow2[..., None] * np.array([0.03, 0.01, 0.08])
    img *= 1 - tint
    img += 255 * tint
    return img


def blend(img: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float = 1.0) -> None:
    a = np.clip(mask * alpha, 0.0, 1.0)[..., None]
    img[:] = img * (1 - a) + color * a


def rounded_rect_mask(x1: int, y1: int, x2: int, y2: int, r: int) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W]
    core = (xx >= x1 + r) & (xx <= x2 - r) & (yy >= y1) & (yy <= y2)
    core |= (xx >= x1) & (xx <= x2) & (yy >= y1 + r) & (yy <= y2 - r)
    tl = ((xx - (x1 + r)) ** 2 + (yy - (y1 + r)) ** 2) <= r**2
    tr = ((xx - (x2 - r)) ** 2 + (yy - (y1 + r)) ** 2) <= r**2
    bl = ((xx - (x1 + r)) ** 2 + (yy - (y2 - r)) ** 2) <= r**2
    br = ((xx - (x2 - r)) ** 2 + (yy - (y2 - r)) ** 2) <= r**2
    return (core | tl | tr | bl | br).astype(np.float32)


def ellipse_mask(cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W]
    return ((((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1).astype(np.float32)


def circle_mask(cx: float, cy: float, r: float) -> np.ndarray:
    return ellipse_mask(cx, cy, r, r)


def line_mask(x1: float, y1: float, x2: float, y2: float, thickness: float) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W]
    px = xx - x1
    py = yy - y1
    dx = x2 - x1
    dy = y2 - y1
    denom = dx * dx + dy * dy + 1e-6
    t = np.clip((px * dx + py * dy) / denom, 0.0, 1.0)
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    dist2 = (xx - proj_x) ** 2 + (yy - proj_y) ** 2
    return (dist2 <= (thickness / 2) ** 2).astype(np.float32)


def tooth_body() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outer = rounded_rect_mask(285, 170, 615, 720, 105)
    inner = rounded_rect_mask(320, 210, 580, 685, 82)
    highlight = ellipse_mask(430, 300, 120, 230) * outer
    return outer, inner, highlight


def save_ppm(path: Path, img: np.ndarray) -> None:
    arr = np.clip(img, 0, 255).astype(np.uint8)
    with path.open("wb") as f:
      f.write(f"P6\n{arr.shape[1]} {arr.shape[0]}\n255\n".encode("ascii"))
      f.write(arr.tobytes())


def write_frames_section_03() -> Path:
    folder = FRAMES / "section-03"
    folder.mkdir(parents=True, exist_ok=True)
    outer, inner, highlight = tooth_body()
    for i in range(32):
        t = i / 31
        img = make_canvas()
        shadow = rounded_rect_mask(295, 185, 625, 730, 105)
        blend(img, shadow, np.array([238, 232, 245], dtype=np.float32), 0.55)
        blend(img, outer, np.array([255, 255, 255], dtype=np.float32), 1.0)
        blend(img, inner, BEIGE, 0.95)
        blend(img, highlight, np.array([255, 255, 255], dtype=np.float32), 0.25)

        stain_band = rounded_rect_mask(320, 210, 580, 325, 82) * inner
        blend(img, stain_band, YELLOW, 0.06 + 0.22 * t)

        for idx in range(14):
            px = 340 + idx * 18
            py = 140 + ((idx % 3) * 14) + (1 - t) * 90 + 6 * math.sin(t * math.pi * 2 + idx)
            particle = circle_mask(px, py, 7 + (idx % 2))
            blend(img, particle, BROWN, 0.85 if py < 330 else 0.45)
            if py >= 250:
                blend(img, particle * stain_band, BROWN, 0.8)

        sparkle = ellipse_mask(470, 260, 70, 110) * inner
        blend(img, sparkle, np.array([255, 255, 255], dtype=np.float32), 0.12)
        save_ppm(folder / f"frame_{i:03d}.ppm", img)
    return folder


def write_frames_section_05() -> Path:
    folder = FRAMES / "section-05"
    folder.mkdir(parents=True, exist_ok=True)
    outer, inner, highlight = tooth_body()
    for i in range(32):
        t = i / 31
        img = make_canvas()
        blend(img, outer, np.array([255, 255, 255], dtype=np.float32), 1.0)
        blend(img, inner, np.array([239, 232, 215], dtype=np.float32), 1.0)
        yellow_cast = rounded_rect_mask(320, 215, 580, 685, 82) * inner
        blend(img, yellow_cast, YELLOW, 0.26)

        sweep_x = 220 + t * 520
        sweep = ellipse_mask(sweep_x, 430, 150, 290) * inner
        blend(img, sweep, PLUM, 0.34)

        neutral_zone = rounded_rect_mask(320, 215, int(min(580, sweep_x)), 685, 82) * inner
        blend(img, neutral_zone, np.array([245, 243, 238], dtype=np.float32), 0.22)
        blend(img, highlight, np.array([255, 255, 255], dtype=np.float32), 0.2)

        for ring_i in range(3):
            ring = circle_mask(170 + ring_i * 82, 450, 36 + ring_i * 10) - circle_mask(170 + ring_i * 82, 450, 25 + ring_i * 10)
            blend(img, np.clip(ring, 0, 1), PLUM, 0.22 - ring_i * 0.04)

        save_ppm(folder / f"frame_{i:03d}.ppm", img)
    return folder


def draw_card(img: np.ndarray, x: int, y: int, active: float) -> None:
    card = rounded_rect_mask(x, y, x + 180, y + 260, 34)
    blend(img, card, np.array([255, 255, 255], dtype=np.float32), 1.0)
    if active > 0:
        glow = rounded_rect_mask(x - 8, y - 8, x + 188, y + 268, 42)
        blend(img, glow, LILAC, 0.22 * active)
        bar = rounded_rect_mask(x + 18, y + 22, x + 162, y + 38, 8)
        blend(img, bar, PLUM, 0.45 * active)


def write_frames_section_08() -> Path:
    folder = FRAMES / "section-08"
    folder.mkdir(parents=True, exist_ok=True)
    steps = [(70, 320), (270, 320), (470, 320), (670, 320)]
    for i in range(36):
        t = i / 35
        img = make_canvas()
        active_step = min(3, int(t * 4))
        step_progress = (t * 4) - active_step

        for idx, (x, y) in enumerate(steps):
            active = 1.0 if idx == active_step else 0.0
            draw_card(img, x, y, active)

        # Pump icon
        x, y = steps[0]
        blend(img, rounded_rect_mask(x + 62, y + 78, x + 120, y + 158, 18), SILVER, 1.0)
        blend(img, rounded_rect_mask(x + 78, y + 56, x + 105, y + 86, 10), PLUM_DARK, 1.0)
        drop = circle_mask(x + 145, y + 145 + 18 * math.sin(t * math.pi), 12)
        blend(img, drop, PLUM, 0.9)

        # Brush icon
        x, y = steps[1]
        handle = line_mask(x + 48, y + 175, x + 136, y + 120, 18)
        brush = rounded_rect_mask(x + 112, y + 96, x + 148, y + 126, 10)
        foam = circle_mask(x + 150, y + 90 + 6 * math.sin(t * math.pi * 4), 18)
        blend(img, handle, PLUM_DARK, 1.0)
        blend(img, brush, LILAC, 1.0)
        blend(img, foam, np.array([255, 255, 255], dtype=np.float32), 0.95)

        # Rinse icon
        x, y = steps[2]
        mouth = rounded_rect_mask(x + 55, y + 110, x + 130, y + 160, 24)
        wave = line_mask(x + 140, y + 115, x + 165, y + 150, 16)
        wave2 = line_mask(x + 132, y + 145, x + 162, y + 182, 12)
        blend(img, mouth, np.array([244, 228, 238], dtype=np.float32), 1.0)
        blend(img, wave, MINT, 0.85)
        blend(img, wave2, MINT, 0.7)

        # Smile icon
        x, y = steps[3]
        arc = ellipse_mask(x + 92, y + 138, 48, 32) - ellipse_mask(x + 92, y + 124, 50, 32)
        spark1 = line_mask(x + 128, y + 82, x + 128, y + 118, 8) + line_mask(x + 110, y + 100, x + 146, y + 100, 8)
        spark2 = line_mask(x + 52, y + 88, x + 52, y + 112, 6) + line_mask(x + 40, y + 100, x + 64, y + 100, 6)
        blend(img, np.clip(arc, 0, 1), PLUM_DARK, 0.9)
        blend(img, np.clip(spark1, 0, 1), YELLOW, 0.9)
        blend(img, np.clip(spark2, 0, 1), YELLOW, 0.8)

        # Progress dot
        y_line = 650
        for idx, (x, _) in enumerate(steps):
            dot = circle_mask(x + 90, y_line, 12 if idx == active_step else 9)
            blend(img, dot, PLUM if idx <= active_step else SILVER, 0.95)
        if active_step < 3:
            cx1 = steps[active_step][0] + 90
            cx2 = steps[active_step + 1][0] + 90
            moving = circle_mask(cx1 + (cx2 - cx1) * step_progress, y_line, 10)
            blend(img, moving, PLUM_DARK, 1.0)

        save_ppm(folder / f"frame_{i:03d}.ppm", img)
    return folder


def main() -> None:
    ensure_dirs()
    write_frames_section_03()
    write_frames_section_05()
    write_frames_section_08()


if __name__ == "__main__":
    main()
