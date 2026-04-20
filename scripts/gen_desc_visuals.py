"""Generate README visualization images from a VLM pipeline JSON output.

Produces ``assets/images/desc{1,2,3}.jpg`` — 1280×720 side-by-side:
  * left 640×720: middle frame of the chosen segment
  * right 640×720: text summary (scene_summary + top key_objects)

Usage (run inside the ds9-vlm-dev container):
    python3 scripts/gen_desc_visuals.py \\
        --json results/pipeline_B_yolo26_detect.json \\
        --video assets/videos/sample.mp4 \\
        --out-dir assets/images \\
        --segments 0 3 6
"""

import argparse
import json
import os
import re
import textwrap

import cv2
from PIL import Image, ImageDraw, ImageFont

PANEL_W = 640
PANEL_H = 720
OUT_W = PANEL_W * 2
OUT_H = PANEL_H
BG = (20, 24, 32)
TEXT_FG = (240, 240, 240)
ACCENT = (102, 204, 255)


def _parse_vlm_json(raw: str) -> dict | None:
    """Extract the JSON payload emitted by the VLM (wrapped in ```json ... ```)."""
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except Exception:
        return None


def _extract_frame(video_path: str, time_sec: float, out_path: str) -> None:
    """Pull a single frame at ``time_sec`` via OpenCV (avoids ffmpeg deps)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = int(round(time_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame at t={time_sec}s (idx={frame_idx})")
    cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])


def _find_font(size: int) -> ImageFont.FreeTypeFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    ):
        if os.path.isfile(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _render_text_panel(parsed: dict, start: float, end: float) -> Image.Image:
    panel = Image.new("RGB", (PANEL_W, PANEL_H), BG)
    draw = ImageDraw.Draw(panel)

    title_font = _find_font(24)
    header_font = _find_font(18)
    body_font = _find_font(16)

    # Header line: segment timing
    draw.text((24, 20), f"Segment [{start:.1f}s — {end:.1f}s]", fill=ACCENT, font=title_font)

    y = 70
    # Scene summary (wrapped)
    draw.text((24, y), "Scene Summary", fill=ACCENT, font=header_font)
    y += 28
    summary = parsed.get("scene_summary", "(no summary)")
    for line in textwrap.wrap(summary, width=58):
        draw.text((24, y), line, fill=TEXT_FG, font=body_font)
        y += 22
    y += 18

    # Key objects (top N)
    draw.text((24, y), "Key Objects (YOLO26 grounded)", fill=ACCENT, font=header_font)
    y += 28
    for obj in (parsed.get("key_objects") or [])[:4]:
        t = obj.get("type", "object")
        desc = obj.get("description", "")
        line = f"• {t}: {desc}"
        for wrapped in textwrap.wrap(line, width=58):
            draw.text((24, y), wrapped, fill=TEXT_FG, font=body_font)
            y += 22
        y += 4

    # Road + weather small footer
    y = PANEL_H - 70
    road = parsed.get("road_type", "")
    weather = parsed.get("weather", "")
    traffic = parsed.get("traffic_density", "")
    footer_font = _find_font(14)
    parts = [
        p
        for p in [f"road: {road}", f"weather: {weather}", f"traffic: {traffic}"]
        if p.split(": ")[1]
    ]
    for i, p in enumerate(parts):
        draw.text((24, y + i * 18), p, fill=(160, 180, 200), font=footer_font)
    return panel


def _compose(frame_path: str, text_panel: Image.Image, out_path: str) -> None:
    frame = Image.open(frame_path).convert("RGB")
    # Letterbox frame into 640x720
    frame.thumbnail((PANEL_W, PANEL_H), Image.LANCZOS)
    canvas = Image.new("RGB", (OUT_W, OUT_H), BG)
    fx = (PANEL_W - frame.width) // 2
    fy = (PANEL_H - frame.height) // 2
    canvas.paste(frame, (fx, fy))
    canvas.paste(text_panel, (PANEL_W, 0))
    canvas.save(out_path, "JPEG", quality=92)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--segments", type=int, nargs="+", default=[0, 3, 6])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.json) as f:
        data = json.load(f)
    segments = data["segments"]

    for i, seg_idx in enumerate(args.segments, start=1):
        seg = segments[seg_idx]
        start = seg["segment"]["start_time"]
        end = seg["segment"]["end_time"]
        mid = (start + end) / 2.0

        parsed = _parse_vlm_json(seg["result"])
        if parsed is None:
            print(f"[skip] seg {seg_idx}: could not parse VLM JSON")
            continue

        frame_tmp = os.path.join(args.out_dir, f"_tmp_frame_{i}.jpg")
        _extract_frame(args.video, mid, frame_tmp)

        panel = _render_text_panel(parsed, start, end)
        out_path = os.path.join(args.out_dir, f"desc{i}.jpg")
        _compose(frame_tmp, panel, out_path)
        os.remove(frame_tmp)
        print(f"[ok] desc{i}.jpg → seg {seg_idx} @ t={mid:.1f}s")


if __name__ == "__main__":
    main()
