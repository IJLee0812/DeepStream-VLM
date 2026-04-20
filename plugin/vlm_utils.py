"""Pure utility functions for VLM plugin.

These functions have ZERO dependency on GStreamer, CUDA, torch, or vLLM,
making them testable on any host machine without a GPU environment.

Extracted from gstnvvllmvlm.py to enable unit testing outside Docker.
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# YOLO26: COCO class ID -> human-readable label (8 target classes)
YOLO26_CLASS_MAPPING = {
    0: "Pedestrian",
    1: "Bike",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
    9: "Trafficlight",
    11: "Trafficsign",
}


def format_detection_hints(
    frames,
    *,
    enabled: bool,
    include_conf: bool = True,
    include_bbox: bool = True,
    max_objects: int = 10,
    detector_name: str = "YOLO26",
) -> str:
    """Format detection results as compact text hints for VLM.

    Args:
        frames: List of objects with a ``detections`` attribute.
            Each detection is a dict with keys: label, confidence, bbox.
        enabled: Master switch. Returns "" immediately when False.
        include_conf: Include confidence score in output.
        include_bbox: Include normalized [0,1] bounding box in output.
        max_objects: Max detections per frame (top-N by confidence).
        detector_name: Name shown in the hints header (e.g. "YOLOE-26").

    Returns:
        Formatted hint string, or "" if disabled / no detections.
    """
    if not enabled:
        return ""
    lines: list[str] = []
    for i, frame in enumerate(frames):
        dets = getattr(frame, "detections", [])
        if not dets:
            continue
        dets = sorted(dets, key=lambda d: d["confidence"], reverse=True)
        dets = dets[:max_objects]
        parts: list[str] = []
        for d in dets:
            s = d["label"]
            if include_conf:
                s += f"({d['confidence']})"
            if include_bbox:
                b = d["bbox"]
                s += f" [{b[0]},{b[1]},{b[2]},{b[3]}]"
            parts.append(s)
        lines.append(f"F{i}: {', '.join(parts)}")
    if not lines:
        return ""
    return f"[Object detection hints from {detector_name}]\n" + "\n".join(lines)


def format_user_prompt(
    user_prompt: str,
    stream_id: int,
    num_frames: int,
    timestamps: str,
    detection_hints: str = "",
) -> str:
    """Format user prompt by replacing placeholders.

    Available placeholders:
        {num_frames}, {stream_id}, {timestamps}, {detection_hints}

    Returns the original prompt on KeyError (unknown placeholder).
    """
    try:
        return user_prompt.format(
            num_frames=num_frames,
            stream_id=stream_id,
            timestamps=timestamps,
            detection_hints=detection_hints,
        )
    except KeyError as e:
        logger.warning("Invalid placeholder in user_prompt: %s", e)
        return user_prompt


def compute_step_ns(segment_length_sec: int, overlap_sec: int) -> int:
    """Compute segment step size in nanoseconds.

    step = max(1, segment_length_sec - overlap_sec) * 1e9
    """
    step = max(1, segment_length_sec - overlap_sec)
    return step * 1_000_000_000


def compute_sample_interval_ns(selection_fps: int) -> int | None:
    """Convert selection FPS to a nanosecond sampling interval.

    Returns None when selection_fps <= 0 (disabled).
    """
    if selection_fps and selection_fps > 0:
        return int(1_000_000_000 / selection_fps)
    return None


def get_stream_config(
    stream_prompts: dict[int, dict[str, Any]],
    stream_id: int,
    setting: str,
    default: Any,
) -> Any:
    """Get config value for a specific stream with fallback to default.

    Priority:
        1. Stream-specific setting (if exists in stream_prompts)
        2. Default value
    """
    if stream_id in stream_prompts and setting in stream_prompts[stream_id]:
        return stream_prompts[stream_id][setting]
    return default


def collect_detections(
    object_items,
    class_mapping: dict[int, str],
    min_confidence: float,
    frame_width: int,
    frame_height: int,
) -> list[dict]:
    """Extract and normalize detections from DeepStream frame metadata.

    Args:
        object_items: Iterable of NvDsObjectMeta-like objects with
            class_id, confidence, rect_params (left, top, width, height).
        class_mapping: class_id -> label mapping.
        min_confidence: Minimum confidence threshold.
        frame_width: Frame width for bbox normalization.
        frame_height: Frame height for bbox normalization.

    Returns:
        List of detection dicts with keys: label, confidence, bbox.
        bbox is (x1, y1, x2, y2) normalized to [0, 1].
    """
    fw = frame_width or 1
    fh = frame_height or 1
    detections: list[dict] = []
    for obj in object_items:
        label = class_mapping.get(obj.class_id)
        if label and obj.confidence >= min_confidence:
            x1 = round(obj.rect_params.left / fw, 3)
            y1 = round(obj.rect_params.top / fh, 3)
            x2 = round((obj.rect_params.left + obj.rect_params.width) / fw, 3)
            y2 = round((obj.rect_params.top + obj.rect_params.height) / fh, 3)
            detections.append(
                {
                    "label": label,
                    "confidence": round(obj.confidence, 2),
                    "bbox": (x1, y1, x2, y2),
                }
            )
    return detections


def to_uri(path_or_uri: str) -> str:
    """Convert a file path or URI string to a GStreamer-compatible URI."""
    if "://" in path_or_uri:
        return path_or_uri
    return "file://" + os.path.abspath(path_or_uri)


def move_built_engine(engine_dest: str | None, cwd: str | None = None) -> str | None:
    """Move TRT engine built by nvinfer to the configured destination path.

    nvinfer writes engines with pattern model_b*_gpu*_fp*.engine into CWD.
    Moves the first match to engine_dest so subsequent runs skip rebuilding.

    Returns engine_dest if moved, None otherwise.
    """
    import glob as _glob

    if not engine_dest or os.path.isfile(engine_dest):
        return None
    search_dir = cwd or "."
    pattern_a = os.path.join(search_dir, "model_b*_gpu*_fp*.engine")
    pattern_b = os.path.join(search_dir, "*.onnx_b*_gpu*_fp*.engine")
    built = _glob.glob(pattern_a) + _glob.glob(pattern_b)
    if not built:
        return None
    dest_dir = os.path.dirname(engine_dest)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)
    os.rename(built[0], engine_dest)
    return engine_dest


def check_onnx_exists(nvinfer_config: str) -> str | None:
    """Parse nvinfer config and return the onnx-file path if it is missing.

    Returns the missing ONNX path string, or None if the file exists or
    the config cannot be parsed.
    """
    import re

    try:
        with open(nvinfer_config) as f:
            for line in f:
                m = re.match(r"onnx-file\s*=\s*(\S+)", line)
                if m:
                    onnx_path = m.group(1)
                    return None if os.path.exists(onnx_path) else onnx_path
    except Exception:
        pass
    return None


def parse_nvinfer_config(nvinfer_config: str) -> dict[str, str]:
    """Parse key=value pairs from an nvinfer [property] section.

    Returns a dict of stripped string values. Only the first occurrence of
    each key is kept. Missing file or parse errors return an empty dict.
    """
    import re

    result: dict[str, str] = {}
    try:
        with open(nvinfer_config) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("["):
                    continue
                m = re.match(r"([a-zA-Z0-9_\-]+)\s*=\s*(\S.*)", line)
                if m:
                    key = m.group(1)
                    if key not in result:
                        result[key] = m.group(2).strip()
    except Exception:
        pass
    return result


def load_class_mapping(labelfile: str | None) -> dict[int, str]:
    """Load ``class_id → label`` from a labels.txt (one name per line).

    Line N (0-indexed) becomes class_id N. Blank lines are skipped but
    still increment the index so indices stay aligned with the file.
    Falls back to ``YOLO26_CLASS_MAPPING`` when labelfile is missing,
    unreadable, or empty — preserving legacy YOLO26 behavior.
    """
    if not labelfile or not os.path.isfile(labelfile):
        return YOLO26_CLASS_MAPPING
    try:
        mapping: dict[int, str] = {}
        with open(labelfile, encoding="utf-8") as f:
            for i, line in enumerate(f):
                name = line.strip()
                if name:
                    mapping[i] = name
        return mapping if mapping else YOLO26_CLASS_MAPPING
    except Exception:
        return YOLO26_CLASS_MAPPING


def is_segmentation_config(nvinfer_config: str) -> bool:
    """Return True if the nvinfer config declares network-type=3 (seg)."""
    props = parse_nvinfer_config(nvinfer_config)
    return props.get("network-type") == "3" or props.get("output-instance-mask") == "1"
