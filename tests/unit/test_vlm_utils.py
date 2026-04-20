"""Tests for plugin/vlm_utils.py — pure logic, no GStreamer/CUDA deps."""

from types import SimpleNamespace

import pytest
from vlm_utils import (
    YOLO26_CLASS_MAPPING,
    collect_detections,
    compute_sample_interval_ns,
    compute_step_ns,
    format_detection_hints,
    format_user_prompt,
    get_stream_config,
    is_segmentation_config,
    load_class_mapping,
    parse_nvinfer_config,
    to_uri,
)

# ── Helpers ──


def _make_frame(detections=None):
    """Create a SimpleNamespace mimicking BufferData with detections."""
    return SimpleNamespace(detections=detections or [])


def _make_detection(label, confidence, bbox):
    return {"label": label, "confidence": confidence, "bbox": bbox}


# ── YOLO26_CLASS_MAPPING ──


class TestYOLO26ClassMapping:
    def test_expected_classes(self):
        assert YOLO26_CLASS_MAPPING[0] == "Pedestrian"
        assert YOLO26_CLASS_MAPPING[2] == "Car"
        assert YOLO26_CLASS_MAPPING[7] == "Truck"
        assert YOLO26_CLASS_MAPPING[11] == "Trafficsign"

    def test_eight_classes(self):
        assert len(YOLO26_CLASS_MAPPING) == 8

    def test_missing_ids_not_present(self):
        for cid in (4, 6, 8, 10, 12, 79):
            assert cid not in YOLO26_CLASS_MAPPING


# ── format_detection_hints ──


class TestFormatDetectionHints:
    def test_disabled_returns_empty(self):
        frames = [_make_frame([_make_detection("Car", 0.9, (0.1, 0.2, 0.3, 0.4))])]
        assert format_detection_hints(frames, enabled=False) == ""

    def test_empty_frames_returns_empty(self):
        assert format_detection_hints([], enabled=True) == ""

    def test_no_detections_returns_empty(self):
        frames = [_make_frame([]), _make_frame([])]
        assert format_detection_hints(frames, enabled=True) == ""

    def test_single_detection_full_format(self):
        frames = [_make_frame([_make_detection("Car", 0.95, (0.1, 0.2, 0.8, 0.9))])]
        result = format_detection_hints(frames, enabled=True)
        assert result == ("[Object detection hints from YOLO26]\nF0: Car(0.95) [0.1,0.2,0.8,0.9]")

    def test_confidence_sort_descending(self):
        dets = [
            _make_detection("Car", 0.5, (0, 0, 1, 1)),
            _make_detection("Truck", 0.9, (0, 0, 1, 1)),
            _make_detection("Bus", 0.7, (0, 0, 1, 1)),
        ]
        result = format_detection_hints([_make_frame(dets)], enabled=True)
        lines = result.split("\n")
        # F0 line should have Truck first (0.9), then Bus (0.7), then Car (0.5)
        assert "Truck" in lines[1]
        f0_parts = lines[1].split(", ")
        assert "Truck" in f0_parts[0]
        assert "Bus" in f0_parts[1]
        assert "Car" in f0_parts[2]

    def test_max_objects_limit(self):
        dets = [_make_detection(f"Obj{i}", 0.9 - i * 0.1, (0, 0, 1, 1)) for i in range(5)]
        result = format_detection_hints([_make_frame(dets)], enabled=True, max_objects=2)
        lines = result.split("\n")
        # F0 should only have 2 items (split by ", " which separates detections)
        f0_line = lines[1].removeprefix("F0: ")
        assert len(f0_line.split("], ")) == 2

    def test_bbox_excluded(self):
        frames = [_make_frame([_make_detection("Car", 0.9, (0.1, 0.2, 0.3, 0.4))])]
        result = format_detection_hints(frames, enabled=True, include_bbox=False)
        assert "[" not in result.split("\n")[1] or "YOLO26]" in result.split("\n")[0]
        assert "0.1" not in result

    def test_confidence_excluded(self):
        frames = [_make_frame([_make_detection("Car", 0.9, (0.1, 0.2, 0.3, 0.4))])]
        result = format_detection_hints(frames, enabled=True, include_conf=False)
        assert "(0.9)" not in result
        assert "Car" in result

    def test_both_excluded_label_only(self):
        frames = [_make_frame([_make_detection("Truck", 0.8, (0.1, 0.2, 0.3, 0.4))])]
        result = format_detection_hints(
            frames, enabled=True, include_conf=False, include_bbox=False
        )
        assert "F0: Truck" in result
        assert "(" not in result.split("\n")[1]

    def test_multi_frame_format(self):
        frames = [
            _make_frame([_make_detection("Car", 0.9, (0, 0, 1, 1))]),
            _make_frame([]),  # empty — should be skipped
            _make_frame([_make_detection("Bus", 0.8, (0, 0, 1, 1))]),
        ]
        result = format_detection_hints(frames, enabled=True)
        assert "F0:" in result
        assert "F1:" not in result  # empty frame skipped
        assert "F2:" in result

    def test_header_format(self):
        frames = [_make_frame([_make_detection("Car", 0.9, (0, 0, 1, 1))])]
        result = format_detection_hints(frames, enabled=True)
        assert result.startswith("[Object detection hints from YOLO26]\n")

    def test_frames_without_detections_attr(self):
        """Frames without .detections attribute should be handled gracefully."""
        frame = SimpleNamespace()  # no detections attr
        result = format_detection_hints([frame], enabled=True)
        assert result == ""


# ── format_user_prompt ──


class TestFormatUserPrompt:
    def test_all_placeholders(self):
        prompt = (
            "Analyze {num_frames} frames from stream {stream_id} at {timestamps}. {detection_hints}"
        )
        result = format_user_prompt(prompt, stream_id=0, num_frames=5, timestamps="1.0s 2.0s")
        assert "5 frames" in result
        assert "stream 0" in result
        assert "1.0s 2.0s" in result

    def test_no_placeholders(self):
        prompt = "Describe the scene in detail."
        result = format_user_prompt(prompt, stream_id=0, num_frames=3, timestamps="0.0s")
        assert result == "Describe the scene in detail."

    def test_invalid_placeholder_returns_original(self):
        prompt = "Test {unknown_placeholder} here."
        result = format_user_prompt(prompt, stream_id=0, num_frames=1, timestamps="0.0s")
        assert result == prompt

    def test_detection_hints_injected(self):
        prompt = "Scene analysis.\n{detection_hints}\nDescribe."
        hints = "[Object detection hints from YOLO26]\nF0: Car(0.9)"
        result = format_user_prompt(
            prompt, stream_id=0, num_frames=1, timestamps="0.0s", detection_hints=hints
        )
        assert "YOLO26" in result
        assert "Car(0.9)" in result

    def test_empty_prompt(self):
        assert format_user_prompt("", stream_id=0, num_frames=0, timestamps="") == ""

    def test_detection_hints_default_empty(self):
        prompt = "Test {detection_hints} end."
        result = format_user_prompt(prompt, stream_id=0, num_frames=1, timestamps="0.0s")
        assert result == "Test  end."


# ── compute_step_ns ──


class TestComputeStepNs:
    def test_normal_case(self):
        assert compute_step_ns(10, 2) == 8_000_000_000

    def test_no_overlap(self):
        assert compute_step_ns(10, 0) == 10_000_000_000

    def test_overlap_equals_segment(self):
        # max(1, 10-10) = 1
        assert compute_step_ns(10, 10) == 1_000_000_000

    def test_overlap_exceeds_segment(self):
        # max(1, 5-10) = 1
        assert compute_step_ns(5, 10) == 1_000_000_000

    def test_negative_overlap(self):
        # max(1, 10-(-5)) = 15
        assert compute_step_ns(10, -5) == 15_000_000_000


# ── compute_sample_interval_ns ──


class TestComputeSampleIntervalNs:
    def test_fps_2(self):
        assert compute_sample_interval_ns(2) == 500_000_000

    def test_fps_1(self):
        assert compute_sample_interval_ns(1) == 1_000_000_000

    def test_fps_30(self):
        result = compute_sample_interval_ns(30)
        assert result == 33_333_333

    def test_fps_zero_disabled(self):
        assert compute_sample_interval_ns(0) is None

    def test_fps_negative_disabled(self):
        assert compute_sample_interval_ns(-1) is None


# ── get_stream_config ──


class TestGetStreamConfig:
    def test_override_exists(self):
        prompts = {0: {"user_prompt": "Override!", "temperature": 0.1}}
        assert get_stream_config(prompts, 0, "user_prompt", "default") == "Override!"

    def test_fallback_to_default(self):
        prompts = {0: {"user_prompt": "Override!"}}
        assert get_stream_config(prompts, 0, "temperature", 0.7) == 0.7

    def test_stream_not_present(self):
        prompts = {0: {"user_prompt": "Override!"}}
        assert get_stream_config(prompts, 1, "user_prompt", "default") == "default"

    def test_empty_prompts(self):
        assert get_stream_config({}, 0, "user_prompt", "default") == "default"

    def test_none_value_override(self):
        """If the override value is None, it should still be returned."""
        prompts = {0: {"system_prompt": None}}
        assert get_stream_config(prompts, 0, "system_prompt", "fallback") is None


# ── collect_detections ──


class TestCollectDetections:
    def _make_obj(self, class_id, confidence, left, top, width, height):
        return SimpleNamespace(
            class_id=class_id,
            confidence=confidence,
            rect_params=SimpleNamespace(left=left, top=top, width=width, height=height),
        )

    def test_known_class_id(self):
        objs = [self._make_obj(2, 0.9, 100, 200, 50, 60)]
        result = collect_detections(objs, YOLO26_CLASS_MAPPING, 0.3, 1920, 1080)
        assert len(result) == 1
        assert result[0]["label"] == "Car"
        assert result[0]["confidence"] == 0.9

    def test_unknown_class_id_ignored(self):
        objs = [self._make_obj(4, 0.9, 100, 200, 50, 60)]  # class 4 not in mapping
        result = collect_detections(objs, YOLO26_CLASS_MAPPING, 0.3, 1920, 1080)
        assert len(result) == 0

    def test_below_min_confidence_filtered(self):
        objs = [self._make_obj(2, 0.2, 100, 200, 50, 60)]  # conf 0.2 < 0.3
        result = collect_detections(objs, YOLO26_CLASS_MAPPING, 0.3, 1920, 1080)
        assert len(result) == 0

    def test_normalized_bbox(self):
        objs = [self._make_obj(2, 0.9, 960, 540, 192, 108)]
        result = collect_detections(objs, YOLO26_CLASS_MAPPING, 0.3, 1920, 1080)
        bbox = result[0]["bbox"]
        assert bbox[0] == 0.5  # x1 = 960/1920
        assert bbox[1] == 0.5  # y1 = 540/1080
        assert bbox[2] == 0.6  # x2 = (960+192)/1920
        assert bbox[3] == 0.6  # y2 = (540+108)/1080

    def test_zero_frame_dimensions(self):
        """frame_width=0 or frame_height=0 should not cause division by zero."""
        objs = [self._make_obj(2, 0.9, 100, 200, 50, 60)]
        result = collect_detections(objs, YOLO26_CLASS_MAPPING, 0.3, 0, 0)
        assert len(result) == 1  # should not crash

    def test_multiple_detections(self):
        objs = [
            self._make_obj(0, 0.9, 100, 200, 50, 60),  # Pedestrian
            self._make_obj(2, 0.8, 300, 400, 70, 80),  # Car
            self._make_obj(7, 0.7, 500, 600, 90, 100),  # Truck
        ]
        result = collect_detections(objs, YOLO26_CLASS_MAPPING, 0.3, 1920, 1080)
        assert len(result) == 3
        labels = [d["label"] for d in result]
        assert "Pedestrian" in labels
        assert "Car" in labels
        assert "Truck" in labels

    def test_empty_object_items(self):
        result = collect_detections([], YOLO26_CLASS_MAPPING, 0.3, 1920, 1080)
        assert result == []


# ── to_uri ──


class TestToUri:
    def test_absolute_path(self):
        result = to_uri("/workspace/video.mp4")
        assert result == "file:///workspace/video.mp4"

    def test_rtsp_passthrough(self):
        uri = "rtsp://192.168.1.100:8554/stream"
        assert to_uri(uri) == uri

    def test_http_passthrough(self):
        uri = "http://example.com/video.mp4"
        assert to_uri(uri) == uri

    def test_file_uri_passthrough(self):
        uri = "file:///workspace/video.mp4"
        assert to_uri(uri) == uri

    def test_relative_path_becomes_absolute(self):
        result = to_uri("video.mp4")
        assert result.startswith("file://")
        assert "/video.mp4" in result
        assert "://" in result


# ── load_class_mapping (YOLOE runtime labels, YOLO26 fallback) ──


class TestLoadClassMapping:
    def test_missing_path_falls_back_to_yolo26(self):
        assert load_class_mapping(None) is YOLO26_CLASS_MAPPING

    def test_empty_string_falls_back_to_yolo26(self):
        assert load_class_mapping("") is YOLO26_CLASS_MAPPING

    def test_nonexistent_file_falls_back_to_yolo26(self):
        assert load_class_mapping("/no/such/labels.txt") is YOLO26_CLASS_MAPPING

    def test_loads_labels_in_order(self, tmp_path):
        labelfile = tmp_path / "labels.txt"
        labelfile.write_text("vehicle\nperson\nmotorcycle\n")
        result = load_class_mapping(str(labelfile))
        assert result == {0: "vehicle", 1: "person", 2: "motorcycle"}

    def test_blank_lines_skip_index(self, tmp_path):
        labelfile = tmp_path / "labels.txt"
        labelfile.write_text("cat\n\ndog\n")
        result = load_class_mapping(str(labelfile))
        assert result == {0: "cat", 2: "dog"}

    def test_empty_file_falls_back_to_yolo26(self, tmp_path):
        labelfile = tmp_path / "labels.txt"
        labelfile.write_text("")
        assert load_class_mapping(str(labelfile)) is YOLO26_CLASS_MAPPING

    def test_single_class(self, tmp_path):
        labelfile = tmp_path / "labels.txt"
        labelfile.write_text("face\n")
        assert load_class_mapping(str(labelfile)) == {0: "face"}

    def test_utf8_labels_preserved(self, tmp_path):
        labelfile = tmp_path / "labels.txt"
        labelfile.write_text("차량\n보행자\n", encoding="utf-8")
        result = load_class_mapping(str(labelfile))
        assert result == {0: "차량", 1: "보행자"}

    def test_unreadable_file_falls_back_to_yolo26(self, tmp_path):
        """An IOError/PermissionError during open falls back to YOLO26 mapping."""
        labelfile = tmp_path / "labels.txt"
        labelfile.write_text("cat\ndog\n")
        labelfile.chmod(0o000)
        try:
            result = load_class_mapping(str(labelfile))
            assert result is YOLO26_CLASS_MAPPING
        finally:
            labelfile.chmod(0o644)


# ── parse_nvinfer_config ──


class TestParseNvinferConfig:
    def test_basic_keyvalue(self, tmp_path):
        cfg = tmp_path / "config.txt"
        cfg.write_text("[property]\nbatch-size=1\nnetwork-type=3\n")
        result = parse_nvinfer_config(str(cfg))
        assert result["batch-size"] == "1"
        assert result["network-type"] == "3"

    def test_skips_comments_and_sections(self, tmp_path):
        cfg = tmp_path / "config.txt"
        cfg.write_text(
            "# comment line\n"
            "[property]\n"
            "labelfile-path=/workspace/models/labels.txt\n"
            "[class-attrs-all]\n"
            "pre-cluster-threshold=0.25\n"
        )
        result = parse_nvinfer_config(str(cfg))
        assert result["labelfile-path"] == "/workspace/models/labels.txt"
        assert result["pre-cluster-threshold"] == "0.25"

    def test_missing_file_returns_empty(self):
        assert parse_nvinfer_config("/no/such/file.txt") == {}

    def test_first_occurrence_wins(self, tmp_path):
        cfg = tmp_path / "config.txt"
        cfg.write_text("batch-size=1\nbatch-size=4\n")
        assert parse_nvinfer_config(str(cfg))["batch-size"] == "1"

    def test_whitespace_around_equals(self, tmp_path):
        cfg = tmp_path / "config.txt"
        cfg.write_text("onnx-file = /models/x.onnx\n")
        assert parse_nvinfer_config(str(cfg))["onnx-file"] == "/models/x.onnx"


# ── is_segmentation_config ──


class TestIsSegmentationConfig:
    def test_detection_config_false(self, tmp_path):
        cfg = tmp_path / "config.txt"
        cfg.write_text("network-type=0\n")
        assert is_segmentation_config(str(cfg)) is False

    def test_seg_network_type_3_true(self, tmp_path):
        cfg = tmp_path / "config.txt"
        cfg.write_text("network-type=3\n")
        assert is_segmentation_config(str(cfg)) is True

    def test_output_instance_mask_true(self, tmp_path):
        cfg = tmp_path / "config.txt"
        cfg.write_text("output-instance-mask=1\n")
        assert is_segmentation_config(str(cfg)) is True

    def test_missing_file_false(self):
        assert is_segmentation_config("/no/such/file.txt") is False


# ── format_detection_hints with custom detector_name ──


class TestFormatDetectionHintsDetectorName:
    def test_default_detector_name_yolo26(self):
        frame = _make_frame([_make_detection("car", 0.9, (0.1, 0.2, 0.3, 0.4))])
        out = format_detection_hints([frame], enabled=True)
        assert "[Object detection hints from YOLO26]" in out

    def test_custom_detector_name(self):
        frame = _make_frame([_make_detection("vehicle", 0.9, (0.1, 0.2, 0.3, 0.4))])
        out = format_detection_hints([frame], enabled=True, detector_name="YOLOE-26")
        assert "[Object detection hints from YOLOE-26]" in out

    def test_seg_detector_name(self):
        frame = _make_frame([_make_detection("person", 0.8, (0.0, 0.0, 1.0, 1.0))])
        out = format_detection_hints([frame], enabled=True, detector_name="YOLOE-26 Seg")
        assert "[Object detection hints from YOLOE-26 Seg]" in out
