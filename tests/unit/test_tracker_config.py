"""Tests for configs/config_tracker_NvDCF_perf.yml — pure config validation, no GPU required."""

import os

import pytest
import yaml

TRACKER_CONFIG = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "configs", "config_tracker_NvDCF_perf.yml")
)


class TestTrackerConfigExists:
    def test_file_exists(self):
        assert os.path.isfile(TRACKER_CONFIG), f"Tracker config not found: {TRACKER_CONFIG}"


class TestTrackerConfigValid:
    @pytest.fixture(autouse=True)
    def _load(self):
        with open(TRACKER_CONFIG) as f:
            content = f.read()
        # Strip OpenCV/ROS %YAML:1.0 directive — not valid in stdlib yaml
        content = "\n".join(
            line for line in content.splitlines() if not line.startswith("%YAML")
        )
        self.cfg = yaml.safe_load(content)

    def test_parses_as_dict(self):
        assert isinstance(self.cfg, dict)

    def test_base_config_present(self):
        assert "BaseConfig" in self.cfg

    def test_min_detector_confidence_present(self):
        assert "minDetectorConfidence" in self.cfg["BaseConfig"]

    def test_min_detector_confidence_value(self):
        assert self.cfg["BaseConfig"]["minDetectorConfidence"] <= 0.25

    def test_tentative_confidence_present(self):
        da = self.cfg.get("DataAssociator", {})
        assert "tentativeDetectorConfidence" in da

    def test_tentative_confidence_matches_min(self):
        min_conf = self.cfg["BaseConfig"]["minDetectorConfidence"]
        tent_conf = self.cfg["DataAssociator"]["tentativeDetectorConfidence"]
        assert tent_conf == min_conf

    def test_visual_tracker_type_nvdcf(self):
        vt = self.cfg.get("VisualTracker", {})
        assert vt.get("visualTrackerType") == 1
