"""Tests for layout model device placement (CPU / CUDA / CUDA:N).

Requires:
  - pip install "glmocr[selfhosted]"  (pulls in torch + transformers)
  - At least one GPU for the CUDA tests (two RTX 3090s recommended)

Run all tests:
    pytest -xvs glmocr/tests/test_layout_device.py

Run just the fast config-level tests (no model download):
    pytest -xvs glmocr/tests/test_layout_device.py -k "TestLayoutDeviceConfig"

Run real model tests (downloads PP-DocLayoutV3 on first run):
    pytest -xvs glmocr/tests/test_layout_device.py -k "TestLayoutDeviceReal"
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Optional
from unittest.mock import patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_IMAGE_DIR = REPO_ROOT / "examples" / "source"


def _find_sample_image() -> Optional[Path]:
    """Return the first sample image in examples/source/."""
    if not SAMPLE_IMAGE_DIR.exists():
        return None
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        images = sorted(SAMPLE_IMAGE_DIR.glob(ext))
        if images:
            return images[0]
    return None


def _gpu_count() -> int:
    """Return the number of CUDA GPUs available (0 if no CUDA)."""
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 1. Config-level tests (no model download, fast)
# ---------------------------------------------------------------------------


class TestLayoutDeviceConfig:
    """Verify LayoutConfig and PPDocLayoutDetector wire-up without loading a real model."""

    def test_layout_config_device_default_is_none(self):
        """LayoutConfig.device defaults to None (auto-select)."""
        from glmocr.config import LayoutConfig

        cfg = LayoutConfig()
        assert cfg.device is None

    def test_layout_config_device_cpu(self):
        """LayoutConfig accepts 'cpu' as device."""
        from glmocr.config import LayoutConfig

        cfg = LayoutConfig(device="cpu")
        assert cfg.device == "cpu"

    def test_layout_config_device_cuda(self):
        """LayoutConfig accepts 'cuda' as device."""
        from glmocr.config import LayoutConfig

        cfg = LayoutConfig(device="cuda")
        assert cfg.device == "cuda"

    def test_layout_config_device_cuda_index(self):
        """LayoutConfig accepts 'cuda:1' as device."""
        from glmocr.config import LayoutConfig

        cfg = LayoutConfig(device="cuda:1")
        assert cfg.device == "cuda:1"

    def test_env_var_sets_device(self, monkeypatch):
        """GLMOCR_LAYOUT_DEVICE env var propagates to config."""
        from glmocr.config import GlmOcrConfig, _ENV_MAP, ENV_PREFIX

        # Clear other GLMOCR_ vars to avoid interference
        for suffix in _ENV_MAP:
            monkeypatch.delenv(f"{ENV_PREFIX}{suffix}", raising=False)
        monkeypatch.setattr("glmocr.config._find_dotenv", lambda: None)

        monkeypatch.setenv("GLMOCR_LAYOUT_DEVICE", "cpu")
        cfg = GlmOcrConfig.from_env()
        assert cfg.pipeline.layout.device == "cpu"

    def test_from_env_layout_device_kwarg(self, monkeypatch):
        """layout_device kwarg in from_env() sets device correctly."""
        from glmocr.config import GlmOcrConfig, _ENV_MAP, ENV_PREFIX

        for suffix in _ENV_MAP:
            monkeypatch.delenv(f"{ENV_PREFIX}{suffix}", raising=False)
        monkeypatch.setattr("glmocr.config._find_dotenv", lambda: None)

        cfg = GlmOcrConfig.from_env(layout_device="cuda:1")
        assert cfg.pipeline.layout.device == "cuda:1"

    # Minimal config kwargs for mocked detector tests
    _MOCK_LAYOUT_KWARGS = dict(
        model_dir="dummy",
        id2label={0: "text"},
        label_task_mapping={"text": ["text"]},
    )

    def _mock_detector(self, device_val):
        """Create a PPDocLayoutDetector with mocked model, ready for start()."""
        from glmocr.config import LayoutConfig
        from glmocr.layout.layout_detector import PPDocLayoutDetector

        cfg = LayoutConfig(device=device_val, **self._MOCK_LAYOUT_KWARGS)
        det = PPDocLayoutDetector(cfg)

        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.id2label = {0: "text"}
        mock_processor = MagicMock()
        return det, mock_model, mock_processor

    def test_detector_device_selection_explicit_cpu(self):
        """When config.device='cpu', detector picks CPU even if CUDA is available."""
        det, mock_model, mock_proc = self._mock_detector("cpu")

        with (
            patch(
                "glmocr.layout.layout_detector.PPDocLayoutV3ForObjectDetection.from_pretrained",
                return_value=mock_model,
            ),
            patch(
                "glmocr.layout.layout_detector.PPDocLayoutV3ImageProcessorFast.from_pretrained",
                return_value=mock_proc,
            ),
        ):
            det.start()

        assert det._device == "cpu"
        mock_model.to.assert_called_with("cpu")

    def test_detector_device_selection_explicit_cuda(self):
        """When config.device='cuda:1', detector picks that device."""
        det, mock_model, mock_proc = self._mock_detector("cuda:1")

        with (
            patch(
                "glmocr.layout.layout_detector.PPDocLayoutV3ForObjectDetection.from_pretrained",
                return_value=mock_model,
            ),
            patch(
                "glmocr.layout.layout_detector.PPDocLayoutV3ImageProcessorFast.from_pretrained",
                return_value=mock_proc,
            ),
        ):
            det.start()

        assert det._device == "cuda:1"
        mock_model.to.assert_called_with("cuda:1")

    def test_detector_device_selection_auto_fallback_cpu(self):
        """When config.device=None and CUDA unavailable, auto-selects CPU."""
        import torch

        det, mock_model, mock_proc = self._mock_detector(None)

        with (
            patch(
                "glmocr.layout.layout_detector.PPDocLayoutV3ForObjectDetection.from_pretrained",
                return_value=mock_model,
            ),
            patch(
                "glmocr.layout.layout_detector.PPDocLayoutV3ImageProcessorFast.from_pretrained",
                return_value=mock_proc,
            ),
            patch.object(torch.cuda, "is_available", return_value=False),
        ):
            det.start()

        assert det._device == "cpu"

    def test_detector_device_selection_auto_cuda(self):
        """When config.device=None and CUDA available, auto-selects cuda:{cuda_visible_devices}."""
        import torch
        from glmocr.config import LayoutConfig
        from glmocr.layout.layout_detector import PPDocLayoutDetector

        cfg = LayoutConfig(
            device=None, cuda_visible_devices="1", **self._MOCK_LAYOUT_KWARGS
        )
        det = PPDocLayoutDetector(cfg)

        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.id2label = {0: "text"}
        mock_proc = MagicMock()

        with (
            patch(
                "glmocr.layout.layout_detector.PPDocLayoutV3ForObjectDetection.from_pretrained",
                return_value=mock_model,
            ),
            patch(
                "glmocr.layout.layout_detector.PPDocLayoutV3ImageProcessorFast.from_pretrained",
                return_value=mock_proc,
            ),
            patch.object(torch.cuda, "is_available", return_value=True),
        ):
            det.start()

        assert det._device == "cuda:1"


# ---------------------------------------------------------------------------
# 2. Real model tests (downloads PP-DocLayoutV3 on first run)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_image():
    """Load a sample PIL image for testing."""
    img_path = _find_sample_image()
    if img_path is None:
        pytest.skip(f"No sample image found in {SAMPLE_IMAGE_DIR}")
    from PIL import Image

    return Image.open(img_path).convert("RGB")


def _make_detector(device: str):
    """Create a PPDocLayoutDetector with a specific device setting."""
    from glmocr.config import load_config

    cfg = load_config()
    cfg.pipeline.layout.device = device
    from glmocr.layout.layout_detector import PPDocLayoutDetector

    return PPDocLayoutDetector(cfg.pipeline.layout)


def _run_detection(detector, image):
    """Run detection on a single image, return (results, elapsed_seconds)."""
    detector.start()
    try:
        t0 = time.perf_counter()
        results = detector.process([image])
        elapsed = time.perf_counter() - t0
        return results, elapsed
    finally:
        detector.stop()
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass


class TestLayoutDeviceReal:
    """Real model tests — actually load PP-DocLayoutV3 and run inference."""

    def test_cpu(self, sample_image):
        """Layout model loads and runs on CPU."""
        detector = _make_detector("cpu")
        results, elapsed = _run_detection(detector, sample_image)

        print(f"\n  [CPU] Inference time: {elapsed:.3f}s")
        print(f"  [CPU] Detected {len(results[0])} regions")

        assert len(results) == 1, "Should return results for 1 image"
        assert isinstance(results[0], list), "Per-image result should be a list"
        # PP-DocLayoutV3 should detect at least something on a real document
        assert len(results[0]) > 0, "Should detect at least one region"

        # Verify structure of each detection
        for det in results[0]:
            assert "label" in det
            assert "score" in det
            assert "bbox_2d" in det
            assert len(det["bbox_2d"]) == 4

    @pytest.mark.skipif(not _has_cuda(), reason="No CUDA available")
    def test_cuda_default(self, sample_image):
        """Layout model loads and runs on default CUDA device."""
        detector = _make_detector("cuda")
        results, elapsed = _run_detection(detector, sample_image)

        print(f"\n  [cuda] Inference time: {elapsed:.3f}s")
        print(f"  [cuda] Detected {len(results[0])} regions")

        assert len(results) == 1
        assert len(results[0]) > 0

    @pytest.mark.skipif(not _has_cuda(), reason="No CUDA available")
    def test_cuda_0(self, sample_image):
        """Layout model loads and runs on cuda:0."""
        detector = _make_detector("cuda:0")
        results, elapsed = _run_detection(detector, sample_image)

        print(f"\n  [cuda:0] Inference time: {elapsed:.3f}s")
        print(f"  [cuda:0] Detected {len(results[0])} regions")

        assert len(results) == 1
        assert len(results[0]) > 0

        # Verify model is actually on GPU 0
        import torch

        for param in (
            detector._model.parameters()
            if hasattr(detector, "_model") and detector._model
            else []
        ):
            assert param.device == torch.device(
                "cuda:0"
            ) or param.device == torch.device("cuda", 0)
            break

    @pytest.mark.skipif(_gpu_count() < 2, reason="Need 2+ GPUs for cuda:1 test")
    def test_cuda_1(self, sample_image):
        """Layout model loads and runs on cuda:1 (second GPU)."""
        detector = _make_detector("cuda:1")
        results, elapsed = _run_detection(detector, sample_image)

        print(f"\n  [cuda:1] Inference time: {elapsed:.3f}s")
        print(f"  [cuda:1] Detected {len(results[0])} regions")

        assert len(results) == 1
        assert len(results[0]) > 0

    @pytest.mark.skipif(not _has_cuda(), reason="No CUDA available")
    def test_auto_selects_cuda(self, sample_image):
        """With device=None (auto), selects CUDA when available."""
        from glmocr.config import load_config

        cfg = load_config()
        cfg.pipeline.layout.device = None  # auto
        cfg.pipeline.layout.cuda_visible_devices = "0"

        from glmocr.layout.layout_detector import PPDocLayoutDetector

        detector = PPDocLayoutDetector(cfg.pipeline.layout)
        results, elapsed = _run_detection(detector, sample_image)

        print(f"\n  [auto → cuda:0] Inference time: {elapsed:.3f}s")
        print(f"  [auto → cuda:0] Detected {len(results[0])} regions")

        assert detector._device is None  # cleaned up by stop()
        assert len(results) == 1
        assert len(results[0]) > 0

    def test_results_consistent_across_devices(self, sample_image):
        """CPU and CUDA produce the same number of detected regions."""
        if not _has_cuda():
            pytest.skip("No CUDA available — cannot compare devices")

        cpu_det = _make_detector("cpu")
        cpu_results, cpu_time = _run_detection(cpu_det, sample_image)

        cuda_det = _make_detector("cuda:0")
        cuda_results, cuda_time = _run_detection(cuda_det, sample_image)

        cpu_labels = sorted(d["label"] for d in cpu_results[0])
        cuda_labels = sorted(d["label"] for d in cuda_results[0])

        print(
            f"\n  [consistency] CPU: {len(cpu_results[0])} regions in {cpu_time:.3f}s"
        )
        print(
            f"  [consistency] CUDA:0: {len(cuda_results[0])} regions in {cuda_time:.3f}s"
        )
        print(f"  [consistency] CPU labels: {cpu_labels}")
        print(f"  [consistency] CUDA labels: {cuda_labels}")

        # Same number of detections (model is deterministic in eval mode)
        assert len(cpu_results[0]) == len(
            cuda_results[0]
        ), f"CPU found {len(cpu_results[0])} regions, CUDA found {len(cuda_results[0])}"
        assert cpu_labels == cuda_labels, "Labels should match across devices"

    @pytest.mark.skipif(_gpu_count() < 2, reason="Need 2+ GPUs")
    def test_both_gpus(self, sample_image):
        """Run on GPU 0, then GPU 1 — both produce valid results."""
        gpu0_det = _make_detector("cuda:0")
        gpu0_results, gpu0_time = _run_detection(gpu0_det, sample_image)

        gpu1_det = _make_detector("cuda:1")
        gpu1_results, gpu1_time = _run_detection(gpu1_det, sample_image)

        print(
            f"\n  [dual-GPU] cuda:0: {len(gpu0_results[0])} regions in {gpu0_time:.3f}s"
        )
        print(
            f"  [dual-GPU] cuda:1: {len(gpu1_results[0])} regions in {gpu1_time:.3f}s"
        )

        assert len(gpu0_results[0]) > 0
        assert len(gpu1_results[0]) > 0
        assert len(gpu0_results[0]) == len(gpu1_results[0])

    @pytest.mark.skipif(_gpu_count() < 2, reason="Need 2+ GPUs")
    def test_benchmark_all_devices(self, sample_image):
        """Benchmark: run inference on CPU, cuda:0, and cuda:1 and print timing comparison."""
        results_summary = []

        for device_str in ["cpu", "cuda:0", "cuda:1"]:
            detector = _make_detector(device_str)
            # Warm-up run (first run includes model load overhead)
            _, _ = _run_detection(detector, sample_image)

            # Timed run
            detector = _make_detector(device_str)
            results, elapsed = _run_detection(detector, sample_image)
            n_regions = len(results[0])
            results_summary.append((device_str, elapsed, n_regions))

        print("\n" + "=" * 60)
        print("  LAYOUT DEVICE BENCHMARK")
        print("=" * 60)
        print(f"  {'Device':<12} {'Time (s)':>10} {'Regions':>10}")
        print("-" * 40)
        for device_str, elapsed, n_regions in results_summary:
            print(f"  {device_str:<12} {elapsed:>10.3f} {n_regions:>10}")
        print("=" * 60)

        # All devices should detect the same number of regions
        region_counts = {r[2] for r in results_summary}
        assert (
            len(region_counts) == 1
        ), f"Inconsistent region counts across devices: {results_summary}"
