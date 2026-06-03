"""Basic smoke tests for the inference engine."""

import numpy as np
import pytest

from petrilya.inference.engine import CellposeEngine
from petrilya.metrics.colony import compute_colony_metrics


@pytest.fixture(scope="module")
def engine():
    return CellposeEngine(use_gpu=False)


def test_engine_initializes(engine):
    assert engine.model is not None


def test_segment_blank_image(engine):
    blank = np.zeros((256, 256), dtype=np.uint8)
    masks, _diam = engine.segment(blank)
    assert masks.shape == (256, 256)
    assert masks.max() == 0


def test_metrics_empty_on_blank():
    blank_masks = np.zeros((128, 128), dtype=np.int32)
    assert compute_colony_metrics(blank_masks) == []
