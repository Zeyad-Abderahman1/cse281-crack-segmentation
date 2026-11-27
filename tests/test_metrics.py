"""Hand-worked checks for src/metrics.py. Run: python -m pytest tests"""
import math

import numpy as np

from src.metrics import count_error, crack_area_ratio, iou, mse, psnr


def test_mse_by_hand():
    # differences are 0, 2, 4, 2 -> squares 0, 4, 16, 4 -> sum 24 / 4 = 6
    a = np.array([[10, 20], [30, 40]])
    b = np.array([[10, 22], [34, 38]])
    assert mse(a, b) == 6.0


def test_psnr_by_hand():
    # MSE = 6 -> 10 * log10(65025 / 6)
    a = np.array([[10, 20], [30, 40]])
    b = np.array([[10, 22], [34, 38]])
    assert math.isclose(psnr(a, b), 10 * math.log10(65025 / 6))


def test_psnr_identical_is_inf():
    a = np.zeros((3, 3))
    assert psnr(a, a) == float("inf")


def test_iou_by_hand():
    # A has 3 pixels, B has 3 pixels, intersection 2, union 4 -> 0.5
    a = np.array([[1, 1, 0], [1, 0, 0]])
    b = np.array([[1, 1, 0], [0, 1, 0]])
    assert iou(a, b) == 0.5


def test_iou_empty_masks():
    z = np.zeros((4, 4))
    assert iou(z, z) == 1.0


def test_iou_disjoint():
    a = np.array([[1, 0]])
    b = np.array([[0, 1]])
    assert iou(a, b) == 0.0


def test_count_error():
    assert count_error(7, 10) == 3


def test_crack_area_ratio():
    m = np.array([[1, 0], [0, 0]])
    assert crack_area_ratio(m) == 0.25
