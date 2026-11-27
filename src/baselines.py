"""Baseline segmenters for the crack task (cracks are DARK on a lighter surface).

Trivial baseline : fixed global threshold of 127 on the raw grayscale image.
Simple baseline  : Otsu threshold followed by one 3x3 opening.
Both return a binary uint8 mask of 0/1 where 1 = crack.
"""
import cv2
import numpy as np


def otsu_threshold_numpy(gray):
    """Otsu threshold written in NumPy (maximises between-class variance).

    Returns the threshold t such that pixels <= t form the dark class.
    """
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    prob = hist / hist.sum()
    levels = np.arange(256)
    w0 = np.cumsum(prob)
    mu = np.cumsum(prob * levels)
    mu_total = mu[-1]
    denom = w0 * (1.0 - w0)
    denom[denom == 0] = np.nan  # classes that are empty are skipped
    between = (mu_total * w0 - mu) ** 2 / denom
    return int(np.nanargmax(between))


def baseline_trivial(gray, thresh=127):
    """Fixed global threshold: dark pixels (<= thresh) are crack."""
    return (gray <= thresh).astype(np.uint8)


def baseline_simple(gray):
    """Otsu threshold (dark class = crack) then a single 3x3 opening."""
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (gray <= t).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
