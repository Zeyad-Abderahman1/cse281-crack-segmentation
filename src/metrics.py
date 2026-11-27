"""Shared metrics: MSE, PSNR, IoU, count error. Written once, used everywhere."""
import numpy as np


def mse(a, b):
    """Mean squared error between two images of the same shape."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return float(np.mean((a - b) ** 2))


def psnr(a, b, peak=255.0):
    """PSNR in dB: 10 * log10(peak^2 / MSE). Returns inf for identical images."""
    err = mse(a, b)
    if err == 0.0:
        return float("inf")
    return float(10.0 * np.log10(peak ** 2 / err))


def iou(pred, truth):
    """Intersection over union of two binary masks: |A and B| / |A or B|.

    Two empty masks count as a perfect match (IoU = 1.0).
    """
    p = np.asarray(pred).astype(bool)
    t = np.asarray(truth).astype(bool)
    if p.shape != t.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {t.shape}")
    union = np.logical_or(p, t).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(p, t).sum() / union)


def count_error(pred_count, true_count):
    """Absolute count error for counting tasks."""
    return abs(int(pred_count) - int(true_count))


def crack_area_ratio(mask):
    """Fraction of pixels marked as crack (the measurement the client wants)."""
    m = np.asarray(mask).astype(bool)
    return float(m.mean())
