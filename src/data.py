"""Data loading, dev/test splitting and controlled degradation.

Every stage must load images through ``load_image`` so that colour order,
dtype and range are identical everywhere.
"""
import os
import random

import cv2
import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def load_image(path, gray=True, size=None):
    """Load an image as uint8, 0-255.

    Parameters
    ----------
    path : str
        File to read.
    gray : bool
        If True return (H, W) grayscale, else (H, W, 3) RGB (BGR is converted).
    size : tuple of int, optional
        (width, height) to resize to, using area interpolation.
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY if gray else cv2.COLOR_BGR2RGB)
    if size is not None:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img.astype(np.uint8)


def load_mask(path, size=None):
    """Load a ground-truth mask as a binary uint8 array of 0/1."""
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"cannot read mask: {path}")
    if size is not None:
        m = cv2.resize(m, size, interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


def list_pairs(image_dir, mask_dir):
    """Match images to masks by file stem. Returns a sorted list of (img, mask)."""
    masks = {os.path.splitext(f)[0]: f for f in os.listdir(mask_dir)
             if f.lower().endswith(IMG_EXTS)}
    pairs = []
    for f in sorted(os.listdir(image_dir)):
        stem, ext = os.path.splitext(f)
        if ext.lower() in IMG_EXTS and stem in masks:
            pairs.append((os.path.join(image_dir, f),
                          os.path.join(mask_dir, masks[stem])))
    return pairs


def split_dev_test(pairs, test_frac=0.30, seed=42):
    """Fixed, seeded dev/test split. Tune on dev only; use test once at the end."""
    items = list(pairs)
    random.Random(seed).shuffle(items)
    n_test = int(round(len(items) * test_frac))
    return items[n_test:], items[:n_test]


def add_noise(img, kind="gaussian", level=0.1, seed=0):
    """Controlled degradation of a uint8 image with a fixed random seed.

    ``level`` is the Gaussian sigma / speckle variance (as a fraction of 255),
    or the corrupted-pixel fraction for salt-and-pepper.
    """
    rng = np.random.default_rng(seed)
    x = img.astype(np.float64) / 255.0
    if kind == "gaussian":
        x = x + rng.normal(0.0, level, x.shape)
    elif kind == "salt_pepper":
        r = rng.random(x.shape)
        x = np.where(r < level / 2, 0.0, np.where(r > 1 - level / 2, 1.0, x))
    elif kind == "speckle":
        x = x + x * rng.normal(0.0, level, x.shape)
    elif kind == "poisson":
        peak = 255.0 * max(1e-3, 1.0 - level) + 1.0
        x = rng.poisson(x * peak) / peak
    else:
        raise ValueError(f"unknown noise kind: {kind}")
    return (np.clip(x, 0.0, 1.0) * 255.0).round().astype(np.uint8)
