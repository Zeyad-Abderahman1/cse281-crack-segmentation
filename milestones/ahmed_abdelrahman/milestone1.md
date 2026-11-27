# Milestone 1: Topic, Dataset and Problem Formulation

**Student:** Ahmed Abdelrahman  |  **Course:** CSE281 Digital Image Processing

## 1. Problem statement
The user is a structural-inspection engineer who photographs concrete surfaces.
The pipeline takes one photo and outputs a binary crack mask plus the cracked-area
ratio (cracked pixels / total pixels). A false negative (missed crack) is the costly
error because an unreported crack is not repaired; a false positive only causes an
extra manual check.

## 2. Dataset
- **Concrete Crack Segmentation** (Mendeley Data), 458 high-resolution images, each
  with a binary crack mask.
- Licence and image count must be verified on the dataset page before use; cite the
  source in the report.
- Checked against the spec criteria:

| Criterion | How the dataset meets it |
|---|---|
| Size | 458 images (at least 100 needed) |
| Ground truth | Pixel-level mask for every image |
| Task | Output is a measurement (crack area) and needs object/background separation |
| Real-world mess | Uneven illumination, low contrast, texture noise, broken cracks after thresholding |
| Not a toy | Real photos, not a built-in sample or synthetic set |
| Speed | Images are resized (for example 448x448) so the test set runs in under 10 min on a Colab CPU |
| No ML | The pipeline is classical (enhancement, thresholding, morphology) only |

## 3. Input, output, metric
- Input: RGB photo, loaded as 8-bit grayscale (uint8, 0-255), resized to 448x448.
- Output: binary mask (uint8, 0/1) and the area ratio.
- Primary metric: **IoU = |A and B| / |A or B|**. Cracks are thin, so pixel accuracy
  would be about 95% for an empty mask. IoU ignores the large true-negative background
  and penalises both missed and spurious crack pixels.

## 4. Data split
Fixed seeded 70/30 dev/test split (`src/data.py::split_dev_test`, seed 42). All
tuning uses the dev set only; the test set is used once at the end.

## 5. Planned pipelines
- A: spatial enhancement (contrast stretch / gamma) then global or Otsu threshold.
- B: adds smoothing, plus adaptive threshold or edge-based segmentation.
- C: adds opening/closing, hole filling and small-object removal, then measurement.

## 6. Repository
Modular layout as required by the spec; `src/metrics.py` is unit-tested on arrays
worked by hand (`tests/test_metrics.py`).
