# Milestone 2: Data Preparation and Baselines

**Student:** Ahmed Abdelrahman

## 1. What is implemented
- `src/data.py`: one loader (BGR to gray/RGB, uint8 0-255, optional resize), mask loader
  (binary 0/1), image/mask pairing by stem, seeded dev/test split, controlled noise
  (Gaussian, salt-and-pepper, speckle, Poisson) with a fixed seed.
- `src/metrics.py`: MSE, PSNR, IoU, count error, crack-area ratio. Checked on small
  arrays worked by hand in `tests/test_metrics.py`
  (for example the IoU example gives 2/4 = 0.5, and MSE = 24/4 = 6).
- `src/baselines.py`:
  - Trivial baseline: fixed threshold 127 on raw grayscale (dark = crack).
  - Simple baseline: Otsu threshold followed by one 3x3 opening.
  - Otsu is also written in NumPy (`otsu_threshold_numpy`) for later comparison
    against OpenCV.
- `experiments/run_baselines.py`: evaluates both baselines on the dev set only and
  appends the runs to `experiments/log.csv`.

## 2. Baseline results (dev set, mean IoU)
| Baseline | Dev mean IoU |
|---|---|
| Trivial (T = 127) | TO BE FILLED from `experiments/log.csv` |
| Simple (Otsu + 3x3 opening) | TO BE FILLED from `experiments/log.csv` |

Reproduce with:
```
python -m experiments.run_baselines --config configs/baseline.json
```

## 3. Note
Every later pipeline is reported as an improvement over these two numbers. If a more
complex pipeline fails to beat them, that is reported and explained, not hidden.
