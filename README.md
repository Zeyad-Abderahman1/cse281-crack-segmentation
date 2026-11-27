# CSE281 Course Project: Concrete Crack Segmentation

Student: **Ahmed Abdelrahman**
Course: CSE281 Digital Image Processing, Alamein International University

## Problem
A civil-inspection client photographs concrete surfaces and needs the cracked
pixels marked and the cracked-area ratio reported. A missed crack (false negative)
is the costly error, since an unreported crack can go unrepaired. Output: a binary
crack mask and the area ratio. Primary metric: **IoU** against the ground-truth mask.
See `milestones/ahmed_abdelrahman/milestone1.md` for the full formulation.

## Status
| Milestone | Content | State |
|---|---|---|
| 1 | Topic, dataset check, metric, repo, data and metrics code | Done |
| 2 | Dev/test split, trivial and simple baselines, experiment log | Code done, numbers pending dataset download |

## Setup
```
pip install -r requirements.txt
```
Download the dataset as described in `data/README.md`.

## Run
```
python -m pytest tests
python -m experiments.run_baselines --config configs/baseline.json
```
Results are appended to `experiments/log.csv`. Seeds are fixed in the config.

## Layout
```
src/data.py        loading, seeded dev/test split, controlled noise
src/metrics.py     MSE, PSNR, IoU, count error (tested by hand in tests/)
src/baselines.py   trivial (T=127) and simple (Otsu + 3x3 opening) baselines
experiments/       run scripts and the experiment CSV
configs/           one file per configuration
```
