# Data

Dataset: **Concrete Crack Segmentation** (Mendeley Data), 458 high-resolution
concrete photos, each with a binary crack mask.

Download it yourself and verify the licence and image count on the dataset page
before use (the spec asks teams to check this). Cite the source in the report.

Expected layout after download (not committed to git):

```
data/raw/images/<name>.jpg
data/raw/masks/<name>.jpg   # same stem as the image
```

Images and masks are matched by file stem in `src/data.py::list_pairs`.
