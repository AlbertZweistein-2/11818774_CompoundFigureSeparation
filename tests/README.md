## Tests

- **test_dataset_integrity.py** — Validates YOLO label files and matching images (format, bounds, existence). Set `DATASET_ROOT` env to point to your dataset copy if not using the default `dataset/` folder.
- **test_gpu_availability.py** — GPU smoke test (CUDA availability + simple tensor op). Skip on CPU-only by setting `SKIP_GPU_TEST=1`.

Run all tests:
```
pytest
```
Run a single test file:
```
pytest tests/test_dataset_integrity.py
```
