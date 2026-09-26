# Joblib model serialization

Fitted Bankai classifiers can be saved with `joblib.dump` and restored with
`joblib.load`. Round-trip tests cover exact and histogram training and verify
parameters, class labels, predictions, probabilities, and feature importances.

```python
import joblib

joblib.dump(classifier, "classifier.joblib")
restored = joblib.load("classifier.joblib")
```

## How restoration works

Bankai does not serialize the live PyO3 forest object. Its pickle state omits
that object and retains the training arrays and fitted parameters instead.
During load, Bankai builds a new native forest from that saved state and
recomputes feature importances. As a result, artifacts contain the training
data and loading includes model reconstruction cost. Round-trip predictions,
probabilities, and feature importances matched exactly in the tested cases.

## Compression and memory mapping

- Uncompressed artifacts load with `mmap_mode="r"`. The stored `_fit_X` array
  is memory mapped, and the native forest is rebuilt from it.
- Compressed artifacts still load and predict correctly. Joblib warns that
  `mmap_mode` is unavailable for compressed files; use an uncompressed file
  when memory mapping is desired.

## Tested environment and compatibility policy

The compatibility tests ran with Bankai 0.1.0, Python 3.11.11, joblib 1.6.0,
and scikit-learn 1.9.1. Exact and `max_bins=8` training both passed. This
confirms joblib compatibility in the tested environment; it does not promise
that artifacts can be loaded across Bankai, Python, or dependency versions.
Keep the Bankai and Python environment consistent when restoring saved models.
