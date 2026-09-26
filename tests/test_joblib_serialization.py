import joblib
import numpy as np
import pytest

from bankai_random_forest import BankaiRandomForestClassifier


@pytest.mark.parametrize("max_bins", [None, 8])
def test_joblib_round_trip_preserves_fitted_classifier(tmp_path, max_bins):
    rng = np.random.RandomState(23)
    x = rng.normal(size=(96, 5))
    y = np.where(x[:, 0] + x[:, 1] > 0, "positive", "negative")
    model = BankaiRandomForestClassifier(
        n_estimators=9,
        random_state=17,
        max_bins=max_bins,
        importance_type="gain",
    ).fit(x[:72], y[:72])
    path = tmp_path / "bankai.joblib"

    joblib.dump(model, path)
    restored = joblib.load(path)

    assert restored.get_params() == model.get_params()
    assert restored.max_bins == max_bins
    assert restored.n_features_in_ == model.n_features_in_
    np.testing.assert_array_equal(restored.classes_, model.classes_)
    np.testing.assert_array_equal(restored.predict(x[72:]), model.predict(x[72:]))
    np.testing.assert_allclose(
        restored.predict_proba(x[72:]), model.predict_proba(x[72:]), atol=0, rtol=0
    )
    np.testing.assert_allclose(
        restored.feature_importances_, model.feature_importances_, atol=1e-15, rtol=0
    )


@pytest.mark.parametrize("max_bins", [None, 8])
def test_joblib_round_trip_preserves_balanced_subsample_models(tmp_path, max_bins):
    rng = np.random.RandomState(53)
    x = rng.normal(size=(96, 5))
    y = np.zeros(96, dtype=np.int64)
    y[-16:] = 1
    x[-16:, 0] += 1.0
    model = BankaiRandomForestClassifier(
        n_estimators=9,
        random_state=59,
        max_bins=max_bins,
        class_weight="balanced_subsample",
        importance_type="permutation",
        oob_score=True,
    ).fit(x[:72], y[:72])
    path = tmp_path / "bankai-balanced-subsample.joblib"

    joblib.dump(model, path)
    restored = joblib.load(path)

    np.testing.assert_array_equal(restored.predict(x[72:]), model.predict(x[72:]))
    np.testing.assert_allclose(
        restored.predict_proba(x[72:]), model.predict_proba(x[72:]), atol=0, rtol=0
    )
    np.testing.assert_allclose(
        restored.feature_importances_, model.feature_importances_, atol=1e-15, rtol=0
    )


def test_compressed_joblib_round_trip_preserves_predictions(tmp_path):
    rng = np.random.RandomState(29)
    x = rng.normal(size=(64, 4))
    y = (x[:, 0] > 0).astype(int)
    model = BankaiRandomForestClassifier(n_estimators=7, random_state=31).fit(x, y)
    path = tmp_path / "bankai-compressed.joblib"

    joblib.dump(model, path, compress=3)
    restored = joblib.load(path)

    np.testing.assert_array_equal(restored.predict(x), model.predict(x))
    np.testing.assert_allclose(restored.predict_proba(x), model.predict_proba(x))


def test_uncompressed_joblib_round_trip_accepts_mmap_mode(tmp_path):
    rng = np.random.RandomState(37)
    x = rng.normal(size=(64, 4))
    y = (x[:, 0] > 0).astype(int)
    model = BankaiRandomForestClassifier(
        n_estimators=7, random_state=41, max_bins=8
    ).fit(x, y)
    path = tmp_path / "bankai-mmap.joblib"

    joblib.dump(model, path, compress=0)
    restored = joblib.load(path, mmap_mode="r")

    np.testing.assert_array_equal(restored.predict(x), model.predict(x))
    np.testing.assert_allclose(restored.predict_proba(x), model.predict_proba(x))
    assert isinstance(restored._fit_X, np.memmap)


def test_compressed_joblib_load_warns_that_mmap_is_unavailable(tmp_path):
    rng = np.random.RandomState(43)
    x = rng.normal(size=(64, 4))
    y = (x[:, 0] > 0).astype(int)
    model = BankaiRandomForestClassifier(n_estimators=7, random_state=47).fit(x, y)
    path = tmp_path / "bankai-compressed-mmap.joblib"
    joblib.dump(model, path, compress=3)

    with pytest.warns(UserWarning, match="mmap_mode"):
        restored = joblib.load(path, mmap_mode="r")

    np.testing.assert_array_equal(restored.predict(x), model.predict(x))
