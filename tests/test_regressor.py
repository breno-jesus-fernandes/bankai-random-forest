import numpy as np
import pytest
from sklearn.base import clone, is_regressor
from sklearn.datasets import make_regression

from bankai_random_forest import BankaiRandomForestRegressor


def test_regressor_api_fit_and_multioutput():
    X, y = make_regression(n_samples=100, n_features=6, random_state=3)
    model = BankaiRandomForestRegressor(n_estimators=12, random_state=4)
    assert is_regressor(model)
    assert clone(model).max_features == 1.0
    model.fit(X, y)
    assert model.predict(X[:4]).shape == (4,)
    assert model.feature_importances_.shape == (6,)

    multi = BankaiRandomForestRegressor(n_estimators=6, random_state=4)
    multi.fit(X, np.column_stack((y, 2 * y)))
    assert multi.predict(X[:4]).shape == (4, 2)


@pytest.mark.parametrize("criterion", ["squared_error", "absolute_error", "poisson"])
def test_supported_regression_criteria(criterion):
    X = np.arange(40, dtype=float).reshape(-1, 1)
    y = np.arange(40, dtype=float) if criterion != "poisson" else np.arange(40, dtype=float)
    model = BankaiRandomForestRegressor(n_estimators=3, criterion=criterion, random_state=1)
    model.fit(X, y)
    assert np.isfinite(model.predict(X)).all()


def test_poisson_validation_and_bankai_options():
    X = np.arange(12, dtype=float).reshape(-1, 1)
    with pytest.raises(ValueError):
        BankaiRandomForestRegressor(criterion="poisson").fit(X, -np.arange(12.0))
    with pytest.raises(ValueError, match="importance_type"):
        BankaiRandomForestRegressor(importance_type="unknown").fit(X, np.arange(12.0))
    with pytest.raises(ValueError, match="criterion"):
        BankaiRandomForestRegressor(criterion="friedman_mse").fit(X, np.arange(12.0))


def test_oob_permutation_importance():
    X, y = make_regression(n_samples=80, n_features=4, random_state=7)
    model = BankaiRandomForestRegressor(
        n_estimators=20, random_state=8, oob_score=True,
        importance_type="permutation",
    ).fit(X, y)
    assert model.feature_importances_.shape == (4,)
    assert np.isfinite(model.feature_importances_).all()
