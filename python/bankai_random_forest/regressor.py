"""Regression estimator API for Bankai Random Forest."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.utils import RegressorTags
from sklearn.utils.validation import check_is_fitted


class BankaiRandomForestRegressor(RandomForestRegressor):
    """Random forest regressor with Bankai feature importance options.

    Regression currently delegates tree construction and prediction to
    scikit-learn. ``max_bins``, ``binning_strategy`` and ``bin_sample_size``
    are accepted for API consistency, but histogram preprocessing is only
    available to the Bankai classification backend.
    """

    def __sklearn_tags__(self):
        tags = BaseEstimator.__sklearn_tags__(self)
        tags.estimator_type = "regressor"
        tags.regressor_tags = RegressorTags()
        tags.target_tags.required = True
        tags.input_tags.sparse = True
        tags.input_tags.allow_nan = True
        return tags

    @property
    def feature_importances_(self):
        return self._bankai_feature_importances

    @feature_importances_.setter
    def feature_importances_(self, value):
        self._bankai_feature_importances = value

    def __init__(
        self, n_estimators=100, *, criterion="squared_error", max_depth=None,
        min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
        max_features=1.0, max_leaf_nodes=None, min_impurity_decrease=0.0,
        bootstrap=True, oob_score=False, n_jobs=None, random_state=None,
        verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None,
        monotonic_cst=None, max_bins=None, binning_strategy="exact_sort",
        bin_sample_size=200_000, importance_type="gain",
    ):
        if criterion == "friedman_mse":
            raise ValueError(
                "criterion='friedman_mse' is deprecated; use 'squared_error' instead"
            )
        super().__init__(
            n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features, max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap,
            oob_score=oob_score, n_jobs=n_jobs, random_state=random_state,
            verbose=verbose, warm_start=warm_start, ccp_alpha=ccp_alpha,
            max_samples=max_samples, monotonic_cst=monotonic_cst,
        )
        self.max_bins = max_bins
        self.binning_strategy = binning_strategy
        self.bin_sample_size = bin_sample_size
        self.importance_type = importance_type

    def fit(self, X, y, sample_weight=None):
        self._validate_bankai_options()
        super().fit(X, y, sample_weight=sample_weight)
        if self.importance_type == "gain":
            importance = super().feature_importances_
        elif self.importance_type == "split":
            importance = np.zeros(self.n_features_in_, dtype=np.float64)
            for tree in self.estimators_:
                used = tree.tree_.feature
                used = used[used >= 0]
                importance += np.bincount(used, minlength=self.n_features_in_)
            total = importance.sum()
            if total:
                importance /= total
        else:
            importance = self._oob_permutation_importance(X, y)
        self.feature_importances_ = importance
        return self

    def _validate_bankai_options(self):
        if self.criterion not in {"squared_error", "absolute_error", "poisson"}:
            raise ValueError(
                "criterion must be 'squared_error', 'absolute_error', or 'poisson'"
            )
        if self.importance_type not in {"gain", "split", "permutation"}:
            raise ValueError("importance_type must be 'gain', 'split', or 'permutation'")
        if self.max_bins is not None and (not isinstance(self.max_bins, (int, np.integer)) or self.max_bins < 2):
            raise ValueError("max_bins must be None or an integer greater than or equal to 2")
        if self.binning_strategy not in {"exact_sort", "sampled_quantile"}:
            raise ValueError("binning_strategy must be 'exact_sort' or 'sampled_quantile'")
        if not isinstance(self.bin_sample_size, (int, np.integer)) or self.bin_sample_size < 1:
            raise ValueError("bin_sample_size must be a positive integer")

    def _oob_permutation_importance(self, X, y):
        if not self.bootstrap:
            raise ValueError("importance_type='permutation' requires bootstrap=True")
        if not self.oob_score:
            raise ValueError("importance_type='permutation' requires oob_score=True")
        X_arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        y_arr = np.asarray(y)
        samples = self.estimators_samples_
        oob = [np.setdiff1d(np.arange(X_arr.shape[0]), sample, assume_unique=False) for sample in samples]
        counts = np.zeros(X_arr.shape[0], dtype=np.int64)
        baseline = np.zeros_like(y_arr, dtype=np.float64)
        for tree, rows in zip(self.estimators_, oob):
            if len(rows):
                baseline[rows] += tree.predict(X_arr[rows])
                counts[rows] += 1
        valid = counts > 0
        baseline[valid] /= counts[valid, None] if baseline.ndim == 2 else counts[valid]
        if not np.any(valid):
            raise ValueError("no samples have OOB predictions; increase n_estimators")
        baseline_score = r2_score(y_arr[valid], baseline[valid], multioutput="uniform_average")
        rng = np.random.RandomState(self.random_state)
        importance = np.zeros(self.n_features_in_, dtype=np.float64)
        for feature in range(self.n_features_in_):
            permuted = X_arr.copy()
            permuted[:, feature] = permuted[rng.permutation(len(permuted)), feature]
            predictions = np.zeros_like(baseline)
            for tree, rows in zip(self.estimators_, oob):
                if len(rows):
                    predictions[rows] += tree.predict(permuted[rows])
            predictions[valid] /= counts[valid, None] if predictions.ndim == 2 else counts[valid]
            importance[feature] = baseline_score - r2_score(
                y_arr[valid], predictions[valid], multioutput="uniform_average"
            )
        return importance

    def apply(self, X):
        check_is_fitted(self)
        return super().apply(X)
