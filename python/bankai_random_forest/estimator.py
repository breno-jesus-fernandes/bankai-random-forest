import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.multiclass import type_of_target
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data

from . import _core


class BankaiRandomForestClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None,
    ):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples
        self.monotonic_cst = monotonic_cst

    def fit(self, X, y, sample_weight=None):
        self._reject_unsupported_baseline_parameters(sample_weight)
        if sparse.issparse(X):
            raise TypeError("sparse input is not supported")

        warm_refit = self.warm_start and hasattr(self, "_forest")
        if warm_refit and self.n_estimators < self._fitted_n_estimators:
            raise ValueError(
                "n_estimators must be greater than or equal to the number of fitted trees "
                "when warm_start=True"
            )
        X, y = validate_data(
            self, X, y, dtype=np.float64, ensure_2d=True, reset=not warm_refit
        )
        target_type = type_of_target(y)
        if target_type.startswith("continuous"):
            raise ValueError(f"Unknown label type: {target_type}")
        sample_weight = self._validate_sample_weight(sample_weight, X.shape[0])
        self.classes_, encoded_y = np.unique(y, return_inverse=True)
        self.n_classes_ = int(self.classes_.shape[0])
        sample_weight = self._combine_class_weight(y, sample_weight)
        min_leaf_weight = self._validate_min_weight_fraction_leaf(
            sample_weight, X.shape[0]
        )
        X, encoded_y, sample_weight = self._prepare_training_data(
            X, encoded_y, sample_weight
        )

        self._fit_seed = self._fit_seed if warm_refit else self._next_seed()
        self._fit_max_features = self._resolve_max_features()
        self._fit_X = X
        self._fit_y = encoded_y.astype(np.int64, copy=False)
        self._fit_sample_weight = sample_weight
        self._fit_min_leaf_weight = min_leaf_weight
        self._fit_max_depth = self._resolve_max_depth()
        self._fit_min_samples_split = self._resolve_min_samples(
            self.min_samples_split, 2, "min_samples_split", X.shape[0]
        )
        self._fit_min_samples_leaf = self._resolve_min_samples(
            self.min_samples_leaf, 1, "min_samples_leaf", X.shape[0]
        )
        self._fit_max_samples = self._resolve_max_samples(X.shape[0])
        self._fit_n_jobs = self._resolve_n_jobs()

        forest = _core.NativeForest()
        forest.fit(
            self._fit_X,
            self._fit_y,
            n_estimators=self.n_estimators,
            max_features=self._fit_max_features,
            random_state=self._fit_seed,
            sample_weight=self._fit_sample_weight,
            min_leaf_weight=self._fit_min_leaf_weight,
            criterion=self.criterion,
            max_depth=self._fit_max_depth,
            min_samples_split=self._fit_min_samples_split,
            min_samples_leaf=self._fit_min_samples_leaf,
            bootstrap=self.bootstrap,
            max_samples=self._fit_max_samples,
            oob=self.oob_score is True,
            n_jobs=self._fit_n_jobs,
        )
        self._forest = forest
        self._fitted_n_estimators = self.n_estimators
        self.feature_importances_ = np.asarray(
            forest.feature_importances(), dtype=np.float64
        )
        if self.oob_score is True:
            self.oob_decision_function_ = np.asarray(
                forest.oob_predict_proba(), dtype=np.float64
            )
            valid = self.oob_decision_function_.sum(axis=1) > 0.0
            self.oob_score_ = float(
                np.mean(
                    self.oob_decision_function_[valid].argmax(axis=1)
                    == self._fit_y[valid]
                )
            )
        return self

    def predict(self, X):
        check_is_fitted(self, "_forest")
        if sparse.issparse(X):
            raise TypeError("sparse input is not supported")

        X = validate_data(self, X, reset=False, dtype=np.float64, ensure_2d=True)
        encoded_y = np.asarray(self._forest.predict(X), dtype=np.intp)
        return self.classes_[encoded_y]

    def predict_proba(self, X):
        check_is_fitted(self, "_forest")
        if sparse.issparse(X):
            raise TypeError("sparse input is not supported")

        X = validate_data(self, X, reset=False, dtype=np.float64, ensure_2d=True)
        return np.asarray(self._forest.predict_proba(X), dtype=np.float64)

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_forest", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if "_fit_X" not in state:
            return

        forest = _core.NativeForest()
        forest.fit(
            self._fit_X,
            self._fit_y,
            n_estimators=self.n_estimators,
            max_features=self._fit_max_features,
            random_state=self._fit_seed,
            sample_weight=self._fit_sample_weight,
            min_leaf_weight=self._fit_min_leaf_weight,
            criterion=self.criterion,
            max_depth=self._fit_max_depth,
            min_samples_split=self._fit_min_samples_split,
            min_samples_leaf=self._fit_min_samples_leaf,
            bootstrap=self.bootstrap,
            max_samples=self._fit_max_samples,
            oob=self.oob_score is True,
            n_jobs=self._fit_n_jobs,
        )
        self._forest = forest
        self.feature_importances_ = np.asarray(
            forest.feature_importances(), dtype=np.float64
        )

    def _next_seed(self):
        random_state = check_random_state(self.random_state)
        return int(random_state.randint(0, np.iinfo(np.uint32).max))

    def _reject_unsupported_baseline_parameters(self, sample_weight):
        unsupported = (
            ("max_leaf_nodes", self.max_leaf_nodes is not None),
            ("min_impurity_decrease", self.min_impurity_decrease != 0.0),
            ("verbose", self.verbose != 0),
            ("ccp_alpha", self.ccp_alpha != 0.0),
            ("monotonic_cst", self.monotonic_cst is not None),
        )
        for name, is_unsupported in unsupported:
            if is_unsupported:
                raise NotImplementedError(f"{name} is not implemented yet")
        if self.oob_score is True and self.bootstrap is not True:
            raise ValueError("Out of bag estimation only available if bootstrap=True")
        if self.oob_score not in (False, True):
            raise ValueError("oob_score must be a boolean")

    def _resolve_max_depth(self):
        if self.max_depth is None:
            return 512
        if isinstance(self.max_depth, (int, np.integer)) and self.max_depth >= 1:
            return int(self.max_depth)
        raise ValueError("max_depth must be an integer greater than or equal to 1")

    @staticmethod
    def _resolve_min_samples(value, minimum, name, n_samples):
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            if value >= minimum:
                return int(value)
        if isinstance(value, (float, np.floating)) and 0.0 < value <= 1.0:
            return max(minimum, int(np.ceil(value * n_samples)))
        raise ValueError(f"{name} must be an integer >= {minimum} or a float in (0, 1]")

    def _resolve_max_features(self):
        max_features = self.max_features
        n_features = self.n_features_in_
        if max_features is None:
            return n_features
        if max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if max_features == "log2":
            return max(1, int(np.log2(n_features)))
        if isinstance(max_features, (int, np.integer)) and not isinstance(max_features, bool):
            if 1 <= max_features <= n_features:
                return int(max_features)
        if isinstance(max_features, (float, np.floating)) and 0.0 < max_features <= 1.0:
            return max(1, int(max_features * n_features))
        raise ValueError(
            "max_features must be an integer in [1, n_features], a float in (0, 1], "
            "'sqrt', 'log2', or None"
        )

    def _resolve_max_samples(self, n_samples):
        if self.max_samples is None:
            return None
        if self.bootstrap is not True:
            raise ValueError("max_samples can only be set if bootstrap=True")
        if isinstance(self.max_samples, (int, np.integer)) and 1 <= self.max_samples <= n_samples:
            return int(self.max_samples)
        if isinstance(self.max_samples, (float, np.floating)) and 0.0 < self.max_samples <= 1.0:
            return max(1, int(round(self.max_samples * n_samples)))
        raise ValueError(
            "max_samples must be an integer in [1, n_samples] or a float in (0, 1]"
        )

    def _resolve_n_jobs(self):
        if self.n_jobs is None:
            return 1
        if not isinstance(self.n_jobs, (int, np.integer)) or isinstance(self.n_jobs, bool):
            raise ValueError("n_jobs must be an integer or None")
        if self.n_jobs == 0:
            raise ValueError("n_jobs == 0 has no meaning")
        cpu_count = __import__("os").cpu_count() or 1
        return max(1, self.n_jobs if self.n_jobs > 0 else cpu_count + 1 + self.n_jobs)

    def _combine_class_weight(self, y, sample_weight):
        if self.class_weight is None:
            return sample_weight

        class_weight = compute_sample_weight(self.class_weight, y)
        if sample_weight is None:
            return class_weight.astype(np.float64, copy=False)
        return sample_weight * class_weight

    def _validate_min_weight_fraction_leaf(self, sample_weight, n_samples):
        fraction = self.min_weight_fraction_leaf
        if not isinstance(fraction, (float, int, np.floating, np.integer)) or not (
            0.0 <= fraction <= 0.5
        ):
            raise ValueError("min_weight_fraction_leaf must be in [0, 0.5]")

        if sample_weight is None:
            return float(fraction) * n_samples
        return float(fraction) * float(sample_weight.sum())

    @staticmethod
    def _validate_sample_weight(sample_weight, n_samples):
        if sample_weight is None:
            return None

        weights = np.asarray(sample_weight, dtype=np.float64)
        if weights.ndim != 1:
            raise ValueError("sample_weight must be one-dimensional")
        if weights.shape[0] != n_samples:
            raise ValueError("sample_weight must have the same length as X")
        if not np.isfinite(weights).all():
            raise ValueError("sample_weight must contain only finite values")
        if np.any(weights < 0.0):
            raise ValueError("sample_weight cannot contain negative values")
        if not np.any(weights > 0.0):
            raise ValueError("sample_weight cannot be all zero")
        return weights

    @staticmethod
    def _prepare_training_data(X, encoded_y, sample_weight):
        if sample_weight is not None and np.equal(
            sample_weight, np.floor(sample_weight)
        ).all():
            repeats = sample_weight.astype(np.intp, copy=False)
            X = np.repeat(X, repeats, axis=0)
            encoded_y = np.repeat(encoded_y, repeats)
            sample_weight = None

        keys = (encoded_y, *(X[:, index] for index in range(X.shape[1] - 1, -1, -1)))
        order = np.lexsort(keys)
        if sample_weight is None:
            return X[order], encoded_y[order], None
        return X[order], encoded_y[order], sample_weight[order]
