import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import ClassifierTags
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.multiclass import type_of_target
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, column_or_1d, validate_data

from . import _core


class _TreeSHAPAdapter(RandomForestClassifier):
    """Benchmark-only sklearn facade for comparing the adapter route."""

    def __init__(self, estimators, n_features, n_classes, criterion):
        self.estimators_ = estimators
        self.n_features_in_ = n_features
        self.n_classes_ = n_classes
        self.classes_ = np.arange(n_classes)
        self.criterion = criterion


def _make_sklearn_trees(tree_arrays, n_features, n_classes):
    """Reconstruct minimal sklearn Tree objects from Rust's exported arrays."""
    trees = []
    for left, right, feature, threshold, cover, flat_values, missing_go_to_left in tree_arrays:
        n_nodes = len(feature)
        nodes = np.zeros(n_nodes, dtype=_tree.NODE_DTYPE)
        nodes["left_child"] = left
        nodes["right_child"] = right
        nodes["feature"] = feature
        nodes["threshold"] = threshold
        nodes["impurity"] = 0.0
        nodes["n_node_samples"] = np.asarray(cover, dtype=np.intp)
        nodes["weighted_n_node_samples"] = cover
        values = np.asarray(flat_values, dtype=np.float64).reshape(n_nodes, 1, n_classes)
        # SHAP recomputes every internal value from child covers. Give the
        # transient sklearn representation a non-zero placeholder so its
        # initial normalization does not emit a divide-by-zero warning.
        internal = np.asarray(feature) >= 0
        values[internal, 0, 0] = 1.0
        tree = _tree.Tree(n_features, np.array([n_classes], dtype=np.intp), 1)
        tree.__setstate__({"max_depth": n_nodes, "node_count": n_nodes, "nodes": nodes, "values": values})
        tree.missing_go_to_left[:] = missing_go_to_left
        estimator = DecisionTreeClassifier()
        estimator.tree_ = tree
        estimator.n_features_in_ = n_features
        estimator.n_outputs_ = 1
        estimator.n_classes_ = n_classes
        estimator.classes_ = np.arange(n_classes)
        trees.append(estimator)
    return trees


class BankaiRandomForestClassifier(RandomForestClassifier):
    def __sklearn_tags__(self):
        # SHAP recognizes this estimator through sklearn's forest tags;
        # fitting and prediction remain implemented by Bankai's native core.
        tags = BaseEstimator.__sklearn_tags__(self)
        tags.estimator_type = "classifier"
        tags.classifier_tags = ClassifierTags()
        tags.target_tags.required = True
        tags.target_tags.one_d_labels = False
        tags.target_tags.single_output = False
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
        importance_type="gain",
        max_bins=None,
        binning_strategy="exact_sort",
        bin_sample_size=200_000,
        shap_mode="sampled",
        shap_sample_size=1000,
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
        self.importance_type = importance_type
        self.max_bins = max_bins
        self.binning_strategy = binning_strategy
        self.bin_sample_size = bin_sample_size
        self.shap_mode = shap_mode
        self.shap_sample_size = shap_sample_size

    @property
    def estimators_(self):
        """Lazily expose native trees in sklearn's format for SHAP TreeExplainer."""
        multioutput_models = getattr(self, "_multioutput_models", None)
        if multioutput_models:
            return multioutput_models[0].estimators_
        check_is_fitted(self, "_forest")
        if getattr(self, "_shap_estimators_cache", None) is None:
            self._shap_estimators_cache = _make_sklearn_trees(
                self._forest.shap_tree_arrays(), self.n_features_in_, self.n_classes_
            )
        return self._shap_estimators_cache

    def _tree_shap_adapter_for_benchmark(self):
        """Build the benchmark-only adapter facade accepted by TreeExplainer."""
        multioutput_models = getattr(self, "_multioutput_models", None)
        if multioutput_models:
            return multioutput_models[0]._tree_shap_adapter_for_benchmark()
        check_is_fitted(self, "_forest")
        return _TreeSHAPAdapter(
            _make_sklearn_trees(
                self._forest.shap_tree_arrays(), self.n_features_in_, self.n_classes_
            ), self.n_features_in_, self.n_classes_, self.criterion
        )

    def _native_tree_shap_for_benchmark(self, X):
        """Return native Rust TreeSHAP values for benchmark comparisons."""
        multioutput_models = getattr(self, "_multioutput_models", None)
        if multioutput_models:
            raise ValueError("native TreeSHAP is not exposed for multioutput models")
        check_is_fitted(self, "_forest")
        X = validate_data(self, X, reset=False, dtype=np.float64, ensure_2d=True, ensure_all_finite="allow-nan")
        values, base_values = self._forest.tree_shap(X)
        return np.asarray(values, dtype=np.float64), np.asarray(base_values, dtype=np.float64)

    def fit(self, X, y, sample_weight=None):
        self._reject_unsupported_baseline_parameters(sample_weight)
        sparse_input = sparse.issparse(X)
        raw_dtype = X.dtype if sparse_input else np.asarray(X).dtype
        if sparse_input:
            X = X.tocsr(copy=True)
            X.sum_duplicates()
            X.sort_indices()
            X.eliminate_zeros()
        if y is None:
            raise ValueError("requires y to be passed, but the target y is None")
        y_array = np.asarray(y)
        if y_array.ndim == 2 and y_array.shape[1] == 1:
            y = column_or_1d(y, warn=True)
        elif y_array.ndim == 2 and y_array.shape[1] > 1:
            return self._fit_multioutput(X, y_array, sample_weight, sparse_input, raw_dtype)
        elif y_array.ndim != 1:
            raise ValueError("multioutput targets are not supported")
        self.__dict__.pop("_multioutput_models", None)

        warm_refit = self.warm_start and hasattr(self, "_forest")
        if warm_refit and self.n_estimators < self._fitted_n_estimators:
            raise ValueError(
                "n_estimators must be greater than or equal to the number of fitted trees "
                "when warm_start=True"
            )
        validation_options = dict(dtype=[np.float64, np.float32], ensure_2d=True, reset=not warm_refit, ensure_all_finite="allow-nan")
        if sparse_input:
            validation_options["accept_sparse"] = ("csr", "csc")
        X, y = validate_data(self, X, y, **validation_options)
        if sparse_input:
            X = X.tocsr(copy=False)
        self.copy_telemetry_ = {
            "input_dtype": str(raw_dtype),
            "core_dtype": str(X.dtype),
            "input_c_contiguous": bool(X.data.flags.c_contiguous) if sparse_input else bool(X.flags.c_contiguous),
            "cast_to_float64": raw_dtype != np.dtype(np.float64),
            "input_sparse": sparse_input,
        }
        target_type = type_of_target(y)
        if target_type.startswith("continuous"):
            raise ValueError(f"Unknown label type: {target_type}")
        sample_weight = self._validate_sample_weight(sample_weight, X.shape[0])
        self.classes_, encoded_y = np.unique(y, return_inverse=True)
        self.n_classes_ = int(self.classes_.shape[0])
        self._fit_monotonic_cst = self._resolve_monotonic_cst(X.shape[1])
        sample_weight = self._combine_class_weight(y, sample_weight)
        min_leaf_weight = self._validate_min_weight_fraction_leaf(
            sample_weight, X.shape[0]
        )
        if getattr(self, "_track_training_row_order", False):
            X, encoded_y, sample_weight, self._training_row_order = (
                self._prepare_training_data(
                    X, encoded_y, sample_weight, return_order=True
                )
            )
        else:
            X, encoded_y, sample_weight = self._prepare_training_data(
                X,
                encoded_y,
                sample_weight,
                defer_dense_sort=not sparse_input,
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
        self._fit_max_samples = self._resolve_max_samples(X.shape[0], sample_weight)
        self._fit_n_jobs = self._resolve_n_jobs()
        self._fit_min_impurity_decrease = self._resolve_min_impurity_decrease()
        self._fit_max_leaves = self._resolve_max_leaf_nodes()
        self._fit_verbose = self._resolve_verbose()
        self._fit_importance_type = self._resolve_importance_type()
        self._fit_shap_mode = self._resolve_shap_mode()
        self._fit_max_bins = self._resolve_max_bins()
        self._fit_binning_strategy = self._resolve_binning_strategy()
        self._fit_bin_sample_size = self._resolve_bin_sample_size()
        if self._fit_verbose:
            print(
                f"[BankaiRandomForestClassifier] building {self.n_estimators} trees",
                flush=True,
            )

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
            max_leaves=self._fit_max_leaves,
            min_samples_split=self._fit_min_samples_split,
            min_samples_leaf=self._fit_min_samples_leaf,
            min_impurity_decrease=self._fit_min_impurity_decrease,
            bootstrap=self.bootstrap,
            max_samples=self._fit_max_samples,
            oob=bool(self.oob_score),
            permutation_importance=self._fit_importance_type == "permutation",
            n_jobs=self._fit_n_jobs,
            max_bins=self._fit_max_bins,
            binning_strategy=self._fit_binning_strategy,
            bin_sample_size=self._fit_bin_sample_size,
            balanced_subsample=(
                self.class_weight == "balanced_subsample" and self.bootstrap is True
            ),
            ccp_alpha=self.ccp_alpha,
            monotonic_cst=self._fit_monotonic_cst,
        )
        self._forest = forest
        self._shap_estimators_cache = None
        self._fitted_n_estimators = self.n_estimators
        if self._fit_importance_type == "shap":
            self.feature_importances_ = (
                None if self._fit_shap_mode == "explicit" else self._sampled_shap_importances()
            )
        else:
            self.feature_importances_ = self._selected_feature_importances(forest)
        if self.oob_score:
            self.oob_decision_function_ = np.asarray(
                forest.oob_predict_proba(), dtype=np.float64
            )
            ordered_y = self._fit_y[np.asarray(forest.training_order(), dtype=np.intp)]
            valid = self.oob_decision_function_.sum(axis=1) > 0.0
            predictions = self.oob_decision_function_.argmax(axis=1)
            if callable(self.oob_score):
                self.oob_score_ = float(self.oob_score(ordered_y, predictions))
            else:
                self.oob_score_ = float(np.mean(predictions[valid] == ordered_y[valid]))
        return self

    def _fit_multioutput(self, X, y, sample_weight, sparse_input, raw_dtype):
        target_type = type_of_target(y)
        if target_type not in ("multilabel-indicator", "multiclass-multioutput"):
            raise ValueError(f"Unknown label type: {target_type}")
        if self.monotonic_cst is not None:
            raise ValueError("monotonic_cst is not supported for multioutput classification")
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y must contain the same number of samples")

        warm_refit = self.warm_start and hasattr(self, "_multioutput_models")
        if warm_refit and self.n_estimators < self._fitted_n_estimators:
            raise ValueError(
                "n_estimators must be greater than or equal to the number of fitted trees "
                "when warm_start=True"
            )
        validation_options = dict(
            dtype=[np.float64, np.float32],
            ensure_2d=True,
            reset=not warm_refit,
            ensure_all_finite="allow-nan",
        )
        if sparse_input:
            validation_options["accept_sparse"] = ("csr", "csc")
        X = validate_data(self, X, **validation_options)
        if sparse_input:
            X = X.tocsr(copy=False)

        sample_weight = self._validate_sample_weight(sample_weight, X.shape[0])
        self.copy_telemetry_ = {
            "input_dtype": str(raw_dtype),
            "core_dtype": str(X.dtype),
            "input_c_contiguous": bool(X.data.flags.c_contiguous)
            if sparse_input
            else bool(X.flags.c_contiguous),
            "cast_to_float64": raw_dtype != np.dtype(np.float64),
            "input_sparse": sparse_input,
        }
        output_classes = [np.unique(y[:, output]) for output in range(y.shape[1])]
        self.classes_ = output_classes
        self.n_classes_ = np.asarray([len(classes) for classes in output_classes])
        self.n_outputs_ = y.shape[1]

        if isinstance(self.class_weight, list):
            if len(self.class_weight) != self.n_outputs_:
                raise ValueError("class_weight list must contain one dictionary per output")
            output_class_weights = self.class_weight
        else:
            output_class_weights = [self.class_weight] * self.n_outputs_

        previous_models = getattr(self, "_multioutput_models", None)
        if warm_refit and previous_models and len(previous_models) == self.n_outputs_:
            models = previous_models
            seeds = self._multioutput_seeds
        else:
            self._fit_seed = self._next_seed()
            rng = np.random.RandomState(self._fit_seed)
            seeds = rng.randint(0, np.iinfo(np.int32).max, size=self.n_outputs_).tolist()
            models = [None] * self.n_outputs_

        for output in range(self.n_outputs_):
            params = self.get_params(deep=False)
            params["random_state"] = int(seeds[output])
            params["class_weight"] = output_class_weights[output]
            if models[output] is None:
                models[output] = BankaiRandomForestClassifier(**params)
            else:
                models[output].set_params(**params)
            models[output]._track_training_row_order = True
            models[output].fit(X, y[:, output], sample_weight=sample_weight)

        self._multioutput_models = models
        self._multioutput_seeds = seeds
        self._forest = models[0]._forest
        self._fitted_n_estimators = self.n_estimators
        self.feature_importances_ = np.mean(
            [model.feature_importances_ for model in models], axis=0
        )

        if self.oob_score:
            self.oob_decision_function_ = [
                self._restore_multioutput_oob_order(
                    model.oob_decision_function_,
                    model._training_row_order,
                    X.shape[0],
                )
                for model in models
            ]
            encoded_oob = np.column_stack(
                [
                    model.classes_[probabilities.argmax(axis=1)]
                    for model, probabilities in zip(models, self.oob_decision_function_)
                ]
            )
            valid = np.logical_and.reduce(
                [probabilities.sum(axis=1) > 0.0 for probabilities in self.oob_decision_function_]
            )
            if callable(self.oob_score):
                self.oob_score_ = float(self.oob_score(y, encoded_oob))
            else:
                self.oob_score_ = float(np.mean(np.all(encoded_oob[valid] == y[valid], axis=1)))
        return self

    @staticmethod
    def _restore_multioutput_oob_order(probabilities, row_order, n_samples):
        restored = np.zeros((n_samples, probabilities.shape[1]), dtype=np.float64)
        counts = np.zeros(n_samples, dtype=np.int64)
        valid = probabilities.sum(axis=1) > 0.0
        np.add.at(restored, row_order[valid], probabilities[valid])
        np.add.at(counts, row_order[valid], 1)
        present = counts > 0
        restored[present] /= counts[present, np.newaxis]
        return restored

    def predict(self, X):
        multioutput_models = getattr(self, "_multioutput_models", None)
        if multioutput_models:
            X = self._validate_multioutput_prediction_input(X)
            outputs = []
            for model in multioutput_models:
                encoded = np.asarray(model._forest.predict(X), dtype=np.intp)
                outputs.append(model.classes_[encoded])
            return np.column_stack(outputs)
        check_is_fitted(self, "_forest")
        sparse_input = sparse.issparse(X)
        if sparse_input:
            X = X.tocsr(copy=True)
            X.sum_duplicates()
            X.sort_indices()
            X.eliminate_zeros()
        validation_options = dict(reset=False, dtype=np.float64, ensure_2d=True, ensure_all_finite="allow-nan")
        if sparse_input:
            validation_options["accept_sparse"] = ("csr", "csc")
        X = validate_data(self, X, **validation_options)
        if sparse_input:
            X = X.tocsr(copy=False)
        encoded_y = np.asarray(self._forest.predict(X), dtype=np.intp)
        return self.classes_[encoded_y]

    def predict_proba(self, X):
        multioutput_models = getattr(self, "_multioutput_models", None)
        if multioutput_models:
            X = self._validate_multioutput_prediction_input(X)
            return [
                np.asarray(model._forest.predict_proba(X), dtype=np.float64)
                for model in multioutput_models
            ]
        check_is_fitted(self, "_forest")
        sparse_input = sparse.issparse(X)
        if sparse_input:
            X = X.tocsr(copy=True)
            X.sum_duplicates()
            X.sort_indices()
            X.eliminate_zeros()
        validation_options = dict(reset=False, dtype=np.float64, ensure_2d=True, ensure_all_finite="allow-nan")
        if sparse_input:
            validation_options["accept_sparse"] = ("csr", "csc")
        X = validate_data(self, X, **validation_options)
        if sparse_input:
            X = X.tocsr(copy=False)
        return np.asarray(self._forest.predict_proba(X), dtype=np.float64)

    def predict_log_proba(self, X):
        probabilities = self.predict_proba(X)
        with np.errstate(divide="ignore"):
            if isinstance(probabilities, list):
                return [np.log(output) for output in probabilities]
            return np.log(probabilities)

    def _validate_multioutput_prediction_input(self, X):
        check_is_fitted(self, "_forest")
        sparse_input = sparse.issparse(X)
        if sparse_input:
            X = X.tocsr(copy=True)
            X.sum_duplicates()
            X.sort_indices()
            X.eliminate_zeros()
        options = dict(
            reset=False,
            dtype=np.float64,
            ensure_2d=True,
            ensure_all_finite="allow-nan",
        )
        if sparse_input:
            options["accept_sparse"] = ("csr", "csc")
        X = validate_data(self, X, **options)
        return X.tocsr(copy=False) if sparse_input else X

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_forest", None)
        if state.get("_multioutput_models"):
            state.pop("_fit_X", None)
            state.pop("_fit_y", None)
            state.pop("_fit_sample_weight", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        multioutput_models = getattr(self, "_multioutput_models", None)
        if multioutput_models:
            self._forest = multioutput_models[0]._forest
            self._shap_estimators_cache = None
            return
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
            max_leaves=self._fit_max_leaves,
            min_samples_split=self._fit_min_samples_split,
            min_samples_leaf=self._fit_min_samples_leaf,
            min_impurity_decrease=self._fit_min_impurity_decrease,
            bootstrap=self.bootstrap,
            max_samples=self._fit_max_samples,
            oob=bool(self.oob_score),
            permutation_importance=self._fit_importance_type == "permutation",
            n_jobs=self._fit_n_jobs,
            max_bins=self._fit_max_bins,
            binning_strategy=getattr(
                self,
                "_fit_binning_strategy",
                getattr(self, "binning_strategy", "exact_sort"),
            ),
            bin_sample_size=getattr(
                self, "_fit_bin_sample_size", getattr(self, "bin_sample_size", 200_000)
            ),
            balanced_subsample=(
                self.class_weight == "balanced_subsample" and self.bootstrap is True
            ),
            ccp_alpha=self.ccp_alpha,
            monotonic_cst=self._fit_monotonic_cst,
        )
        self._forest = forest
        self._shap_estimators_cache = None
        if self._fit_importance_type == "shap":
            self.feature_importances_ = (
                None if self._fit_shap_mode == "explicit" else self._sampled_shap_importances()
            )
        else:
            self.feature_importances_ = self._selected_feature_importances(forest)

    def _next_seed(self):
        random_state = check_random_state(self.random_state)
        # RandomState's default integer dtype is int32 on 32-bit Python.
        return int(random_state.randint(0, np.iinfo(np.int32).max))

    def _reject_unsupported_baseline_parameters(self, sample_weight):
        if (
            not isinstance(self.ccp_alpha, (int, float, np.integer, np.floating))
            or isinstance(self.ccp_alpha, (bool, np.bool_))
            or not np.isfinite(self.ccp_alpha)
            or self.ccp_alpha < 0.0
        ):
            raise ValueError("ccp_alpha must be a finite non-negative number")
        if self.oob_score and self.bootstrap is not True:
            raise ValueError("Out of bag estimation only available if bootstrap=True")
        if not isinstance(self.oob_score, (bool, np.bool_)) and not callable(self.oob_score):
            raise ValueError("oob_score must be a boolean or callable")

    def _resolve_monotonic_cst(self, n_features):
        if self.monotonic_cst is None:
            return None
        if self.n_classes_ != 2:
            raise ValueError("monotonic_cst is supported only for binary classification")
        constraints = np.asarray(self.monotonic_cst)
        if constraints.ndim != 1 or constraints.shape[0] != n_features:
            raise ValueError(
                "monotonic_cst must have one value per feature (" + str(n_features) + ")"
            )
        try:
            constraints = constraints.astype(np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("monotonic_cst values must be -1, 0, or 1") from error
        if not np.isfinite(constraints).all() or not np.isin(constraints, [-1, 0, 1]).all():
            raise ValueError("monotonic_cst values must be -1, 0, or 1")
        return constraints.astype(np.int8).tolist()

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

    def _resolve_max_samples(self, n_samples, sample_weight=None):
        if self.max_samples is None:
            return None
        if self.bootstrap is not True:
            raise ValueError("max_samples can only be set if bootstrap=True")
        effective_samples = float(sample_weight.sum()) if sample_weight is not None else n_samples
        if isinstance(self.max_samples, (int, np.integer)) and 1 <= self.max_samples <= effective_samples:
            return int(self.max_samples)
        if isinstance(self.max_samples, (float, np.floating)) and 0.0 < self.max_samples <= 1.0:
            return max(1, int(self.max_samples * effective_samples))
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

    def _resolve_min_impurity_decrease(self):
        if isinstance(self.min_impurity_decrease, (float, int, np.floating, np.integer)) and (
            self.min_impurity_decrease >= 0.0
        ):
            return float(self.min_impurity_decrease)
        raise ValueError("min_impurity_decrease must be a non-negative number")

    def _resolve_max_leaf_nodes(self):
        if self.max_leaf_nodes is None:
            return None
        if isinstance(self.max_leaf_nodes, (int, np.integer)) and self.max_leaf_nodes >= 1:
            return int(self.max_leaf_nodes)
        raise ValueError("max_leaf_nodes must be an integer greater than or equal to 1")

    def _resolve_verbose(self):
        if isinstance(self.verbose, (int, np.integer)) and self.verbose >= 0:
            return int(self.verbose)
        raise ValueError("verbose must be a non-negative integer")

    def _resolve_importance_type(self):
        if self.importance_type not in ("gain", "split", "permutation", "shap"):
            raise ValueError("importance_type must be 'gain', 'split', 'permutation', or 'shap'")
        if self.importance_type == "permutation" and self.bootstrap is not True:
            raise ValueError("importance_type='permutation' requires bootstrap=True")
        return self.importance_type

    def _resolve_shap_mode(self):
        if self.shap_mode not in ("sampled", "all", "explicit"):
            raise ValueError("shap_mode must be 'sampled', 'all', or 'explicit'")
        if not isinstance(self.shap_sample_size, (int, np.integer)) or isinstance(self.shap_sample_size, (bool, np.bool_)) or self.shap_sample_size < 1:
            raise ValueError("shap_sample_size must be a positive integer")
        return self.shap_mode

    def _sampled_shap_importances(self):
        n_rows = self._fit_X.shape[0]
        if self._fit_shap_mode == "all" or n_rows <= self.shap_sample_size:
            selected = self._fit_X
        else:
            rng = check_random_state(self.random_state)
            rows = np.sort(rng.choice(n_rows, size=self.shap_sample_size, replace=False))
            selected = self._fit_X[rows]
        return self.shap_importances(selected)

    def shap_importances(self, X):
        """Return normalized mean absolute native TreeSHAP importance."""
        check_is_fitted(self, "_forest")
        options = dict(reset=False, dtype=np.float64, ensure_2d=True, ensure_all_finite="allow-nan")
        if sparse.issparse(X):
            options["accept_sparse"] = ("csr", "csc")
        X = validate_data(self, X, **options)
        if sparse.issparse(X):
            X = X.tocsr(copy=False)
        return np.asarray(
            self._forest.shap_importances(X, n_jobs=self._resolve_n_jobs()),
            dtype=np.float64,
        )

    def _resolve_max_bins(self):
        if self.max_bins is None:
            return None
        if (
            isinstance(self.max_bins, (int, np.integer))
            and not isinstance(self.max_bins, (bool, np.bool_))
            and 2 <= self.max_bins <= 255
        ):
            return int(self.max_bins)
        raise ValueError("max_bins must be None or an integer in [2, 255]")

    def _resolve_binning_strategy(self):
        allowed = ("exact_sort", "sampled_sort", "exact_select", "sampled_select")
        if self.binning_strategy not in allowed:
            choices = ", ".join(repr(value) for value in allowed)
            raise ValueError(f"binning_strategy must be one of {choices}")
        return self.binning_strategy

    def _resolve_bin_sample_size(self):
        if (
            isinstance(self.bin_sample_size, (int, np.integer))
            and not isinstance(self.bin_sample_size, (bool, np.bool_))
            and self.bin_sample_size > 0
        ):
            return int(self.bin_sample_size)
        raise ValueError("bin_sample_size must be a positive integer")

    def _selected_feature_importances(self, forest):
        method = (
            forest.permutation_importances
            if self._fit_importance_type == "permutation"
            else forest.gain_importances
            if self._fit_importance_type == "gain"
            else forest.feature_importances
        )
        return np.asarray(method(), dtype=np.float64)

    def _combine_class_weight(self, y, sample_weight):
        if self.class_weight is None:
            return sample_weight

        if self.class_weight == "balanced_subsample":
            if self.bootstrap:
                # The native forest computes class factors from each tree's bag.
                return sample_weight
            class_weight = compute_sample_weight("balanced", y)
            if sample_weight is None:
                return class_weight.astype(np.float64, copy=False)
            return sample_weight * class_weight

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
    def _prepare_training_data(
        X, encoded_y, sample_weight, return_order=False, defer_dense_sort=False
    ):
        source_rows = np.arange(X.shape[0], dtype=np.intp) if return_order else None
        if sample_weight is not None and np.equal(
            sample_weight, np.floor(sample_weight)
        ).all():
            repeats = sample_weight.astype(np.intp, copy=False)
            if return_order:
                source_rows = np.repeat(source_rows, repeats)
            if sparse.issparse(X):
                X = sparse.vstack([X[index] for index, count in enumerate(repeats) for _ in range(count)], format="csr")
            else:
                X = np.repeat(X, repeats, axis=0)
            encoded_y = np.repeat(encoded_y, repeats)
            sample_weight = None

        if sparse.issparse(X):
            order = np.argsort(encoded_y, kind="stable")
            columns = X.tocsc(copy=False)
            # Reproduce lexsort's tie ordering with one temporary feature vector
            # at a time, so sparse input never becomes a dense feature matrix.
            for feature in range(X.shape[1] - 1, -1, -1):
                values = np.zeros(X.shape[0], dtype=X.dtype)
                start, end = columns.indptr[feature : feature + 2]
                values[columns.indices[start:end]] = columns.data[start:end]
                order = order[np.argsort(values[order], kind="stable")]
        elif defer_dense_sort:
            # The native boundary computes the same stable canonical order and
            # copies rows directly into its owned matrix, avoiding NumPy's
            # full-size ``X[order]`` materialization here.
            return X, encoded_y, sample_weight
        else:
            keys = (encoded_y, *(X[:, index] for index in range(X.shape[1] - 1, -1, -1)))
            order = np.lexsort(keys)
        if return_order:
            source_rows = source_rows[order]
        if sample_weight is None:
            result = (X[order], encoded_y[order], None)
        else:
            result = (X[order], encoded_y[order], sample_weight[order])
        return (*result, source_rows) if return_order else result
