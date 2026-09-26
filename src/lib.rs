pub mod dense;

use dense::{ClassVotes, Criterion, DenseInput};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use xrf::{Forest, Walk};

#[pyclass]
struct NativeForest {
    forest: Option<Forest<DenseInput>>,
    n_classes: usize,
    n_features: usize,
    n_samples: usize,
}

#[pymethods]
impl NativeForest {
    #[new]
    fn new() -> Self {
        Self {
            forest: None,
            n_classes: 0,
            n_features: 0,
            n_samples: 0,
        }
    }

    fn is_fitted(&self) -> bool {
        self.forest.is_some()
    }

    #[pyo3(signature = (x, y, n_estimators, max_features, random_state, sample_weight=None, min_leaf_weight=0.0, criterion="gini", max_depth=512, max_leaves=None, min_samples_split=2, min_samples_leaf=1, min_impurity_decrease=0.0, bootstrap=true, max_samples=None, oob=false, permutation_importance=false, n_jobs=1))]
    fn fit(
        &mut self,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, i64>,
        n_estimators: usize,
        max_features: usize,
        random_state: u64,
        sample_weight: Option<PyReadonlyArray1<'_, f64>>,
        min_leaf_weight: f64,
        criterion: &str,
        max_depth: usize,
        max_leaves: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
        min_impurity_decrease: f64,
        bootstrap: bool,
        max_samples: Option<usize>,
        oob: bool,
        permutation_importance: bool,
        n_jobs: usize,
    ) -> PyResult<()> {
        if n_estimators == 0 {
            return Err(PyValueError::new_err("n_estimators must be at least 1"));
        }

        let (values, rows, columns) = matrix_to_owned(x)?;
        if max_features == 0 || max_features > columns {
            return Err(PyValueError::new_err(
                "max_features must be between 1 and the number of features",
            ));
        }

        let labels: Vec<usize> = y
            .as_array()
            .iter()
            .copied()
            .map(|label| {
                usize::try_from(label)
                    .map_err(|_| PyValueError::new_err("y must contain non-negative labels"))
            })
            .collect::<PyResult<_>>()?;
        let sample_weights = match sample_weight {
            Some(weights) => weights.as_array().iter().copied().collect(),
            None => vec![1.0; rows],
        };
        let n_classes = labels
            .iter()
            .copied()
            .max()
            .and_then(|label| label.checked_add(1))
            .ok_or_else(|| PyValueError::new_err("y must contain at least one class"))?;
        let criterion = match criterion {
            "gini" => Criterion::Gini,
            "entropy" | "log_loss" => Criterion::Entropy,
            _ => return Err(PyValueError::new_err("unsupported criterion")),
        };
        let input = DenseInput::training(
            values,
            rows,
            columns,
            labels,
            sample_weights,
            n_classes,
            min_leaf_weight,
            criterion,
            min_samples_split,
            min_samples_leaf,
            min_impurity_decrease,
        )
        .map_err(PyValueError::new_err)?;

        self.forest = Some(if n_jobs == 1 {
            Forest::new_with_settings(
                &input,
                n_estimators,
                max_features,
                true,
                permutation_importance,
                oob,
                random_state,
                max_depth,
                max_leaves.unwrap_or(usize::MAX),
                bootstrap,
                max_samples,
            )
        } else {
            Forest::new_parallel_with_settings(
                &input,
                n_estimators,
                max_features,
                true,
                permutation_importance,
                oob,
                random_state,
                n_jobs,
                max_depth,
                max_leaves.unwrap_or(usize::MAX),
                bootstrap,
                max_samples,
            )
        });
        self.n_classes = n_classes;
        self.n_features = columns;
        self.n_samples = rows;
        Ok(())
    }

    fn predict(&self, x: PyReadonlyArray2<'_, f64>) -> PyResult<Vec<usize>> {
        let forest = self
            .forest
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("forest is not fitted"))?;
        let (values, rows, columns) = matrix_to_owned(x)?;
        if columns != self.n_features {
            return Err(PyValueError::new_err(format!(
                "X has {columns} features, but the fitted forest expects {}",
                self.n_features
            )));
        }

        let input = DenseInput::prediction(values, rows, columns, self.n_classes)
            .map_err(PyValueError::new_err)?;
        let prediction = forest.predict(&input);
        let mut labels = vec![0; input.rows()];
        for (row, votes) in prediction.predictions() {
            labels[row] = class_from_votes(votes);
        }
        Ok(labels)
    }

    fn predict_proba(&self, x: PyReadonlyArray2<'_, f64>) -> PyResult<Vec<Vec<f64>>> {
        let forest = self
            .forest
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("forest is not fitted"))?;
        let (values, rows, columns) = matrix_to_owned(x)?;
        if columns != self.n_features {
            return Err(PyValueError::new_err(format!(
                "X has {columns} features, but the fitted forest expects {}",
                self.n_features
            )));
        }

        let input = DenseInput::prediction(values, rows, columns, self.n_classes)
            .map_err(PyValueError::new_err)?;
        Ok(forest
            .predict(&input)
            .predictions()
            .map(|(_, votes)| votes.probabilities())
            .collect())
    }

    fn feature_importances(&self) -> PyResult<Vec<f64>> {
        let forest = self
            .forest
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("forest is not fitted"))?;
        let mut importances = vec![0.0; self.n_features];
        for walk in forest.walk() {
            if let Walk::VisitBranch(feature, _, _) = walk {
                importances[feature] += 1.0;
            }
        }
        let total = importances.iter().sum::<f64>();
        if total > 0.0 {
            for importance in &mut importances {
                *importance /= total;
            }
        }
        Ok(importances)
    }

    fn gain_importances(&self) -> PyResult<Vec<f64>> {
        let forest = self
            .forest
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("forest is not fitted"))?;
        let mut importances = vec![0.0; self.n_features];
        for (feature, importance) in forest.gain_importance_normalised() {
            importances[feature] = importance;
        }
        Ok(importances)
    }

    fn permutation_importances(&self) -> PyResult<Vec<f64>> {
        let forest = self
            .forest
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("forest is not fitted"))?;
        let mut importances = vec![0.0; self.n_features];
        for (feature, importance) in forest.importance() {
            importances[feature] = importance;
        }
        Ok(importances)
    }

    fn oob_predict_proba(&self) -> PyResult<Vec<Vec<f64>>> {
        let forest = self
            .forest
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("forest is not fitted"))?;
        if !forest.has_oob() {
            return Err(PyRuntimeError::new_err(
                "out-of-bag predictions are unavailable",
            ));
        }
        let mut probabilities = vec![vec![0.0; self.n_classes]; self.n_samples];
        for (row, votes) in forest.oob() {
            probabilities[row] = votes.probabilities();
        }
        Ok(probabilities)
    }
}

fn matrix_to_owned(x: PyReadonlyArray2<'_, f64>) -> PyResult<(Vec<f64>, usize, usize)> {
    let array = x.as_array();
    let rows = array.nrows();
    let columns = array.ncols();
    let values: Vec<f64> = array.iter().copied().collect();

    if values.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("X must contain only finite values"));
    }

    Ok((values, rows, columns))
}

fn class_from_votes(votes: &ClassVotes) -> usize {
    votes.winner()
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeForest>()?;
    Ok(())
}
