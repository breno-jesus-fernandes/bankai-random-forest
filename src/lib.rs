mod dense;

use dense::{ClassVotes, DenseInput};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use xrf::{Forest, Walk};

#[pyclass]
struct NativeForest {
    forest: Option<Forest<DenseInput>>,
    n_classes: usize,
    n_features: usize,
}

#[pymethods]
impl NativeForest {
    #[new]
    fn new() -> Self {
        Self {
            forest: None,
            n_classes: 0,
            n_features: 0,
        }
    }

    fn is_fitted(&self) -> bool {
        self.forest.is_some()
    }

    #[pyo3(signature = (x, y, n_estimators, max_features, random_state, sample_weight=None))]
    fn fit(
        &mut self,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, i64>,
        n_estimators: usize,
        max_features: usize,
        random_state: u64,
        sample_weight: Option<PyReadonlyArray1<'_, f64>>,
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
        let input = DenseInput::training(
            values,
            rows,
            columns,
            labels,
            sample_weights,
            n_classes,
        )
            .map_err(PyValueError::new_err)?;

        self.forest = Some(Forest::new(
            &input,
            n_estimators,
            max_features,
            true,
            false,
            false,
            random_state,
        ));
        self.n_classes = n_classes;
        self.n_features = columns;
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
            if let Walk::VisitBranch(feature, _) = walk {
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
