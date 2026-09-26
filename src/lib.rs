pub mod dense;

use dense::{ClassVotes, Criterion, DenseInput};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use xrf::{Forest, Walk};

#[derive(Clone)]
struct ShapNode {
    feature: i32,
    threshold: f64,
    left: usize,
    right: usize,
    cover: f64,
    vote: Option<usize>,
}

#[pyclass]
struct NativeForest {
    forest: Option<Forest<DenseInput>>,
    n_classes: usize,
    n_features: usize,
    n_samples: usize,
    histogram_edges: Option<Vec<Vec<f64>>>,
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
            histogram_edges: None,
        }
    }

    fn is_fitted(&self) -> bool {
        self.forest.is_some()
    }

    #[pyo3(signature = (x, y, n_estimators, max_features, random_state, sample_weight=None, min_leaf_weight=0.0, criterion="gini", max_depth=512, max_leaves=None, min_samples_split=2, min_samples_leaf=1, min_impurity_decrease=0.0, bootstrap=true, max_samples=None, oob=false, permutation_importance=false, n_jobs=1, max_bins=None, balanced_subsample=false, ccp_alpha=0.0))]
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
        max_bins: Option<usize>,
        balanced_subsample: bool,
        ccp_alpha: f64,
    ) -> PyResult<()> {
        if n_estimators == 0 {
            return Err(PyValueError::new_err("n_estimators must be at least 1"));
        }
        if !ccp_alpha.is_finite() || ccp_alpha < 0.0 {
            return Err(PyValueError::new_err(
                "ccp_alpha must be a finite non-negative number",
            ));
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
            max_bins,
        )
        .map_err(PyValueError::new_err)?;
        let input = if balanced_subsample {
            input.with_balanced_subsample()
        } else {
            input
        };
        let input = input.with_ccp_alpha(ccp_alpha);

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
        self.histogram_edges = input.bin_edges().map(|edges| edges.to_vec());
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

        let input = DenseInput::prediction(
            values,
            rows,
            columns,
            self.n_classes,
            self.histogram_edges.as_deref(),
        )
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

        let input = DenseInput::prediction(
            values,
            rows,
            columns,
            self.n_classes,
            self.histogram_edges.as_deref(),
        )
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

    /// Export compact tree arrays for the experimental Python TreeSHAP facade.
    /// Children are reversed because XRF's left branch is `value > pivot`,
    /// while sklearn tree arrays define left as `value <= threshold`.
    fn shap_tree_arrays(&self) -> PyResult<Vec<(Vec<i32>, Vec<i32>, Vec<i32>, Vec<f64>, Vec<f64>, Vec<f64>)>> {
        let forest = self.forest.as_ref().ok_or_else(|| PyRuntimeError::new_err("forest is not fitted"))?;
        Ok(forest.tree_refs().map(|tree| {
            let mut nodes = Vec::new();
            flatten_tree(tree, &mut nodes);
            let mut left = Vec::with_capacity(nodes.len());
            let mut right = Vec::with_capacity(nodes.len());
            let mut feature = Vec::with_capacity(nodes.len());
            let mut threshold = Vec::with_capacity(nodes.len());
            let mut cover = Vec::with_capacity(nodes.len());
            let mut values = Vec::with_capacity(nodes.len() * self.n_classes);
            for node in nodes {
                // See method documentation: swap child indexes for sklearn.
                left.push(if node.feature < 0 { -1 } else { node.right as i32 });
                right.push(if node.feature < 0 { -1 } else { node.left as i32 });
                feature.push(node.feature);
                let exported_threshold = self.histogram_edges.as_ref().and_then(|edges| {
                    let feature_edges = edges.get(node.feature.max(0) as usize)?;
                    feature_edges.get(node.threshold.floor() as usize).copied()
                }).unwrap_or(node.threshold);
                threshold.push(exported_threshold);
                cover.push(node.cover);
                for class in 0..self.n_classes {
                    values.push(f64::from(node.vote == Some(class)));
                }
            }
            (left, right, feature, threshold, cover, values)
        }).collect())
    }

    /// Experimental native TreeSHAP (`tree_path_dependent`) for the forest's
    /// vote probabilities. Returns `(values, base_values)`.
    fn tree_shap(&self, x: PyReadonlyArray2<'_, f64>) -> PyResult<(Vec<Vec<Vec<f64>>>, Vec<f64>)> {
        let forest = self.forest.as_ref().ok_or_else(|| PyRuntimeError::new_err("forest is not fitted"))?;
        let (raw, rows, columns) = matrix_to_owned(x)?;
        if columns != self.n_features { return Err(PyValueError::new_err("X has a different number of features")); }
        let input = DenseInput::prediction(raw, rows, columns, self.n_classes, self.histogram_edges.as_deref()).map_err(PyValueError::new_err)?;
        let trees: Vec<Vec<ShapNode>> = forest.tree_refs().map(|tree| { let mut nodes = Vec::new(); flatten_tree(tree, &mut nodes); nodes }).collect();
        let count = trees.len() as f64;
        let mut base = vec![0.0; self.n_classes];
        for tree in &trees {
            let mut tree_base = vec![0.0; self.n_classes];
            leaf_vote_covers(tree, 0, &mut tree_base);
            for (value, contribution) in base.iter_mut().zip(tree_base) {
                *value += contribution / tree[0].cover.max(1.0);
            }
        }
        for value in &mut base { *value /= count; }
        let mut output = vec![vec![vec![0.0; self.n_classes]; self.n_features]; rows];
        for row in 0..rows {
            for tree in &trees { tree_shap_one(tree, &input, row, self.n_features, self.n_classes, &mut output[row]); }
            for feature in &mut output[row] { for value in feature { *value /= count; } }
        }
        Ok((output, base))
    }
}

fn flatten_tree(tree: &xrf::Tree<DenseInput>, nodes: &mut Vec<ShapNode>) -> usize {
    let index = nodes.len();
    match tree {
        xrf::Tree::Leaf(vote, cover) => nodes.push(ShapNode { feature: -1, threshold: 0.0, left: 0, right: 0, cover: *cover as f64, vote: Some(*vote) }),
        xrf::Tree::Branch(feature, pivot, _, cover, left, right) => {
            nodes.push(ShapNode { feature: *feature as i32, threshold: *pivot, left: 0, right: 0, cover: *cover as f64, vote: None });
            let left_index = flatten_tree(left, nodes);
            let right_index = flatten_tree(right, nodes);
            nodes[index].left = left_index; nodes[index].right = right_index;
        }
    }
    index
}

fn leaf_vote_covers(tree: &[ShapNode], index: usize, out: &mut [f64]) {
    let node = &tree[index];
    if let Some(vote) = node.vote { out[vote] += node.cover; return; }
    leaf_vote_covers(tree, node.left, out);
    leaf_vote_covers(tree, node.right, out);
}

fn tree_shap_one(tree: &[ShapNode], input: &DenseInput, row: usize, features: usize, classes: usize, out: &mut [Vec<f64>]) {
    let max_depth = tree.len() + 2;
    let mut path_features = vec![0usize; max_depth]; let mut zero = vec![0.0; max_depth]; let mut one = vec![0.0; max_depth]; let mut weights = vec![0.0; max_depth];
    recurse_shap(tree, input, row, features, classes, out, 0, 0, &mut path_features, &mut zero, &mut one, &mut weights, 1.0, 1.0, usize::MAX);
}

#[allow(clippy::too_many_arguments)]
fn recurse_shap(tree: &[ShapNode], input: &DenseInput, row: usize, _features: usize, classes: usize, out: &mut [Vec<f64>], index: usize, depth: usize, pf: &mut [usize], z: &mut [f64], o: &mut [f64], w: &mut [f64], parent_zero: f64, parent_one: f64, parent_feature: usize) {
    pf[depth] = parent_feature; z[depth] = parent_zero; o[depth] = parent_one;
    if depth == 0 { w[depth] = 1.0; } else { w[depth] = 0.0; }
    for i in (0..depth).rev() { w[i + 1] += parent_one * w[i] * (i + 1) as f64 / (depth + 1) as f64; w[i] = parent_zero * w[i] * (depth - i) as f64 / (depth + 1) as f64; }
    let node = &tree[index];
    if let Some(vote) = node.vote {
        for i in 1..=depth { let s = unwound_sum(z, o, w, depth, i); out[pf[i]][vote] += s * (o[i] - z[i]); }
        return;
    }
    let feature = node.feature as usize;
    let hot = if input.value_at(row, feature) > node.threshold { node.left } else { node.right };
    let cold = if hot == node.left { node.right } else { node.left };
    let cover = node.cover.max(1.0);
    let hot_zero = tree[hot].cover / cover; let cold_zero = tree[cold].cover / cover;
    let mut incoming_zero = 1.0; let mut incoming_one = 1.0; let mut new_depth = depth;
    if let Some(path_index) = (0..=depth).find(|&i| pf[i] == feature) { incoming_zero = z[path_index]; incoming_one = o[path_index]; unwind(pf, z, o, w, depth, path_index); new_depth -= 1; }
    // Each branch extends its own copy of the unique path. Sharing these
    // buffers would let the hot branch overwrite the cold branch's path.
    let (mut hot_pf, mut hot_z, mut hot_o, mut hot_w) = (pf.to_vec(), z.to_vec(), o.to_vec(), w.to_vec());
    recurse_shap(tree, input, row, _features, classes, out, hot, new_depth + 1, &mut hot_pf, &mut hot_z, &mut hot_o, &mut hot_w, hot_zero * incoming_zero, incoming_one, feature);
    let (mut cold_pf, mut cold_z, mut cold_o, mut cold_w) = (pf.to_vec(), z.to_vec(), o.to_vec(), w.to_vec());
    recurse_shap(tree, input, row, _features, classes, out, cold, new_depth + 1, &mut cold_pf, &mut cold_z, &mut cold_o, &mut cold_w, cold_zero * incoming_zero, 0.0, feature);
}

fn unwind(pf: &mut [usize], z: &mut [f64], o: &mut [f64], w: &mut [f64], depth: usize, at: usize) { let one = o[at]; let zero = z[at]; let mut next = w[depth]; for i in (0..depth).rev() { if one != 0.0 { let tmp = w[i]; w[i] = next * (depth + 1) as f64 / ((i + 1) as f64 * one); next = tmp - w[i] * zero * (depth - i) as f64 / (depth + 1) as f64; } else { w[i] = w[i] * (depth + 1) as f64 / (zero * (depth - i) as f64); } } for i in at..depth { pf[i] = pf[i + 1]; z[i] = z[i + 1]; o[i] = o[i + 1]; } }
fn unwound_sum(z: &[f64], o: &[f64], w: &[f64], depth: usize, at: usize) -> f64 { let one = o[at]; let zero = z[at]; let mut next = w[depth]; let mut total = 0.0; for i in (0..depth).rev() { if one != 0.0 { let tmp = next * (depth + 1) as f64 / ((i + 1) as f64 * one); total += tmp; next = w[i] - tmp * zero * (depth - i) as f64 / (depth + 1) as f64; } else { total += (w[i] / zero) / ((depth - i) as f64 / (depth + 1) as f64); } } total }

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
