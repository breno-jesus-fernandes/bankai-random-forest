use std::collections::HashMap;
use std::sync::Arc;
use std::thread;
use xrf::{
    AccuracyDecreaseAggregator, DecisionSlice, FairBest, FeatureSampler, Mask, RfInput, RfRng,
    VoteAggregator,
};

#[derive(Clone)]
pub struct DenseInput {
    values: DenseValues,
    labels: Option<Arc<Vec<usize>>>,
    sample_weights: Option<Arc<Vec<f64>>>,
    rows: usize,
    columns: usize,
    n_classes: usize,
    min_leaf_weight: f64,
    criterion: Criterion,
    min_samples_split: usize,
    min_samples_leaf: usize,
    min_impurity_decrease: f64,
    total_weight: f64,
    histogram_bins: Option<usize>,
    bin_edges: Option<Arc<Vec<Vec<f64>>>>,
    active_features: Vec<usize>,
    balanced_subsample: bool,
    ccp_alpha: f64,
    monotonic_constraints: Option<Arc<Vec<i8>>>,
    has_missing_values: bool,
    histogram_threads: usize,
}

#[derive(Clone)]
enum DenseValues {
    Exact(Arc<Vec<f64>>),
    Binned(Arc<Vec<u8>>),
    Sparse(Arc<CsrValues>),
}

#[derive(Clone)]
struct CsrValues {
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<f64>,
}

struct FeatureHistogram<'a> {
    class_weights: &'a [f64],
    sample_counts: &'a [usize],
}

#[derive(Clone, Debug, PartialEq)]
struct HistogramCache {
    // One contiguous allocation per statistic, shared feature boundaries.
    bin_offsets: Vec<usize>,
    class_weights: Vec<f64>,
    sample_counts: Vec<usize>,
}

const HISTOGRAM_PARALLEL_MIN_CELLS: usize = 32 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinningStrategy {
    ExactSort,
    SampledSort,
    ExactSelect,
    SampledSelect,
}

impl BinningStrategy {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "exact_sort" => Ok(Self::ExactSort),
            "sampled_sort" => Ok(Self::SampledSort),
            "exact_select" => Ok(Self::ExactSelect),
            "sampled_select" => Ok(Self::SampledSelect),
            _ => Err("binning_strategy must be 'exact_sort', 'sampled_sort', 'exact_select', or 'sampled_select'".into()),
        }
    }

    fn samples(self) -> bool {
        matches!(self, Self::SampledSort | Self::SampledSelect)
    }

    fn selects(self) -> bool {
        matches!(self, Self::ExactSelect | Self::SampledSelect)
    }
}

impl HistogramCache {
    fn feature(&self, feature: usize, classes: usize) -> FeatureHistogram<'_> {
        let start = self.bin_offsets[feature];
        let end = self.bin_offsets[feature + 1];
        FeatureHistogram {
            class_weights: &self.class_weights[start * classes..end * classes],
            sample_counts: &self.sample_counts[start..end],
        }
    }
}

pub struct DenseSplitCache(Option<HistogramCache>);

enum DenseSplitIter<'a> {
    Lazy {
        input: &'a DenseInput,
        rows: std::slice::Iter<'a, usize>,
        feature: usize,
        pivot: f64,
    },
    Buffered(std::vec::IntoIter<bool>),
}

impl Iterator for DenseSplitIter<'_> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Lazy {
                input,
                rows,
                feature,
                pivot,
            } => rows.next().map(|&row| input.value(row, *feature) > *pivot),
            Self::Buffered(values) => values.next(),
        }
    }
}

#[derive(Clone, Copy)]
pub enum Criterion {
    Gini,
    Entropy,
}

impl DenseInput {
    pub fn training(
        values: Vec<f64>,
        rows: usize,
        columns: usize,
        labels: Vec<usize>,
        sample_weights: Vec<f64>,
        n_classes: usize,
        min_leaf_weight: f64,
        criterion: Criterion,
        min_samples_split: usize,
        min_samples_leaf: usize,
        min_impurity_decrease: f64,
        max_bins: Option<usize>,
    ) -> Result<Self, String> {
        Self::training_with_threads(
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
            1,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn training_with_threads(
        values: Vec<f64>,
        rows: usize,
        columns: usize,
        labels: Vec<usize>,
        sample_weights: Vec<f64>,
        n_classes: usize,
        min_leaf_weight: f64,
        criterion: Criterion,
        min_samples_split: usize,
        min_samples_leaf: usize,
        min_impurity_decrease: f64,
        max_bins: Option<usize>,
        threads: usize,
    ) -> Result<Self, String> {
        Self::training_with_binning_options(
            values, rows, columns, labels, sample_weights, n_classes, min_leaf_weight,
            criterion, min_samples_split, min_samples_leaf, min_impurity_decrease,
            max_bins, threads, BinningStrategy::ExactSort, 200_000,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn training_with_binning_options(
        values: Vec<f64>,
        rows: usize,
        columns: usize,
        labels: Vec<usize>,
        sample_weights: Vec<f64>,
        n_classes: usize,
        min_leaf_weight: f64,
        criterion: Criterion,
        min_samples_split: usize,
        min_samples_leaf: usize,
        min_impurity_decrease: f64,
        max_bins: Option<usize>,
        threads: usize,
        binning_strategy: BinningStrategy,
        bin_sample_size: usize,
    ) -> Result<Self, String> {
        if threads == 0 {
            return Err("histogram preprocessing requires at least one thread".into());
        }
        let has_missing_values = values.iter().any(|value| value.is_nan());
        validate_matrix(&values, rows, columns)?;
        if labels.len() != rows {
            return Err("y must contain one label per row".to_string());
        }
        if sample_weights.len() != rows {
            return Err("sample_weight must contain one value per row".to_string());
        }
        let total_weight = validate_sample_weights(&sample_weights)?;
        if n_classes == 0 {
            return Err("at least one class is required".to_string());
        }
        if !min_leaf_weight.is_finite() || min_leaf_weight < 0.0 {
            return Err("min_leaf_weight must be finite and non-negative".to_string());
        }
        if min_samples_split < 2 || min_samples_leaf == 0 {
            return Err("invalid minimum sample control".to_string());
        }
        if !min_impurity_decrease.is_finite() || min_impurity_decrease < 0.0 {
            return Err("min_impurity_decrease must be finite and non-negative".to_string());
        }
        if labels.iter().any(|&label| label >= n_classes) {
            return Err("labels must be encoded in 0..n_classes".to_string());
        }
        if max_bins.is_some_and(|bins| !(2..=255).contains(&bins)) {
            return Err("max_bins must be None or an integer in [2, 255]".to_string());
        }
        if bin_sample_size == 0 {
            return Err("bin_sample_size must be a positive integer".into());
        }

        let (values, bin_edges, active_features) = match max_bins {
            Some(max_bins) => {
                let (binned, edges) = histogramize(
                    &values, rows, columns, max_bins, threads, binning_strategy, bin_sample_size,
                );
                let active = edges
                    .iter()
                    .enumerate()
                    .filter_map(|(feature, feature_edges)| {
                        (!feature_edges.is_empty()).then_some(feature)
                    })
                    .collect();
                (
                    DenseValues::Binned(Arc::new(binned)),
                    Some(Arc::new(edges)),
                    active,
                )
            }
            None => (
                DenseValues::Exact(Arc::new(values)),
                None,
                (0..columns).collect(),
            ),
        };
        Ok(Self {
            values,
            labels: Some(Arc::new(labels)),
            sample_weights: Some(Arc::new(sample_weights)),
            rows,
            columns,
            n_classes,
            min_leaf_weight,
            criterion,
            min_samples_split,
            min_samples_leaf,
            min_impurity_decrease,
            total_weight,
            histogram_bins: max_bins,
            bin_edges,
            active_features,
            balanced_subsample: false,
            ccp_alpha: 0.0,
            monotonic_constraints: None,
            has_missing_values,
            histogram_threads: 1,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn training_binned_with_accessor<F>(
        rows: usize,
        columns: usize,
        labels: Vec<usize>,
        sample_weights: Vec<f64>,
        n_classes: usize,
        min_leaf_weight: f64,
        criterion: Criterion,
        min_samples_split: usize,
        min_samples_leaf: usize,
        min_impurity_decrease: f64,
        max_bins: usize,
        threads: usize,
        strategy: BinningStrategy,
        sample_size: usize,
        value_at: F,
    ) -> Result<Self, String>
    where
        F: Fn(usize, usize) -> f64 + Sync,
    {
        if threads == 0 {
            return Err("histogram preprocessing requires at least one thread".into());
        }
        if rows == 0 {
            return Err("X must contain at least one row".into());
        }
        if columns == 0 {
            return Err("X must contain at least one feature".into());
        }
        rows.checked_mul(columns)
            .ok_or_else(|| "X shape is too large".to_string())?;
        if !(2..=255).contains(&max_bins) {
            return Err("max_bins must be an integer in [2, 255]".into());
        }
        if sample_size == 0 {
            return Err("bin_sample_size must be a positive integer".into());
        }
        if labels.len() != rows {
            return Err("y must contain one label per row".into());
        }
        if sample_weights.len() != rows {
            return Err("sample_weight must contain one value per row".into());
        }
        let total_weight = validate_sample_weights(&sample_weights)?;
        if n_classes == 0 {
            return Err("at least one class is required".into());
        }
        if !min_leaf_weight.is_finite() || min_leaf_weight < 0.0 {
            return Err("min_leaf_weight must be finite and non-negative".into());
        }
        if min_samples_split < 2 || min_samples_leaf == 0 {
            return Err("invalid minimum sample control".into());
        }
        if !min_impurity_decrease.is_finite() || min_impurity_decrease < 0.0 {
            return Err("min_impurity_decrease must be finite and non-negative".into());
        }
        if labels.iter().any(|&label| label >= n_classes) {
            return Err("labels must be encoded in 0..n_classes".into());
        }

        let (binned, edges, has_missing_values) = histogramize_from_accessor(
            rows,
            columns,
            max_bins,
            threads,
            strategy,
            sample_size,
            &value_at,
        )?;
        let active_features = edges
            .iter()
            .enumerate()
            .filter_map(|(feature, feature_edges)| {
                (!feature_edges.is_empty()).then_some(feature)
            })
            .collect();
        Ok(Self {
            values: DenseValues::Binned(Arc::new(binned)),
            labels: Some(Arc::new(labels)),
            sample_weights: Some(Arc::new(sample_weights)),
            rows,
            columns,
            n_classes,
            min_leaf_weight,
            criterion,
            min_samples_split,
            min_samples_leaf,
            min_impurity_decrease,
            total_weight,
            histogram_bins: Some(max_bins),
            bin_edges: Some(Arc::new(edges)),
            active_features,
            balanced_subsample: false,
            ccp_alpha: 0.0,
            monotonic_constraints: None,
            has_missing_values,
            histogram_threads: 1,
        })
    }

    pub fn prediction(
        values: Vec<f64>,
        rows: usize,
        columns: usize,
        n_classes: usize,
        bin_edges: Option<&[Vec<f64>]>,
    ) -> Result<Self, String> {
        let has_missing_values = values.iter().any(|value| value.is_nan());
        validate_matrix(&values, rows, columns)?;
        if n_classes == 0 {
            return Err("at least one class is required".to_string());
        }

        let values = match bin_edges {
            Some(edges) => {
                if edges.len() != columns {
                    return Err("histogram edges do not match the fitted feature count".to_string());
                }
                DenseValues::Binned(Arc::new(apply_histogram_edges(
                    &values, rows, columns, edges,
                )))
            }
            None => DenseValues::Exact(Arc::new(values)),
        };

        Ok(Self {
            values,
            labels: None,
            sample_weights: None,
            rows,
            columns,
            n_classes,
            min_leaf_weight: 0.0,
            criterion: Criterion::Gini,
            min_samples_split: 2,
            min_samples_leaf: 1,
            min_impurity_decrease: 0.0,
            total_weight: 0.0,
            histogram_bins: None,
            bin_edges: bin_edges.map(|edges| Arc::new(edges.to_vec())),
            active_features: (0..columns).collect(),
            balanced_subsample: false,
            ccp_alpha: 0.0,
            monotonic_constraints: None,
            has_missing_values,
            histogram_threads: 1,
        })
    }

    pub fn training_csr(
        indptr: Vec<usize>,
        indices: Vec<usize>,
        data: Vec<f64>,
        rows: usize,
        columns: usize,
        labels: Vec<usize>,
        sample_weights: Vec<f64>,
        n_classes: usize,
        min_leaf_weight: f64,
        criterion: Criterion,
        min_samples_split: usize,
        min_samples_leaf: usize,
        min_impurity_decrease: f64,
        max_bins: Option<usize>,
    ) -> Result<Self, String> {
        Self::training_csr_with_threads(
            indptr,
            indices,
            data,
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
            1,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn training_csr_with_threads(
        indptr: Vec<usize>,
        indices: Vec<usize>,
        data: Vec<f64>,
        rows: usize,
        columns: usize,
        labels: Vec<usize>,
        sample_weights: Vec<f64>,
        n_classes: usize,
        min_leaf_weight: f64,
        criterion: Criterion,
        min_samples_split: usize,
        min_samples_leaf: usize,
        min_impurity_decrease: f64,
        max_bins: Option<usize>,
        threads: usize,
    ) -> Result<Self, String> {
        Self::training_csr_with_binning_options(
            indptr, indices, data, rows, columns, labels, sample_weights, n_classes,
            min_leaf_weight, criterion, min_samples_split, min_samples_leaf,
            min_impurity_decrease, max_bins, threads, BinningStrategy::ExactSort, 200_000,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn training_csr_with_binning_options(
        indptr: Vec<usize>,
        indices: Vec<usize>,
        data: Vec<f64>,
        rows: usize,
        columns: usize,
        labels: Vec<usize>,
        sample_weights: Vec<f64>,
        n_classes: usize,
        min_leaf_weight: f64,
        criterion: Criterion,
        min_samples_split: usize,
        min_samples_leaf: usize,
        min_impurity_decrease: f64,
        max_bins: Option<usize>,
        threads: usize,
        binning_strategy: BinningStrategy,
        bin_sample_size: usize,
    ) -> Result<Self, String> {
        if threads == 0 {
            return Err("histogram preprocessing requires at least one thread".into());
        }
        let has_missing_values = data.iter().any(|value| value.is_nan());
        let sparse = validate_csr(indptr, indices, data, rows, columns)?;
        if labels.len() != rows {
            return Err("y must contain one label per row".into());
        }
        if sample_weights.len() != rows {
            return Err("sample_weight must contain one value per row".into());
        }
        let total_weight = validate_sample_weights(&sample_weights)?;
        if n_classes == 0 || labels.iter().any(|&label| label >= n_classes) {
            return Err("labels must be encoded in 0..n_classes".into());
        }
        if !min_leaf_weight.is_finite() || min_leaf_weight < 0.0 {
            return Err("min_leaf_weight must be finite and non-negative".into());
        }
        if min_samples_split < 2 || min_samples_leaf == 0 {
            return Err("invalid minimum sample control".into());
        }
        if !min_impurity_decrease.is_finite() || min_impurity_decrease < 0.0 {
            return Err("min_impurity_decrease must be finite and non-negative".into());
        }
        if max_bins.is_some_and(|bins| !(2..=255).contains(&bins)) {
            return Err("max_bins must be None or an integer in [2, 255]".into());
        }
        if bin_sample_size == 0 {
            return Err("bin_sample_size must be a positive integer".into());
        }
        let bin_edges = max_bins.map(|bins| {
            sparse_histogram_edges(
                &sparse, rows, columns, bins, threads, binning_strategy, bin_sample_size,
            )
        });
        let active_features = bin_edges.as_ref().map_or_else(
            || (0..columns).collect(),
            |edges| {
                edges
                    .iter()
                    .enumerate()
                    .filter_map(|(i, e)| (!e.is_empty()).then_some(i))
                    .collect()
            },
        );
        Ok(Self {
            values: DenseValues::Sparse(Arc::new(sparse)),
            labels: Some(Arc::new(labels)),
            sample_weights: Some(Arc::new(sample_weights)),
            rows,
            columns,
            n_classes,
            min_leaf_weight,
            criterion,
            min_samples_split,
            min_samples_leaf,
            min_impurity_decrease,
            total_weight,
            histogram_bins: max_bins,
            bin_edges: bin_edges.map(Arc::new),
            active_features,
            balanced_subsample: false,
            ccp_alpha: 0.0,
            monotonic_constraints: None,
            has_missing_values,
            histogram_threads: 1,
        })
    }

    pub fn prediction_csr(
        indptr: Vec<usize>,
        indices: Vec<usize>,
        data: Vec<f64>,
        rows: usize,
        columns: usize,
        n_classes: usize,
        bin_edges: Option<&[Vec<f64>]>,
    ) -> Result<Self, String> {
        let has_missing_values = data.iter().any(|value| value.is_nan());
        let sparse = validate_csr(indptr, indices, data, rows, columns)?;
        if n_classes == 0 {
            return Err("at least one class is required".into());
        }
        if bin_edges.is_some_and(|edges| edges.len() != columns) {
            return Err("histogram edges do not match the fitted feature count".into());
        }
        Ok(Self {
            values: DenseValues::Sparse(Arc::new(sparse)),
            labels: None,
            sample_weights: None,
            rows,
            columns,
            n_classes,
            min_leaf_weight: 0.0,
            criterion: Criterion::Gini,
            min_samples_split: 2,
            min_samples_leaf: 1,
            min_impurity_decrease: 0.0,
            total_weight: 0.0,
            histogram_bins: None,
            bin_edges: bin_edges.map(|e| Arc::new(e.to_vec())),
            active_features: (0..columns).collect(),
            balanced_subsample: false,
            ccp_alpha: 0.0,
            monotonic_constraints: None,
            has_missing_values,
            histogram_threads: 1,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn bin_edges(&self) -> Option<&[Vec<f64>]> {
        self.bin_edges.as_ref().map(|edges| edges.as_slice())
    }

    pub fn value_at(&self, row: usize, column: usize) -> f64 {
        self.value(row, column)
    }

    pub fn with_balanced_subsample(mut self) -> Self {
        self.balanced_subsample = true;
        self
    }

    pub fn with_ccp_alpha(mut self, ccp_alpha: f64) -> Self {
        self.ccp_alpha = ccp_alpha;
        self
    }

    pub fn with_histogram_threads(mut self, threads: usize) -> Self {
        self.histogram_threads = threads.max(1);
        self
    }

    pub fn with_monotonic_constraints(mut self, constraints: Vec<i8>) -> Self {
        self.monotonic_constraints = Some(Arc::new(constraints));
        self
    }

    #[cfg(test)]
    fn shares_feature_storage_with(&self, other: &Self) -> bool {
        match (&self.values, &other.values) {
            (DenseValues::Exact(left), DenseValues::Exact(right)) => Arc::ptr_eq(left, right),
            (DenseValues::Binned(left), DenseValues::Binned(right)) => Arc::ptr_eq(left, right),
            (DenseValues::Sparse(left), DenseValues::Sparse(right)) => Arc::ptr_eq(left, right),
            _ => false,
        }
    }

    fn balanced_tree_view(&self, bag: &Mask) -> Self {
        let labels = self.labels();
        let mut class_counts = vec![0usize; self.n_classes];
        for &row in bag.iter() {
            class_counts[labels[row]] += 1;
        }
        let present_classes = class_counts.iter().filter(|&&count| count > 0).count();
        let sample_count = bag.len();
        let mut weights = self
            .sample_weights
            .as_ref()
            .expect("balanced bootstrap requires training weights")
            .as_ref()
            .clone();
        for (row, weight) in weights.iter_mut().enumerate() {
            let count = class_counts[labels[row]];
            *weight = if count == 0 {
                0.0
            } else {
                *weight * sample_count as f64 / (present_classes * count) as f64
            };
        }
        let tree_total_weight = bag.iter().map(|&row| weights[row]).sum::<f64>();
        let min_leaf_fraction = if self.total_weight > 0.0 {
            self.min_leaf_weight / self.total_weight
        } else {
            0.0
        };
        Self {
            sample_weights: Some(Arc::new(weights)),
            total_weight: tree_total_weight,
            min_leaf_weight: min_leaf_fraction * tree_total_weight,
            balanced_subsample: false,
            ccp_alpha: self.ccp_alpha,
            monotonic_constraints: self.monotonic_constraints.clone(),
            ..self.clone()
        }
    }

    fn labels(&self) -> &[usize] {
        self.labels
            .as_deref()
            .expect("prediction input cannot be used for training")
    }

    fn sample_weight(&self, row: usize) -> f64 {
        self.sample_weights
            .as_deref()
            .expect("prediction input cannot be used for training")[row]
    }

    fn value(&self, row: usize, column: usize) -> f64 {
        match &self.values {
            DenseValues::Exact(values) => values[row * self.columns + column],
            DenseValues::Binned(values) => {
                let bin = values[row * self.columns + column];
                let missing_bin = self.bin_edges.as_ref().map_or(usize::MAX, |e| e[column].len() + 1);
                if bin as usize == missing_bin { f64::NAN } else { bin as f64 }
            }
            DenseValues::Sparse(values) => {
                let value = values.get(row, column);
                if value.is_nan() {
                    value
                } else {
                    self.bin_edges.as_ref().map_or(value, |edges| {
                        edges[column].partition_point(|edge| value > *edge) as f64
                    })
                }
            }
        }
    }

    #[inline]
    fn routes_left(&self, row: usize, column: usize, pivot: f64, missing_left: bool) -> bool {
        if !self.has_missing_values {
            return match &self.values {
                DenseValues::Exact(values) => values[row * self.columns + column] > pivot,
                DenseValues::Binned(values) => (values[row * self.columns + column] as f64) > pivot,
                DenseValues::Sparse(values) => {
                    let value = values.get(row, column);
                    let value = self.bin_edges.as_ref().map_or(value, |edges| edges[column].partition_point(|edge| value > *edge) as f64);
                    value > pivot
                }
            };
        }
        match &self.values {
            DenseValues::Exact(values) => {
                let value = values[row * self.columns + column];
                if value.is_nan() { missing_left } else { value > pivot }
            }
            DenseValues::Binned(values) => {
                let bin = values[row * self.columns + column];
                let missing_bin = self.bin_edges.as_ref().map_or(usize::MAX, |e| e[column].len() + 1);
                if bin as usize == missing_bin { missing_left } else { (bin as f64) > pivot }
            }
            DenseValues::Sparse(values) => {
                let raw = values.get(row, column);
                if raw.is_nan() { return missing_left; }
                let value = self.bin_edges.as_ref().map_or(raw, |edges| edges[column].partition_point(|edge| raw > *edge) as f64);
                value > pivot
            }
        }
    }

    fn bin_value(&self, row: usize, column: usize) -> usize {
        match &self.values {
            DenseValues::Binned(values) => values[row * self.columns + column] as usize,
            DenseValues::Sparse(values) => {
                let value = values.get(row, column);
                if value.is_nan() { self.bin_edges.as_ref().unwrap()[column].len() + 1 }
                else { self.bin_edges.as_ref().unwrap()[column].partition_point(|edge| value > *edge) }
            }
            DenseValues::Exact(_) => unreachable!("histograms require binned input"),
        }
    }
}

impl CsrValues {
    fn get(&self, row: usize, column: usize) -> f64 {
        let start = self.indptr[row];
        let end = self.indptr[row + 1];
        match self.indices[start..end].binary_search(&column) {
            Ok(i) => self.data[start + i],
            Err(_) => 0.0,
        }
    }
}

fn validate_sample_weights(sample_weights: &[f64]) -> Result<f64, String> {
    if sample_weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight < 0.0)
    {
        return Err("sample_weight must contain finite non-negative values".into());
    }
    if sample_weights.iter().all(|weight| *weight == 0.0) {
        return Err("sample_weight cannot be all zero".into());
    }
    let total_weight = sample_weights.iter().sum::<f64>();
    if !total_weight.is_finite() {
        return Err("sample_weight sum must be finite".into());
    }
    if total_weight <= 0.0 {
        return Err("sample_weight sum must be positive".into());
    }
    Ok(total_weight)
}

fn validate_csr(
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<f64>,
    rows: usize,
    columns: usize,
) -> Result<CsrValues, String> {
    if rows == 0 {
        return Err("X must contain at least one row".into());
    }
    if columns == 0 {
        return Err("X must contain at least one feature".into());
    }
    if rows.checked_add(1) != Some(indptr.len())
        || indptr.first() != Some(&0)
        || indptr.last() != Some(&data.len())
        || indices.len() != data.len()
        || indptr.windows(2).any(|w| w[0] > w[1])
    {
        return Err("invalid CSR matrix structure".into());
    }
    if indices.iter().any(|&i| i >= columns)
        || indptr
            .windows(2)
            .any(|w| indices[w[0]..w[1]].windows(2).any(|v| v[0] >= v[1]))
    {
        return Err("CSR column indices must be sorted and unique".into());
    }
    if data.iter().any(|v| v.is_infinite()) {
        return Err("X must not contain infinite values".into());
    }
    Ok(CsrValues {
        indptr,
        indices,
        data,
    })
}

fn sparse_histogram_edges(
    values: &CsrValues,
    rows: usize,
    columns: usize,
    max_bins: usize,
    threads: usize,
    strategy: BinningStrategy,
    sample_size: usize,
) -> Vec<Vec<f64>> {
    let preprocessing_threads = if rows.saturating_mul(columns) >= HISTOGRAM_PARALLEL_MIN_CELLS {
        threads
    } else {
        1
    };
    let sampled_rows = (strategy.samples() && sample_size < rows)
        .then(|| deterministic_sample_rows(rows, sample_size));
    let workers = preprocessing_threads.max(1).min(columns);
    let build = |column| {
        let mut observations = Vec::with_capacity(sampled_rows.as_ref().map_or(rows, Vec::len));
        if let Some(sampled_rows) = &sampled_rows {
            observations.extend(sampled_rows.iter().map(|&row| values.get(row, column)));
        } else {
            observations.extend((0..rows).map(|row| values.get(row, column)));
        }
        edges_from_observations(observations, max_bins, strategy)
    };
    if workers == 1 {
        return (0..columns).map(build).collect();
    }
    let chunk_size = columns.div_ceil(workers);
    std::thread::scope(|scope| {
        let handles = (0..columns)
            .step_by(chunk_size)
            .map(|start| {
                let end = (start + chunk_size).min(columns);
                let build = &build;
                scope.spawn(move || (start..end).map(build).collect::<Vec<_>>())
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("sparse histogram worker panicked"))
            .collect()
    })
}

fn deterministic_sample_rows(rows: usize, sample_size: usize) -> Vec<usize> {
    let count = rows.min(sample_size);
    let mut indices = (0..rows).collect::<Vec<_>>();
    let mut state = 0x6a09_e667_f3bc_c909;
    for index in 0..count {
        state = splitmix64(state);
        let remaining = rows - index;
        let selected = index + state as usize % remaining;
        indices.swap(index, selected);
    }
    indices.truncate(count);
    indices.sort_unstable();
    indices
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn edges_from_observations(
    mut observations: Vec<f64>,
    max_bins: usize,
    strategy: BinningStrategy,
) -> Vec<f64> {
    observations.retain(|value| !value.is_nan());
    if observations.is_empty() {
        return Vec::new();
    }

    if !strategy.selects() || observations.len() <= max_bins {
        observations.sort_unstable_by(f64::total_cmp);
        observations.dedup_by(|left, right| left.total_cmp(right).is_eq());
        return edges_from_sorted_unique(&observations, max_bins);
    }

    let mut positions = Vec::with_capacity((max_bins - 1) * 2);
    for bin in 1..max_bins {
        let rank = bin * observations.len() / max_bins;
        if rank > 0 && rank < observations.len() {
            positions.push(rank - 1);
            positions.push(rank);
        }
    }
    positions.sort_unstable();
    positions.dedup();
    multi_select_positions(&mut observations, &positions, 0);

    let mut edges = Vec::with_capacity(max_bins.saturating_sub(1));
    for bin in 1..max_bins {
        let rank = bin * observations.len() / max_bins;
        if rank == 0 || rank >= observations.len() {
            continue;
        }
        let left = observations[rank - 1];
        let right = observations[rank];
        if left.total_cmp(&right).is_eq() {
            continue;
        }
        let edge = midpoint(left, right);
        if edges.last().is_none_or(|previous| *previous < edge) {
            edges.push(edge);
        }
    }
    edges
}

fn edges_from_sorted_unique(sorted: &[f64], max_bins: usize) -> Vec<f64> {
    let mut edges = Vec::with_capacity(max_bins.saturating_sub(1));
    if sorted.len() <= max_bins {
        for adjacent in sorted.windows(2) {
            edges.push(midpoint(adjacent[0], adjacent[1]));
        }
    } else {
        for bin in 1..max_bins {
            let index = bin * sorted.len() / max_bins;
            if index > 0 && index < sorted.len() {
                let edge = midpoint(sorted[index - 1], sorted[index]);
                if edges.last().is_none_or(|previous| *previous < edge) {
                    edges.push(edge);
                }
            }
        }
    }
    edges
}

fn multi_select_positions(values: &mut [f64], targets: &[usize], offset: usize) {
    if targets.is_empty() {
        return;
    }
    let middle = targets.len() / 2;
    let pivot_index = targets[middle] - offset;
    let (lower, _, upper) = values.select_nth_unstable_by(pivot_index, f64::total_cmp);
    multi_select_positions(lower, &targets[..middle], offset);
    multi_select_positions(
        upper,
        &targets[middle + 1..],
        targets[middle] + 1,
    );
}

fn validate_matrix(values: &[f64], rows: usize, columns: usize) -> Result<(), String> {
    if rows == 0 {
        return Err("X must contain at least one row".to_string());
    }
    if columns == 0 {
        return Err("X must contain at least one feature".to_string());
    }
    let expected = rows
        .checked_mul(columns)
        .ok_or_else(|| "X shape is too large".to_string())?;
    if values.len() != expected {
        return Err("X buffer length does not match its shape".to_string());
    }
    if values.iter().any(|value| value.is_infinite()) {
        return Err("X must not contain infinite values".to_string());
    }

    Ok(())
}

fn histogramize(
    values: &[f64],
    rows: usize,
    columns: usize,
    max_bins: usize,
    threads: usize,
    strategy: BinningStrategy,
    sample_size: usize,
) -> (Vec<u8>, Vec<Vec<f64>>) {
    let value_at = |row, feature| values[row * columns + feature];
    let (binned, edges, _) = histogramize_from_accessor(
        rows,
        columns,
        max_bins,
        threads,
        strategy,
        sample_size,
        &value_at,
    )
    .expect("training values were validated before histogramization");
    (binned, edges)
}

fn histogramize_from_accessor<F>(
    rows: usize,
    columns: usize,
    max_bins: usize,
    threads: usize,
    strategy: BinningStrategy,
    sample_size: usize,
    value_at: &F,
) -> Result<(Vec<u8>, Vec<Vec<f64>>, bool), String>
where
    F: Fn(usize, usize) -> f64 + Sync,
{
    let preprocessing_threads = if rows.saturating_mul(columns) >= HISTOGRAM_PARALLEL_MIN_CELLS {
        threads
    } else {
        1
    };
    let sampled_rows = (strategy.samples() && sample_size < rows)
        .then(|| deterministic_sample_rows(rows, sample_size));
    let sampled_rows = sampled_rows.as_deref();
    let workers = preprocessing_threads.max(1).min(columns);
    let edges_by_feature: Vec<Vec<f64>> = if workers == 1 {
        (0..columns)
            .map(|feature| {
                histogram_edges_for_feature(
                    rows,
                    feature,
                    max_bins,
                    strategy,
                    sampled_rows,
                    value_at,
                )
            })
            .collect()
    } else {
        let chunk_size = columns.div_ceil(workers);
        std::thread::scope(|scope| {
            let handles = (0..columns)
                .step_by(chunk_size)
                .map(|start| {
                    let end = (start + chunk_size).min(columns);
                    scope.spawn(move || {
                        (start..end)
                            .map(|feature| {
                                histogram_edges_for_feature(
                                    rows,
                                    feature,
                                    max_bins,
                                    strategy,
                                    sampled_rows,
                                    value_at,
                                )
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .flat_map(|handle| handle.join().expect("histogram edge worker panicked"))
                .collect()
        })
    };
    let (binned, has_missing_values, has_infinite_values) = apply_histogram_accessor_parallel(
        rows,
        columns,
        &edges_by_feature,
        preprocessing_threads,
        value_at,
    );
    if has_infinite_values {
        return Err("X must not contain infinite values".into());
    }
    Ok((binned, edges_by_feature, has_missing_values))
}

fn histogram_edges_for_feature(
    rows: usize,
    feature: usize,
    max_bins: usize,
    strategy: BinningStrategy,
    sampled_rows: Option<&[usize]>,
    value_at: &(impl Fn(usize, usize) -> f64 + Sync),
) -> Vec<f64> {
    let mut observations = Vec::with_capacity(sampled_rows.map_or(rows, <[usize]>::len));
    if let Some(sampled_rows) = sampled_rows {
        observations.extend(sampled_rows.iter().map(|&row| value_at(row, feature)));
    } else {
        observations.extend((0..rows).map(|row| value_at(row, feature)));
    }
    edges_from_observations(observations, max_bins, strategy)
}

fn midpoint(left: f64, right: f64) -> f64 {
    let midpoint = left * 0.5 + right * 0.5;
    if midpoint >= right {
        left
    } else {
        midpoint
    }
}

fn apply_histogram_edges(
    values: &[f64],
    rows: usize,
    columns: usize,
    edges_by_feature: &[Vec<f64>],
) -> Vec<u8> {
    let mut binned = Vec::with_capacity(values.len());
    for row in 0..rows {
        for feature in 0..columns {
            let value = values[row * columns + feature];
            let bin = if value.is_nan() { edges_by_feature[feature].len() + 1 } else { edges_by_feature[feature].partition_point(|edge| value > *edge) };
            binned.push(bin as u8);
        }
    }
    binned
}

fn apply_histogram_accessor_parallel<F>(
    rows: usize,
    columns: usize,
    edges_by_feature: &[Vec<f64>],
    threads: usize,
    value_at: &F,
) -> (Vec<u8>, bool, bool)
where
    F: Fn(usize, usize) -> f64 + Sync,
{
    let workers = threads.max(1).min(rows);
    if workers == 1 {
        let mut binned = Vec::with_capacity(rows.saturating_mul(columns));
        let mut has_missing_values = false;
        let mut has_infinite_values = false;
        for row in 0..rows {
            for feature in 0..columns {
                let value = value_at(row, feature);
                let is_missing = value.is_nan();
                has_missing_values |= is_missing;
                has_infinite_values |= value.is_infinite();
                let bin = if is_missing {
                    edges_by_feature[feature].len() + 1
                } else {
                    edges_by_feature[feature].partition_point(|edge| value > *edge)
                };
                binned.push(bin as u8);
            }
        }
        return (binned, has_missing_values, has_infinite_values);
    }
    let rows_per_worker = rows.div_ceil(workers);
    let output_len = rows.saturating_mul(columns);
    let mut binned = vec![0u8; output_len];
    let chunk_len = rows_per_worker * columns;
    let (has_missing_values, has_infinite_values) = thread::scope(|scope| {
        let handles = binned
            .chunks_mut(chunk_len)
            .enumerate()
            .map(|(chunk_index, output)| {
                let start_row = chunk_index * rows_per_worker;
                scope.spawn(move || {
                    let mut has_missing_values = false;
                    let mut has_infinite_values = false;
                    for (index, bin_out) in output.iter_mut().enumerate() {
                        let row = start_row + index / columns;
                        let feature = index % columns;
                        let value = value_at(row, feature);
                        let is_missing = value.is_nan();
                        has_missing_values |= is_missing;
                        has_infinite_values |= value.is_infinite();
                        *bin_out = if is_missing {
                            (edges_by_feature[feature].len() + 1) as u8
                        } else {
                            edges_by_feature[feature].partition_point(|edge| value > *edge) as u8
                        };
                    }
                    (has_missing_values, has_infinite_values)
                })
            })
            .collect::<Vec<_>>();
        let mut has_missing_values = false;
        let mut has_infinite_values = false;
        for handle in handles {
            let (chunk_has_missing, chunk_has_infinite) =
                handle.join().expect("histogram bin worker panicked");
            has_missing_values |= chunk_has_missing;
            has_infinite_values |= chunk_has_infinite;
        }
        (has_missing_values, has_infinite_values)
    });
    (binned, has_missing_values, has_infinite_values)
}

fn build_histograms(input: &DenseInput, mask: &Mask) -> HistogramCache {
    let edges = input.bin_edges.as_ref().expect("histogram cache requires fitted bin edges");
    let mut bin_offsets = Vec::with_capacity(edges.len() + 1);
    bin_offsets.push(0);
    for feature_edges in edges.iter() {
        bin_offsets.push(bin_offsets.last().unwrap() + feature_edges.len() + 1
            + usize::from(input.has_missing_values));
    }
    let bins = *bin_offsets.last().unwrap();
    let mut histogram = HistogramCache {
        bin_offsets,
        class_weights: vec![0.0; bins * input.n_classes],
        sample_counts: vec![0; bins],
    };
    match &input.values {
        DenseValues::Binned(values) => match input.n_classes {
            2 => accumulate_dense_histograms::<2>(input, mask, values, &mut histogram),
            4 => accumulate_dense_histograms::<4>(input, mask, values, &mut histogram),
            _ => accumulate_dense_histograms::<0>(input, mask, values, &mut histogram),
        },
        DenseValues::Sparse(_) => {
            for &row in mask.iter() {
                let label = input.labels()[row];
                let weight = input.sample_weight(row);
                for feature in 0..input.columns {
                    let bin = histogram.bin_offsets[feature] + input.bin_value(row, feature);
                    histogram.class_weights[bin * input.n_classes + label] += weight;
                    histogram.sample_counts[bin] += 1;
                }
            }
        }
        DenseValues::Exact(_) => unreachable!("histograms require binned input"),
    }
    histogram
}

fn accumulate_dense_histograms<const CLASSES: usize>(
    input: &DenseInput, mask: &Mask, values: &[u8], histogram: &mut HistogramCache,
) {
    let classes = if CLASSES == 0 { input.n_classes } else { CLASSES };
    let workers = input.histogram_threads.min(input.columns);
    if workers > 1
        && mask.len().saturating_mul(input.columns) >= HISTOGRAM_PARALLEL_MIN_CELLS
    {
        accumulate_dense_histograms_parallel::<CLASSES>(
            input, mask, values, histogram, classes, workers,
        );
        return;
    }
    let labels = input.labels();
    for &row in mask.iter() {
        let label = labels[row];
        let weight = input.sample_weight(row);
        let offset = row * input.columns;
        let row_bins = &values[offset..offset + input.columns];
        for (&bin, &start) in row_bins.iter().zip(&histogram.bin_offsets) {
            let index = start + bin as usize;
            histogram.class_weights[index * classes + label] += weight;
            histogram.sample_counts[index] += 1;
        }
    }
}

fn accumulate_dense_histograms_parallel<const CLASSES: usize>(
    input: &DenseInput,
    mask: &Mask,
    values: &[u8],
    histogram: &mut HistogramCache,
    classes: usize,
    workers: usize,
) {
    let features_per_worker = input.columns.div_ceil(workers);
    let offsets = &histogram.bin_offsets;
    let labels = input.labels();
    let n_classes = input.n_classes;
    let columns = input.columns;
    let mut class_weights = histogram.class_weights.as_mut_slice();
    let mut sample_counts = histogram.sample_counts.as_mut_slice();

    std::thread::scope(|scope| {
        let mut start_feature = 0;
        while start_feature < columns {
            let end_feature = (start_feature + features_per_worker).min(columns);
            let start_bin = offsets[start_feature];
            let end_bin = offsets[end_feature];
            let weight_len = (end_bin - start_bin) * classes;
            let (worker_weights, remaining_weights) = class_weights.split_at_mut(weight_len);
            class_weights = remaining_weights;
            let (worker_counts, remaining_counts) = sample_counts.split_at_mut(end_bin - start_bin);
            sample_counts = remaining_counts;
            let feature_offsets = &offsets[start_feature..end_feature];
            let input_weights = input.sample_weights.as_deref();

            scope.spawn(move || {
                for &row in mask.iter() {
                    let label = labels[row];
                    let weight = input_weights.map_or(1.0, |weights| weights[row]);
                    let row_start = row * columns + start_feature;
                    let row_bins = &values[row_start..row_start + (end_feature - start_feature)];
                    for (&bin, &feature_start) in row_bins.iter().zip(feature_offsets) {
                        let local_bin = feature_start - start_bin + bin as usize;
                        worker_weights[local_bin * n_classes + label] += weight;
                        worker_counts[local_bin] += 1;
                    }
                }
            });
            start_feature = end_feature;
        }
    });
}

fn subtract_histograms(mut parent: HistogramCache, smaller_child: &HistogramCache) -> HistogramCache {
    debug_assert_eq!(parent.bin_offsets, smaller_child.bin_offsets);
    // The parent buffer becomes the larger child's buffer. These independent,
    // contiguous elementwise loops are suitable for LLVM auto-vectorization.
    for (parent, smaller) in parent.class_weights.iter_mut().zip(&smaller_child.class_weights) {
        *parent = (*parent - smaller).max(0.0);
    }
    for (parent, smaller) in parent.sample_counts.iter_mut().zip(&smaller_child.sample_counts) {
        *parent -= smaller;
    }
    parent
}

#[derive(Clone)]
pub struct ClassVotes(Vec<usize>);

impl ClassVotes {
    fn zeroed(n_classes: usize) -> Self {
        Self(vec![0; n_classes])
    }

    pub fn winner(&self) -> usize {
        let mut winner = 0;
        for (class, &count) in self.0.iter().enumerate() {
            if count > self.0[winner] {
                winner = class;
            }
        }
        winner
    }

    pub fn probabilities(&self) -> Vec<f64> {
        let total = self.0.iter().sum::<usize>();
        if total == 0 {
            return vec![0.0; self.0.len()];
        }
        self.0
            .iter()
            .map(|&count| count as f64 / total as f64)
            .collect()
    }

    fn add(&mut self, class: usize) {
        self.0[class] += 1;
    }
}

impl VoteAggregator<DenseInput> for ClassVotes {
    fn new(input: &DenseInput) -> Self {
        Self::zeroed(input.n_classes)
    }

    fn ingest_vote(&mut self, vote: usize) {
        self.add(vote);
    }

    fn merge(&mut self, other: &Self) {
        for (into, from) in self.0.iter_mut().zip(&other.0) {
            *into += *from;
        }
    }
}

pub struct DenseDecisionSlice {
    labels: Vec<usize>,
    class_weights: Vec<f64>,
    total_weight: f64,
    criterion: Criterion,
}

impl DenseDecisionSlice {
    fn new(input: &DenseInput, mask: &Mask) -> Self {
        let mut class_weights = vec![0.0; input.n_classes];
        let mut total_weight = 0.0;
        let labels = mask
            .iter()
            .map(|&row| {
                let label = input.labels()[row];
                let weight = input.sample_weight(row);
                class_weights[label] += weight;
                total_weight += weight;
                label
            })
            .collect();

        Self {
            labels,
            class_weights,
            total_weight,
            criterion: input.criterion,
        }
    }
}

impl DecisionSlice<usize> for DenseDecisionSlice {
    fn is_pure(&self) -> bool {
        self.class_weights
            .iter()
            .filter(|&&weight| weight > 0.0)
            .count()
            <= 1
    }

    fn condense(&self, rng: &mut RfRng) -> usize {
        let mut best = FairBest::new();
        for (class, &weight) in self.class_weights.iter().enumerate() {
            best.ingest(weight, class, rng);
        }
        best.consume().map(|(_, class)| class).unwrap_or(0)
    }

    fn condense_with_bounds(&self, rng: &mut RfRng, lower_bound: f64, upper_bound: f64) -> usize {
        if self.class_weights.len() != 2 || self.total_weight <= 0.0 {
            return self.condense(rng);
        }
        let probability =
            (self.class_weights[1] / self.total_weight).clamp(lower_bound, upper_bound);
        usize::from(probability > 0.5)
    }

    fn positive_probability(&self) -> Option<f64> {
        (self.class_weights.len() == 2 && self.total_weight > 0.0)
            .then(|| self.class_weights[1] / self.total_weight)
    }

    fn pruning_weight(&self) -> Option<f64> {
        Some(self.total_weight)
    }

    fn pruning_risk(&self, root_weight: f64) -> Option<f64> {
        if root_weight <= 0.0 {
            return Some(0.0);
        }
        Some(
            self.total_weight * impurity(self.criterion, &self.class_weights, self.total_weight)
                / root_weight,
        )
    }

    fn pruning_vote(&self) -> Option<usize> {
        let mut best_class = 0;
        for (class, &weight) in self.class_weights.iter().enumerate().skip(1) {
            if weight > self.class_weights[best_class] {
                best_class = class;
            }
        }
        Some(best_class)
    }
}

fn impurity(criterion: Criterion, class_weights: &[f64], total_weight: f64) -> f64 {
    if total_weight == 0.0 {
        return 0.0;
    }
    match criterion {
        Criterion::Gini => {
            1.0 - class_weights
                .iter()
                .map(|&weight| {
                    let probability = weight / total_weight;
                    probability * probability
                })
                .sum::<f64>()
        }
        Criterion::Entropy => class_weights
            .iter()
            .filter(|&&weight| weight > 0.0)
            .map(|&weight| {
                let probability = weight / total_weight;
                -probability * probability.ln()
            })
            .sum(),
    }
}

fn best_split(
    input: &DenseInput,
    mask: &Mask,
    feature: usize,
    target: &DenseDecisionSlice,
    split_cache: &DenseSplitCache,
    monotonic_constraint: i8,
    lower_bound: f64,
    upper_bound: f64,
) -> Option<(f64, bool, f64)> {
    let best = if input.histogram_bins.is_some() {
        best_split_histogram(
            input,
            target,
            &split_cache.0.as_ref()?.feature(feature, input.n_classes),
            monotonic_constraint,
            lower_bound,
            upper_bound,
        )
    } else {
        best_split_exact(
            input,
            mask,
            feature,
            target,
            monotonic_constraint,
            lower_bound,
            upper_bound,
        )
    }?;
    Some((best.0, best.1, best.2 * target.total_weight / input.total_weight))
}

fn best_split_exact(
    input: &DenseInput,
    mask: &Mask,
    feature: usize,
    target: &DenseDecisionSlice,
    monotonic_constraint: i8,
    lower_bound: f64,
    upper_bound: f64,
) -> Option<(f64, bool, f64)> {
    let mut ranked: Vec<(f64, usize, f64)> = match &input.values {
        DenseValues::Exact(values) => mask
            .iter()
            .zip(&target.labels)
            .map(|(&row, &label)| {
                (
                    values[row * input.columns + feature],
                    label,
                    input.sample_weight(row),
                )
            })
            .collect(),
        DenseValues::Binned(values) => mask
            .iter()
            .zip(&target.labels)
            .map(|(&row, &label)| {
                (
                    values[row * input.columns + feature] as f64,
                    label,
                    input.sample_weight(row),
                )
            })
            .collect(),
        DenseValues::Sparse(values) => mask
            .iter()
            .zip(&target.labels)
            .map(|(&row, &label)| (values.get(row, feature), label, input.sample_weight(row)))
            .collect(),
    };
    let mut missing = input.has_missing_values.then(|| vec![0.0; input.n_classes]);
    let mut missing_count = 0usize;
    let mut missing_weight = 0.0;
    if input.has_missing_values {
        ranked.retain(|(value, label, weight)| {
            if value.is_nan() {
                missing.as_mut().unwrap()[*label] += *weight;
                missing_count += 1;
                missing_weight += *weight;
                false
            } else {
                true
            }
        });
    }
    ranked.sort_unstable_by(|left, right| left.0.total_cmp(&right.0));

    if ranked.len() < 2 {
        return None;
    }

    let total_weight = target.total_weight;
    let parent_impurity = impurity(input.criterion, &target.class_weights, total_weight);
    let mut left = vec![0.0; input.n_classes];
    let mut right = target.class_weights.clone();
    if let Some(missing) = &missing {
        for class in 0..input.n_classes { right[class] -= missing[class]; }
    }
    let mut left_weight = 0.0;
    let mut best: Option<(f64, bool, f64)> = None;

    for index in 0..(ranked.len() - 1) {
        let (value, label, weight) = ranked[index];
        left[label] += weight;
        right[label] -= weight;
        left_weight += weight;

        let next = ranked[index + 1].0;
        if value.total_cmp(&next).is_eq() {
            continue;
        }

        let midpoint = value * 0.5 + next * 0.5;
        let pivot = if midpoint >= next { value } else { midpoint };
        let finite_left_count = index + 1;
        let finite_right_count = ranked.len() - finite_left_count;
        if missing_count == 0 {
            let right_weight = total_weight - left_weight;
            if finite_left_count < input.min_samples_leaf || finite_right_count < input.min_samples_leaf
                || left_weight < input.min_leaf_weight || right_weight < input.min_leaf_weight
                || left_weight == 0.0 || right_weight == 0.0 { continue; }
            if monotonic_constraint != 0 {
                let lp = left[1] / left_weight; let rp = right[1] / right_weight;
                if lp < lower_bound || lp > upper_bound || rp < lower_bound || rp > upper_bound
                    || (lp - rp) * monotonic_constraint as f64 > 0.0 { continue; }
            }
            let child_impurity = (left_weight / total_weight) * impurity(input.criterion, &left, left_weight)
                + (right_weight / total_weight) * impurity(input.criterion, &right, right_weight);
            let gain = parent_impurity - child_impurity;
            if gain >= input.min_impurity_decrease && best.as_ref().is_none_or(|(_, _, g)| gain > *g) {
                best = Some((pivot, finite_left_count <= finite_right_count, gain));
            }
            continue;
        }
        for missing_left in [false, true] {
            if missing_count == 0 && missing_left != (finite_left_count > finite_right_count) { continue; }
            let (left_classes, right_classes, left_count, right_count, left_weight) = if missing_left {
                let mut l = left.clone();
                for class in 0..input.n_classes { l[class] += missing.as_ref().unwrap()[class]; }
                (l, right.clone(), finite_left_count + missing_count, finite_right_count, left_weight + missing_weight)
            } else {
                let mut r = right.clone();
                for class in 0..input.n_classes { r[class] += missing.as_ref().unwrap()[class]; }
                (left.clone(), r, finite_left_count, finite_right_count + missing_count, left_weight)
            };
            let right_weight = total_weight - left_weight;
            if left_count < input.min_samples_leaf || right_count < input.min_samples_leaf
                || left_weight < input.min_leaf_weight || right_weight < input.min_leaf_weight
                || left_weight == 0.0 || right_weight == 0.0 { continue; }
            if monotonic_constraint != 0 {
                let lp = left_classes[1] / left_weight;
                let rp = right_classes[1] / right_weight;
                if lp < lower_bound || lp > upper_bound || rp < lower_bound || rp > upper_bound
                    || (lp - rp) * monotonic_constraint as f64 > 0.0 { continue; }
            }
            let child_impurity = (left_weight / total_weight) * impurity(input.criterion, &left_classes, left_weight)
                + (right_weight / total_weight) * impurity(input.criterion, &right_classes, right_weight);
            let gain = parent_impurity - child_impurity;
            if gain >= input.min_impurity_decrease && best.as_ref().is_none_or(|(_, _, g)| gain > *g) {
                // Split iterators route values above the pivot to XRF's left
                // child, while the running histogram accumulates low bins first.
                best = Some((pivot, !missing_left, gain));
            }
        }
    }

    best
}

fn best_split_histogram(
    input: &DenseInput,
    target: &DenseDecisionSlice,
    histogram: &FeatureHistogram<'_>,
    monotonic_constraint: i8,
    lower_bound: f64,
    upper_bound: f64,
) -> Option<(f64, bool, f64)> {
    if input.n_classes == 2 {
        best_split_histogram_classes::<2>(
            input,
            target,
            histogram,
            monotonic_constraint,
            lower_bound,
            upper_bound,
        )
    } else {
        best_split_histogram_classes::<0>(
            input,
            target,
            histogram,
            monotonic_constraint,
            lower_bound,
            upper_bound,
        )
    }
}
fn best_split_histogram_classes<const CLASSES: usize>(
    input: &DenseInput,
    target: &DenseDecisionSlice,
    histogram: &FeatureHistogram<'_>,
    monotonic_constraint: i8,
    lower_bound: f64,
    upper_bound: f64,
) -> Option<(f64, bool, f64)> {
    let classes = if CLASSES == 0 {
        input.n_classes
    } else {
        CLASSES
    };
    let bin_count = histogram.sample_counts.len();
    if bin_count < 2 {
        return None;
    }

    let total_weight = target.total_weight;
    let parent_impurity = impurity(input.criterion, &target.class_weights, total_weight);
    // Binary splits need no heap allocation for their class accumulators.
    // Keep the generic path for arbitrary class counts and identical arithmetic.
    let mut left_stack = [0.0; CLASSES];
    let mut right_stack = [0.0; CLASSES];
    let mut left_heap = if CLASSES == 0 {
        vec![0.0; classes]
    } else {
        Vec::new()
    };
    let mut right_heap = if CLASSES == 0 {
        vec![0.0; classes]
    } else {
        Vec::new()
    };
    let left: &mut [f64] = if CLASSES == 0 {
        &mut left_heap
    } else {
        &mut left_stack
    };
    let right: &mut [f64] = if CLASSES == 0 {
        &mut right_heap
    } else {
        &mut right_stack
    };
    right.copy_from_slice(&target.class_weights);
    let mut left_weight = 0.0;
    let mut left_count = 0;
    let missing_bin = if input.has_missing_values {
        bin_count - 1
    } else {
        bin_count
    };
    let missing_classes = input.has_missing_values.then(|| {
        (0..classes)
            .map(|class| histogram.class_weights[missing_bin * classes + class])
            .collect::<Vec<_>>()
    });
    let missing_count = if input.has_missing_values {
        histogram.sample_counts[missing_bin]
    } else {
        0
    };
    if let Some(missing_classes) = &missing_classes {
        for class in 0..classes {
            right[class] -= missing_classes[class];
        }
    }
    let total_count = histogram.sample_counts[..missing_bin].iter().sum::<usize>();
    let mut best: Option<(f64, bool, f64)> = None;

    let mut candidate_left = if missing_count > 0 {
        vec![0.0; classes]
    } else {
        Vec::new()
    };
    let mut candidate_right = if missing_count > 0 {
        vec![0.0; classes]
    } else {
        Vec::new()
    };
    let split_bin_end = if input.has_missing_values {
        missing_bin.saturating_sub(1)
    } else {
        bin_count.saturating_sub(1)
    };
    for bin in 0..split_bin_end {
        for class in 0..classes {
            let weight = histogram.class_weights[bin * classes + class];
            left[class] += weight;
            right[class] -= weight;
            left_weight += weight;
        }
        left_count += histogram.sample_counts[bin];
        let pivot = bin as f64 + 0.5;
        if missing_count == 0 {
            let right_count = total_count - left_count;
            let right_weight = total_weight - left_weight;
            if left_count < input.min_samples_leaf
                || right_count < input.min_samples_leaf
                || left_weight == 0.0
                || right_weight == 0.0
                || left_weight < input.min_leaf_weight
                || right_weight < input.min_leaf_weight
            {
                continue;
            }
            if monotonic_constraint != 0 {
                let lp = left[1] / left_weight;
                let rp = right[1] / right_weight;
                if lp < lower_bound
                    || lp > upper_bound
                    || rp < lower_bound
                    || rp > upper_bound
                    || (lp - rp) * monotonic_constraint as f64 > 0.0
                {
                    continue;
                }
            }
            let child_impurity = (left_weight / total_weight)
                * impurity(input.criterion, &left, left_weight)
                + (right_weight / total_weight) * impurity(input.criterion, &right, right_weight);
            let gain = parent_impurity - child_impurity;
            if gain >= input.min_impurity_decrease
                && best.as_ref().is_none_or(|(_, _, g)| gain > *g)
            {
                best = Some((pivot, left_count <= right_count, gain));
            }
            continue;
        }
        for missing_left in [false, true] {
            if missing_count == 0 && missing_left != (left_count > total_count - left_count) {
                continue;
            }
            candidate_left.copy_from_slice(&left);
            candidate_right.copy_from_slice(&right);
            let (lcount, rcount, lw) = if missing_left {
                let missing_classes = missing_classes.as_ref().unwrap();
                for class in 0..classes {
                    candidate_left[class] += missing_classes[class];
                }
                (
                    left_count + missing_count,
                    total_count - left_count,
                    left_weight + missing_classes.iter().sum::<f64>(),
                )
            } else {
                for class in 0..classes {
                    candidate_right[class] += missing_classes.as_ref().unwrap()[class];
                }
                (
                    left_count,
                    total_count - left_count + missing_count,
                    left_weight,
                )
            };
            let lc = &candidate_left;
            let rc = &candidate_right;
            let rw = total_weight - lw;
            if lcount < input.min_samples_leaf
                || rcount < input.min_samples_leaf
                || lw == 0.0
                || rw == 0.0
                || lw < input.min_leaf_weight
                || rw < input.min_leaf_weight
            {
                continue;
            }
            if monotonic_constraint != 0 {
                let lp = lc[1] / lw;
                let rp = rc[1] / rw;
                if lp < lower_bound
                    || lp > upper_bound
                    || rp < lower_bound
                    || rp > upper_bound
                    || (lp - rp) * monotonic_constraint as f64 > 0.0
                {
                    continue;
                }
            }
            let child_impurity = (lw / total_weight) * impurity(input.criterion, &lc, lw)
                + (rw / total_weight) * impurity(input.criterion, &rc, rw);
            let gain = parent_impurity - child_impurity;
            if gain >= input.min_impurity_decrease
                && best.as_ref().is_none_or(|(_, _, g)| gain > *g)
            {
                best = Some((pivot, !missing_left, gain));
            }
        }
    }
    best
}
pub struct PermutationImportance {
    direct: Vec<Option<usize>>,
    drops: HashMap<usize, isize>,
    n: usize,
    true_labels: Vec<usize>,
}

impl AccuracyDecreaseAggregator<DenseInput> for PermutationImportance {
    fn new(input: &DenseInput, on: &Mask, n: usize) -> Self {
        Self {
            direct: vec![None; n],
            drops: HashMap::new(),
            n: on.len(),
            true_labels: input.labels().to_vec(),
        }
    }

    fn ingest(&mut self, permuted: Option<usize>, mask: &Mask, vote: &usize) {
        if let Some(feature) = permuted {
            let difference = mask
                .iter()
                .map(|&row| {
                    let direct = self.direct[row]
                        .expect("direct OOB votes must be collected before permutation");
                    if direct == *vote {
                        return 0;
                    }
                    match (
                        self.true_labels[row] == *vote,
                        self.true_labels[row] == direct,
                    ) {
                        (true, false) => -1,
                        (false, true) => 1,
                        (false, false) => 0,
                        (true, true) => unreachable!("identical predictions cannot differ"),
                    }
                })
                .sum::<isize>();
            *self.drops.entry(feature).or_insert(0) += difference;
        } else {
            for &row in mask.iter() {
                self.direct[row] = Some(*vote);
            }
        }
    }

    fn mda_iter(&self) -> impl Iterator<Item = (usize, f64)> {
        self.drops
            .iter()
            .map(|(&feature, &drop)| (feature, drop as f64 / self.n as f64))
    }

    fn get_direct_vote(&self, row: usize) -> usize {
        self.direct[row].expect("direct OOB vote must be available")
    }
}

pub struct UniformFeatureSampler {
    active_features: Vec<usize>,
}

impl FeatureSampler<DenseInput> for UniformFeatureSampler {
    fn random_feature(&mut self, rng: &mut RfRng) -> usize {
        self.active_features[rng.up_to(self.active_features.len())]
    }

    fn reload(&mut self) {}

    fn reset(&mut self) {}
}

impl RfInput for DenseInput {
    type FeatureId = usize;
    type Pivot = f64;
    type Vote = usize;
    type VoteAggregator = ClassVotes;
    type DecisionSlice = DenseDecisionSlice;
    type AccuracyDecreaseAggregator = PermutationImportance;
    type FeatureSampler = UniformFeatureSampler;
    type SplitCache = DenseSplitCache;

    fn observation_count(&self) -> usize {
        self.rows
    }

    fn ccp_alpha(&self) -> f64 {
        self.ccp_alpha
    }

    fn tree_input(&self, bag: &Mask) -> Option<Self> {
        self.balanced_subsample
            .then(|| self.balanced_tree_view(bag))
    }

    fn feature_count(&self) -> usize {
        self.columns
    }

    fn decision_slice(&self, mask: &Mask) -> Self::DecisionSlice {
        DenseDecisionSlice::new(self, mask)
    }

    fn can_split(&self, mask: &Mask) -> bool {
        mask.len() >= self.min_samples_split && !self.active_features.is_empty()
    }

    fn feature_sampler(&self) -> Self::FeatureSampler {
        UniformFeatureSampler {
            active_features: self.active_features.clone(),
        }
    }

    fn split_cache(&self, on: &Mask) -> Self::SplitCache {
        DenseSplitCache(
            self.histogram_bins
                .is_some()
                .then(|| build_histograms(self, on)),
        )
    }

    fn split_cache_children(
        &self,
        parent_cache: Self::SplitCache,
        _: &Mask,
        left: &Mask,
        right: &Mask,
    ) -> (Self::SplitCache, Self::SplitCache) {
        let Some(parent_histograms) = parent_cache.0 else {
            return (DenseSplitCache(None), DenseSplitCache(None));
        };
        if left.len() <= right.len() {
            let left_histograms = build_histograms(self, left);
            let right_histograms = subtract_histograms(parent_histograms, &left_histograms);
            (
                DenseSplitCache(Some(left_histograms)),
                DenseSplitCache(Some(right_histograms)),
            )
        } else {
            let right_histograms = build_histograms(self, right);
            let left_histograms = subtract_histograms(parent_histograms, &right_histograms);
            (
                DenseSplitCache(Some(left_histograms)),
                DenseSplitCache(Some(right_histograms)),
            )
        }
    }

    fn new_split(
        &self,
        mask: &Mask,
        feature: Self::FeatureId,
        target: &Self::DecisionSlice,
        split_cache: &Self::SplitCache,
        _: &mut RfRng,
    ) -> Option<(Self::Pivot, bool, f64)> {
        best_split(self, mask, feature, target, split_cache, 0, 0.0, 1.0)
    }

    fn new_split_with_bounds(
        &self,
        mask: &Mask,
        feature: Self::FeatureId,
        target: &Self::DecisionSlice,
        split_cache: &Self::SplitCache,
        _: &mut RfRng,
        lower_bound: f64,
        upper_bound: f64,
    ) -> Option<(Self::Pivot, bool, f64)> {
        best_split(
            self,
            mask,
            feature,
            target,
            split_cache,
            self.monotonic_constraints
                .as_ref()
                .map_or(0, |c| c[feature]),
            lower_bound,
            upper_bound,
        )
    }

    fn monotonic_constraint(&self, feature: Self::FeatureId) -> i8 {
        self.monotonic_constraints
            .as_ref()
            .map_or(0, |c| c[feature])
    }

    fn split_iter<'a>(
        &'a self,
        mask: &'a Mask,
        feature: Self::FeatureId,
        pivot: &'a Self::Pivot,
        missing_left: bool,
    ) -> impl Iterator<Item = bool> + 'a {
        if self.labels.is_none() {
            let values = match &self.values {
                DenseValues::Exact(values) => {
                    if self.has_missing_values { mask.iter().map(|&row| { let value = values[row * self.columns + feature]; if value.is_nan() { missing_left } else { value > *pivot } }).collect::<Vec<_>>().into_iter() }
                    else { mask.iter().map(|&row| values[row * self.columns + feature] > *pivot).collect::<Vec<_>>().into_iter() }
                }
                DenseValues::Binned(values) => {
                    if self.has_missing_values { mask.iter().map(|&row| { let bin = values[row * self.columns + feature]; let missing_bin = self.bin_edges.as_ref().map_or(usize::MAX, |e| e[feature].len() + 1); if bin as usize == missing_bin { missing_left } else { (bin as f64) > *pivot } }).collect::<Vec<_>>().into_iter() }
                    else { mask.iter().map(|&row| (values[row * self.columns + feature] as f64) > *pivot).collect::<Vec<_>>().into_iter() }
                }
                DenseValues::Sparse(_) => mask
                    .iter()
                    .map(|&row| self.routes_left(row, feature, *pivot, missing_left))
                    .collect::<Vec<_>>()
                    .into_iter(),
            };
            DenseSplitIter::Buffered(values)
        } else {
            if self.has_missing_values {
                DenseSplitIter::Buffered(
                    mask.iter()
                        .map(|&row| self.routes_left(row, feature, *pivot, missing_left))
                        .collect::<Vec<_>>()
                        .into_iter(),
                )
            } else {
                DenseSplitIter::Lazy { input: self, rows: mask.iter(), feature, pivot: *pivot }
            }
        }
    }
}

#[cfg(test)]
mod histogram_optimization_tests {
    use super::*;

    fn binary_input(max_bins: Option<usize>) -> DenseInput {
        DenseInput::training(
            (0..8).map(|row| row as f64).collect(),
            8,
            1,
            vec![0, 0, 0, 0, 1, 1, 1, 1],
            vec![1.0; 8],
            2,
            0.0,
            Criterion::Gini,
            2,
            1,
            0.0,
            max_bins,
        )
        .unwrap()
    }

    fn training_input() -> DenseInput {
        DenseInput::training(
            vec![0.0, 7.0, 1.0, 7.0, 2.0, 7.0, 3.0, 7.0, 4.0, 7.0, 5.0, 7.0],
            6,
            2,
            vec![0, 0, 0, 1, 1, 1],
            vec![1.0; 6],
            2,
            0.0,
            Criterion::Gini,
            2,
            1,
            0.0,
            Some(4),
        )
        .unwrap()
    }

    #[test]
    fn histogram_bins_use_compact_storage() {
        let input = training_input();
        assert!(matches!(input.values, DenseValues::Binned(ref values) if values.len() == 12));
    }

    #[test]
    fn direct_histogram_accessor_matches_owned_matrix_binning() {
        let rows = 600;
        let columns = 64;
        let mut values = (0..rows * columns)
            .map(|index| ((index * 17) % 257) as f64)
            .collect::<Vec<_>>();
        for index in (37..values.len()).step_by(211) {
            values[index] = f64::NAN;
        }
        let labels = (0..rows).map(|row| row % 2).collect::<Vec<_>>();
        let weights = vec![1.0; rows];
        let reference = DenseInput::training_with_binning_options(
            values.clone(),
            rows,
            columns,
            labels.clone(),
            weights.clone(),
            2,
            0.0,
            Criterion::Gini,
            2,
            1,
            0.0,
            Some(8),
            4,
            BinningStrategy::SampledSelect,
            200,
        )
        .unwrap();
        let direct = DenseInput::training_binned_with_accessor(
            rows,
            columns,
            labels,
            weights,
            2,
            0.0,
            Criterion::Gini,
            2,
            1,
            0.0,
            8,
            4,
            BinningStrategy::SampledSelect,
            200,
            |row, feature| values[row * columns + feature],
        )
        .unwrap();

        assert_eq!(direct.bin_edges(), reference.bin_edges());
        assert!(direct.has_missing_values);
        match (&direct.values, &reference.values) {
            (DenseValues::Binned(direct), DenseValues::Binned(reference)) => {
                assert_eq!(direct, reference);
            }
            _ => panic!("both histogram inputs should use compact binned storage"),
        }
    }

    #[test]
    fn direct_histogram_accessor_rejects_infinite_values() {
        let values = [0.0, 1.0, f64::INFINITY, 3.0];
        let result = DenseInput::training_binned_with_accessor(
            4,
            1,
            vec![0, 0, 1, 1],
            vec![1.0; 4],
            2,
            0.0,
            Criterion::Gini,
            2,
            1,
            0.0,
            8,
            2,
            BinningStrategy::ExactSort,
            4,
            |row, _| values[row],
        );

        assert!(matches!(result, Err(message) if message == "X must not contain infinite values"));
    }

    #[test]
    fn multiselect_matches_full_sort_for_unique_values() {
        let values = (0..4096)
            .map(|index| (index as f64 * 1.25) - 1000.0)
            .collect::<Vec<_>>();
        let sorted = edges_from_observations(values.clone(), 63, BinningStrategy::ExactSort);
        let selected = edges_from_observations(values, 63, BinningStrategy::ExactSelect);
        assert_eq!(selected, sorted);
    }

    #[test]
    fn sampled_binning_is_deterministic_and_produces_ordered_edges() {
        let first = deterministic_sample_rows(1000, 200);
        assert_eq!(first, deterministic_sample_rows(1000, 200));
        assert_eq!(first.len(), 200);
        assert!(first.windows(2).all(|pair| pair[0] < pair[1]));

        let values = (0..1000).map(|index| index as f64).collect::<Vec<_>>();
        let edges = edges_from_observations(values, 32, BinningStrategy::SampledSelect);
        assert!(edges.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(edges.len() <= 31);
    }

    #[test]
    fn multiselect_handles_duplicate_values_without_duplicate_cuts() {
        let values = (0..1000)
            .map(|index| match index % 10 {
                0..=6 => 0.0,
                7..=8 => 1.0,
                _ => 2.0,
            })
            .collect::<Vec<_>>();
        let edges = edges_from_observations(values, 32, BinningStrategy::ExactSelect);
        assert!(edges.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(edges.len() <= 2);
    }

    #[test]
    fn binning_strategy_parser_rejects_unknown_values() {
        assert_eq!(
            BinningStrategy::parse("sampled_select").unwrap(),
            BinningStrategy::SampledSelect
        );
        assert!(BinningStrategy::parse("unknown").is_err());
    }

    #[test]
    fn parallel_histogram_preprocessing_matches_serial() {
        let rows = 37;
        let columns = 5;
        let values = (0..rows * columns)
            .map(|i| {
                if i % 29 == 0 {
                    f64::NAN
                } else {
                    ((i * 17) % 41) as f64 / 3.0
                }
            })
            .collect::<Vec<_>>();
        let make = |threads| {
            DenseInput::training_with_threads(
                values.clone(),
                rows,
                columns,
                (0..rows).map(|i| i % 3).collect(),
                vec![1.0; rows],
                3,
                0.0,
                Criterion::Gini,
                2,
                1,
                0.0,
                Some(16),
                threads,
            )
            .unwrap()
        };
        let serial = make(1);
        let parallel = make(4);
        assert_eq!(serial.bin_edges(), parallel.bin_edges());
        match (serial.values, parallel.values) {
            (DenseValues::Binned(left), DenseValues::Binned(right)) => assert_eq!(left, right),
            _ => panic!("histogram inputs must use compact binned storage"),
        }
    }

    #[test]
    fn parallel_sparse_histogram_edges_match_serial() {
        let make = |threads| {
            DenseInput::training_csr_with_threads(
                vec![0, 2, 4, 6, 8],
                vec![0, 1, 0, 2, 1, 2, 0, 2],
                vec![1.0, 2.0, 2.0, 1.0, 3.0, 4.0, 4.0, 2.0],
                4,
                3,
                vec![0, 0, 1, 1],
                vec![1.0; 4],
                2,
                0.0,
                Criterion::Gini,
                2,
                1,
                0.0,
                Some(4),
                threads,
            )
            .unwrap()
        };
        let serial = make(1);
        let parallel = make(3);
        assert_eq!(serial.bin_edges(), parallel.bin_edges());
    }

    #[test]
    fn constant_features_are_excluded_from_histogram_split_sampling() {
        let input = training_input();
        assert_eq!(input.active_features, vec![0]);
        assert_eq!(input.feature_count(), 2);
    }

    #[test]
    fn child_histogram_subtraction_matches_direct_histogram() {
        let input = training_input();
        let parent = Mask::new_all(6);
        let left = Mask::from_vec(vec![0, 1, 2]);
        let right = Mask::from_vec(vec![3, 4, 5]);
        let parent_histogram = build_histograms(&input, &parent);
        let weights_ptr = parent_histogram.class_weights.as_ptr();
        let counts_ptr = parent_histogram.sample_counts.as_ptr();
        let left_histogram = build_histograms(&input, &left);
        let derived_right = subtract_histograms(parent_histogram, &left_histogram);
        let direct_right = build_histograms(&input, &right);

        assert_eq!(derived_right, direct_right);
        assert_eq!(derived_right.class_weights.as_ptr(), weights_ptr);
        assert_eq!(derived_right.sample_counts.as_ptr(), counts_ptr);
    }

    #[test]
    fn parallel_child_histogram_matches_serial() {
        let rows = 512;
        let columns = 64;
        let values = (0..rows * columns)
            .map(|index| ((index * 37 % 997) as f64) / 997.0)
            .collect::<Vec<_>>();
        let labels = (0..rows).map(|row| row % 3).collect::<Vec<_>>();
        let weights = (0..rows)
            .map(|row| 0.5 + (row % 7) as f64 / 10.0)
            .collect::<Vec<_>>();
        let input = DenseInput::training_with_threads(
            values,
            rows,
            columns,
            labels,
            weights,
            3,
            0.0,
            Criterion::Gini,
            2,
            1,
            0.0,
            Some(32),
            1,
        )
        .unwrap();
        let mask = Mask::from_vec((0..rows).step_by(2).collect());
        let serial = build_histograms(&input, &mask);
        let parallel = build_histograms(&input.clone().with_histogram_threads(4), &mask);
        assert_eq!(serial, parallel);
    }

    #[test]
    fn child_caches_preserve_bootstrap_counts_in_both_size_orders() {
        let input = training_input();
        for (left_rows, right_rows) in [
            (vec![0, 0], vec![1, 2, 3, 3, 4, 5]),
            (vec![1, 2, 3, 3, 4, 5], vec![0, 0]),
        ] {
            let parent = Mask::from_vec([left_rows.clone(), right_rows.clone()].concat());
            let left = Mask::from_vec(left_rows);
            let right = Mask::from_vec(right_rows);
            let (left_cache, right_cache) = input.split_cache_children(
                input.split_cache(&parent), &parent, &left, &right,
            );
            assert_eq!(left_cache.0.unwrap(), build_histograms(&input, &left));
            assert_eq!(right_cache.0.unwrap(), build_histograms(&input, &right));
        }
    }

    #[test]
    fn finite_and_nan_training_split_iterators_route_rows() {
        let make_input = |values| {
            DenseInput::training(
                values,
                3,
                1,
                vec![0, 1, 1],
                vec![1.0; 3],
                2,
                0.0,
                Criterion::Gini,
                2,
                1,
                0.0,
                None,
            )
            .unwrap()
        };
        let mask = Mask::new_all(3);
        let finite = make_input(vec![0.0, 1.0, 2.0]);
        assert_eq!(
            finite
                .split_iter(&mask, 0, &1.0, false)
                .collect::<Vec<_>>(),
            vec![false, false, true]
        );

        let with_nan = make_input(vec![0.0, f64::NAN, 2.0]);
        assert_eq!(
            with_nan
                .split_iter(&mask, 0, &1.0, true)
                .collect::<Vec<_>>(),
            vec![false, true, true]
        );
    }

    #[test]
    fn bootstrap_balancing_weights_each_tree_from_its_bag_without_copying_features() {
        let bag = Mask::from_vec(vec![0, 0, 1, 2, 2, 0]);
        for max_bins in [None, Some(4)] {
            let input = DenseInput::training(
                vec![0.0, 1.0, 2.0],
                3,
                1,
                vec![0, 0, 1],
                vec![2.0, 3.0, 5.0],
                2,
                0.0,
                Criterion::Gini,
                2,
                1,
                0.0,
                max_bins,
            )
            .unwrap()
            .with_balanced_subsample();

            let tree_input = input.tree_input(&bag).unwrap();
            assert_eq!(tree_input.sample_weight(0), 1.5);
            assert_eq!(tree_input.sample_weight(1), 2.25);
            assert_eq!(tree_input.sample_weight(2), 7.5);
            assert!(input.shares_feature_storage_with(&tree_input));
        }
    }

    #[test]
    fn training_rejects_weight_sums_that_overflow_in_dense_csr_and_accessor_paths() {
        let error = "sample_weight sum must be finite";
        let dense = DenseInput::training(
            vec![0.0, 1.0],
            2,
            1,
            vec![0, 1],
            vec![f64::MAX, f64::MAX],
            2,
            0.0,
            Criterion::Gini,
            2,
            1,
            0.0,
            None,
        );
        assert_eq!(dense.err().as_deref(), Some(error));

        let csr = DenseInput::training_csr(
            vec![0, 1, 2],
            vec![0, 0],
            vec![0.0, 1.0],
            2,
            1,
            vec![0, 1],
            vec![f64::MAX, f64::MAX],
            2,
            0.0,
            Criterion::Gini,
            2,
            1,
            0.0,
            None,
        );
        assert_eq!(csr.err().as_deref(), Some(error));

        let accessor = DenseInput::training_binned_with_accessor(
            2,
            1,
            vec![0, 1],
            vec![f64::MAX, f64::MAX],
            2,
            0.0,
            Criterion::Gini,
            2,
            1,
            0.0,
            4,
            1,
            BinningStrategy::ExactSort,
            2,
            |row, _| row as f64,
        );
        assert_eq!(accessor.err().as_deref(), Some(error));
    }

    #[test]
    fn csr_validation_rejects_shape_overflow_duplicate_indices_and_infinity() {
        let shape_overflow = validate_csr(Vec::new(), Vec::new(), Vec::new(), usize::MAX, 1);
        assert_eq!(shape_overflow.err().as_deref(), Some("invalid CSR matrix structure"));

        let duplicate_columns = validate_csr(
            vec![0, 2, 2],
            vec![1, 1],
            vec![2.0, 3.0],
            2,
            2,
        );
        assert_eq!(
            duplicate_columns.err().as_deref(),
            Some("CSR column indices must be sorted and unique")
        );

        let infinite_data = validate_csr(
            vec![0, 1],
            vec![0],
            vec![f64::INFINITY],
            1,
            1,
        );
        assert_eq!(infinite_data.err().as_deref(), Some("X must not contain infinite values"));
    }

    #[test]
    fn direct_binning_accessor_rejects_zero_rows_before_sampling() {
        let result = DenseInput::training_binned_with_accessor(
            0,
            1,
            Vec::new(),
            Vec::new(),
            2,
            0.0,
            Criterion::Gini,
            2,
            1,
            0.0,
            4,
            1,
            BinningStrategy::ExactSort,
            1,
            |_, _| 0.0,
        );

        assert_eq!(result.err().as_deref(), Some("X must contain at least one row"));
    }

    #[test]
    fn csr_prediction_uses_implicit_zeros_and_routes_nan_consistently() {
        let input = DenseInput::prediction_csr(
            vec![0, 1, 1, 2],
            vec![1, 0],
            vec![3.0, f64::NAN],
            3,
            2,
            2,
            None,
        )
        .unwrap();

        assert_eq!(input.value_at(0, 0), 0.0);
        assert_eq!(input.value_at(0, 1), 3.0);
        assert_eq!(input.value_at(1, 0), 0.0);
        assert!(input.value_at(2, 0).is_nan());
        assert_eq!(
            input.split_iter(&Mask::new_all(3), 0, &0.0, true).collect::<Vec<_>>(),
            vec![false, false, true]
        );
    }

    #[test]
    fn exact_and_histogram_forests_fit_predict_in_parallel_and_prune() {
        for max_bins in [None, Some(4)] {
            let input = binary_input(max_bins);
            let forest = xrf::Forest::new_with_settings(
                &input,
                1,
                1,
                true,
                false,
                false,
                23,
                4,
                2,
                false,
                None,
            );
            let expected = input.labels().to_vec();
            let predict = |parallel: bool| {
                let result = if parallel {
                    forest.predict_parallel(&input, 3)
                } else {
                    forest.predict(&input)
                };
                result
                    .predictions()
                    .map(|(_, votes)| votes.winner())
                    .collect::<Vec<_>>()
            };

            assert_eq!(predict(false), expected);
            assert_eq!(predict(true), expected);
            let gain = forest.gain_importance_normalised().collect::<Vec<_>>();
            assert_eq!(gain, vec![(0, 1.0)]);

            let pruned_input = binary_input(max_bins).with_ccp_alpha(1.0);
            let pruned = xrf::Forest::new_with_settings(
                &pruned_input,
                1,
                1,
                true,
                false,
                false,
                23,
                4,
                2,
                false,
                None,
            );
            assert!(matches!(
                &pruned.tree_refs().next().unwrap().nodes[0],
                xrf::Node::Leaf(..)
            ));
            assert_eq!(
                pruned
                    .predict(&pruned_input)
                    .predictions()
                    .map(|(_, votes)| votes.winner())
                    .collect::<Vec<_>>(),
                vec![0; 8]
            );
        }
    }

    #[test]
    fn serial_and_parallel_forests_match_predictions_oob_votes_and_importance() {
        let input = binary_input(Some(4));
        let serial = xrf::Forest::new_with_settings(
            &input, 24, 1, true, true, true, 91, 4, 2, true, Some(6),
        );
        let parallel = xrf::Forest::new_parallel_with_settings(
            &input, 24, 1, true, true, true, 91, 3, 4, 2, true, Some(6),
        );

        let collect_probabilities = |forest: &xrf::Forest<DenseInput>| {
            forest
                .predict(&input)
                .predictions()
                .map(|(_, votes)| votes.probabilities())
                .collect::<Vec<_>>()
        };
        assert_eq!(collect_probabilities(&serial), collect_probabilities(&parallel));
        assert!(serial.has_oob() && parallel.has_oob());
        assert!(serial.has_importance() && parallel.has_importance());
        assert_eq!(serial.oob().count(), input.rows());
        assert_eq!(parallel.oob().count(), input.rows());
        for forest in [&serial, &parallel] {
            for (_, votes) in forest.oob() {
                let probabilities = votes.probabilities();
                assert!(probabilities.iter().all(|value| value.is_finite()));
                assert!(probabilities.iter().sum::<f64>() <= 1.0);
            }
        }
        let serial_importance = serial.importance().collect::<Vec<_>>();
        let parallel_importance = parallel.importance().collect::<Vec<_>>();
        assert_eq!(serial_importance.len(), parallel_importance.len());
        for (feature, serial_value) in serial_importance {
            let parallel_value = parallel_importance
                .iter()
                .find(|(parallel_feature, _)| *parallel_feature == feature)
                .unwrap()
                .1;
            assert!((serial_value - parallel_value).abs() < 1e-12);
        }
    }
}
