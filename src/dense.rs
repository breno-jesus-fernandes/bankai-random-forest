use std::collections::HashMap;
use std::sync::Arc;
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

#[derive(Clone, Debug, PartialEq)]
struct FeatureHistogram {
    class_weights: Vec<f64>,
    sample_counts: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
struct HistogramCache {
    features: Vec<FeatureHistogram>,
}

pub struct DenseSplitCache(Option<HistogramCache>);

enum DenseSplitIter<'a> {
    Lazy {
        input: &'a DenseInput,
        rows: std::slice::Iter<'a, usize>,
        feature: usize,
        pivot: f64,
    },
    LazyMissing {
        input: &'a DenseInput,
        rows: std::slice::Iter<'a, usize>,
        feature: usize,
        pivot: f64,
        missing_left: bool,
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
            Self::LazyMissing { input, rows, feature, pivot, missing_left } => rows.next().map(|&row| input.routes_left(row, *feature, *pivot, *missing_left)),
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
        let has_missing_values = values.iter().any(|value| value.is_nan());
        validate_matrix(&values, rows, columns)?;
        if labels.len() != rows {
            return Err("y must contain one label per row".to_string());
        }
        if sample_weights.len() != rows {
            return Err("sample_weight must contain one value per row".to_string());
        }
        if sample_weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        {
            return Err("sample_weight must contain finite non-negative values".to_string());
        }
        if sample_weights.iter().all(|weight| *weight == 0.0) {
            return Err("sample_weight cannot be all zero".to_string());
        }
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

        let total_weight = sample_weights.iter().sum();
        let (values, bin_edges, active_features) = match max_bins {
            Some(max_bins) => {
                let (binned, edges) = histogramize(&values, rows, columns, max_bins);
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
        let has_missing_values = data.iter().any(|value| value.is_nan());
        let sparse = validate_csr(indptr, indices, data, rows, columns)?;
        if labels.len() != rows {
            return Err("y must contain one label per row".into());
        }
        if sample_weights.len() != rows {
            return Err("sample_weight must contain one value per row".into());
        }
        if sample_weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
            return Err("sample_weight must contain finite non-negative values".into());
        }
        if sample_weights.iter().all(|w| *w == 0.0) {
            return Err("sample_weight cannot be all zero".into());
        }
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
        let total_weight = sample_weights.iter().sum();
        let bin_edges = max_bins.map(|bins| sparse_histogram_edges(&sparse, rows, columns, bins));
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
                self.bin_edges.as_ref().map_or(value, |edges| {
                    if value.is_nan() { (edges[column].len() + 1) as f64 } else { edges[column].partition_point(|edge| value > *edge) as f64 }
                })
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
    if indptr.len() != rows + 1
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
) -> Vec<Vec<f64>> {
    (0..columns)
        .map(|column| {
            let mut sorted = (0..rows)
                .map(|row| values.get(row, column))
                .filter(|value| !value.is_nan())
                .collect::<Vec<_>>();
            sorted.sort_unstable_by(f64::total_cmp);
            sorted.dedup_by(|a, b| a.total_cmp(b).is_eq());
            let mut edges = Vec::new();
            if sorted.len() <= max_bins {
                for pair in sorted.windows(2) {
                    edges.push(midpoint(pair[0], pair[1]));
                }
            } else {
                for bin in 1..max_bins {
                    let index = bin * sorted.len() / max_bins;
                    if index > 0 && index < sorted.len() {
                        let edge = midpoint(sorted[index - 1], sorted[index]);
                        if edges.last().is_none_or(|prev| *prev < edge) {
                            edges.push(edge);
                        }
                    }
                }
            }
            edges
        })
        .collect()
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
) -> (Vec<u8>, Vec<Vec<f64>>) {
    let mut edges_by_feature = Vec::with_capacity(columns);
    for feature in 0..columns {
        let mut sorted: Vec<_> = (0..rows)
            .map(|row| values[row * columns + feature])
            .filter(|value| !value.is_nan())
            .collect();
        sorted.sort_unstable_by(f64::total_cmp);
        sorted.dedup_by(|left, right| left.total_cmp(right).is_eq());

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
        edges_by_feature.push(edges);
    }

    (
        apply_histogram_edges(values, rows, columns, &edges_by_feature),
        edges_by_feature,
    )
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

fn build_histograms(input: &DenseInput, mask: &Mask) -> HistogramCache {
    let mut features = input
        .bin_edges
        .as_ref()
        .expect("histogram cache requires fitted bin edges")
        .iter()
        .map(|edges| {
            let bins = edges.len() + 1 + usize::from(input.has_missing_values);
            FeatureHistogram {
                class_weights: vec![0.0; bins * input.n_classes],
                sample_counts: vec![0; bins],
            }
        })
        .collect::<Vec<_>>();

    match &input.values {
        DenseValues::Binned(values) => {
            for &row in mask.iter() {
                let label = input.labels()[row];
                let weight = input.sample_weight(row);
                let offset = row * input.columns;
                for (feature, histogram) in features.iter_mut().enumerate() {
                    let bin = values[offset + feature] as usize;
                    histogram.class_weights[bin * input.n_classes + label] += weight;
                    histogram.sample_counts[bin] += 1;
                }
            }
        }
        DenseValues::Sparse(_) => {
            for &row in mask.iter() {
                let label = input.labels()[row];
                let weight = input.sample_weight(row);
                for (feature, histogram) in features.iter_mut().enumerate() {
                    let bin = input.bin_value(row, feature);
                    histogram.class_weights[bin * input.n_classes + label] += weight;
                    histogram.sample_counts[bin] += 1;
                }
            }
        }
        DenseValues::Exact(_) => unreachable!("histograms require binned input"),
    }
    HistogramCache { features }
}

fn subtract_histograms(parent: &HistogramCache, smaller_child: &HistogramCache) -> HistogramCache {
    let features = parent
        .features
        .iter()
        .zip(&smaller_child.features)
        .map(|(parent, smaller)| FeatureHistogram {
            class_weights: parent
                .class_weights
                .iter()
                .zip(&smaller.class_weights)
                .map(|(parent, smaller)| (parent - smaller).max(0.0))
                .collect(),
            sample_counts: parent
                .sample_counts
                .iter()
                .zip(&smaller.sample_counts)
                .map(|(parent, smaller)| parent - smaller)
                .collect(),
        })
        .collect();
    HistogramCache { features }
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
            &split_cache.0.as_ref()?.features[feature],
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
    histogram: &FeatureHistogram,
    monotonic_constraint: i8,
    lower_bound: f64,
    upper_bound: f64,
) -> Option<(f64, bool, f64)> {
    let bin_count = histogram.sample_counts.len();
    if bin_count < 2 {
        return None;
    }

    let total_weight = target.total_weight;
    let parent_impurity = impurity(input.criterion, &target.class_weights, total_weight);
    let mut left = vec![0.0; input.n_classes];
    let mut right = target.class_weights.clone();
    let mut left_weight = 0.0;
    let mut left_count = 0;
    let missing_bin = if input.has_missing_values { bin_count - 1 } else { bin_count };
    let missing_classes = input.has_missing_values.then(|| (0..input.n_classes).map(|class| histogram.class_weights[missing_bin * input.n_classes + class]).collect::<Vec<_>>());
    let missing_count = if input.has_missing_values { histogram.sample_counts[missing_bin] } else { 0 };
    if let Some(missing_classes) = &missing_classes {
        for class in 0..input.n_classes { right[class] -= missing_classes[class]; }
    }
    let total_count = histogram.sample_counts[..missing_bin].iter().sum::<usize>();
    let mut best: Option<(f64, bool, f64)> = None;

    let split_bin_end = if input.has_missing_values { missing_bin.saturating_sub(1) } else { bin_count.saturating_sub(1) };
    for bin in 0..split_bin_end {
        for class in 0..input.n_classes {
            let weight = histogram.class_weights[bin * input.n_classes + class];
            left[class] += weight;
            right[class] -= weight;
            left_weight += weight;
        }
        left_count += histogram.sample_counts[bin];
        let pivot = bin as f64 + 0.5;
        if missing_count == 0 {
            let right_count = total_count - left_count;
            let right_weight = total_weight - left_weight;
            if left_count < input.min_samples_leaf || right_count < input.min_samples_leaf
                || left_weight == 0.0 || right_weight == 0.0
                || left_weight < input.min_leaf_weight || right_weight < input.min_leaf_weight { continue; }
            if monotonic_constraint != 0 {
                let lp = left[1] / left_weight; let rp = right[1] / right_weight;
                if lp < lower_bound || lp > upper_bound || rp < lower_bound || rp > upper_bound
                    || (lp - rp) * monotonic_constraint as f64 > 0.0 { continue; }
            }
            let child_impurity = (left_weight / total_weight) * impurity(input.criterion, &left, left_weight)
                + (right_weight / total_weight) * impurity(input.criterion, &right, right_weight);
            let gain = parent_impurity - child_impurity;
            if gain >= input.min_impurity_decrease && best.as_ref().is_none_or(|(_, _, g)| gain > *g) {
                best = Some((pivot, left_count <= right_count, gain));
            }
            continue;
        }
        for missing_left in [false, true] {
            if missing_count == 0 && missing_left != (left_count > total_count - left_count) { continue; }
            let (lc, rc, lcount, rcount, lw) = if missing_left {
                let mut lc = left.clone();
                let missing_classes = missing_classes.as_ref().unwrap();
                for class in 0..input.n_classes { lc[class] += missing_classes[class]; }
                (lc, right.clone(), left_count + missing_count, total_count - left_count, left_weight + missing_classes.iter().sum::<f64>())
            } else {
                let mut rc = right.clone();
                for class in 0..input.n_classes { rc[class] += missing_classes.as_ref().unwrap()[class]; }
                (left.clone(), rc, left_count, total_count - left_count + missing_count, left_weight)
            };
            let rw = total_weight - lw;
            if lcount < input.min_samples_leaf || rcount < input.min_samples_leaf || lw == 0.0 || rw == 0.0
                || lw < input.min_leaf_weight || rw < input.min_leaf_weight { continue; }
            if monotonic_constraint != 0 {
                let lp = lc[1] / lw; let rp = rc[1] / rw;
                if lp < lower_bound || lp > upper_bound || rp < lower_bound || rp > upper_bound
                    || (lp - rp) * monotonic_constraint as f64 > 0.0 { continue; }
            }
            let child_impurity = (lw / total_weight) * impurity(input.criterion, &lc, lw)
                + (rw / total_weight) * impurity(input.criterion, &rc, rw);
            let gain = parent_impurity - child_impurity;
            if gain >= input.min_impurity_decrease && best.as_ref().is_none_or(|(_, _, g)| gain > *g) {
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
        parent_cache: &Self::SplitCache,
        _: &Mask,
        left: &Mask,
        right: &Mask,
    ) -> (Self::SplitCache, Self::SplitCache) {
        let Some(parent_histograms) = &parent_cache.0 else {
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
                DenseSplitIter::LazyMissing { input: self, rows: mask.iter(), feature, pivot: *pivot, missing_left }
            } else {
                DenseSplitIter::Lazy { input: self, rows: mask.iter(), feature, pivot: *pivot }
            }
        }
    }
}

#[cfg(test)]
mod histogram_optimization_tests {
    use super::*;

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
        let left_histogram = build_histograms(&input, &left);
        let derived_right = subtract_histograms(&parent_histogram, &left_histogram);
        let direct_right = build_histograms(&input, &right);

        assert_eq!(derived_right, direct_right);
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
}
