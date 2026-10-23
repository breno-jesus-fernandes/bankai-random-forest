use std::collections::HashMap;
use std::cell::RefCell;

thread_local! {
    // One contiguous feature workspace per tree worker; retained across nodes.
    static EXTRA_FEATURE_VALUES: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    static EXTRA_FEATURE_VALUES_F32: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}
use std::sync::Arc;
use std::thread;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::time::Instant;

#[derive(Default)]
pub struct FitProfile {
    split_ns: AtomicU64,
    bootstrap_ns: AtomicU64,
    partition_ns: AtomicU64,
    feature_visits: AtomicU64,
    sample_visits: AtomicU64,
    rejected: AtomicU64,
    buffer_bytes: AtomicU64,
    layout_ns: AtomicU64,
}

impl FitProfile {
    pub fn snapshot(&self) -> HashMap<String, f64> {
        [("bin_layout_conversion_seconds", self.layout_ns.load(AtomicOrdering::Relaxed) as f64 / 1e9),
         ("split_search_seconds", self.split_ns.load(AtomicOrdering::Relaxed) as f64 / 1e9),
         ("bootstrap_seconds", self.bootstrap_ns.load(AtomicOrdering::Relaxed) as f64 / 1e9),
         ("partition_seconds", self.partition_ns.load(AtomicOrdering::Relaxed) as f64 / 1e9),
         ("feature_visits", self.feature_visits.load(AtomicOrdering::Relaxed) as f64),
         ("sample_visits", self.sample_visits.load(AtomicOrdering::Relaxed) as f64),
         ("rejected_candidates", self.rejected.load(AtomicOrdering::Relaxed) as f64),
         ("feature_buffer_peak_bytes_per_worker", self.buffer_bytes.load(AtomicOrdering::Relaxed) as f64)]
         .into_iter().map(|(k,v)| (k.to_string(),v)).collect()
    }
}

struct ProfileTimer<'a>(Option<(&'a AtomicU64, Instant)>);
impl Drop for ProfileTimer<'_> {
    fn drop(&mut self) {
        if let Some((counter, start)) = self.0 { counter.fetch_add(start.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed); }
    }
}
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
    extra_trees: bool,
    column_major: bool,
    extra_seed: u64,
    profile: Option<Arc<FitProfile>>,
    /// Every training weight is a whole bootstrap count (no balanced or user
    /// scaling), so weight sums are exact integers in any order.
    integral_weights: bool,
    /// Whole bootstrap counts standing in for `sample_weights` (4 bytes a row).
    count_weights: Option<Arc<Vec<u32>>>,
}

#[derive(Clone)]
enum DenseValues {
    Exact(Arc<ExactValues>),
    Binned(Arc<Vec<u8>>),
    Sparse(Arc<CsrValues>),
}

#[derive(Clone)]
enum ExactValues {
    F64(Vec<f64>),
    F32(Vec<f32>),
    Virtual(ExactVirtualColumns),
    Columns(ExactColumnTable),
}

impl ExactValues {
    #[inline]
    fn at(&self, index: usize) -> f64 {
        match self { Self::F64(values) => values[index], Self::F32(values) => f64::from(values[index]), Self::Virtual(values) => f64::from(values.at(index)), Self::Columns(values) => f64::from(values.at(index)) }
    }
}

/// Immutable original-precision columns shared by every round and tree.
/// Sorting uses `value_at`; training reproduces the a15 float32 conversion.
#[derive(Clone)]
pub struct ExactColumnBank {
    values: Arc<ExactValues>,
    rows: usize,
    columns: usize,
}

impl ExactColumnBank {
    pub fn from_f32(values: Vec<f32>, rows: usize, columns: usize) -> Result<Self, String> {
        Self::new(ExactValues::F32(values), rows, columns)
    }
    pub fn from_f64(values: Vec<f64>, rows: usize, columns: usize) -> Result<Self, String> {
        Self::new(ExactValues::F64(values), rows, columns)
    }
    fn new(values: ExactValues, rows: usize, columns: usize) -> Result<Self, String> {
        let len = match &values { ExactValues::F32(v) => v.len(), ExactValues::F64(v) => v.len(), _ => unreachable!() };
        if rows == 0 || columns == 0 || rows.checked_mul(columns) != Some(len) {
            return Err("invalid exact column bank shape".into());
        }
        if (0..len).any(|i| !values.at(i).is_finite() || !(values.at(i) as f32).is_finite()) {
            return Err("exact column bank requires finite float32-representable values".into());
        }
        Ok(Self { values: Arc::new(values), rows, columns })
    }
    pub fn rows(&self) -> usize { self.rows }
    pub fn columns(&self) -> usize { self.columns }
    pub fn bytes(&self) -> usize {
        self.rows * self.columns * match self.values.as_ref() { ExactValues::F64(_) => 8, _ => 4 }
    }
    pub fn value_at(&self, row: usize, column: usize) -> f64 {
        self.values.at(column * self.rows + row)
    }
}

/// Explicit row maps. Compact indices support exactly 2^32 rows on 64-bit hosts.
#[derive(Clone)]
pub enum ExactRowIndices { U32(Vec<u32>), Usize(Vec<usize>) }
impl ExactRowIndices {
    pub fn len(&self) -> usize { match self { Self::U32(v) => v.len(), Self::Usize(v) => v.len() } }
    pub fn is_empty(&self) -> bool { self.len() == 0 }
    pub fn at(&self, i: usize) -> usize { match self { Self::U32(v) => v[i] as usize, Self::Usize(v) => v[i] } }
    pub fn bytes(&self) -> usize { match self { Self::U32(v) => v.len() * 4, Self::Usize(v) => v.len() * std::mem::size_of::<usize>() } }
    pub fn allocated_bytes(&self) -> usize { match self { Self::U32(v) => v.capacity() * 4, Self::Usize(v) => v.capacity() * std::mem::size_of::<usize>() } }
}

#[derive(Clone)]
pub struct ExactVirtualColumns {
    bank: ExactColumnBank,
    columns: Vec<usize>,
    real_columns: usize,
    order: Arc<Vec<usize>>,
    permutations: Arc<ExactRowIndices>,
}
impl ExactVirtualColumns {
    pub fn new(bank: ExactColumnBank, active: Vec<usize>, sources: Vec<usize>,
        order: Arc<Vec<usize>>, permutations: Arc<ExactRowIndices>) -> Result<Self, String> {
        let rows = bank.rows;
        if active.is_empty() || sources.is_empty() || order.len() != rows
            || rows.checked_mul(sources.len()) != Some(permutations.len())
            || active.iter().chain(&sources).any(|&c| c >= bank.columns)
            || order.iter().any(|&r| r >= rows)
            || (0..permutations.len()).any(|i| permutations.at(i) >= rows) {
            return Err("invalid virtual exact maps".into());
        }
        let mut seen = vec![false; rows];
        for &r in order.iter() { if std::mem::replace(&mut seen[r], true) { return Err("invalid row order".into()); } }
        for offset in (0..permutations.len()).step_by(rows) {
            seen.fill(false);
            for i in offset..offset + rows {
                if std::mem::replace(&mut seen[permutations.at(i)], true) { return Err("invalid shadow permutation".into()); }
            }
        }
        let real_columns = active.len();
        let columns = active.into_iter().chain(sources).collect();
        Ok(Self { bank, columns, real_columns, order, permutations })
    }
    #[inline]
    fn at(&self, index: usize) -> f32 {
        self.value_at(index % self.bank.rows, index / self.bank.rows)
    }
    #[inline]
    fn value_at(&self, row: usize, column: usize) -> f32 {
        let row = self.order[row];
        let row = if column < self.real_columns { row } else {
            self.permutations.at((column - self.real_columns) * self.bank.rows + row)
        };
        self.bank.value_at(row, self.columns[column]) as f32
    }
}

/// Float32 columns already in training-row order: real columns from a bank
/// shared by every round, shadows from a per-round buffer. Each feature is one
/// contiguous slice, so training reads it without row or permutation maps.
#[derive(Clone)]
pub struct ExactColumnTable {
    rows: usize,
    real: Arc<Vec<f32>>,
    shadows: Arc<Vec<f32>>,
    offsets: Vec<(bool, usize)>,
}
impl ExactColumnTable {
    /// `active` indexes real bank columns; shadow `j` is buffer column `j`.
    pub fn new(rows: usize, real: Arc<Vec<f32>>, shadows: Arc<Vec<f32>>, active: &[usize], shadow_count: usize) -> Result<Self, String> {
        if rows == 0 || active.is_empty() || shadow_count == 0
            || real.len() % rows != 0 || shadows.len() < rows.checked_mul(shadow_count).ok_or("shadow buffer is too large")?
            || active.iter().any(|&c| c >= real.len() / rows) {
            return Err("invalid exact column table".into());
        }
        let offsets = active.iter().map(|&c| (false, c * rows))
            .chain((0..shadow_count).map(|j| (true, j * rows))).collect();
        Ok(Self { rows, real, shadows, offsets })
    }
    #[inline]
    fn column(&self, feature: usize) -> &[f32] {
        let (shadow, offset) = self.offsets[feature];
        let values = if shadow { &self.shadows } else { &self.real };
        &values[offset..offset + self.rows]
    }
    #[inline]
    fn value_at(&self, row: usize, column: usize) -> f32 { self.column(column)[row] }
    #[inline]
    fn at(&self, index: usize) -> f32 { self.value_at(index % self.rows, index / self.rows) }
}

// Parallelize independent column blocks, sharing a single output allocation.
fn transpose_exact<T: Copy + Default + Send + Sync>(values: &[T], rows: usize, columns: usize, threads: usize) -> Vec<T> {
    transpose_exact_map(values, rows, columns, threads, |v| v)
}

fn transpose_exact_map<T: Copy + Sync, U: Copy + Default + Send, F: Fn(T) -> U + Copy + Send>(values: &[T], rows: usize, columns: usize, threads: usize, convert: F) -> Vec<U> {
    let mut output = vec![U::default(); values.len()];
    let columns_per_worker = columns.div_ceil(threads.max(1)).max(1);
    thread::scope(|scope| {
        for (block, dst) in output.chunks_mut(columns_per_worker * rows).enumerate() {
            scope.spawn(move || {
                // Revisit nearby columns while their source cache lines are hot.
                for row_start in (0..rows).step_by(64) {
                    let row_end = (row_start + 64).min(rows);
                    for (local_column, column_dst) in dst.chunks_mut(rows).enumerate() {
                        let column = block * columns_per_worker + local_column;
                        for row in row_start..row_end {
                            column_dst[row] = convert(values[row * columns + column]);
                        }
                    }
                }
            });
        }
    });
    output
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
const DEFAULT_BINNING_SEED: u64 = 6;
const LEGACY_BINNING_SAMPLE_STATE: u64 = 0x6a09_e667_f3bc_c909;

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

pub struct DenseSplitCache(Option<HistogramCache>, RefCell<Vec<usize>>);
impl DenseSplitCache {
    fn new(histogram: Option<HistogramCache>) -> Self { Self(histogram, RefCell::new(Vec::new())) }
    fn constant(&self, feature: usize) -> bool { self.1.borrow().contains(&feature) }
    fn mark_constant(&self, feature: usize) {
        if !self.constant(feature) { self.1.borrow_mut().push(feature); }
    }
}

enum DenseSplitIter<'a> {
    F32Column {
        column: &'a [f32], rows: std::slice::Iter<'a, usize>, pivot: f64,
    },
    VirtualF32Real {
        column: &'a [f32], order: &'a [usize], rows: std::slice::Iter<'a, usize>, pivot: f64,
    },
    VirtualF32Shadow {
        column: &'a [f32], order: &'a [usize], permutation: &'a [u32], rows: std::slice::Iter<'a, usize>, pivot: f64,
    },
    Lazy {
        input: &'a DenseInput,
        rows: std::slice::Iter<'a, usize>,
        feature: usize,
        pivot: f64,
    },
    Buffered(std::vec::IntoIter<bool>),
}

struct ProfiledSplitIter<'a> {
    inner: DenseSplitIter<'a>,
    _timer: ProfileTimer<'a>,
}
impl Iterator for ProfiledSplitIter<'_> {
    type Item = bool;
    fn next(&mut self) -> Option<bool> { self.inner.next() }
}

impl Iterator for DenseSplitIter<'_> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::F32Column { column, rows, pivot } => rows.next().map(|&r| f64::from(column[r]) > *pivot),
            Self::VirtualF32Real { column, order, rows, pivot } => rows.next().map(|&r| f64::from(column[order[r]]) > *pivot),
            Self::VirtualF32Shadow { column, order, permutation, rows, pivot } => rows.next().map(|&r| f64::from(column[permutation[order[r]] as usize]) > *pivot),
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
    /// Bag, out-of-bag rows (when kept) and the tree view weighted by bootstrap
    /// draw counts.
    fn bootstrap_masks(&self, keep_oob: bool, count: impl Fn(usize) -> usize) -> (Mask, Mask) {
        let in_bag = (0..self.rows).filter(|&row| count(row) > 0).count();
        let mut bag = Vec::with_capacity(in_bag);
        let mut oob = Vec::with_capacity(if keep_oob { self.rows - in_bag } else { 0 });
        for row in 0..self.rows {
            if count(row) > 0 { bag.push(row) } else if keep_oob { oob.push(row) }
        }
        (Mask::from_vec(bag), Mask::from_vec(oob))
    }

    /// Unbalanced bootstrap with u32 counts: the counts are the tree weights.
    fn bootstrap_from_u32_counts(&self, keep_oob: bool, counts: Vec<u32>) -> (Mask, Mask, Self) {
        let (bag, oob) = self.bootstrap_masks(keep_oob, |row| counts[row] as usize);
        // Whole counts sum exactly, matching the f64 sum of the same weights.
        let total_weight = counts.iter().map(|&n| u64::from(n)).sum::<u64>() as f64;
        let tree = Self {
            sample_weights: None, count_weights: Some(Arc::new(counts)), integral_weights: true, total_weight,
            min_leaf_weight: self.min_leaf_weight / self.total_weight * total_weight,
            ..self.clone()
        };
        (bag, oob, tree)
    }

    fn bootstrap_from_counts(&self, size: usize, keep_oob: bool, count: impl Fn(usize) -> usize) -> (Mask, Mask, Self) {
        let (bag, oob) = self.bootstrap_masks(keep_oob, &count);
        let mut tree_weights: Vec<f64> = (0..self.rows).map(|row| count(row) as f64).collect();
        if self.balanced_subsample {
            let mut classes = vec![0usize; self.n_classes];
            for row in 0..self.rows { classes[self.labels()[row]] += count(row); }
            let present = classes.iter().filter(|&&n| n > 0).count();
            for row in 0..self.rows {
                let n = classes[self.labels()[row]];
                if n > 0 { tree_weights[row] *= size as f64 / (present * n) as f64; }
            }
        }
        let total_weight = tree_weights.iter().sum();
        let tree = Self {
            sample_weights: Some(Arc::new(tree_weights)), total_weight,
            min_leaf_weight: self.min_leaf_weight / self.total_weight * total_weight,
            integral_weights: !self.balanced_subsample,
            balanced_subsample: false, ..self.clone()
        };
        (bag, oob, tree)
    }

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
        Self::training_with_binning_seed_options(
            values, rows, columns, labels, sample_weights, n_classes, min_leaf_weight,
            criterion, min_samples_split, min_samples_leaf, min_impurity_decrease,
            max_bins, threads, binning_strategy, bin_sample_size, DEFAULT_BINNING_SEED,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn training_with_binning_seed_options(
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
        binning_seed: u64,
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
                    binning_seed,
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
                DenseValues::Exact(Arc::new(ExactValues::F64(values))),
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
            extra_trees: false,
            column_major: false,
            extra_seed: 6,
            profile: None,
            integral_weights: false,
            count_weights: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn training_exact_f32_with_accessor<F>(
        rows: usize, columns: usize, labels: Vec<usize>, sample_weights: Vec<f64>,
        n_classes: usize, min_leaf_weight: f64, criterion: Criterion,
        min_samples_split: usize, min_samples_leaf: usize, min_impurity_decrease: f64,
        threads: usize, value_at: F,
    ) -> Result<Self, String>
    where F: Fn(usize, usize) -> f32 + Sync,
    {
        if rows == 0 || columns == 0 {
            return Err("X must contain at least one row and feature".into());
        }
        let cells = rows.checked_mul(columns).ok_or("X shape is too large")?;
        let mut values = vec![0.0f32; cells];
        let workers = if cells >= HISTOGRAM_PARALLEL_MIN_CELLS { threads.max(1) } else { 1 };
        let chunk_columns = columns.div_ceil(workers).max(1);
        thread::scope(|scope| {
            for (block, dst) in values.chunks_mut(chunk_columns * rows).enumerate() {
                let value_at = &value_at;
                scope.spawn(move || {
                    for row_start in (0..rows).step_by(64) {
                        for (offset, column_dst) in dst.chunks_mut(rows).enumerate() {
                            for row in row_start..(row_start + 64).min(rows) {
                                column_dst[row] = value_at(row, block * chunk_columns + offset);
                            }
                        }
                    }
                });
            }
        });
        if threads == 0 {
            return Err("histogram preprocessing requires at least one thread".into());
        }
        let has_missing_values = values.iter().any(|value| value.is_nan());
        validate_matrix(&values, rows, columns)?;
        Self::training_exact_storage(ExactValues::F32(values), rows, columns, labels, sample_weights,
            n_classes, min_leaf_weight, criterion, min_samples_split, min_samples_leaf,
            min_impurity_decrease, has_missing_values)
    }

    #[allow(clippy::too_many_arguments)]
    fn training_exact_storage(values: ExactValues, rows: usize, columns: usize,
        labels: Vec<usize>, sample_weights: Vec<f64>, n_classes: usize,
        min_leaf_weight: f64, criterion: Criterion, min_samples_split: usize,
        min_samples_leaf: usize, min_impurity_decrease: f64, has_missing_values: bool,
    ) -> Result<Self, String> {
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

        Ok(Self {
            values: DenseValues::Exact(Arc::new(values)),
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
            histogram_bins: None,
            bin_edges: None,
            active_features: (0..columns).collect(),
            balanced_subsample: false,
            ccp_alpha: 0.0,
            monotonic_constraints: None,
            has_missing_values,
            histogram_threads: 1,
            extra_trees: false,
            column_major: true,
            extra_seed: 6,
            profile: None,
            integral_weights: false,
            count_weights: None,
        })
    }
    /// Bin every column once: column-major codes plus per-column edges, the
    /// same codes and edges `training_binned_with_accessor_and_seed` builds.
    #[allow(clippy::too_many_arguments)]
    pub fn bin_columns<F>(
        rows: usize, columns: usize, max_bins: usize, threads: usize, strategy: BinningStrategy,
        sample_size: usize, binning_seed: u64, value_at: F,
    ) -> Result<(Vec<u8>, Vec<Vec<f64>>), String>
    where F: Fn(usize, usize) -> f64 + Sync,
    {
        if threads == 0 || rows == 0 || columns == 0 || !(2..=255).contains(&max_bins) || sample_size == 0 {
            return Err("invalid binning settings".into());
        }
        let (binned, edges, has_missing_values) = histogramize_from_accessor(
            rows, columns, max_bins, threads, strategy, sample_size, binning_seed, &value_at)?;
        if has_missing_values {
            return Err("column binning requires finite values".into());
        }
        Ok((transpose_exact(&binned, rows, columns, threads), edges))
    }

    /// Binned training input from column-major codes made by `bin_columns`
    /// (columns may repeat or be permuted); every sample weight is one.
    #[allow(clippy::too_many_arguments)]
    pub fn training_binned_columns(
        rows: usize, codes: Vec<u8>, edges: Vec<Vec<f64>>, labels: Vec<usize>, n_classes: usize,
        min_leaf_weight: f64, criterion: Criterion, min_samples_split: usize,
        min_samples_leaf: usize, min_impurity_decrease: f64, max_bins: usize,
    ) -> Result<Self, String> {
        let columns = edges.len();
        if rows == 0 || columns == 0 || rows.checked_mul(columns) != Some(codes.len()) || labels.len() != rows
            || !(2..=255).contains(&max_bins) || edges.iter().any(|e| e.len() >= max_bins)
            || n_classes == 0 || labels.iter().any(|&l| l >= n_classes)
            || !min_leaf_weight.is_finite() || min_leaf_weight < 0.0
            || min_samples_split < 2 || min_samples_leaf == 0
            || !min_impurity_decrease.is_finite() || min_impurity_decrease < 0.0
            || codes.chunks(rows).zip(&edges).any(|(column, e)| column.iter().any(|&c| c as usize > e.len())) {
            return Err("invalid binned column codes".into());
        }
        let active_features = edges.iter().enumerate()
            .filter_map(|(feature, e)| (!e.is_empty()).then_some(feature)).collect();
        Ok(Self {
            values: DenseValues::Binned(Arc::new(codes)),
            labels: Some(Arc::new(labels)),
            sample_weights: Some(Arc::new(vec![1.; rows])),
            rows,
            columns,
            n_classes,
            min_leaf_weight,
            criterion,
            min_samples_split,
            min_samples_leaf,
            min_impurity_decrease,
            total_weight: rows as f64,
            histogram_bins: Some(max_bins),
            bin_edges: Some(Arc::new(edges)),
            active_features,
            balanced_subsample: false,
            ccp_alpha: 0.0,
            monotonic_constraints: None,
            has_missing_values: false,
            histogram_threads: 1,
            extra_trees: false,
            column_major: true,
            extra_seed: 6,
            profile: None,
            integral_weights: false,
            count_weights: None,
        })
    }

    /// Private integration storage over borrowed column slices; rows are
    /// already in training order and every sample weight is one.
    #[allow(clippy::too_many_arguments)]
    pub fn training_exact_columns(
        values: ExactColumnTable, labels: Vec<usize>, n_classes: usize,
        min_leaf_weight: f64, criterion: Criterion, min_samples_split: usize,
        min_samples_leaf: usize, min_impurity_decrease: f64,
    ) -> Result<Self, String> {
        let rows = values.rows;
        let columns = values.offsets.len();
        Self::training_exact_storage(ExactValues::Columns(values), rows, columns,
            labels, vec![1.; rows], n_classes, min_leaf_weight, criterion,
            min_samples_split, min_samples_leaf, min_impurity_decrease, false)
    }
    /// Private integration storage: no real/shadow augmented matrix is allocated.
    #[allow(clippy::too_many_arguments)]
    pub fn training_exact_virtual(
        values: ExactVirtualColumns, labels: Vec<usize>, n_classes: usize,
        min_leaf_weight: f64, criterion: Criterion, min_samples_split: usize,
        min_samples_leaf: usize, min_impurity_decrease: f64,
    ) -> Result<Self, String> {
        let rows = values.bank.rows;
        let columns = values.columns.len();
        Self::training_exact_storage(ExactValues::Virtual(values), rows, columns,
            labels, vec![1.; rows], n_classes, min_leaf_weight, criterion,
            min_samples_split, min_samples_leaf, min_impurity_decrease, false)
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
        Self::training_binned_with_accessor_and_seed(
            rows, columns, labels, sample_weights, n_classes, min_leaf_weight,
            criterion, min_samples_split, min_samples_leaf, min_impurity_decrease,
            max_bins, threads, strategy, sample_size, DEFAULT_BINNING_SEED, value_at,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn training_binned_with_accessor_and_seed<F>(
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
        binning_seed: u64,
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
            binning_seed,
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
            extra_trees: false,
            column_major: false,
            extra_seed: 6,
            profile: None,
            integral_weights: false,
            count_weights: None,
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
            None => DenseValues::Exact(Arc::new(ExactValues::F64(values))),
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
            extra_trees: false,
            column_major: false,
            extra_seed: 6,
            profile: None,
            integral_weights: false,
            count_weights: None,
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
        Self::training_csr_with_binning_seed_options(
            indptr, indices, data, rows, columns, labels, sample_weights, n_classes,
            min_leaf_weight, criterion, min_samples_split, min_samples_leaf,
            min_impurity_decrease, max_bins, threads, binning_strategy, bin_sample_size,
            DEFAULT_BINNING_SEED,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn training_csr_with_binning_seed_options(
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
        binning_seed: u64,
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
                binning_seed,
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
            extra_trees: false,
            column_major: false,
            extra_seed: 6,
            profile: None,
            integral_weights: false,
            count_weights: None,
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
            extra_trees: false,
            column_major: false,
            extra_seed: 6,
            profile: None,
            integral_weights: false,
            count_weights: None,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn bin_edges(&self) -> Option<&[Vec<f64>]> {
        self.bin_edges.as_ref().map(|edges| edges.as_slice())
    }

    pub fn with_float32_column_storage(mut self, threads: usize) -> Self {
        if let DenseValues::Exact(values) = &self.values {
            if let ExactValues::F64(values) = values.as_ref() {
                let converted = transpose_exact_map(values, self.rows, self.columns, threads, |v| v as f32);
                self.values = DenseValues::Exact(Arc::new(ExactValues::F32(converted)));
                self.column_major = true;
            }
        }
        self
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

    pub fn with_profiling(mut self, profile: Option<Arc<FitProfile>>) -> Self {
        self.profile = profile;
        self
    }

    pub fn with_extra_trees(mut self, enabled: bool, seed: u64) -> Self {
        self.extra_trees = enabled;
        self.extra_seed = seed;
        // Exact Extra Trees scan one feature at a time. Store its original values
        // contiguously; no quantization or change to threshold sampling.
        if enabled && !self.column_major {
            if let DenseValues::Exact(values) = &self.values {
                let transposed = match values.as_ref() {
                    ExactValues::F32(v) => ExactValues::F32(transpose_exact(v, self.rows, self.columns, self.histogram_threads)),
                    ExactValues::F64(v) => ExactValues::F64(transpose_exact(v, self.rows, self.columns, self.histogram_threads)),
                    ExactValues::Virtual(v) => ExactValues::Virtual(v.clone()),
                    ExactValues::Columns(v) => ExactValues::Columns(v.clone()),
                };
                self.values = DenseValues::Exact(Arc::new(transposed));
                self.column_major = true;
            }
        }
        self
    }

    /// Dense binned Extra Trees only: keep RF histogram storage row-major.
    pub fn with_binned_extra_trees(mut self, threads: usize) -> Self {
        if self.extra_trees {
            if let DenseValues::Binned(values) = &self.values {
                let _timer = ProfileTimer(self.profile.as_ref().map(|p| (&p.layout_ns, Instant::now())));
                let mut transposed = vec![0u8; self.rows * self.columns];
                let columns_per_worker = self.columns.div_ceil(threads.max(1)).max(1);
                let rows = self.rows; let columns = self.columns;
                thread::scope(|scope| {
                    for (block, dst) in transposed.chunks_mut(columns_per_worker * rows).enumerate() {
                        let values = values.as_slice();
                        scope.spawn(move || {
                            for (local_column, column_dst) in dst.chunks_mut(rows).enumerate() {
                                let column = block * columns_per_worker + local_column;
                                for (row, slot) in column_dst.iter_mut().enumerate() { *slot = values[row * columns + column]; }
                            }
                        });
                    }
                });
                self.values = DenseValues::Binned(Arc::new(transposed));
                self.column_major = true;
            }
        }
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

    fn balanced_tree_view(&self, bag: &[usize]) -> Self {
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
        if let Some(counts) = &self.count_weights { return f64::from(counts[row]); }
        self.sample_weights
            .as_deref()
            .expect("prediction input cannot be used for training")[row]
    }

    #[inline]
    fn bin_index(&self, row: usize, column: usize) -> usize {
        if self.column_major { column * self.rows + row } else { row * self.columns + column }
    }

    #[inline]
    fn storage_index(&self, row: usize, column: usize) -> usize {
        if self.column_major { column * self.rows + row } else { row * self.columns + column }
    }

    #[inline]
    fn exact_value(&self, values: &ExactValues, row: usize, column: usize) -> f64 {
        match values {
            ExactValues::Virtual(v) => f64::from(v.value_at(row, column)),
            ExactValues::Columns(v) => f64::from(v.value_at(row, column)),
            _ => values.at(self.storage_index(row, column)),
        }
    }

    fn value(&self, row: usize, column: usize) -> f64 {
        match &self.values {
            DenseValues::Exact(values) => self.exact_value(values, row, column),
            DenseValues::Binned(values) => {
                let bin = values[self.bin_index(row, column)];
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
                DenseValues::Exact(values) => self.exact_value(values, row, column) > pivot,
                DenseValues::Binned(values) => (values[self.bin_index(row, column)] as f64) > pivot,
                DenseValues::Sparse(values) => {
                    let value = values.get(row, column);
                    let value = self.bin_edges.as_ref().map_or(value, |edges| edges[column].partition_point(|edge| value > *edge) as f64);
                    value > pivot
                }
            };
        }
        match &self.values {
            DenseValues::Exact(values) => {
                let value = self.exact_value(values, row, column);
                if value.is_nan() { missing_left } else { value > pivot }
            }
            DenseValues::Binned(values) => {
                let bin = values[self.bin_index(row, column)];
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
            DenseValues::Binned(values) => values[self.bin_index(row, column)] as usize,
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
    binning_seed: u64,
) -> Vec<Vec<f64>> {
    let preprocessing_threads = if rows.saturating_mul(columns) >= HISTOGRAM_PARALLEL_MIN_CELLS {
        threads
    } else {
        1
    };
    let sampled_rows = (strategy.samples() && sample_size < rows)
        .then(|| deterministic_sample_rows_with_seed(rows, sample_size, binning_seed));
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

fn deterministic_sample_rows_with_seed(
    rows: usize,
    sample_size: usize,
    seed: u64,
) -> Vec<usize> {
    let count = rows.min(sample_size);
    let mut indices = (0..rows).collect::<Vec<_>>();
    // Seed 6 retains the historical sampled-binning row order.
    let mut state = LEGACY_BINNING_SAMPLE_STATE ^ seed ^ DEFAULT_BINNING_SEED;
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

fn validate_matrix<T: Copy + Into<f64>>(values: &[T], rows: usize, columns: usize) -> Result<(), String> {
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
    if values.iter().any(|&value| value.into().is_infinite()) {
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
    binning_seed: u64,
) -> (Vec<u8>, Vec<Vec<f64>>) {
    let value_at = |row, feature| values[row * columns + feature];
    let (binned, edges, _) = histogramize_from_accessor(
        rows,
        columns,
        max_bins,
        threads,
        strategy,
        sample_size,
        binning_seed,
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
    binning_seed: u64,
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
        .then(|| deterministic_sample_rows_with_seed(rows, sample_size, binning_seed));
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

fn build_histograms(input: &DenseInput, mask: &[usize]) -> HistogramCache {
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
    input: &DenseInput, mask: &[usize], values: &[u8], histogram: &mut HistogramCache,
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
    if input.column_major {
        // Each bin still receives its rows in mask order, so sums match the
        // row-major layout bit for bit.
        for (feature, &start) in histogram.bin_offsets[..input.columns].iter().enumerate() {
            let column = &values[feature * input.rows..(feature + 1) * input.rows];
            for &row in mask.iter() {
                let index = start + column[row] as usize;
                histogram.class_weights[index * classes + labels[row]] += input.sample_weight(row);
                histogram.sample_counts[index] += 1;
            }
        }
        return;
    }
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
    mask: &[usize],
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
    let rows = input.rows;
    let column_major = input.column_major;
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
            debug_assert!(input.count_weights.is_none(), "count-weighted views are Extra Trees only");
            let input_weights = input.sample_weights.as_deref();

            scope.spawn(move || {
                if column_major {
                    for (local, &feature_start) in feature_offsets.iter().enumerate() {
                        let feature = start_feature + local;
                        let column = &values[feature * rows..(feature + 1) * rows];
                        for &row in mask.iter() {
                            let local_bin = feature_start - start_bin + column[row] as usize;
                            worker_weights[local_bin * n_classes + labels[row]] += input_weights.map_or(1.0, |weights| weights[row]);
                            worker_counts[local_bin] += 1;
                        }
                    }
                    return;
                }
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

/// Integral binary node: per-row weights and the positive-class share.
struct BinaryCounts {
    weights: Vec<u32>,
    positive: Vec<u32>,
}

pub struct DenseDecisionSlice {
    labels: Vec<usize>,
    exact_binary_weights: Option<Vec<f64>>,
    binary_counts: Option<BinaryCounts>,
    class_weights: Vec<f64>,
    total_weight: f64,
    criterion: Criterion,
}

impl DenseDecisionSlice {
    fn new(input: &DenseInput, mask: &[usize]) -> Self {
        if input.integral_weights && input.extra_trees && input.n_classes == 2 && !input.has_missing_values
            && matches!(input.values, DenseValues::Exact(_)) {
            return Self::new_binary_counts(input, mask);
        }
        let mut class_weights = vec![0.0; input.n_classes];
        let mut total_weight = 0.0;
        let mut exact_binary_weights = (input.extra_trees && input.n_classes == 2 && matches!(input.values, DenseValues::Exact(_)))
            .then(|| Vec::with_capacity(mask.len()));
        let labels = mask
            .iter()
            .map(|&row| {
                let label = input.labels()[row];
                let weight = input.sample_weight(row);
                if let Some(weights) = &mut exact_binary_weights { weights.push(weight); }
                class_weights[label] += weight;
                total_weight += weight;
                label
            })
            .collect();

        Self {
            labels,
            exact_binary_weights,
            binary_counts: None,
            class_weights,
            total_weight,
            criterion: input.criterion,
        }
    }

    /// Whole-count weights sum exactly as integers, so the f64 class and total
    /// weights equal the row-order f64 sums of the general constructor.
    fn new_binary_counts(input: &DenseInput, mask: &[usize]) -> Self {
        let labels = input.labels();
        let mut weights = Vec::with_capacity(mask.len());
        let mut positive = Vec::with_capacity(mask.len());
        let (mut total, mut ones) = (0u64, 0u64);
        for &row in mask.iter() {
            let weight = input.sample_weight(row) as u32;
            let share = if labels[row] == 0 { 0 } else { weight };
            weights.push(weight);
            positive.push(share);
            total += u64::from(weight);
            ones += u64::from(share);
        }
        Self {
            labels: Vec::new(),
            exact_binary_weights: None,
            binary_counts: Some(BinaryCounts { weights, positive }),
            class_weights: vec![(total - ones) as f64, ones as f64],
            total_weight: total as f64,
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
    mask: &[usize],
    feature: usize,
    target: &DenseDecisionSlice,
    split_cache: &DenseSplitCache,
    monotonic_constraint: i8,
    lower_bound: f64,
    upper_bound: f64,
    rng: &mut RfRng,
) -> Option<(f64, bool, f64)> {
    if input.extra_trees && split_cache.constant(feature) { return None; }
    let _timer = ProfileTimer(input.profile.as_ref().map(|p| (&p.split_ns, Instant::now())));
    if let Some(p) = &input.profile { p.feature_visits.fetch_add(1, AtomicOrdering::Relaxed); }
    // Extra Trees only needs one random threshold for the selected feature.
    // In histogram mode, building a complete cache for every feature at every
    // node is strictly more work than scanning the node for the sampled
    // feature. Keep this branch ahead of the histogram path so the specialized
    // implementation is also used for binned dense inputs.
    let best = if input.extra_trees {
        best_random_split(
            input, mask, feature, target, monotonic_constraint, lower_bound, upper_bound, rng, split_cache,
        )
    } else if input.histogram_bins.is_some() {
        best_split_histogram(
            input,
            target,
            &split_cache.0.as_ref()?.feature(feature, input.n_classes),
            monotonic_constraint,
            lower_bound,
            upper_bound,
            rng,
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
            rng,
        )
    }?;
    Some((best.0, best.1, best.2 * target.total_weight / input.total_weight))
}

fn best_split_exact(
    input: &DenseInput,
    mask: &[usize],
    feature: usize,
    target: &DenseDecisionSlice,
    monotonic_constraint: i8,
    lower_bound: f64,
    upper_bound: f64,
    rng: &mut RfRng,
) -> Option<(f64, bool, f64)> {
    let mut ranked: Vec<(f64, usize, f64)> = match &input.values {
        DenseValues::Exact(values) => mask
            .iter()
            .zip(&target.labels)
            .map(|(&row, &label)| {
                (
                    input.exact_value(values, row, feature),
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
                    values[input.bin_index(row, feature)] as f64,
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

    // Sparse inputs retain the original sorted Extra Trees implementation.
    let random_threshold = if input.extra_trees {
        let min = ranked.first()?.0;
        let max = ranked.last()?.0;
        if min >= max { return None; }
        let unit = ((rng.get_u64() ^ input.extra_seed.wrapping_mul(0x9e37_79b9_7f4a_7c15)) >> 11) as f64
            * (1.0 / ((1u64 << 53) as f64));
        Some(min * (1.0 - unit) + max * unit)
    } else { None };

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
        if random_threshold.is_some_and(|threshold| !(value <= threshold && threshold < next)) {
            continue;
        }
        let midpoint = value * 0.5 + next * 0.5;
        let pivot = random_threshold.unwrap_or_else(|| if midpoint >= next { value } else { midpoint });
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

/// Extra Trees only needs one random threshold, so avoid constructing and sorting
/// the per-feature sample list used by the exact CART search. This deliberately
/// also accepts binned values: the bin ids are the values used by split_iter,
/// which avoids materializing histogram statistics for all features.
fn best_random_split(
    input: &DenseInput,
    mask: &[usize],
    feature: usize,
    target: &DenseDecisionSlice,
    monotonic_constraint: i8,
    lower_bound: f64,
    upper_bound: f64,
    rng: &mut RfRng,
    split_cache: &DenseSplitCache,
) -> Option<(f64, bool, f64)> {
    if input.n_classes == 2 && matches!(input.values, DenseValues::Binned(_)) {
        return best_random_split_binned_binary(input, mask, feature, target, monotonic_constraint, lower_bound, upper_bound, rng, split_cache);
    }
    if input.n_classes == 2 && matches!(input.values, DenseValues::Exact(_)) {
        return best_random_split_binary(
            input, mask, feature, target, monotonic_constraint, lower_bound, upper_bound, rng, split_cache,
        );
    }
    best_random_split_generic(input, mask, feature, target, monotonic_constraint, lower_bound, upper_bound, rng, split_cache)
}

fn best_random_split_generic(
    input: &DenseInput, mask: &[usize], feature: usize, target: &DenseDecisionSlice,
    monotonic_constraint: i8, lower_bound: f64, upper_bound: f64, rng: &mut RfRng,
    split_cache: &DenseSplitCache,
) -> Option<(f64, bool, f64)> {
    let value_at = |row: usize| match &input.values {
        DenseValues::Exact(values) => input.exact_value(values, row, feature),
        DenseValues::Binned(values) => {
            let bin = values[input.bin_index(row, feature)] as usize;
            let missing_bin = input
                .bin_edges
                .as_ref()
                .map_or(usize::MAX, |edges| edges[feature].len() + 1);
            if input.has_missing_values && bin == missing_bin {
                f64::NAN
            } else {
                bin as f64
            }
        }
        DenseValues::Sparse(values) => values.get(row, feature),
    };
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut missing_classes = vec![0.0; input.n_classes];
    let mut missing_count = 0usize;
    let mut missing_weight = 0.0;
    for (&row, &label) in mask.iter().zip(&target.labels) {
        let value = value_at(row);
        if value.is_nan() {
            missing_count += 1;
            let weight = input.sample_weight(row);
            missing_weight += weight;
            missing_classes[label] += weight;
        } else {
            min = min.min(value);
            max = max.max(value);
        }
    }
    if min >= max { split_cache.mark_constant(feature); return None; }
    let unit = ((rng.get_u64() ^ input.extra_seed.wrapping_mul(0x9e37_79b9_7f4a_7c15)) >> 11) as f64
        * (1.0 / ((1u64 << 53) as f64));
    let threshold = min * (1.0 - unit) + max * unit;

    let mut low_classes = vec![0.0; input.n_classes];
    let mut low_count = 0usize;
    let mut low_weight = 0.0;
    for (&row, &label) in mask.iter().zip(&target.labels) {
        let value = value_at(row);
        if !value.is_nan() && value <= threshold {
            low_count += 1;
            let weight = input.sample_weight(row);
            low_weight += weight;
            low_classes[label] += weight;
        }
    }
    let finite_count = mask.len() - missing_count;
    let high_count = finite_count - low_count;
    if low_count == 0 || high_count == 0 { return None; }
    let total_weight = target.total_weight;
    let parent = impurity(input.criterion, &target.class_weights, total_weight);
    let mut best = None;
    // `missing_left` means NaNs route with values above the threshold.
    for missing_to_low in [false, true] {
        let left_count = low_count + if missing_to_low { missing_count } else { 0 };
        let right_count = high_count + if missing_to_low { 0 } else { missing_count };
        let left_weight = low_weight + if missing_to_low { missing_weight } else { 0.0 };
        let right_weight = total_weight - left_weight;
        if left_count < input.min_samples_leaf || right_count < input.min_samples_leaf
            || left_weight < input.min_leaf_weight || right_weight < input.min_leaf_weight
            || left_weight <= 0.0 || right_weight <= 0.0 { continue; }
        let mut left = low_classes.clone();
        let mut right = target.class_weights.clone();
        for c in 0..input.n_classes { right[c] -= low_classes[c]; }
        if missing_to_low {
            for c in 0..input.n_classes { left[c] += missing_classes[c]; right[c] -= missing_classes[c]; }
        }
        if monotonic_constraint != 0 {
            let lp = left[1] / left_weight;
            let rp = right[1] / right_weight;
            if lp < lower_bound || lp > upper_bound || rp < lower_bound || rp > upper_bound
                || (lp - rp) * monotonic_constraint as f64 > 0.0 { continue; }
        }
        let gain = parent - (left_weight / total_weight) * impurity(input.criterion, &left, left_weight)
            - (right_weight / total_weight) * impurity(input.criterion, &right, right_weight);
        if gain >= input.min_impurity_decrease && best.as_ref().is_none_or(|(_, _, g)| gain > *g) {
            best = Some((threshold, !missing_to_low, gain));
        }
    }
    best
}

fn best_random_split_binary(
    input: &DenseInput,
    mask: &[usize],
    feature: usize,
    target: &DenseDecisionSlice,
    monotonic_constraint: i8,
    lower_bound: f64,
    upper_bound: f64,
    rng: &mut RfRng,
    split_cache: &DenseSplitCache,
) -> Option<(f64, bool, f64)> {
    let DenseValues::Exact(values) = &input.values else { unreachable!() };
    macro_rules! search {
        ($values:expr, $workspace:ident) => {
            $workspace.with(|workspace| {
                let mut workspace = workspace.borrow_mut();
                workspace.clear();
                if input.has_missing_values {
                    best_random_split_binary_buffered::<_, true>(input, mask, feature, target, monotonic_constraint, lower_bound, upper_bound, rng, $values, &mut workspace, split_cache)
                } else {
                    best_random_split_binary_buffered::<_, false>(input, mask, feature, target, monotonic_constraint, lower_bound, upper_bound, rng, $values, &mut workspace, split_cache)
                }
            })
        };
    }
    match values.as_ref() {
        ExactValues::F32(values) => search!(|row| values[input.storage_index(row, feature)], EXTRA_FEATURE_VALUES_F32),
        ExactValues::F64(values) => search!(|row| values[input.storage_index(row, feature)], EXTRA_FEATURE_VALUES),
        ExactValues::Columns(values) => {
            let column = values.column(feature);
            if let (false, Some(counts)) = (input.has_missing_values, &target.binary_counts) {
                return EXTRA_FEATURE_VALUES_F32.with(|workspace| best_random_split_binary_counts_f32(
                    input, mask, feature, target, counts, monotonic_constraint, lower_bound, upper_bound,
                    rng, column, &mut workspace.borrow_mut(), split_cache));
            }
            search!(|row| column[row], EXTRA_FEATURE_VALUES_F32)
        },
        ExactValues::Virtual(values) => {
            // Resolve dtype, source column and permutation width once per feature,
            // so the sample gather only follows typed slices.
            let rows = values.bank.rows;
            let order = values.order.as_slice();
            let source = values.columns[feature];
            macro_rules! gather {
                ($bank:expr) => {{
                    let column = &$bank[source * rows..(source + 1) * rows];
                    if feature < values.real_columns {
                        search!(|row| column[order[row]] as f32, EXTRA_FEATURE_VALUES_F32)
                    } else {
                        let offset = (feature - values.real_columns) * rows;
                        match values.permutations.as_ref() {
                            ExactRowIndices::U32(p) => {
                                let permutation = &p[offset..offset + rows];
                                search!(|row| column[permutation[order[row]] as usize] as f32, EXTRA_FEATURE_VALUES_F32)
                            },
                            ExactRowIndices::Usize(p) => {
                                let permutation = &p[offset..offset + rows];
                                search!(|row| column[permutation[order[row]]] as f32, EXTRA_FEATURE_VALUES_F32)
                            },
                        }
                    }
                }};
            }
            match values.bank.values.as_ref() {
                ExactValues::F32(bank) => gather!(bank),
                ExactValues::F64(bank) => gather!(bank),
                ExactValues::Virtual(_) | ExactValues::Columns(_) => unreachable!("bank cannot contain virtual storage"),
            }
        },
    }
}

/// `best_random_split_binary_buffered` for f32 columns without missing values
/// and whole-count weights, with the same decisions. f32 to f64 is exact and
/// monotonic, so the f32 range converts to the f64 range, and `value <= threshold`
/// holds exactly when `value <= below`, the largest f32 not above the threshold.
#[allow(clippy::too_many_arguments)]
fn best_random_split_binary_counts_f32(
    input: &DenseInput, mask: &[usize], feature: usize, target: &DenseDecisionSlice, counts: &BinaryCounts,
    monotonic_constraint: i8, lower_bound: f64, upper_bound: f64, rng: &mut RfRng,
    column: &[f32], workspace: &mut Vec<f32>, split_cache: &DenseSplitCache,
) -> Option<(f64, bool, f64)> {
    workspace.clear();
    workspace.extend(mask.iter().map(|&row| column[row]));
    let (min, max) = workspace.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &value| (min.min(value), max.max(value)));
    if let Some(p) = &input.profile {
        p.sample_visits.fetch_add(mask.len() as u64, AtomicOrdering::Relaxed);
        p.buffer_bytes.fetch_max((workspace.capacity() * std::mem::size_of::<f32>()) as u64, AtomicOrdering::Relaxed);
    }
    let (min, max) = (f64::from(min), f64::from(max));
    if min >= max {
        split_cache.mark_constant(feature);
        if let Some(p) = &input.profile { p.rejected.fetch_add(1, AtomicOrdering::Relaxed); }
        return None;
    }
    let unit = ((rng.get_u64() ^ input.extra_seed.wrapping_mul(0x9e37_79b9_7f4a_7c15)) >> 11) as f64
        * (1.0 / ((1u64 << 53) as f64));
    let threshold = min * (1.0 - unit) + max * unit;
    let rounded = threshold as f32;
    let below = if f64::from(rounded) > threshold { rounded.next_down() } else { rounded };
    let (mut low_count, mut low, mut ones) = (0usize, 0u64, 0u64);
    for ((&weight, &share), &value) in counts.weights.iter().zip(&counts.positive).zip(workspace.iter()) {
        let is_low = value <= below;
        low_count += usize::from(is_low);
        low += if is_low { u64::from(weight) } else { 0 };
        ones += if is_low { u64::from(share) } else { 0 };
    }
    binary_split_gain(input, target, mask.len(), threshold, (low_count, low as f64, (low - ones) as f64, ones as f64),
        (0, 0.0, 0.0, 0.0), monotonic_constraint, lower_bound, upper_bound)
}

#[allow(clippy::too_many_arguments)]
fn best_random_split_binary_buffered<T: Copy + Into<f64>, const MISSING: bool>(
    input: &DenseInput, mask: &[usize], feature: usize, target: &DenseDecisionSlice,
    monotonic_constraint: i8, lower_bound: f64, upper_bound: f64, rng: &mut RfRng,
    values: impl Fn(usize) -> T, workspace: &mut Vec<T>,
    split_cache: &DenseSplitCache,
) -> Option<(f64, bool, f64)> {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut missing_count = 0usize;
    let mut missing_weight = 0.0;
    let mut missing_class_0 = 0.0;
    let mut missing_class_1 = 0.0;
    workspace.reserve(mask.len());
    if MISSING {
        let weights = target.exact_binary_weights.as_ref().expect("exact binary node caches its weights");
        for ((&row, &label), &weight) in mask.iter().zip(&target.labels).zip(weights) {
            let stored = values(row);
            workspace.push(stored);
            let value: f64 = stored.into();
            if value.is_nan() {
                missing_count += 1;
                missing_weight += weight;
                if label == 0 { missing_class_0 += weight; } else { missing_class_1 += weight; }
            } else {
                min = min.min(value);
                max = max.max(value);
            }
        }
    } else {
        for &row in mask.iter() {
            let stored = values(row);
            workspace.push(stored);
            let value: f64 = stored.into();
            min = min.min(value);
            max = max.max(value);
        }
    }
    if let Some(p) = &input.profile {
        p.sample_visits.fetch_add(mask.len() as u64, AtomicOrdering::Relaxed);
        p.buffer_bytes.fetch_max((workspace.capacity() * std::mem::size_of::<T>()) as u64, AtomicOrdering::Relaxed);
    }
    if min >= max {
        split_cache.mark_constant(feature);
        if let Some(p) = &input.profile { p.rejected.fetch_add(1, AtomicOrdering::Relaxed); }
        return None;
    }
    let unit = ((rng.get_u64() ^ input.extra_seed.wrapping_mul(0x9e37_79b9_7f4a_7c15)) >> 11) as f64
        * (1.0 / ((1u64 << 53) as f64));
    let threshold = min * (1.0 - unit) + max * unit;

    let mut low_count = 0usize;
    let mut low_weight = 0.0;
    let mut low_class_0 = 0.0;
    let mut low_class_1 = 0.0;
    if let (false, Some(counts)) = (MISSING, &target.binary_counts) {
        // Integer sums of whole-count weights are exact, so they equal the f64
        // row-order sums below bit for bit once converted.
        let (mut low, mut ones) = (0u64, 0u64);
        for ((&weight, &share), &stored) in counts.weights.iter().zip(&counts.positive).zip(workspace.iter()) {
            let value: f64 = stored.into();
            let is_low = value <= threshold;
            low_count += usize::from(is_low);
            low += if is_low { u64::from(weight) } else { 0 };
            ones += if is_low { u64::from(share) } else { 0 };
        }
        low_weight = low as f64;
        low_class_0 = (low - ones) as f64;
        low_class_1 = ones as f64;
    } else if MISSING {
        let weights = target.exact_binary_weights.as_ref().expect("exact binary node caches its weights");
        for ((&weight, &label), &stored) in weights.iter().zip(&target.labels).zip(workspace.iter()) {
            let value: f64 = stored.into();
            if !value.is_nan() && value <= threshold {
                low_count += 1;
                low_weight += weight;
                if label == 0 { low_class_0 += weight; } else { low_class_1 += weight; }
            }
        }
    } else {
        // A uniform random threshold makes both comparisons unpredictable, so
        // select instead of branching. Adding 0.0 leaves every accumulator
        // bit-identical to the branching sums, in the same order.
        let weights = target.exact_binary_weights.as_ref().expect("exact binary node caches its weights");
        for ((&weight, &label), &stored) in weights.iter().zip(&target.labels).zip(workspace.iter()) {
            let value: f64 = stored.into();
            let low = value <= threshold;
            let low_weight_here = if low { weight } else { 0.0 };
            low_count += usize::from(low);
            low_weight += low_weight_here;
            low_class_0 += if label == 0 { low_weight_here } else { 0.0 };
            low_class_1 += if label == 0 { 0.0 } else { low_weight_here };
        }
    }
    binary_split_gain(input, target, mask.len(), threshold, (low_count, low_weight, low_class_0, low_class_1),
        (missing_count, missing_weight, missing_class_0, missing_class_1), monotonic_constraint, lower_bound, upper_bound)
}

/// Best side for the missing rows of a binary threshold split and its gain,
/// from the low side's (count, weight, class 0, class 1) and the missing rows'.
#[allow(clippy::too_many_arguments)]
fn binary_split_gain(
    input: &DenseInput, target: &DenseDecisionSlice, rows: usize, threshold: f64,
    (low_count, low_weight, low_class_0, low_class_1): (usize, f64, f64, f64),
    (missing_count, missing_weight, missing_class_0, missing_class_1): (usize, f64, f64, f64),
    monotonic_constraint: i8, lower_bound: f64, upper_bound: f64,
) -> Option<(f64, bool, f64)> {
    if let Some(p) = &input.profile { p.sample_visits.fetch_add(rows as u64, AtomicOrdering::Relaxed); }
    let high_count = rows - missing_count - low_count;
    if low_count == 0 || high_count == 0 { return None; }
    let total_weight = target.total_weight;
    let parent_impurity = impurity(input.criterion, &target.class_weights, total_weight);
    let mut best: Option<(f64, bool, f64)> = None;
    for missing_to_low in [false, true] {
        let left_count = low_count + usize::from(missing_to_low) * missing_count;
        let right_count = high_count + usize::from(!missing_to_low) * missing_count;
        let left_weight = low_weight + if missing_to_low { missing_weight } else { 0.0 };
        let right_weight = total_weight - left_weight;
        if left_count < input.min_samples_leaf || right_count < input.min_samples_leaf
            || left_weight < input.min_leaf_weight || right_weight < input.min_leaf_weight
            || left_weight <= 0.0 || right_weight <= 0.0 { continue; }
        let left_classes = if missing_to_low {
            [low_class_0 + missing_class_0, low_class_1 + missing_class_1]
        } else { [low_class_0, low_class_1] };
        let right_classes = [
            target.class_weights[0] - left_classes[0],
            target.class_weights[1] - left_classes[1],
        ];
        if monotonic_constraint != 0 {
            let lp = left_classes[1] / left_weight;
            let rp = right_classes[1] / right_weight;
            if lp < lower_bound || lp > upper_bound || rp < lower_bound || rp > upper_bound
                || (lp - rp) * monotonic_constraint as f64 > 0.0 { continue; }
        }
        let child_impurity = (left_weight / total_weight)
            * impurity(input.criterion, &left_classes, left_weight)
            + (right_weight / total_weight)
                * impurity(input.criterion, &right_classes, right_weight);
        let gain = parent_impurity - child_impurity;
        if gain >= input.min_impurity_decrease && best.as_ref().is_none_or(|(_, _, g)| gain > *g) {
            best = Some((threshold, !missing_to_low, gain));
        }
    }
    if best.is_none() { if let Some(p) = &input.profile { p.rejected.fetch_add(1, AtomicOrdering::Relaxed); } }
    best
}

/// Experimental one-pass statistics for a visited binned feature, never a global cache.
fn best_random_split_binned_binary(
    input: &DenseInput, mask: &[usize], feature: usize, target: &DenseDecisionSlice,
    monotonic_constraint: i8, lower_bound: f64, upper_bound: f64, rng: &mut RfRng,
    split_cache: &DenseSplitCache,
) -> Option<(f64, bool, f64)> {
    let DenseValues::Binned(values) = &input.values else { unreachable!() };
    let missing_bin = input.bin_edges.as_ref().unwrap()[feature].len() + 1;
    let mut counts = [0usize; 256];
    let mut classes = [[0.0f64; 2]; 256];
    let mut min = usize::MAX;
    let mut max = 0usize;
    let mut missing_count = 0usize;
    let mut missing_weight = 0.0;
    let mut missing_class_0 = 0.0;
    let mut missing_class_1 = 0.0;
    for (&row, &label) in mask.iter().zip(&target.labels) {
        let bin = values[input.bin_index(row, feature)] as usize;
        let weight = input.sample_weight(row);
        if input.has_missing_values && bin == missing_bin {
            missing_count += 1;
            missing_weight += weight;
            if label == 0 { missing_class_0 += weight; } else { missing_class_1 += weight; }
        } else {
            counts[bin] += 1;
            classes[bin][label] += weight;
            min = min.min(bin); max = max.max(bin);
        }
    }
    if let Some(p) = &input.profile { p.sample_visits.fetch_add(mask.len() as u64, AtomicOrdering::Relaxed); }
    if min >= max {
        split_cache.mark_constant(feature);
        if let Some(p) = &input.profile { p.rejected.fetch_add(1, AtomicOrdering::Relaxed); }
        return None;
    }
    let unit = ((rng.get_u64() ^ input.extra_seed.wrapping_mul(0x9e37_79b9_7f4a_7c15)) >> 11) as f64
        * (1.0 / ((1u64 << 53) as f64));
    let threshold = min as f64 * (1.0 - unit) + max as f64 * unit;
    let mut low_count = 0usize;
    let mut low_class_0 = 0.0;
    let mut low_class_1 = 0.0;
    for bin in min..=threshold.floor() as usize {
        low_count += counts[bin]; low_class_0 += classes[bin][0]; low_class_1 += classes[bin][1];
    }
    let low_weight = low_class_0 + low_class_1;
    let high_count = mask.len() - missing_count - low_count;
    if low_count == 0 || high_count == 0 { return None; }
    let total_weight = target.total_weight;
    let parent_impurity = impurity(input.criterion, &target.class_weights, total_weight);
    let mut best: Option<(f64, bool, f64)> = None;
    for missing_to_low in [false, true] {
        let left_count = low_count + usize::from(missing_to_low) * missing_count;
        let right_count = high_count + usize::from(!missing_to_low) * missing_count;
        let left_weight = low_weight + if missing_to_low { missing_weight } else { 0.0 };
        let right_weight = total_weight - left_weight;
        if left_count < input.min_samples_leaf || right_count < input.min_samples_leaf
            || left_weight < input.min_leaf_weight || right_weight < input.min_leaf_weight
            || left_weight <= 0.0 || right_weight <= 0.0 { continue; }
        let left_classes = if missing_to_low {
            [low_class_0 + missing_class_0, low_class_1 + missing_class_1]
        } else { [low_class_0, low_class_1] };
        let right_classes = [
            target.class_weights[0] - left_classes[0],
            target.class_weights[1] - left_classes[1],
        ];
        if monotonic_constraint != 0 {
            let lp = left_classes[1] / left_weight;
            let rp = right_classes[1] / right_weight;
            if lp < lower_bound || lp > upper_bound || rp < lower_bound || rp > upper_bound
                || (lp - rp) * monotonic_constraint as f64 > 0.0 { continue; }
        }
        let child_impurity = (left_weight / total_weight)
            * impurity(input.criterion, &left_classes, left_weight)
            + (right_weight / total_weight)
                * impurity(input.criterion, &right_classes, right_weight);
        let gain = parent_impurity - child_impurity;
        if gain >= input.min_impurity_decrease && best.as_ref().is_none_or(|(_, _, g)| gain > *g) {
            best = Some((threshold, !missing_to_low, gain));
        }
    }
    if best.is_none() { if let Some(p) = &input.profile { p.rejected.fetch_add(1, AtomicOrdering::Relaxed); } }
    best
}


fn best_split_histogram(
    input: &DenseInput,
    target: &DenseDecisionSlice,
    histogram: &FeatureHistogram<'_>,
    monotonic_constraint: i8,
    lower_bound: f64,
    upper_bound: f64,
    rng: &mut RfRng,
) -> Option<(f64, bool, f64)> {
    if input.n_classes == 2 {
        best_split_histogram_classes::<2>(
            input,
            target,
            histogram,
            monotonic_constraint,
            lower_bound,
            upper_bound,
            rng,
        )
    } else {
        best_split_histogram_classes::<0>(
            input,
            target,
            histogram,
            monotonic_constraint,
            lower_bound,
            upper_bound,
            rng,
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
    rng: &mut RfRng,
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
    let random_bin = if input.extra_trees && split_bin_end > 0 {
        Some(((rng.get_u64() ^ input.extra_seed.wrapping_mul(0x9e37_79b9_7f4a_7c15)) % split_bin_end as u64) as usize)
    } else {
        None
    };
    for bin in 0..split_bin_end {
        for class in 0..classes {
            let weight = histogram.class_weights[bin * classes + class];
            left[class] += weight;
            right[class] -= weight;
            left_weight += weight;
        }
        left_count += histogram.sample_counts[bin];
        if random_bin.is_some_and(|selected| selected != bin) {
            continue;
        }
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
    fn new(input: &DenseInput, on: &[usize], n: usize) -> Self {
        Self {
            direct: vec![None; n],
            drops: HashMap::new(),
            n: on.len(),
            true_labels: input.labels().to_vec(),
        }
    }

    fn ingest(&mut self, permuted: Option<usize>, mask: &[usize], vote: &usize) {
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
    shuffled_features: Vec<usize>,
    remaining: usize,
    without_replacement: bool,
}

impl FeatureSampler<DenseInput> for UniformFeatureSampler {
    fn random_feature(&mut self, rng: &mut RfRng) -> usize {
        if self.without_replacement && self.remaining > 0 {
            let selected = rng.up_to(self.remaining);
            self.remaining -= 1;
            self.shuffled_features.swap(selected, self.remaining);
            self.shuffled_features[self.remaining]
        } else {
            self.active_features[rng.up_to(self.active_features.len())]
        }
    }

    fn reload(&mut self) { self.remaining = self.shuffled_features.len(); }

    fn reset(&mut self) {
        if self.without_replacement {
            self.shuffled_features.clone_from(&self.active_features);
            self.remaining = self.active_features.len();
        }
    }
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

    fn training_mask(&self) -> Mask {
        if self.extra_trees {
            Mask::from_vec((0..self.rows).filter(|&row| self.sample_weight(row) > 0.0).collect())
        } else { Mask::new_all(self.rows) }
    }

    fn ccp_alpha(&self) -> f64 {
        self.ccp_alpha
    }

    fn tree_input(&self, bag: &[usize]) -> Option<Self> {
        self.balanced_subsample
            .then(|| self.balanced_tree_view(bag))
    }

    fn bootstrap_input(&self, size: usize, keep_oob: bool, rng: &mut RfRng) -> Option<(Mask, Mask, Self)> {
        if !self.extra_trees { return None; }
        let _timer = ProfileTimer(self.profile.as_ref().map(|p| (&p.bootstrap_ns, Instant::now())));
        let weights = self.sample_weights.as_ref()?;
        let draw = |rng: &mut RfRng| (rng.get_u64() >> 11) as f64 / (1u64 << 53) as f64;
        if size <= u32::MAX as usize && weights.iter().all(|&weight| weight == 1.0) {
            // Unit weights make the cumulative sums the exact integers 1..=rows,
            // so the binary search below is floor(unit * rows): the same rows
            // without the cumulative vector or a cache-missing search per draw.
            let sum = self.rows as f64;
            let mut counts = vec![0u32; self.rows];
            for _ in 0..size {
                counts[((draw(rng) * sum) as usize).min(self.rows - 1)] += 1;
            }
            if !self.balanced_subsample {
                return Some(self.bootstrap_from_u32_counts(keep_oob, counts));
            }
            return Some(self.bootstrap_from_counts(size, keep_oob, |row| counts[row] as usize));
        }
        let mut cumulative = Vec::with_capacity(self.rows);
        let mut sum = 0.0;
        for &weight in weights.iter() { sum += weight; cumulative.push(sum); }
        let mut counts = vec![0usize; self.rows];
        for _ in 0..size {
            let unit = draw(rng);
            let row = cumulative.partition_point(|&value| value <= unit * sum).min(self.rows - 1);
            counts[row] += 1;
        }
        Some(self.bootstrap_from_counts(size, keep_oob, |row| counts[row]))
    }

    fn known_constant(&self, feature: usize, cache: &DenseSplitCache) -> bool { self.extra_trees && cache.constant(feature) }
    fn feature_search_continues(&self, visited: usize, nonconstant: usize, tries: usize, has_split: bool) -> bool {
        if self.extra_trees { visited < self.active_features.len() && (nonconstant < tries || !has_split) }
        else { visited < tries }
    }

    fn feature_count(&self) -> usize {
        self.columns
    }

    fn decision_slice(&self, mask: &[usize]) -> Self::DecisionSlice {
        DenseDecisionSlice::new(self, mask)
    }

    fn can_split(&self, mask: &[usize]) -> bool {
        mask.len() >= self.min_samples_split
            && (!self.extra_trees || mask.len() >= self.min_samples_leaf.saturating_mul(2))
            && !self.active_features.is_empty()
    }

    fn feature_sampler(&self) -> Self::FeatureSampler {
        UniformFeatureSampler {
            active_features: self.active_features.clone(),
            shuffled_features: self.active_features.clone(),
            remaining: self.active_features.len(),
            // Extra Trees samples candidate features without replacement at
            // each node, regardless of whether the input is exact or binned.
            // Once all active features have been visited, the sampler keeps
            // the existing fallback to replacement for tries > n_features.
            without_replacement: self.extra_trees,
        }
    }

    fn split_cache(&self, on: &[usize]) -> Self::SplitCache {
        DenseSplitCache::new(
            ((!self.extra_trees) && self.histogram_bins.is_some())
                .then(|| build_histograms(self, on)),
        )
    }

    fn split_cache_children(
        &self,
        parent_cache: Self::SplitCache,
        _: &[usize],
        left: &[usize],
        right: &[usize],
    ) -> (Self::SplitCache, Self::SplitCache) {
        let Some(parent_histograms) = parent_cache.0 else {
            let constants = parent_cache.1.into_inner();
            return (DenseSplitCache(None, RefCell::new(constants.clone())), DenseSplitCache(None, RefCell::new(constants)));
        };
        if left.len() <= right.len() {
            let left_histograms = build_histograms(self, left);
            let right_histograms = subtract_histograms(parent_histograms, &left_histograms);
            (
                DenseSplitCache::new(Some(left_histograms)),
                DenseSplitCache::new(Some(right_histograms)),
            )
        } else {
            let right_histograms = build_histograms(self, right);
            let left_histograms = subtract_histograms(parent_histograms, &right_histograms);
            (
                DenseSplitCache::new(Some(left_histograms)),
                DenseSplitCache::new(Some(right_histograms)),
            )
        }
    }

    fn new_split(
        &self,
        mask: &[usize],
        feature: Self::FeatureId,
        target: &Self::DecisionSlice,
        split_cache: &Self::SplitCache,
        rng: &mut RfRng,
    ) -> Option<(Self::Pivot, bool, f64)> {
        best_split(self, mask, feature, target, split_cache, 0, 0.0, 1.0, rng)
    }

    fn new_split_with_bounds(
        &self,
        mask: &[usize],
        feature: Self::FeatureId,
        target: &Self::DecisionSlice,
        split_cache: &Self::SplitCache,
        rng: &mut RfRng,
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
            rng,
        )
    }

    fn monotonic_constraint(&self, feature: Self::FeatureId) -> i8 {
        self.monotonic_constraints
            .as_ref()
            .map_or(0, |c| c[feature])
    }

    fn split_iter<'a>(
        &'a self,
        mask: &'a [usize],
        feature: Self::FeatureId,
        pivot: &'a Self::Pivot,
        missing_left: bool,
    ) -> impl Iterator<Item = bool> + 'a {
        let timer = ProfileTimer(self.profile.as_ref().map(|p| (&p.partition_ns, Instant::now())));
        let virtual_f32 = match &self.values {
            DenseValues::Exact(values) => match values.as_ref() {
                ExactValues::Virtual(v) => match v.bank.values.as_ref() {
                    ExactValues::F32(bank) => Some((v, bank)),
                    _ => None,
                },
                _ => None,
            },
            _ => None,
        };
        let columns = match &self.values {
            DenseValues::Exact(values) => match values.as_ref() { ExactValues::Columns(v) => Some(v), _ => None },
            _ => None,
        };
        let inner = if let Some(v) = columns {
            DenseSplitIter::F32Column { column: v.column(feature), rows: mask.iter(), pivot: *pivot }
        } else if let Some((v, bank)) = virtual_f32 {
            let source = v.columns[feature];
            let column = &bank[source * self.rows..(source + 1) * self.rows];
            if feature < v.real_columns {
                DenseSplitIter::VirtualF32Real { column, order: &v.order, rows: mask.iter(), pivot: *pivot }
            } else if let ExactRowIndices::U32(p) = v.permutations.as_ref() {
                let offset = (feature - v.real_columns) * self.rows;
                DenseSplitIter::VirtualF32Shadow { column, order: &v.order, permutation: &p[offset..offset+self.rows], rows: mask.iter(), pivot: *pivot }
            } else {
                DenseSplitIter::Lazy { input: self, rows: mask.iter(), feature, pivot: *pivot }
            }
        } else if self.labels.is_none() {
            let values = match &self.values {
                DenseValues::Exact(values) => {
                    if self.has_missing_values { mask.iter().map(|&row| { let value = values.at(self.storage_index(row, feature)); if value.is_nan() { missing_left } else { value > *pivot } }).collect::<Vec<_>>().into_iter() }
                    else { mask.iter().map(|&row| values.at(self.storage_index(row, feature)) > *pivot).collect::<Vec<_>>().into_iter() }
                }
                DenseValues::Binned(values) => {
                    if self.has_missing_values { mask.iter().map(|&row| { let bin = values[self.bin_index(row, feature)]; let missing_bin = self.bin_edges.as_ref().map_or(usize::MAX, |e| e[feature].len() + 1); if bin as usize == missing_bin { missing_left } else { (bin as f64) > *pivot } }).collect::<Vec<_>>().into_iter() }
                    else { mask.iter().map(|&row| (values[self.bin_index(row, feature)] as f64) > *pivot).collect::<Vec<_>>().into_iter() }
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
        };
        ProfiledSplitIter { inner, _timer: timer }
    }
}

#[cfg(test)]
mod histogram_optimization_tests {
    use super::*;

    #[test]
    fn exact_column_layout_preserves_rectangular_values_and_nan_for_both_dtypes() {
        for float32 in [false, true] {
            let raw = vec![1.0, f64::NAN, 3.0, 4.0, 5.0, 6.0];
            let input = DenseInput::training(raw.clone(), 2, 3, vec![0, 1],
                vec![1.0; 2], 2, 0.0, Criterion::Gini, 2, 1, 0.0, None).unwrap();
            let input = if float32 { input.with_float32_column_storage(2) } else { input };
            let input = input.with_histogram_threads(2).with_extra_trees(true, 6);
            assert!(input.column_major);
            let shared = input.clone();
            assert!(input.shares_feature_storage_with(&shared));
            for row in 0..2 {
                for col in 0..3 {
                    let expected = raw[row * 3 + col];
                    let actual = input.value_at(row, col);
                    assert!(actual == expected || (actual.is_nan() && expected.is_nan()));
                }
            }
            // Reapplying configuration must not transpose a second time.
            let again = input.clone().with_extra_trees(true, 6);
            assert!(input.shares_feature_storage_with(&again));
        }
    }

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
        let first = deterministic_sample_rows_with_seed(1000, 200, DEFAULT_BINNING_SEED);
        assert_eq!(
            first,
            deterministic_sample_rows_with_seed(1000, 200, DEFAULT_BINNING_SEED)
        );
        assert_eq!(first, deterministic_sample_rows_with_seed(1000, 200, 6));
        assert_ne!(first, deterministic_sample_rows_with_seed(1000, 200, 7));
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
    fn binned_one_pass_matches_single_candidate_statistics() {
        for criterion in [Criterion::Gini, Criterion::Entropy] {
            for with_nan in [false, true] {
                let rows = 512;
                let values = (0..rows).map(|i| if with_nan && i % 17 == 0 { f64::NAN } else { (i % 97) as f64 }).collect();
                let input = DenseInput::training(values, rows, 1, (0..rows).map(|i| i % 2).collect(),
                    (0..rows).map(|i| 0.5 + (i % 9) as f64 / 10.0).collect(), 2, 0.0, criterion, 2, 3, 0.0, Some(63)).unwrap().with_extra_trees(true, 6);
                let mask = Mask::new_all(rows);
                let target = input.decision_slice(&mask);
                for seed in 0..100 {
                    let generic = input.clone();
                    let optimized = input.clone();
                    let a = best_random_split_generic(&generic, &mask, 0, &target, 0, 0.0, 1.0, &mut RfRng::from_seed(seed,1), &generic.split_cache(&mask));
                    let b = best_random_split(&optimized, &mask, 0, &target, 0, 0.0, 1.0, &mut RfRng::from_seed(seed,1), &optimized.split_cache(&mask));
                    match (a,b) {
                        (Some((ta,_,ga)),Some((tb,_,gb))) => { assert_eq!(ta,tb); assert!((ga-gb).abs()<1e-12); },
                        (None,None) => (), _ => panic!("candidate viability differs"),
                    }
                }
            }
        }
    }

    #[test]
    fn extra_trees_histograms_use_direct_node_scans_without_a_global_cache() {
        let input = binary_input(Some(4)).with_extra_trees(true, 17);
        let mask = Mask::from_vec((0..8).collect());
        let cache = input.split_cache(&mask);

        assert!(cache.0.is_none());

        let target = input.decision_slice(&mask);
        let mut rng = RfRng::from_seed(31, 1);
        let split = input.new_split(&mask, 0, &target, &cache, &mut rng);
        assert!(
            split.is_some(),
            "the direct binned Extra Trees split must remain viable"
        );
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
    fn direct_exact_float32_accessor_preserves_values_and_validation() {
        let raw = [0.0f32, -0.0, f32::from_bits(1), f32::NAN, 4.0, 9.0];
        let build = |weights, infinite| DenseInput::training_exact_f32_with_accessor(
            3, 2, vec![0, 1, 1], weights, 2, 0.0, Criterion::Gini, 2, 1, 0.0, 2,
            |row, col| if infinite { f32::INFINITY } else { raw[row * 2 + col] },
        );
        let input = build(vec![1.0; 3], false).unwrap();
        assert!(input.column_major && input.has_missing_values);
        for row in 0..3 {
            for col in 0..2 {
                let expected = f64::from(raw[row * 2 + col]);
                assert_eq!(input.value_at(row, col).to_bits(), expected.to_bits());
            }
        }
        assert_eq!(build(vec![1.0; 3], true).err().as_deref(), Some("X must not contain infinite values"));
        assert_eq!(build(vec![f64::MAX; 3], false).err().as_deref(), Some("sample_weight sum must be finite"));
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

#[cfg(test)]
mod virtual_exact_tests {
    use super::*;
    use xrf::Forest;

    #[test]
    fn virtual_matches_materialized_trees_binary_and_multiclass() {
        let rows = 96;
        for bank in [
            ExactColumnBank::from_f64((0..rows * 3).map(|i| (i % 19) as f64 / 7.).collect(), rows, 3).unwrap(),
            ExactColumnBank::from_f32((0..rows * 3).map(|i| (i % 19) as f32 / 7.).collect(), rows, 3).unwrap(),
        ] {
            check_tree_parity(bank);
        }
    }

    fn check_tree_parity(bank: ExactColumnBank) {
        let rows = bank.rows();
        for classes in [2, 3] {
            for bootstrap in [false, true] {
                for criterion in [Criterion::Gini, Criterion::Entropy] {
                    let order = Arc::new((0..rows).rev().collect::<Vec<_>>());
                    let permutations = Arc::new(ExactRowIndices::U32((0..rows).map(|r| ((r + 7) % rows) as u32).collect()));
                    let virtuals = ExactVirtualColumns::new(bank.clone(), vec![0, 2], vec![1], order, permutations).unwrap();
                    let labels = (0..rows).map(|r| r % classes).collect::<Vec<_>>();
                    let materialized = DenseInput::training_exact_f32_with_accessor(rows, 3, labels.clone(), vec![1.; rows], classes, 0., criterion, 2, 1, 0., 2, |r,c| virtuals.at(c * rows + r)).unwrap().with_extra_trees(true, 6);
                    let indirect = DenseInput::training_exact_virtual(virtuals, labels, classes, 0., criterion, 2, 1, 0.).unwrap().with_extra_trees(true, 6);
                    let train = |input: &DenseInput| Forest::new_parallel_with_settings(input, 7, 2, true, false, false, 42, 2, 5, usize::MAX, bootstrap, None);
                    let a = train(&materialized);
                    let b = train(&indirect);
                    // XRF concatenates worker chunks in completion order. Compare
                    // the exact tree multiset, including leaf votes and covers.
                    let signatures = |forest: &Forest<DenseInput>| {
                        let mut trees = forest.tree_refs().map(|tree| {
                            tree.nodes.iter().map(|node| match node {
                                xrf::Node::Branch(f,t,g,l,r,m) => format!("B:{f}:{:x}:{:x}:{l}:{r}:{m}", t.to_bits(), g.to_bits()),
                                xrf::Node::Leaf(v,c) => format!("L:{v}:{c}"),
                            }).collect::<Vec<_>>().join(";")
                        }).collect::<Vec<_>>();
                        trees.sort(); trees
                    };
                    assert_eq!(signatures(&a), signatures(&b));
                }
            }
        }
    }

    #[test]
    fn column_table_matches_materialized_trees() {
        let rows = 96;
        let real = Arc::new((0..rows * 3).map(|i| ((i * 7) % 23) as f32 / 5.).collect::<Vec<_>>());
        let shadows = Arc::new((0..rows * 2).map(|i| ((i * 11) % 17) as f32 / 3.).collect::<Vec<_>>());
        for classes in [2, 3] {
            for bootstrap in [false, true] {
                let table = ExactColumnTable::new(rows, real.clone(), shadows.clone(), &[2, 0], 2).unwrap();
                let labels = (0..rows).map(|r| (r * 5 / 3) % classes).collect::<Vec<_>>();
                let materialized = DenseInput::training_exact_f32_with_accessor(rows, 4, labels.clone(), vec![1.; rows], classes, 0., Criterion::Gini, 2, 1, 0., 2, |r, c| table.value_at(r, c)).unwrap().with_extra_trees(true, 6);
                let direct = DenseInput::training_exact_columns(table, labels, classes, 0., Criterion::Gini, 2, 1, 0.).unwrap().with_extra_trees(true, 6);
                let train = |input: &DenseInput| Forest::new_parallel_with_settings(input, 7, 2, true, false, false, 42, 2, 5, usize::MAX, bootstrap, None);
                let signatures = |forest: &Forest<DenseInput>| {
                    let mut trees = forest.tree_refs().map(|tree| {
                        tree.nodes.iter().map(|node| match node {
                            xrf::Node::Branch(f,t,g,l,r,m) => format!("B:{f}:{:x}:{:x}:{l}:{r}:{m}", t.to_bits(), g.to_bits()),
                            xrf::Node::Leaf(v,c) => format!("L:{v}:{c}"),
                        }).collect::<Vec<_>>().join(";")
                    }).collect::<Vec<_>>();
                    trees.sort(); trees
                };
                assert_eq!(signatures(&train(&materialized)), signatures(&train(&direct)));
            }
        }
        assert!(ExactColumnTable::new(rows, real.clone(), shadows.clone(), &[3], 1).is_err());
        assert!(ExactColumnTable::new(rows, real, shadows, &[0], 3).is_err());
    }

    #[test]
    fn binned_columns_match_per_fit_binning() {
        let rows = 120;
        let value = |r: usize, c: usize| (((r * (c + 3) * 7) % 41) as f64 / 3.).sin();
        let (codes, edges) = DenseInput::bin_columns(rows, 3, 15, 2, BinningStrategy::ExactSort, 1000, 6, value).unwrap();
        for bootstrap in [false, true] {
            let labels = (0..rows).map(|r| (r * 7 / 5) % 2).collect::<Vec<_>>();
            let reference = DenseInput::training_binned_with_accessor_and_seed(rows, 3, labels.clone(), vec![1.; rows], 2, 0., Criterion::Gini, 2, 1, 0., 15, 2, BinningStrategy::ExactSort, 1000, 6, value)
                .unwrap().with_extra_trees(true, 6).with_float32_column_storage(2).with_histogram_threads(1).with_binned_extra_trees(2);
            let direct = DenseInput::training_binned_columns(rows, codes.clone(), edges.clone(), labels, 2, 0., Criterion::Gini, 2, 1, 0., 15)
                .unwrap().with_extra_trees(true, 6).with_histogram_threads(1);
            let train = |input: &DenseInput| Forest::new_parallel_with_settings(input, 7, 2, true, false, false, 42, 2, 5, usize::MAX, bootstrap, None);
            let signatures = |forest: &Forest<DenseInput>| {
                let mut trees = forest.tree_refs().map(|tree| {
                    tree.nodes.iter().map(|node| match node {
                        xrf::Node::Branch(f,t,g,l,r,m) => format!("B:{f}:{:x}:{:x}:{l}:{r}:{m}", t.to_bits(), g.to_bits()),
                        xrf::Node::Leaf(v,c) => format!("L:{v}:{c}"),
                    }).collect::<Vec<_>>().join(";")
                }).collect::<Vec<_>>();
                trees.sort(); trees
            };
            assert_eq!(signatures(&train(&reference)), signatures(&train(&direct)));
        }
        let mut bad = codes.clone(); bad[0] = 200;
        assert!(DenseInput::training_binned_columns(rows, bad, edges, vec![0; rows], 2, 0., Criterion::Gini, 2, 1, 0., 15).is_err());
    }

    #[test]
    fn binned_random_forest_on_column_codes_matches_row_major() {
        let rows = 2000;
        let value = |r: usize, c: usize| (((r * (c + 3) * 7) % 97) as f64 / 5.).sin() + c as f64;
        let (codes, edges) = DenseInput::bin_columns(rows, 20, 31, 2, BinningStrategy::ExactSort, 100_000, 6, value).unwrap();
        for classes in [2, 3] {
            let labels = (0..rows).map(|r| ((r * 7 / 5) + (value(r, 1) > 1.2) as usize) % classes).collect::<Vec<_>>();
            for threads in [1, 4] {
                let reference = DenseInput::training_binned_with_accessor_and_seed(rows, 20, labels.clone(), vec![1.; rows], classes, 0., Criterion::Gini, 2, 1, 0., 31, 2, BinningStrategy::ExactSort, 100_000, 6, value)
                    .unwrap().with_histogram_threads(threads);
                let direct = DenseInput::training_binned_columns(rows, codes.clone(), edges.clone(), labels.clone(), classes, 0., Criterion::Gini, 2, 1, 0., 31)
                    .unwrap().with_histogram_threads(threads);
                assert!(!reference.column_major && direct.column_major);
                let mask = Mask::from_vec((0..rows).step_by(3).chain([5, 5, 8]).collect());
                assert_eq!(build_histograms(&reference, &mask), build_histograms(&direct, &mask));
                for bootstrap in [false, true] {
                    let train = |input: &DenseInput| Forest::new_parallel_with_settings(input, 7, 5, true, false, false, 42, 2, 6, usize::MAX, bootstrap, None);
                    let signatures = |forest: &Forest<DenseInput>| {
                        let mut trees = forest.tree_refs().map(|tree| {
                            tree.nodes.iter().map(|node| match node {
                                xrf::Node::Branch(f,t,g,l,r,m) => format!("B:{f}:{:x}:{:x}:{l}:{r}:{m}", t.to_bits(), g.to_bits()),
                                xrf::Node::Leaf(v,c) => format!("L:{v}:{c}"),
                            }).collect::<Vec<_>>().join(";")
                        }).collect::<Vec<_>>();
                        trees.sort(); trees
                    };
                    let expected = signatures(&train(&reference));
                    assert!(expected.iter().all(|tree| tree.contains("B:")));
                    assert_eq!(expected, signatures(&train(&direct)));
                }
            }
        }
    }

    #[test]
    fn rejects_invalid_banks_and_maps() {
        assert!(ExactColumnBank::from_f64(vec![f64::MAX], 1, 1).is_err());
        assert!(ExactColumnBank::from_f32(vec![f32::NAN], 1, 1).is_err());
        assert!(ExactColumnBank::from_f32(vec![], usize::MAX, 2).is_err());
        let bank = ExactColumnBank::from_f32(vec![0.; 4], 4, 1).unwrap();
        for (order, permutation) in [(vec![0,1,2,3], vec![0,0,2,3]), (vec![0,0,2,3], vec![0,1,2,3]), (vec![0,1,2,3], vec![0,1,2,4])] {
            assert!(ExactVirtualColumns::new(bank.clone(), vec![0], vec![0], Arc::new(order), Arc::new(ExactRowIndices::U32(permutation))).is_err());
        }
    }
}
