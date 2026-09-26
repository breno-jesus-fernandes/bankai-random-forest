use std::collections::HashMap;
use xrf::{
    AccuracyDecreaseAggregator, DecisionSlice, FairBest, FeatureSampler, Mask, RfInput, RfRng,
    VoteAggregator,
};

pub struct DenseInput {
    values: DenseValues,
    labels: Option<Vec<usize>>,
    sample_weights: Option<Vec<f64>>,
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
    bin_edges: Option<Vec<Vec<f64>>>,
    active_features: Vec<usize>,
}

enum DenseValues {
    Exact(Vec<f64>),
    Binned(Vec<u8>),
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
                (DenseValues::Binned(binned), Some(edges), active)
            }
            None => (DenseValues::Exact(values), None, (0..columns).collect()),
        };
        Ok(Self {
            values,
            labels: Some(labels),
            sample_weights: Some(sample_weights),
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
        })
    }

    pub fn prediction(
        values: Vec<f64>,
        rows: usize,
        columns: usize,
        n_classes: usize,
        bin_edges: Option<&[Vec<f64>]>,
    ) -> Result<Self, String> {
        validate_matrix(&values, rows, columns)?;
        if n_classes == 0 {
            return Err("at least one class is required".to_string());
        }

        let values = match bin_edges {
            Some(edges) => {
                if edges.len() != columns {
                    return Err("histogram edges do not match the fitted feature count".to_string());
                }
                DenseValues::Binned(apply_histogram_edges(&values, rows, columns, edges))
            }
            None => DenseValues::Exact(values),
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
            bin_edges: None,
            active_features: (0..columns).collect(),
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn bin_edges(&self) -> Option<&[Vec<f64>]> {
        self.bin_edges.as_deref()
    }

    pub fn value_at(&self, row: usize, column: usize) -> f64 {
        self.value(row, column)
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
        let index = row * self.columns + column;
        match &self.values {
            DenseValues::Exact(values) => values[index],
            DenseValues::Binned(values) => values[index] as f64,
        }
    }

    fn bin_value(&self, row: usize, column: usize) -> usize {
        match &self.values {
            DenseValues::Binned(values) => values[row * self.columns + column] as usize,
            DenseValues::Exact(_) => unreachable!("histograms require binned input"),
        }
    }
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
    if values.iter().any(|value| !value.is_finite()) {
        return Err("X must contain only finite values".to_string());
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
    if midpoint >= right { left } else { midpoint }
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
            let bin = edges_by_feature[feature].partition_point(|edge| value > *edge);
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
            let bins = edges.len() + 1;
            FeatureHistogram {
                class_weights: vec![0.0; bins * input.n_classes],
                sample_counts: vec![0; bins],
            }
        })
        .collect::<Vec<_>>();

    for &row in mask.iter() {
        let label = input.labels()[row];
        let weight = input.sample_weight(row);
        for (feature, histogram) in features.iter_mut().enumerate() {
            let bin = input.bin_value(row, feature);
            histogram.class_weights[bin * input.n_classes + label] += weight;
            histogram.sample_counts[bin] += 1;
        }
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
) -> Option<(f64, f64)> {
    let best = if input.histogram_bins.is_some() {
        best_split_histogram(input, target, &split_cache.0.as_ref()?.features[feature])
    } else {
        best_split_exact(input, mask, feature, target)
    }?;
    Some((best.0, best.1 * target.total_weight / input.total_weight))
}

fn best_split_exact(
    input: &DenseInput,
    mask: &Mask,
    feature: usize,
    target: &DenseDecisionSlice,
) -> Option<(f64, f64)> {
    let mut ranked: Vec<(f64, usize, f64)> = mask
        .iter()
        .zip(&target.labels)
        .map(|(&row, &label)| (input.value(row, feature), label, input.sample_weight(row)))
        .collect();
    ranked.sort_unstable_by(|left, right| left.0.total_cmp(&right.0));

    if ranked.len() < 2 {
        return None;
    }

    let total_weight = target.total_weight;
    let parent_impurity = impurity(input.criterion, &target.class_weights, total_weight);
    let mut left = vec![0.0; input.n_classes];
    let mut right = target.class_weights.clone();
    let mut left_weight = 0.0;
    let mut best: Option<(f64, f64)> = None;

    for index in 0..(ranked.len() - 1) {
        let (value, label, weight) = ranked[index];
        left[label] += weight;
        right[label] -= weight;
        left_weight += weight;

        let next = ranked[index + 1].0;
        if value.total_cmp(&next).is_eq() {
            continue;
        }

        let right_weight = total_weight - left_weight;
        if left_weight == 0.0 || right_weight == 0.0 {
            continue;
        }
        let left_count = index + 1;
        let right_count = ranked.len() - left_count;
        if left_count < input.min_samples_leaf || right_count < input.min_samples_leaf {
            continue;
        }
        if left_weight < input.min_leaf_weight || right_weight < input.min_leaf_weight {
            continue;
        }
        let child_impurity = (left_weight / total_weight)
            * impurity(input.criterion, &left, left_weight)
            + (right_weight / total_weight) * impurity(input.criterion, &right, right_weight);
        let gain = parent_impurity - child_impurity;
        if gain < input.min_impurity_decrease {
            continue;
        }
        let midpoint = value * 0.5 + next * 0.5;
        let pivot = if midpoint >= next { value } else { midpoint };

        if best
            .as_ref()
            .is_none_or(|(_, previous_gain)| gain > *previous_gain)
        {
            best = Some((pivot, gain));
        }
    }

    best
}

fn best_split_histogram(
    input: &DenseInput,
    target: &DenseDecisionSlice,
    histogram: &FeatureHistogram,
) -> Option<(f64, f64)> {
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
    let total_count = histogram.sample_counts.iter().sum::<usize>();
    let mut best: Option<(f64, f64)> = None;

    for bin in 0..(bin_count - 1) {
        for class in 0..input.n_classes {
            let weight = histogram.class_weights[bin * input.n_classes + class];
            left[class] += weight;
            right[class] -= weight;
            left_weight += weight;
        }
        left_count += histogram.sample_counts[bin];
        let right_count = total_count - left_count;
        if left_count < input.min_samples_leaf || right_count < input.min_samples_leaf {
            continue;
        }

        let right_weight = total_weight - left_weight;
        if left_weight == 0.0
            || right_weight == 0.0
            || left_weight < input.min_leaf_weight
            || right_weight < input.min_leaf_weight
        {
            continue;
        }
        let child_impurity = (left_weight / total_weight)
            * impurity(input.criterion, &left, left_weight)
            + (right_weight / total_weight) * impurity(input.criterion, &right, right_weight);
        let gain = parent_impurity - child_impurity;
        if gain < input.min_impurity_decrease {
            continue;
        }
        let pivot = bin as f64 + 0.5;
        if best
            .as_ref()
            .is_none_or(|(_, previous_gain)| gain > *previous_gain)
        {
            best = Some((pivot, gain));
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
    ) -> Option<(Self::Pivot, f64)> {
        best_split(self, mask, feature, target, split_cache)
    }

    fn split_iter(
        &self,
        mask: &Mask,
        feature: Self::FeatureId,
        pivot: &Self::Pivot,
    ) -> impl Iterator<Item = bool> {
        mask.iter()
            .map(move |&row| self.value(row, feature) > *pivot)
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
}
