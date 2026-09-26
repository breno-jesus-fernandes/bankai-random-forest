use xrf::{
    AccuracyDecreaseAggregator, DecisionSlice, FairBest, FeatureSampler, Mask, RfInput, RfRng,
    VoteAggregator,
};

pub struct DenseInput {
    values: Vec<f64>,
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
        })
    }

    pub fn prediction(
        values: Vec<f64>,
        rows: usize,
        columns: usize,
        n_classes: usize,
    ) -> Result<Self, String> {
        validate_matrix(&values, rows, columns)?;
        if n_classes == 0 {
            return Err("at least one class is required".to_string());
        }

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
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
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
        self.values[row * self.columns + column]
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
            + (right_weight / total_weight)
                * impurity(input.criterion, &right, right_weight);
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

pub struct NoImportance;

impl AccuracyDecreaseAggregator<DenseInput> for NoImportance {
    fn new(_: &DenseInput, _: &Mask, _: usize) -> Self {
        Self
    }

    fn ingest(&mut self, _: Option<usize>, _: &Mask, _: &usize) {
        unreachable!("importance is disabled for the baseline backend")
    }

    fn mda_iter(&self) -> impl Iterator<Item = (usize, f64)> {
        std::iter::empty()
    }

    fn get_direct_vote(&self, _: usize) -> usize {
        unreachable!("importance is disabled for the baseline backend")
    }
}

pub struct UniformFeatureSampler {
    n_features: usize,
}

impl FeatureSampler<DenseInput> for UniformFeatureSampler {
    fn random_feature(&mut self, rng: &mut RfRng) -> usize {
        rng.up_to(self.n_features)
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
    type AccuracyDecreaseAggregator = NoImportance;
    type FeatureSampler = UniformFeatureSampler;

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
        mask.len() >= self.min_samples_split
    }

    fn feature_sampler(&self) -> Self::FeatureSampler {
        UniformFeatureSampler {
            n_features: self.columns,
        }
    }

    fn new_split(
        &self,
        mask: &Mask,
        feature: Self::FeatureId,
        target: &Self::DecisionSlice,
        _: &mut RfRng,
    ) -> Option<(Self::Pivot, f64)> {
        best_split(self, mask, feature, target)
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
