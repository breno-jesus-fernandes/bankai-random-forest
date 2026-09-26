use crate::{FeatureSampler, RfInput, RfRng};

pub struct FYSampler<I: RfInput> {
    mixed: Vec<I::FeatureId>,
    left: usize,
}

impl<I: RfInput<FeatureId = usize>> FYSampler<I> {
    pub fn new(input: &I) -> Self {
        Self {
            mixed: (0..input.feature_count()).collect(),
            left: input.feature_count(),
        }
    }
}

impl<I: RfInput<FeatureId = usize>> FeatureSampler<I> for FYSampler<I> {
    fn random_feature(&mut self, rng: &mut RfRng) -> I::FeatureId {
        let sel = rng.up_to(self.left);
        self.left = self.left.checked_sub(1).unwrap();
        self.mixed.swap(sel, self.left);
        sel
    }
    fn reset(&mut self) {
        self.mixed = (0..self.mixed.len()).collect();
        self.left = self.mixed.len();
    }
    fn reload(&mut self) {
        self.left = self.mixed.len();
    }
}
