use crate::Mask;
use crate::RfRng;
use std::hash::Hash;

/// Xrf can built a model on any data object that implements this trait
pub trait RfInput: Sized {
    /// Id of a feature, usually column in data frame
    type FeatureId: Copy + Hash + Eq;
    /// Criterion applied in a branch of the decision tree; for instance, for a numerical feature, it may be "> 2.0", and that would mean sending object with this feature value larger than two left and other right.
    type Pivot;
    /// Content of the leaves of decision tree; usually majority class of mean of decision values of observation that ended up in this leaf during training.
    /// In prediction, a value cast for observation reaching this leaf, and ingested into VoteAggregator.
    type Vote: Copy;
    /// Aggregator for Votes; see its corresponding trait
    type VoteAggregator: VoteAggregator<Self> + Clone;
    /// An abstraction of a chunk of decision feature selected by a mask; separate object to facilitate caching some properties.
    type DecisionSlice: DecisionSlice<Self::Vote>;
    /// Aggregator for vote differences; can be () if importance is not needed.
    type AccuracyDecreaseAggregator: AccuracyDecreaseAggregator<Self>;
    /// A generator of random features for training.
    /// Useful to implement sampling without replacement; can be () where simple sampling is enough, for instance for Extra Trees.
    type FeatureSampler: FeatureSampler<Self>;
    /// Number of observations, used to generate masks; has to be accurate.
    fn observation_count(&self) -> usize;
    /// Number of features; not actually used by xrf, but usually useful for implementers.
    fn feature_count(&self) -> usize;
    /// Constructor of the DecisionSlice object for elements in the mask.
    fn decision_slice(&self, mask: &Mask) -> Self::DecisionSlice;
    /// Constructor of FeatureSampler.
    fn feature_sampler(&self) -> Self::FeatureSampler;
    /// Constructor of a split, which is (usually optimal) pivot using certain feature that splits observations in a possibly most uniform subsets.
    fn new_split(
        &self,
        on: &Mask,
        using: Self::FeatureId,
        y: &Self::DecisionSlice,
        rng: &mut RfRng,
    ) -> Option<(Self::Pivot, f64)>;
    /// Application of pivot to a given subset of the data; returns of iterator that sends observations left (for true) or right (for false).
    fn split_iter(
        &self,
        on: &Mask,
        using: Self::FeatureId,
        by: &Self::Pivot,
    ) -> impl Iterator<Item = bool>;
}

pub trait FeatureSampler<I: RfInput> {
    /// Generate a new feature to try to build a split; will be executed tries times for each tree level.
    /// Usually sampling with replacement is used, since split generation is deterministic and trying the same feature and observations multiple times will yield the same outcome --- sampling with replacement will work in non-deterministic scenarios or as a fast-and-dirty solution (resulting forest will be less accurate).
    fn random_feature(&mut self, rng: &mut RfRng) -> I::FeatureId;
    /// Called before each tree level is created; all features should be available after reload, but streams of features after reload may be influenced by the structure history.
    fn reload(&mut self);
    /// Called before each tree is created; like reload, but streams of features after resets must not depend on the object history.
    fn reset(&mut self);
}

/// An abstraction of decision values for a group of observations
pub trait DecisionSlice<Vote> {
    /// Tests the decision pureness of the subset of observations in a slice; when this function returns true, the subset is collapsed into a leaf.
    fn is_pure(&self) -> bool;
    /// Generates a value for the leaf; note that condense may happen not only after is_pure is true but also when leaf is enforced by other circumstances, in particular the exhaustion of the allowed tree depth or failure to generate any split in the tree level scan.
    fn condense(&self, rng: &mut RfRng) -> Vote;
}

/// Aggregator of votes coming from different trees, represents the result collected over the whole forest
pub trait VoteAggregator<I: RfInput> {
    fn new(input: &I) -> Self;
    /// Add a next vote to the aggregator; note that the order of ingestion events should not matter (ignoring some hard to avoid numerical issues).
    fn ingest_vote(&mut self, v: I::Vote);
    /// Should work equivalent as if self ingested all votes ingested into other.
    fn merge(&mut self, other: &Self);
}

pub trait AccuracyDecreaseAggregator<I: RfInput> {
    fn new(input: &I, on: &Mask, n: usize) -> Self;
    fn ingest(&mut self, permuted: Option<I::FeatureId>, mask: &Mask, vote: &I::Vote);
    fn mda_iter(&self) -> impl Iterator<Item = (I::FeatureId, f64)>;
    fn get_direct_vote(&self, e: usize) -> I::Vote;
}
