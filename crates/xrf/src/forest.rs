use crate::rfinput::{RfInput, VoteAggregator};
use crate::{Mask, MaskCache, RfRng};

mod maybe_vec;
use maybe_vec::MaybeVec;

mod tree;
pub use tree::Tree;

mod importance;

mod importance_aggregator;
pub use importance_aggregator::ImportanceAggregator;

mod prediction;
pub use prediction::Prediction;

use std::collections::HashMap;

use crate::Walk;

/// Trained XRF model
///
/// It may contain (all optional), a model itself, a collection of decision trees, feature importance and out-of-bag (OOB) predictions (an internal accuracy estimation, similar to cross-validation).
///
/// What this crate is doing mainly depends on a given data set structure implementing RfInput trait; in particular, pivots used by trees and features are all abstract and are filled by the implementer.
pub struct Forest<I: RfInput> {
    trees: MaybeVec<Tree<I>>,
    importance: Option<HashMap<I::FeatureId, ImportanceAggregator>>,
    oob_votes: Option<Vec<I::VoteAggregator>>,
}

impl<I: RfInput> Forest<I> {
    /// Train a new model
    pub fn new(
        input: &I,
        trees: usize,
        tries: usize,
        save_forest: bool,
        importance: bool,
        oob: bool,
        seed: u64,
    ) -> Self {
        let num_trees = trees;
        let mut trees = MaybeVec::new(save_forest);
        let mut importance: Option<HashMap<I::FeatureId, ImportanceAggregator>> = if importance {
            Some(HashMap::new())
        } else {
            None
        };
        let mut oob_votes: Option<Vec<_>> = if oob {
            Some(
                (0..input.observation_count())
                    .map(|_| <I::VoteAggregator>::new(input))
                    .collect(),
            )
        } else {
            None
        };
        let mut mask_cache = MaskCache::new();
        let mut feature_sampler = input.feature_sampler();
        for tree_id in 0..num_trees {
            let mut rng = RfRng::from_seed(seed, 1 + tree_id as u64);
            let (bag, oob) = Mask::new_bag_oob(input.observation_count(), &mut rng);
            let tree = Tree::new(
                input,
                &bag,
                tries,
                &mut feature_sampler,
                512,
                &mut mask_cache,
                &mut rng,
            );
            if let Some(importance) = importance.as_mut() {
                tree.permutational_importance(
                    input,
                    &oob,
                    &mut mask_cache,
                    &mut rng,
                    importance,
                    oob_votes.as_deref_mut(),
                );
            } else if let Some(oob_votes) = oob_votes.as_mut() {
                tree.cast_votes(input, &oob, oob_votes, &mut mask_cache);
            }
            trees.push(tree);
        }
        Self {
            trees,
            importance,
            oob_votes,
        }
    }
    /// Train a new model using multiple threads
    #[allow(clippy::too_many_arguments)]
    pub fn new_parallel(
        input: &I,
        trees: usize,
        tries: usize,
        save_forest: bool,
        importance: bool,
        oob: bool,
        seed: u64,
        threads: usize,
    ) -> Self
    where
        I: Send + Sync,
        I::Pivot: Send + Sync,
        I::Vote: Send + Sync,
        I::FeatureId: Send + Sync,
        I::VoteAggregator: Send + Sync,
    {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::mpsc;
        use std::thread;
        let (collect_sink, collect_source) = mpsc::channel();
        let tree_idx = Arc::new(AtomicUsize::new(0));

        assert!(threads > 0);
        let num_trees = trees;

        thread::scope(move |s| {
            for _thread in 0..threads {
                let collect_sink = collect_sink.clone();
                let tree_idx = tree_idx.clone();
                s.spawn(move || {
                    let mut trees: MaybeVec<Tree<I>> = MaybeVec::new(save_forest);
                    let mut importance: Option<HashMap<I::FeatureId, ImportanceAggregator>> =
                        if importance {
                            Some(HashMap::new())
                        } else {
                            None
                        };
                    let mut oob_votes: Option<Vec<_>> = if oob {
                        Some(
                            (0..input.observation_count())
                                .map(|_| <I::VoteAggregator>::new(input))
                                .collect(),
                        )
                    } else {
                        None
                    };
                    let mut mask_cache = MaskCache::new();
                    let mut feature_sampler = input.feature_sampler();
                    loop {
                        let e = tree_idx.fetch_add(1, Ordering::SeqCst);
                        if e < num_trees {
                            let mut rng = RfRng::from_seed(seed, 1 + e as u64);
                            let (bag, oob) = Mask::new_bag_oob(input.observation_count(), &mut rng);
                            let tree = Tree::new(
                                input,
                                &bag,
                                tries,
                                &mut feature_sampler,
                                512,
                                &mut mask_cache,
                                &mut rng,
                            );
                            if let Some(importance) = importance.as_mut() {
                                tree.permutational_importance(
                                    input,
                                    &oob,
                                    &mut mask_cache,
                                    &mut rng,
                                    importance,
                                    oob_votes.as_deref_mut(),
                                );
                            } else if let Some(oob_votes) = oob_votes.as_mut() {
                                tree.cast_votes(input, &oob, oob_votes, &mut mask_cache);
                            }
                            trees.push(tree);
                        } else {
                            break;
                        }
                    }

                    //No more work to do, try to merge what we have
                    let chunk = Self {
                        trees,
                        importance,
                        oob_votes,
                    };
                    collect_sink.send(chunk).unwrap();
                });
            }
        });
        collect_source
            .iter()
            .fold(Self::new_empty(), |mut acc, chunk| {
                acc.merge(chunk);
                acc
            })
    }
    /// Merges two forests together
    pub fn merge(&mut self, mut other: Self) {
        if self.trees.len() > 0 {
            self.trees.merge(other.trees);
            self.oob_votes = self
                .oob_votes
                .take()
                .zip(other.oob_votes.take())
                .map(|(mut a, b)| {
                    a.iter_mut().zip(b.iter()).for_each(|(a, b)| a.merge(b));
                    a
                });
            self.importance =
                self.importance
                    .take()
                    .zip(other.importance.take())
                    .map(|(mut a, b)| {
                        b.into_iter().for_each(|(feature, aggregator)| {
                            a.entry(feature)
                                .and_modify(|x| x.merge(&aggregator))
                                .or_insert_with(|| aggregator);
                        });
                        a
                    });
        } else {
            //Self was empty
            *self = other;
        }
    }
    /// A zero model for merge
    fn new_empty() -> Self {
        Self::new_with_num_trees(0)
    }
    /// To deserialise forest with no trees element
    pub fn new_with_num_trees(num_trees: usize) -> Self {
        Self {
            oob_votes: None,
            trees: MaybeVec::from_just_length(num_trees),
            importance: None,
        }
    }
    pub fn has_oob(&self) -> bool {
        self.oob_votes.is_some()
    }
    pub fn has_trees(&self) -> bool {
        self.trees.is_some()
    }
    pub fn has_importance(&self) -> bool {
        self.importance.is_some()
    }
    pub fn trees(&self) -> usize {
        self.trees.len()
    }
    pub fn predict(&self, input: &I) -> Prediction<I> {
        Prediction::new(self, input)
    }
    pub fn predict_parallel(&self, input: &I, threads: usize) -> Prediction<I>
    where
        I: Send + Sync,
        I::FeatureId: Send + Sync,
        I::Pivot: Send + Sync,
        I::Vote: Send + Sync,
        I::VoteAggregator: Send + Sync,
    {
        assert!(threads > 0);
        Prediction::new_parallel(self, input, threads)
    }
    /// Walk over all trees in the forest in a DFS manner,
    ///  producing an iterator of Walk steps
    pub fn walk(&self) -> impl Iterator<Item = Walk<I>>
    where
        I::Pivot: Clone,
    {
        self.trees.iter().flat_map(|x| x.walk())
    }
    /// Generate a forest object from a tree walk.
    /// OOB and importance slots are going to be empty,
    ///  use `replace_importance()` and `add_oob()` to recreate them
    pub fn from_walk<W>(forest_walk: W) -> Result<Self, crate::XrfError>
    where
        W: Iterator<Item = Walk<I>>,
        I::Pivot: Clone,
    {
        let mut trees: MaybeVec<Tree<I>> = MaybeVec::new(true);
        let mut forest_walk = forest_walk.peekable();
        loop {
            let tree = Tree::from_walk(&mut forest_walk)?;
            trees.push(tree);
            if forest_walk.peek().is_none() {
                break;
            }
        }
        Ok(Self {
            trees,
            importance: None,
            oob_votes: None,
        })
    }
    /// Replaces or takes importance aggregators
    pub fn replace_importance<II>(
        &mut self,
        importance_iterator: II,
    ) -> Option<HashMap<I::FeatureId, ImportanceAggregator>>
    where
        II: Iterator<Item = (I::FeatureId, ImportanceAggregator)>,
    {
        let imp: HashMap<_, _> = importance_iterator.collect();
        self.importance.replace(imp)
    }
    /// Replaces or takes OOB prediction aggregators
    pub fn replace_oob<II>(&mut self, oob_iterator: II) -> Option<Vec<I::VoteAggregator>>
    where
        II: Iterator<Item = I::VoteAggregator>,
    {
        let oob: Vec<_> = oob_iterator.collect();
        self.oob_votes.replace(oob)
    }
    /// Iterator over references for raw VoteAggregators for each object
    pub fn oob(&self) -> impl Iterator<Item = (usize, &I::VoteAggregator)> {
        self.oob_votes.iter().flat_map(|x| x.iter().enumerate())
    }
    /// Iterator over references to ImportancAggregators for each feature
    pub fn raw_importance(&self) -> impl Iterator<Item = (I::FeatureId, &ImportanceAggregator)> {
        self.importance
            .iter()
            .flat_map(|x| x.iter())
            .map(|(&fid, a)| (fid, a))
    }
    /// Iterator producing permutational importance for each feature
    pub fn importance(&self) -> impl Iterator<Item = (I::FeatureId, f64)> {
        self.raw_importance()
            .map(|(feature, value)| (feature, value.value(self.trees.len())))
    }
    /// Iterator producing normalised permutational importance for each feature
    pub fn importance_normalised(&self) -> impl Iterator<Item = (I::FeatureId, f64)> {
        self.raw_importance().filter_map(|(feature, value)| {
            value
                .value_normalised(self.trees.len())
                .map(|val| (feature, val))
        })
    }
}
