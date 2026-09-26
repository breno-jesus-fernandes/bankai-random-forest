use super::Forest;
use crate::rfinput::{RfInput, VoteAggregator};
use crate::{Mask, MaskCache};

/// A representation of model predictions
pub struct Prediction<I: RfInput>(Vec<I::VoteAggregator>);

impl<I: RfInput> Prediction<I> {
    /// Predict elements in input with a model forest
    pub fn new(forest: &Forest<I>, input: &I) -> Self {
        let on = Mask::new_all(input.observation_count());
        let mut mask_cache = MaskCache::new();
        let mut onto: Vec<_> = std::iter::repeat_with(|| <I::VoteAggregator>::new(input))
            .take(on.len())
            .collect();
        forest
            .trees
            .iter()
            .for_each(|tree| tree.cast_votes(input, &on, &mut onto, &mut mask_cache));
        Prediction(onto)
    }
    /// Predict elements in input with a model forest using parallel execution
    pub fn new_parallel(forest: &Forest<I>, input: &I, threads: usize) -> Self
    where
        I: Send + Sync,
        I::Pivot: Send + Sync,
        I::FeatureId: Send + Sync,
        I::Vote: Send + Sync,
        I::VoteAggregator: Send + Sync,
    {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::mpsc;
        use std::thread;
        let (collect_sink, collect_source) = mpsc::channel();
        let tree_idx = Arc::new(AtomicUsize::new(0));

        thread::scope(move |s| {
            for _thread in 0..threads {
                let collect_sink = collect_sink.clone();
                let tree_idx = tree_idx.clone();
                let on = Mask::new_all(input.observation_count());
                s.spawn(move || {
                    let mut mask_cache = MaskCache::new();
                    let mut onto: Vec<_> =
                        std::iter::repeat_with(|| <I::VoteAggregator>::new(input))
                            .take(on.len())
                            .collect();
                    loop {
                        let e = tree_idx.fetch_add(1, Ordering::SeqCst);
                        if e < forest.trees.len() {
                            if let Some(tree) = forest.trees.get(e) {
                                tree.cast_votes(input, &on, &mut onto, &mut mask_cache)
                            }
                        } else {
                            break;
                        }
                    }
                    //No more work to do, try to merge what we have
                    collect_sink.send(onto).unwrap();
                });
            }
        });

        Prediction(
            collect_source
                .iter()
                .fold(None, |acc: Option<Vec<I::VoteAggregator>>, va| {
                    if let Some(mut merged) = acc {
                        merged
                            .iter_mut()
                            .zip(va.iter())
                            .for_each(|(a, b)| a.merge(b));
                        Some(merged)
                    } else {
                        Some(va)
                    }
                })
                .unwrap_or(Vec::new()),
        )
    }
    /// Iterate over predictions; you'll get VoteAggregators, the structure provided by the user via RfInput trait
    pub fn predictions(&self) -> impl Iterator<Item = (usize, &I::VoteAggregator)> {
        self.0.iter().enumerate()
    }
}
