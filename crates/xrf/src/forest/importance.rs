use crate::rfinput::AccuracyDecreaseAggregator;
use crate::rfinput::RfInput;
use crate::rfinput::VoteAggregator;
use crate::{Mask, MaskCache, RfRng};

use std::collections::HashMap;

use super::ImportanceAggregator;
use super::tree::Tree;

impl<I: RfInput> Tree<I> {
    pub fn permutational_importance(
        &self,
        input: &I,
        on: &Mask,
        mask_cache: &mut MaskCache,
        rng: &mut RfRng,
        imp_aggregator: &mut HashMap<I::FeatureId, ImportanceAggregator>,
        oob_aggregator: Option<&mut [I::VoteAggregator]>,
    ) {
        let mut ad_aggregator =
            <I::AccuracyDecreaseAggregator>::new(input, on, input.observation_count());
        let perm = on.permute(rng);
        self.cast_votes_permutational(
            input,
            on,
            &perm,
            None,
            &mut Vec::with_capacity(32),
            mask_cache,
            &mut |permuted, mask, vote| {
                ad_aggregator.ingest(permuted, mask, vote);
            },
        );
        ad_aggregator.mda_iter().for_each(|(feature, value)| {
            imp_aggregator
                .entry(feature)
                .or_insert(ImportanceAggregator::new())
                .ingest(value);
        });
        if let Some(oob_aggregator) = oob_aggregator {
            on.iter()
                .for_each(|&e| oob_aggregator[e].ingest_vote(ad_aggregator.get_direct_vote(e)));
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn cast_votes_permutational<F>(
        &self,
        input: &I,
        on: &Mask,
        perm: &Mask,
        upstream_permuted: Option<I::FeatureId>,
        upstream_used: &mut Vec<I::FeatureId>,
        mask_cache: &mut MaskCache,
        collector: &mut F,
    ) where
        F: FnMut(Option<I::FeatureId>, &Mask, &I::Vote),
    {
        match self {
            Self::Leaf(vote) => collector(upstream_permuted, on, vote),
            Self::Branch(feature_id, pivot, left, right) => {
                let mut left_mask = mask_cache.provide();
                let mut right_mask = mask_cache.provide();
                let mut left_perm = mask_cache.provide();
                let mut right_perm = mask_cache.provide();

                if upstream_permuted.is_none_or(|x| !x.eq(feature_id)) {
                    //Normal split
                    upstream_used.push(*feature_id);
                    on.split_together_into(
                        perm,
                        input.split_iter(on, *feature_id, pivot),
                        &mut left_mask,
                        &mut left_perm,
                        &mut right_mask,
                        &mut right_perm,
                    );
                    if !left_mask.is_empty() {
                        left.cast_votes_permutational(
                            input,
                            &left_mask,
                            &left_perm,
                            upstream_permuted,
                            upstream_used,
                            mask_cache,
                            collector,
                        );
                    }
                    if !right_mask.is_empty() {
                        right.cast_votes_permutational(
                            input,
                            &right_mask,
                            &right_perm,
                            upstream_permuted,
                            upstream_used,
                            mask_cache,
                            collector,
                        );
                    }
                    upstream_used.pop().unwrap();
                }
                let revisit = upstream_used.iter().any(|&x| x.eq(feature_id));

                //Permuted split
                if (!revisit) && upstream_permuted.is_none_or(|x| x.eq(feature_id)) {
                    //We make a permuted split
                    on.split_together_into(
                        perm,
                        input.split_iter(perm, *feature_id, pivot),
                        &mut left_mask,
                        &mut left_perm,
                        &mut right_mask,
                        &mut right_perm,
                    );
                    if !left_mask.is_empty() {
                        left.cast_votes_permutational(
                            input,
                            &left_mask,
                            &left_perm,
                            Some(*feature_id),
                            upstream_used,
                            mask_cache,
                            collector,
                        );
                    }
                    if !right_mask.is_empty() {
                        right.cast_votes_permutational(
                            input,
                            &right_mask,
                            &right_perm,
                            Some(*feature_id),
                            upstream_used,
                            mask_cache,
                            collector,
                        );
                    }
                }

                mask_cache.release(right_perm);
                mask_cache.release(left_perm);
                mask_cache.release(right_mask);
                mask_cache.release(left_mask);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mockups::generate_ident;

    #[test]
    fn imp() {
        let n = 1200;
        let nc = 4;
        let mut rng = RfRng::from_seed(21, 1);
        let mut mask_cache = MaskCache::new();
        let bag = Mask::new_all(n);
        let input = generate_ident(nc, n);
        let mut feature_sampler = input.feature_sampler();
        let tree = Tree::new(
            &input,
            &bag,
            1,
            &mut feature_sampler,
            512,
            &mut mask_cache,
            &mut rng,
        );
        let mut importance_aggregator = HashMap::new();
        tree.permutational_importance(
            &input,
            &bag,
            &mut mask_cache,
            &mut rng,
            &mut importance_aggregator,
            None,
        );
        let ia = importance_aggregator.get(&0).unwrap();
        let imp = ia.value(1);
        let nc = nc as f64;
        assert!((imp - (nc - 1.) / nc).abs() < 0.05);
    }
}
