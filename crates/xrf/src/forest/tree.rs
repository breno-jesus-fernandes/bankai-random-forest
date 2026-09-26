use crate::rfinput::VoteAggregator;
use crate::rfinput::{DecisionSlice, RfInput};
use crate::{FairBest, FeatureSampler, Mask, MaskCache, RfRng, XrfError};

pub enum Tree<I: RfInput> {
    Leaf(I::Vote),
    Branch(I::FeatureId, I::Pivot, Box<Tree<I>>, Box<Tree<I>>),
}

use crate::walk::{Walk, WalkIter};

impl<I: RfInput> Tree<I> {
    pub fn new(
        input: &I,
        bag: &Mask,
        tries: usize,
        feature_sampler: &mut I::FeatureSampler,
        max_depth: usize,
        mask_cache: &mut MaskCache,
        rng: &mut RfRng,
    ) -> Self {
        feature_sampler.reset();
        Self::new_rec(
            input,
            bag,
            tries,
            feature_sampler,
            max_depth,
            mask_cache,
            rng,
        )
    }
    fn new_rec(
        input: &I,
        mask: &Mask,
        tries: usize,
        feature_sampler: &mut I::FeatureSampler,
        depth_left: usize,
        mask_cache: &mut MaskCache,
        rng: &mut RfRng,
    ) -> Self {
        let y = input.decision_slice(mask);
        if depth_left == 0 || y.is_pure() {
            Self::Leaf(y.condense(rng))
        } else {
            feature_sampler.reload();
            std::iter::repeat_n((), tries)
                .fold(FairBest::new(), |mut fair_best: FairBest<_, f64>, _| {
                    let feature = feature_sampler.random_feature(rng);
                    if let Some((pivot, score)) = input.new_split(mask, feature, &y, rng) {
                        fair_best.ingest(score, (feature, pivot), rng);
                    }
                    fair_best
                })
                .consume()
                .map(|best| {
                    let (_best_score, (feature, pivot)) = best;
                    let mut left = mask_cache.provide();
                    let mut right = mask_cache.provide();
                    mask.split_into(
                        input.split_iter(mask, feature, &pivot),
                        &mut left,
                        &mut right,
                    );
                    let branch = Self::Branch(
                        feature,
                        pivot,
                        Box::new(Self::new_rec(
                            input,
                            &left,
                            tries,
                            feature_sampler,
                            depth_left - 1,
                            mask_cache,
                            rng,
                        )),
                        Box::new(Self::new_rec(
                            input,
                            &right,
                            tries,
                            feature_sampler,
                            depth_left - 1,
                            mask_cache,
                            rng,
                        )),
                    );
                    mask_cache.release(left);
                    mask_cache.release(right);
                    branch
                })
                //No split mean a third way to make a leaf
                .unwrap_or_else(|| Self::Leaf(y.condense(rng)))
        }
    }
    pub fn from_walk<W: Iterator<Item = Walk<I>>>(iter: &mut W) -> Result<Self, XrfError> {
        let b = iter.next();
        match b {
            Some(Walk::VisitLeaf(v)) => Ok(Tree::Leaf(v)),
            Some(Walk::VisitBranch(fid, piv)) => {
                let left = Self::from_walk(iter)?;
                let right = Self::from_walk(iter)?;
                Ok(Tree::Branch(fid, piv, Box::new(left), Box::new(right)))
            }
            None => Err(XrfError::WalkAggregationFailure),
        }
    }
    pub fn cast_votes(
        &self,
        input: &I,
        on: &Mask,
        onto: &mut [I::VoteAggregator],
        mask_cache: &mut MaskCache,
    ) {
        match self {
            Self::Leaf(vote) => on.iter().for_each(|e| onto[*e].ingest_vote(*vote)),
            Self::Branch(feature_id, pivot, left, right) => {
                let mut left_on = mask_cache.provide();
                let mut right_on = mask_cache.provide();
                on.split_into(
                    input.split_iter(on, *feature_id, pivot),
                    &mut left_on,
                    &mut right_on,
                );
                if !left_on.is_empty() {
                    left.cast_votes(input, &left_on, onto, mask_cache);
                }
                if !right_on.is_empty() {
                    right.cast_votes(input, &right_on, onto, mask_cache);
                }
                mask_cache.release(right_on);
                mask_cache.release(left_on);
            }
        }
    }
    pub fn walk(&self) -> WalkIter<'_, I>
    where
        I::Pivot: Clone,
    {
        WalkIter::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mockups::generate_ident;

    #[test]
    fn ident() {
        let n = 8;
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
        let tw: Vec<_> = tree.walk().collect();

        use crate::mockups::simple_cls::DataFrame;
        let ref_walk = vec![
            Walk::<DataFrame>::VisitBranch(0, 3.5),
            Walk::<DataFrame>::VisitBranch(0, 5.5),
            Walk::<DataFrame>::VisitLeaf(3),
            Walk::<DataFrame>::VisitLeaf(2),
            Walk::<DataFrame>::VisitBranch(0, 1.5),
            Walk::<DataFrame>::VisitLeaf(1),
            Walk::<DataFrame>::VisitLeaf(0),
        ];
        let ok = tw.iter().zip(ref_walk.iter()).all(|x| match x {
            (Walk::VisitLeaf(x), Walk::VisitLeaf(y)) => x == y,
            (Walk::VisitBranch(xf, xp), Walk::VisitBranch(yf, yp)) => (xf == yf) && (xp == yp),
            _ => false,
        });
        assert!(ok);
    }
}
