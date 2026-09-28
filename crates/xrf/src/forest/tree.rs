use crate::rfinput::VoteAggregator;
use crate::rfinput::{DecisionSlice, RfInput};
use crate::{FairBest, FeatureSampler, Mask, MaskCache, RfRng, XrfError};

/// Nodes are appended in postorder; child indices survive arena growth.
pub struct Tree<I: RfInput> {
    pub nodes: Vec<Node<I>>,
    pub root: usize,
}

pub enum Node<I: RfInput> {
    /// Node cover is retained for path-dependent tree explainers.
    Leaf(I::Vote, usize),
    Branch(I::FeatureId, I::Pivot, f64, usize, usize, usize),
}

use crate::walk::{Walk, WalkIter};

impl<I: RfInput> Tree<I> {
    pub fn new(
        input: &I,
        bag: &Mask,
        tries: usize,
        feature_sampler: &mut I::FeatureSampler,
        max_depth: usize,
        max_leaves: usize,
        mask_cache: &mut MaskCache,
        rng: &mut RfRng,
    ) -> Self {
        feature_sampler.reset();
        let mut leaf_count = 1;
        let split_cache = input.split_cache(bag);
        // Bound the initial reservation for unrestricted trees. Larger trees grow
        // geometrically without invalidating indices or allocating individual nodes.
        let mut nodes = Vec::with_capacity(
            bag.len()
                .min(max_leaves)
                .saturating_mul(2)
                .saturating_sub(1)
                .min(8191),
        );
        let root = Self::new_rec(
            &mut nodes,
            input,
            bag,
            tries,
            feature_sampler,
            max_depth,
            max_leaves,
            &mut leaf_count,
            mask_cache,
            split_cache,
            rng,
            0.0,
            1.0,
        );
        Self { nodes, root }
    }

    /// Prune this tree using the normalized cost-complexity threshold.
    pub fn prune_with_ccp_alpha(
        &mut self,
        input: &I,
        bag: &Mask,
        ccp_alpha: f64,
        mask_cache: &mut MaskCache,
    ) {
        if ccp_alpha <= 0.0 {
            return;
        }
        let root = input.decision_slice(bag);
        let Some(root_weight) = root.pruning_weight() else {
            return;
        };
        if root.pruning_risk(root_weight).is_none() {
            return;
        }
        Self::prune_rec(
            &mut self.nodes,
            self.root,
            input,
            bag,
            root_weight,
            ccp_alpha,
            mask_cache,
        );
    }

    fn prune_rec(
        nodes: &mut [Node<I>],
        index: usize,
        input: &I,
        mask: &Mask,
        root_weight: f64,
        ccp_alpha: f64,
        mask_cache: &mut MaskCache,
    ) -> (f64, usize) {
        let node_slice = input.decision_slice(mask);
        let Some(node_risk) = node_slice.pruning_risk(root_weight) else {
            return (0.0, 1);
        };
        let (subtree_risk, leaf_count, collapse_vote) = match &nodes[index] {
            Node::Leaf(_, _) => return (node_risk, 1),
            Node::Branch(feature, pivot, score, _, left, right) => {
                let (left, right) = (*left, *right);
                let mut left_mask = mask_cache.provide();
                let mut right_mask = mask_cache.provide();
                mask.split_into(
                    input.split_iter(mask, *feature, pivot, score.is_sign_negative()),
                    &mut left_mask,
                    &mut right_mask,
                );
                let (left_risk, left_leaves) = Self::prune_rec(
                    nodes,
                    left,
                    input,
                    &left_mask,
                    root_weight,
                    ccp_alpha,
                    mask_cache,
                );
                let (right_risk, right_leaves) = Self::prune_rec(
                    nodes,
                    right,
                    input,
                    &right_mask,
                    root_weight,
                    ccp_alpha,
                    mask_cache,
                );
                mask_cache.release(right_mask);
                mask_cache.release(left_mask);

                let leaves = left_leaves + right_leaves;
                let effective_alpha =
                    (node_risk - left_risk - right_risk) / (leaves.saturating_sub(1) as f64);
                let collapse_vote = if effective_alpha <= ccp_alpha {
                    node_slice.pruning_vote()
                } else {
                    None
                };
                (left_risk + right_risk, leaves, collapse_vote)
            }
        };

        if let Some(vote) = collapse_vote {
            nodes[index] = Node::Leaf(vote, mask.len());
            (node_risk, 1)
        } else {
            (subtree_risk, leaf_count)
        }
    }

    fn new_rec(
        nodes: &mut Vec<Node<I>>,
        input: &I,
        mask: &Mask,
        tries: usize,
        feature_sampler: &mut I::FeatureSampler,
        depth_left: usize,
        max_leaves: usize,
        leaf_count: &mut usize,
        mask_cache: &mut MaskCache,
        split_cache: I::SplitCache,
        rng: &mut RfRng,
        lower_bound: f64,
        upper_bound: f64,
    ) -> usize {
        let y = input.decision_slice(mask);
        let node = if depth_left == 0
            || y.is_pure()
            || !input.can_split(mask)
            || *leaf_count >= max_leaves
        {
            Node::Leaf(
                y.condense_with_bounds(rng, lower_bound, upper_bound),
                mask.len(),
            )
        } else {
            feature_sampler.reload();
            std::iter::repeat_n((), tries)
                .fold(FairBest::new(), |mut fair_best: FairBest<_, f64>, _| {
                    let feature = feature_sampler.random_feature(rng);
                    if let Some((pivot, missing_left, score)) = input.new_split_with_bounds(
                        mask,
                        feature,
                        &y,
                        &split_cache,
                        rng,
                        lower_bound,
                        upper_bound,
                    ) {
                        fair_best.ingest(score, (feature, pivot, missing_left), rng);
                    }
                    fair_best
                })
                .consume()
                .map(|best| {
                    *leaf_count += 1;
                    let (best_score, (feature, pivot, missing_left)) = best;
                    let mut left = mask_cache.provide();
                    let mut right = mask_cache.provide();
                    mask.split_into(
                        input.split_iter(mask, feature, &pivot, missing_left),
                        &mut left,
                        &mut right,
                    );
                    let (left_split_cache, right_split_cache) =
                        input.split_cache_children(split_cache, mask, &left, &right);
                    let constraint = input.monotonic_constraint(feature);
                    let (left_bounds, right_bounds) = if constraint == 0 {
                        ((lower_bound, upper_bound), (lower_bound, upper_bound))
                    } else {
                        let left_probability = input
                            .decision_slice(&left)
                            .positive_probability()
                            .unwrap_or(lower_bound);
                        let right_probability = input
                            .decision_slice(&right)
                            .positive_probability()
                            .unwrap_or(upper_bound);
                        let middle = (left_probability + right_probability) / 2.0;
                        if constraint > 0 {
                            // XRF routes values above the pivot to the left child.
                            ((middle, upper_bound), (lower_bound, middle))
                        } else {
                            ((lower_bound, middle), (middle, upper_bound))
                        }
                    };
                    let stored_score = if missing_left {
                        -best_score.abs()
                    } else {
                        best_score.abs()
                    };
                    let left_index = Self::new_rec(
                        nodes,
                        input,
                        &left,
                        tries,
                        feature_sampler,
                        depth_left - 1,
                        max_leaves,
                        leaf_count,
                        mask_cache,
                        left_split_cache,
                        rng,
                        left_bounds.0,
                        left_bounds.1,
                    );
                    let right_index = Self::new_rec(
                        nodes,
                        input,
                        &right,
                        tries,
                        feature_sampler,
                        depth_left - 1,
                        max_leaves,
                        leaf_count,
                        mask_cache,
                        right_split_cache,
                        rng,
                        right_bounds.0,
                        right_bounds.1,
                    );
                    let branch = Node::Branch(
                        feature,
                        pivot,
                        stored_score,
                        mask.len(),
                        left_index,
                        right_index,
                    );
                    mask_cache.release(left);
                    mask_cache.release(right);
                    branch
                })
                //No split mean a third way to make a leaf
                .unwrap_or_else(|| {
                    Node::Leaf(
                        y.condense_with_bounds(rng, lower_bound, upper_bound),
                        mask.len(),
                    )
                })
        };
        let index = nodes.len();
        nodes.push(node);
        index
    }
    pub fn from_walk<W: Iterator<Item = Walk<I>>>(iter: &mut W) -> Result<Self, XrfError> {
        fn append<I: RfInput, W: Iterator<Item = Walk<I>>>(
            iter: &mut W,
            nodes: &mut Vec<Node<I>>,
        ) -> Result<usize, XrfError> {
            let node = match iter.next() {
                Some(Walk::VisitLeaf(v)) => Node::Leaf(v, 0),
                Some(Walk::VisitBranch(fid, pivot, score)) => {
                    let left = append(iter, nodes)?;
                    let right = append(iter, nodes)?;
                    Node::Branch(fid, pivot, score, 0, left, right)
                }
                None => return Err(XrfError::WalkAggregationFailure),
            };
            let index = nodes.len();
            nodes.push(node);
            Ok(index)
        }
        let mut nodes = Vec::new();
        let root = append(iter, &mut nodes)?;
        Ok(Self { nodes, root })
    }
    pub fn cast_votes(
        &self,
        input: &I,
        on: &Mask,
        onto: &mut [I::VoteAggregator],
        mask_cache: &mut MaskCache,
    ) {
        self.cast_votes_at(self.root, input, on, onto, mask_cache);
    }

    fn cast_votes_at(
        &self,
        index: usize,
        input: &I,
        on: &Mask,
        onto: &mut [I::VoteAggregator],
        mask_cache: &mut MaskCache,
    ) {
        match &self.nodes[index] {
            Node::Leaf(vote, _) => on.iter().for_each(|e| onto[*e].ingest_vote(*vote)),
            Node::Branch(feature_id, pivot, score, _, left, right) => {
                let mut left_on = mask_cache.provide();
                let mut right_on = mask_cache.provide();
                on.split_into(
                    input.split_iter(on, *feature_id, pivot, score.is_sign_negative()),
                    &mut left_on,
                    &mut right_on,
                );
                if !left_on.is_empty() {
                    self.cast_votes_at(*left, input, &left_on, onto, mask_cache);
                }
                if !right_on.is_empty() {
                    self.cast_votes_at(*right, input, &right_on, onto, mask_cache);
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
            usize::MAX,
            &mut mask_cache,
            &mut rng,
        );
        let tw: Vec<_> = tree.walk().collect();
        assert_eq!(tree.nodes.len(), 7);
        for (index, node) in tree.nodes.iter().enumerate() {
            if let Node::Branch(_, _, _, _, left, right) = node {
                assert!(*left < index && *right < index);
            }
        }
        let restored = Tree::from_walk(&mut tree.walk()).unwrap();
        assert_eq!(restored.walk().count(), tw.len());

        use crate::mockups::simple_cls::DataFrame;
        let ref_walk = vec![
            Walk::<DataFrame>::VisitBranch(0, 3.5, 0.0),
            Walk::<DataFrame>::VisitBranch(0, 5.5, 0.0),
            Walk::<DataFrame>::VisitLeaf(3),
            Walk::<DataFrame>::VisitLeaf(2),
            Walk::<DataFrame>::VisitBranch(0, 1.5, 0.0),
            Walk::<DataFrame>::VisitLeaf(1),
            Walk::<DataFrame>::VisitLeaf(0),
        ];
        let ok = tw.iter().zip(ref_walk.iter()).all(|x| match x {
            (Walk::VisitLeaf(x), Walk::VisitLeaf(y)) => x == y,
            (Walk::VisitBranch(xf, xp, _), Walk::VisitBranch(yf, yp, _)) => {
                (xf == yf) && (xp == yp)
            }
            _ => false,
        });
        assert!(ok);
    }
}
