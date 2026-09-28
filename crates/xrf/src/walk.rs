use crate::forest::{Node, Tree};
use crate::rfinput::RfInput;

/// A portable representation of a vertex of a trained forest; useful for serialisation or model analysis
///
/// This enum is usually an item in iterator performing left-first, depth-first walk over the whole forest; trees are binary, so this is enough to represent the whole graph.
pub enum Walk<I: RfInput> {
    /// The currently visited vertex is a leaf in a decision tree
    VisitLeaf(I::Vote),
    /// The currently visited vertex is a branch in a decision tree
    VisitBranch(I::FeatureId, I::Pivot, f64),
}

pub struct WalkIter<'a, I>
where
    I: RfInput,
    I::Pivot: Clone,
{
    tree: &'a Tree<I>,
    on: Option<usize>,
    stack: Vec<usize>,
}

impl<'a, I> WalkIter<'a, I>
where
    I: RfInput,
    I::Pivot: Clone,
{
    pub fn new(tree: &'a Tree<I>) -> Self {
        Self {
            tree,
            on: Some(tree.root),
            stack: Vec::new(),
        }
    }
}

impl<'a, I: RfInput> Iterator for WalkIter<'a, I>
where
    I::Pivot: Clone,
{
    type Item = Walk<I>;
    fn next(&mut self) -> Option<Self::Item> {
        match &self.tree.nodes[self.on?] {
            Node::Leaf(v, _) => {
                self.on = self.stack.pop();
                Some(Walk::VisitLeaf(*v))
            }
            Node::Branch(fid, pivot, score, _, left, right) => {
                self.stack.push(*right);
                self.on = Some(*left);
                Some(Walk::VisitBranch(*fid, (*pivot).clone(), *score))
            }
        }
    }
}
