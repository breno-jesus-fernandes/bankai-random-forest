use crate::forest::Tree;
use crate::rfinput::RfInput;

/// A portable representation of a vertex of a trained forest; useful for serialisation or model analysis
///
/// This enum is usually an item in iterator performing left-first, depth-first walk over the whole forest; trees are binary, so this is enough to represent the whole graph.
pub enum Walk<I: RfInput> {
    /// The currently visited vertex is a leaf in a decision tree
    VisitLeaf(I::Vote),
    /// The currently visited vertex is a branch in a decision tree
    VisitBranch(I::FeatureId, I::Pivot),
}

pub struct WalkIter<'a, I>
where
    I: RfInput,
    I::Pivot: Clone,
{
    on: Option<&'a Tree<I>>,
    stack: Vec<&'a Tree<I>>,
}

impl<'a, I> WalkIter<'a, I>
where
    I: RfInput,
    I::Pivot: Clone,
{
    pub fn new(tree: &'a Tree<I>) -> Self {
        Self {
            on: Some(tree),
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
        match self.on? {
            Tree::Leaf(v) => {
                self.on = self.stack.pop();
                Some(Walk::VisitLeaf(*v))
            }
            Tree::Branch(fid, pivot, left, right) => {
                self.stack.push(right);
                self.on = Some(left);
                Some(Walk::VisitBranch(*fid, (*pivot).clone()))
            }
        }
    }
}
