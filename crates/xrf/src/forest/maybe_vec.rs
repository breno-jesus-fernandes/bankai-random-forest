/// Holds a vector of something or just its length
pub enum MaybeVec<T> {
    Vector(Vec<T>),
    JustLength(usize),
}

use MaybeVec::*;
impl<T> MaybeVec<T> {
    pub fn len(&self) -> usize {
        match self {
            Vector(x) => x.len(),
            JustLength(x) => *x,
        }
    }
    pub fn push(&mut self, x: T) {
        match self {
            Vector(v) => v.push(x),
            JustLength(l) => *l += 1,
        }
    }
    pub fn new(real: bool) -> Self {
        if real {
            Vector(Vec::new())
        } else {
            JustLength(0)
        }
    }
    pub fn from_just_length(len: usize) -> Self {
        JustLength(len)
    }
    pub fn is_some(&self) -> bool {
        matches!(self, Vector(_))
    }
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        match self {
            Vector(x) => x.iter(),
            _ => panic!("Iterating over trees non existing"),
        }
    }
    pub fn get(&self, e: usize) -> Option<&T> {
        match self {
            Vector(x) => x.get(e),
            _ => None,
        }
    }
    pub fn merge(&mut self, other: Self) {
        match (self, other) {
            (Vector(a), Vector(mut b)) => {
                a.append(&mut b);
            }
            (JustLength(a), JustLength(b)) => *a += b,
            _ => panic!("Merging forests with and without trees"),
        }
    }
}
