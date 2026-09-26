use crate::RfRng;

/// Mask is xrf's abstraction of the collection of observations (usually data frame rows); the crate assumes they are numbered with usize indices, and mask is a vector of said indices.
/// Masks are not necessary sorted and they can contain single element multiple times.
#[derive(Clone)]
pub struct Mask(Vec<usize>);

impl Mask {
    /// Permute the mask, i.e., change the order to random.
    /// Uses Fisher-Yates shuffle.
    pub fn permute(&self, rng: &mut RfRng) -> Self {
        let mut ans = self.clone();
        if !ans.0.is_empty() {
            for e in 0..(ans.0.len() - 1) {
                let ee = e + rng.up_to(ans.0.len() - e);
                ans.0.swap(e, ee);
            }
        }
        ans
    }
    /// Creates 0..n mask
    pub fn new_all(n: usize) -> Self {
        Mask((0..n).collect())
    }
    /// Split mask into two parts; mask is zipped with iter and every time iter returns true the corresponding index goes to a left mask, while the elements with false move to the right mask.
    /// Iterator is usually a result of pivot classifying a stream of feature values.
    pub fn split_into<I>(&self, iter: I, left: &mut Mask, right: &mut Mask)
    where
        I: Iterator<Item = bool>,
    {
        left.0.clear();
        right.0.clear();
        for (lr, &e) in iter.zip(self.iter()) {
            if lr {
                left.0.push(e);
            } else {
                right.0.push(e);
            }
        }
    }
    /// Some as a split, but splits a pair of masks into a pair of right/left sub-masks.
    pub fn split_together_into<I>(
        &self,
        other: &Mask,
        iter: I,
        left: &mut Mask,
        left_other: &mut Mask,
        right: &mut Mask,
        right_other: &mut Mask,
    ) where
        I: Iterator<Item = bool>,
    {
        assert_eq!(self.len(), other.len());
        left.0.clear();
        left_other.0.clear();
        right.0.clear();
        right_other.0.clear();
        for ((lr, &e), &ee) in iter.zip(self.iter()).zip(other.iter()) {
            if lr {
                left.0.push(e);
                left_other.0.push(ee);
            } else {
                right.0.push(e);
                right_other.0.push(ee);
            }
        }
    }
    /// Redistribute observations into bag and out-of-bag (OOB) masks.
    /// Bag contains as many observations as in the original mask, but sampled with resampling, thus only about 63.2% of unique observation remain there, yet they are multiplied.
    /// The other about 36.8% is called OOB and stored in the second slot of the resulting pair.
    pub fn new_bag_oob(n: usize, rng: &mut RfRng) -> (Self, Self) {
        let mut bag = Vec::with_capacity(n);
        let mut hits: Vec<usize> = vec![0; n];
        // Hits[e] is the number of times e is in bag]
        // this can be optimised for large n and make both the bag & oob masks to be sorted for some cache locality maybe
        for _ in 0..n {
            hits[rng.up_to(n)] += 1;
        }

        let mut e = 0;
        hits.retain_mut(|h| {
            let oob = *h == 0;
            if oob {
                //It is gonna be retained so we change it into its index
                *h = e;
            } else {
                bag.resize(bag.len() + *h, e);
            }
            e += 1;
            oob
        });

        let oob = hits;
        (Mask(bag), Mask(oob))
    }
    /// Constructor simply converting an index vector
    #[inline]
    pub fn from_vec(x: Vec<usize>) -> Self {
        Self(x)
    }
}

impl std::ops::Deref for Mask {
    type Target = [usize];
    #[inline]
    fn deref(&self) -> &[usize] {
        self.0.as_slice()
    }
}
impl std::iter::FromIterator<usize> for Mask {
    fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        Mask::from_vec(iter.into_iter().collect::<Vec<usize>>())
    }
}

/// A simple allocation arena for Masks, to lift some stress on allocator.
pub struct MaskCache(Vec<Mask>);

impl MaskCache {
    pub fn new() -> Self {
        MaskCache(Vec::new())
    }
    /// Provide a mask; either create new or give back some of released ones
    pub fn provide(&mut self) -> Mask {
        self.0.pop().unwrap_or_else(|| Mask(Vec::new()))
    }
    /// Move a mask into the cache for later use
    pub fn release(&mut self, what: Mask) {
        self.0.push(what);
    }
}

impl Default for MaskCache {
    fn default() -> Self {
        Self::new()
    }
}
