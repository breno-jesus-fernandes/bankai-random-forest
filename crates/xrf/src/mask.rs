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
        Self::new_bag_oob_with_size(n, n, rng)
    }
    /// Same as `new_bag_oob`, with an explicit bootstrap sample size.
    pub fn new_bag_oob_with_size(n: usize, sample_size: usize, rng: &mut RfRng) -> (Self, Self) {
        let mut bag = Vec::with_capacity(n);
        let mut hits: Vec<usize> = vec![0; n];
        // Hits[e] is the number of times e is in bag]
        // this can be optimised for large n and make both the bag & oob masks to be sorted for some cache locality maybe
        for _ in 0..sample_size {
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
    /// Empty and move a mask into the cache for later reuse.
    pub fn release(&mut self, mut what: Mask) {
        what.0.clear();
        self.0.push(what);
    }
}

impl Default for MaskCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permutation_preserves_duplicate_indices_and_handles_short_masks() {
        let mut rng = RfRng::from_seed(17, 3);
        let input = Mask::from_vec(vec![4, 4, 9, 12, 9]);
        let mut expected = input.to_vec();
        expected.sort_unstable();
        let mut actual = input.permute(&mut rng).to_vec();
        actual.sort_unstable();

        assert_eq!(actual, expected);
        assert!(Mask::from_vec(Vec::new()).permute(&mut rng).is_empty());
        assert_eq!(&*Mask::from_vec(vec![7]).permute(&mut rng), &[7]);
    }

    #[test]
    fn bootstrap_masks_partition_rows_and_keep_the_requested_bag_size() {
        let mut rng = RfRng::from_seed(31, 2);
        let (bag, oob) = Mask::new_bag_oob_with_size(12, 25, &mut rng);

        assert_eq!(bag.len(), 25);
        assert!(bag.iter().all(|&row| row < 12));
        assert!(oob.iter().all(|&row| row < 12));
        for row in 0..12 {
            assert!(bag.contains(&row) ^ oob.contains(&row));
        }

        let (empty_bag, all_oob) = Mask::new_bag_oob_with_size(4, 0, &mut rng);
        assert!(empty_bag.is_empty());
        assert_eq!(&*all_oob, &[0, 1, 2, 3]);
    }

    #[test]
    fn split_operations_clear_reused_masks_and_keep_parallel_rows_aligned() {
        let rows = Mask::from_vec(vec![8, 3, 8, 1]);
        let companion = Mask::from_vec(vec![80, 30, 81, 10]);
        let mut left = Mask::from_vec(vec![999]);
        let mut left_companion = Mask::from_vec(vec![999]);
        let mut right = Mask::from_vec(vec![999]);
        let mut right_companion = Mask::from_vec(vec![999]);

        rows.split_together_into(
            &companion,
            [false, true, true, false].into_iter(),
            &mut left,
            &mut left_companion,
            &mut right,
            &mut right_companion,
        );

        assert_eq!(&*left, &[3, 8]);
        assert_eq!(&*left_companion, &[30, 81]);
        assert_eq!(&*right, &[8, 1]);
        assert_eq!(&*right_companion, &[80, 10]);
        rows.split_into([true, false, true, false].into_iter(), &mut left, &mut right);
        assert_eq!(&*left, &[8, 8]);
        assert_eq!(&*right, &[3, 1]);
    }

    #[test]
    fn mask_cache_reuses_released_capacity() {
        let mut cache = MaskCache::default();
        let mut mask = cache.provide();
        mask.0.reserve(32);
        let capacity = mask.0.capacity();
        mask.0.extend([1, 2, 3]);
        cache.release(mask);

        let reused = cache.provide();

        assert!(reused.is_empty());
        assert!(reused.0.capacity() >= capacity);
    }
}
