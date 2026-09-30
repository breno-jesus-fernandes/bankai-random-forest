pub struct ImportanceAggregator {
    n: usize,
    mean: f64,
    sum_sq: f64,
}

impl Default for ImportanceAggregator {
    fn default() -> Self {
        ImportanceAggregator::new()
    }
}

impl ImportanceAggregator {
    pub fn new() -> Self {
        ImportanceAggregator {
            n: 0,
            mean: 0.,
            sum_sq: 0.,
        }
    }
    /// Ingests a value into the aggeragator
    pub fn ingest(&mut self, x: f64) {
        self.n += 1;
        let old_mean = self.mean;
        self.mean += (x - old_mean) / (self.n as f64);
        self.sum_sq += (x - old_mean) * (x - self.mean);
    }
    /// Merges in a second aggregator; after this operation, self is in the state as if it had ingested all the values ingested into `other` (including those acquired via merges).
    pub fn merge(&mut self, other: &Self) {
        if other.n == 0 {
            return;
        }
        if self.n == 0 {
            self.n = other.n;
            self.mean = other.mean;
            self.sum_sq = other.sum_sq;
            return;
        }
        let n = self.n + other.n;
        let delta = other.mean - self.mean;
        self.mean = ((self.n as f64) * self.mean + (other.n as f64) * other.mean) / (n as f64);
        self.sum_sq +=
            other.sum_sq + delta * delta * ((self.n as f64) * (other.n as f64)) / (n as f64);
        self.n += other.n;
    }
    /// Returns the number of direct and merged-in ingests
    pub fn samples(&self) -> usize {
        self.n
    }
    /// Calculates the mean of all values that were ingested into this aggregator and into aggregators that were merged with it,
    ///  as well as as many zeroes as it takes to have `count` ingested elements in total.
    /// This is due to a fact that average should be taken over all trees in the ensemble, yet not all trees use each feature.
    pub fn value(&self, count: usize) -> f64 {
        assert!(count >= self.n);
        (self.n as f64) / (count as f64) * self.mean
    }
    /// Calculates the mean and standard deviation of all values that were ingested into this aggregator and into aggregators that were merged with it,
    ///  as well as as many zeroes as it takes to have `count` ingested elements in total.
    /// Returns mean divided by standard deviation, or `None` when the standard deviation is zero or undefined.
    pub fn value_normalised(&self, count: usize) -> Option<f64> {
        assert!(count >= self.n);
        let on = count - self.n;
        if self.n <= 2 {
            None
        } else {
            let sum_sq = self.sum_sq
                + self.mean * self.mean * ((self.n as f64) * (on as f64)) / (count as f64);
            let mean = (self.n as f64) / (count as f64) * self.mean;
            let sd = (sum_sq / ((count - 1) as f64)).sqrt();
            if sd == 0. { None } else { Some(mean / sd) }
        }
    }
    /// Tries to regenerate the object from a 24-byte chunk, which should be created by the `into_raw()` method;
    /// may panic if given bytes are not a valid representation
    pub fn from_raw(raw: &[u8; 24]) -> Self {
        Self {
            n: (u64::from_le_bytes(raw[0..8].try_into().unwrap()))
                .try_into()
                .unwrap(),
            mean: f64::from_le_bytes(raw[8..16].try_into().unwrap()),
            sum_sq: f64::from_le_bytes(raw[16..24].try_into().unwrap()),
        }
    }
    /// Converts the object into a vector of 24 bytes
    pub fn into_raw(&self) -> [u8; 24] {
        [
            (self.n as u64).to_le_bytes(),
            self.mean.to_le_bytes(),
            self.sum_sq.to_le_bytes(),
        ]
        .concat()
        .try_into()
        .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn works() {
        let p: Vec<_> = (1..=48).map(|x| x as f64).collect();
        let mut ia = ImportanceAggregator::new();
        p.iter().for_each(|&x| ia.ingest(x));
        assert_eq!(ia.value(48), 24.5);
        assert_eq!(ia.value_normalised(48).unwrap(), 1.75);
    }
    #[test]
    fn merges() {
        let mut ia1 = ImportanceAggregator::new();
        let mut ia2 = ImportanceAggregator::new();
        (1..27).for_each(|x| ia1.ingest(x as f64));
        (27..=48).for_each(|x| ia2.ingest(x as f64));
        ia1.merge(&ia2);
        assert_eq!(ia1.value(48), 24.5);
        assert_eq!(ia1.value_normalised(48).unwrap(), 1.75);
    }
    #[test]
    fn add_zeroes() {
        let mut ia1 = ImportanceAggregator::new();
        let mut ia2 = ImportanceAggregator::new();
        let c1 = 48;
        let c2 = 12;
        (1..c1)
            .chain(std::iter::repeat_n(0, c2))
            .for_each(|x| ia1.ingest(x as f64));
        (1..c1).for_each(|x| ia2.ingest(x as f64));
        assert_eq!(ia1.value(c1 + c2), ia2.value(c1 + c2));
        assert_eq!(ia1.value_normalised(c1 + c2), ia2.value_normalised(c1 + c2));
    }

    #[test]
    fn merging_empty_aggregators_preserves_the_empty_state() {
        let mut left = ImportanceAggregator::new();
        let right = ImportanceAggregator::new();

        left.merge(&right);

        assert_eq!(left.samples(), 0);
        assert_eq!(left.mean, 0.0);
        assert_eq!(left.sum_sq, 0.0);
        assert_eq!(left.value_normalised(0), None);
    }

    #[test]
    fn merging_into_empty_copies_state_and_merging_empty_is_a_noop() {
        let mut empty = ImportanceAggregator::new();
        let mut populated = ImportanceAggregator::new();
        populated.ingest(2.5);
        populated.ingest(4.5);

        empty.merge(&populated);
        let after_copy = empty.into_raw();
        empty.merge(&ImportanceAggregator::new());

        assert_eq!(empty.into_raw(), after_copy);
        assert_eq!(empty.samples(), 2);
        assert_eq!(empty.value(2), 3.5);
    }

    #[test]
    fn raw_state_round_trips_exactly() {
        let mut original = ImportanceAggregator::new();
        for value in [1.25, -3.5, 8.0, 2.25] {
            original.ingest(value);
        }

        let restored = ImportanceAggregator::from_raw(&original.into_raw());

        assert_eq!(restored.samples(), original.samples());
        assert_eq!(restored.mean, original.mean);
        assert_eq!(restored.sum_sq, original.sum_sq);
        assert_eq!(restored.value(6), original.value(6));
        assert_eq!(restored.value_normalised(6), original.value_normalised(6));
    }
}
