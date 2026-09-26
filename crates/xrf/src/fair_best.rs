use crate::RfRng;

/// An accumulator finding best element, with fair, random tie resolution
pub struct FairBest<T, S = f64> {
    best: Option<(S, usize, T)>,
}

impl<T, S: PartialEq + PartialOrd> FairBest<T, S> {
    pub fn new() -> Self {
        FairBest { best: None }
    }
    /// Ingest a new candidate; when score is better than any seen previously, an element is stored as best.
    /// In case of tie, rng is consulted to resolve it randomly --- the resolution is fair in a sense that selection of each element with a score equal to maximal is equally probable.  
    pub fn ingest(&mut self, score: S, candidate: T, rng: &mut RfRng) {
        self.best = if let Some((cur_score, hits, cur_best)) = self.best.take() {
            if cur_score == score {
                //There is a tie, use reservoir to break
                let new_best = if rng.up_to(hits + 1) == 0 {
                    candidate
                } else {
                    cur_best
                };
                Some((score, hits + 1, new_best))
            } else if score > cur_score {
                //New score is better, move and reset hits
                Some((score, 1, candidate))
            } else {
                //New score is worse, do nothing
                Some((cur_score, hits, cur_best))
            }
        } else {
            Some((score, 1, candidate))
        }
    }
    /// Collapses the accumulator into a pair of best score and best element
    pub fn consume(self) -> Option<(S, T)> {
        self.best.map(|(score, _, best)| (score, best))
    }
}

impl<T, S: PartialEq + PartialOrd> Default for FairBest<T, S> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fair_best_fairness() {
        let n = 7;
        let s = 10000;
        let mut rng = RfRng::from_seed(1, 21);
        let mut counts = vec![0; n];
        for _ in 1..(n * s) {
            let mut ans = FairBest::new();
            (0..n).for_each(|x| ans.ingest(17., x, &mut rng));
            counts[ans.consume().unwrap().1] += 1;
        }
        let max_dev = counts
            .iter()
            .map(|x| isize::abs((*x as isize) - (s as isize)))
            .max()
            .unwrap();
        // eprintln!("{:?}, md={}", counts, max_dev);
        assert!(max_dev < s as isize / 10);
    }

    #[test]
    fn fair_best_basic() {
        let mut rng = RfRng::from_seed(1, 21);
        let mut ans = FairBest::new();
        ans.ingest(1., 1, &mut rng);
        ans.ingest(1., 2, &mut rng);
        ans.ingest(2., 3, &mut rng);
        ans.ingest(0., 4, &mut rng);
        ans.ingest(1., 5, &mut rng);
        let ans = ans.consume().unwrap();
        assert_eq!(ans.0, 2.);
        assert_eq!(ans.1, 3);
    }
}
