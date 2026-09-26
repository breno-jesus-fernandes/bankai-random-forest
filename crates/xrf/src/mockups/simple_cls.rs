use super::fy_sampler::FYSampler;
use crate::{AccuracyDecreaseAggregator, FairBest, Mask, RfInput, RfRng, VoteAggregator};
use std::collections::HashMap;

#[derive(Clone)]
pub struct DataFrame {
    x: Vec<Vec<f64>>,
    y: Vec<usize>,
    n_cat: usize,
}

impl RfInput for DataFrame {
    type FeatureId = usize;
    type DecisionSlice = DecisionSlice;
    type Pivot = f64;
    type Vote = usize;
    type VoteAggregator = Votes;
    type AccuracyDecreaseAggregator = DaAggregator;
    type FeatureSampler = FYSampler<Self>;
    fn observation_count(&self) -> usize {
        self.y.len()
    }
    fn feature_count(&self) -> usize {
        self.x.len()
    }
    fn feature_sampler(&self) -> Self::FeatureSampler {
        FYSampler::new(self)
    }
    fn decision_slice(&self, mask: &Mask) -> Self::DecisionSlice {
        DecisionSlice::new(self, mask)
    }
    fn new_split(
        &self,
        on: &Mask,
        using: Self::FeatureId,
        y: &Self::DecisionSlice,
        _rng: &mut crate::RfRng,
    ) -> Option<(Self::Pivot, f64)> {
        let feature = &self.x[using];
        scan(&feature, y, on)
    }
    fn split_iter(
        &self,
        on: &Mask,
        using: Self::FeatureId,
        by: &Self::Pivot,
    ) -> impl Iterator<Item = bool> {
        let feature = &self.x[using];
        on.iter().map(|&e| feature[e] > *by)
    }
}

impl DataFrame {
    pub fn new(x: Vec<Vec<f64>>, y: Vec<usize>, n_cat: usize) -> Self {
        assert!(x.len() > 0);
        assert!(y.len() > 3);
        assert!(x[0].len() == y.len());
        DataFrame { x, y, n_cat }
    }
    pub fn subset_objects(&self, mask: &Mask) -> Self {
        let xn = self
            .x
            .iter()
            .map(|x| mask.iter().map(|&e| x[e]).collect())
            .collect();
        let yn = mask.iter().map(|&e| self.y[e]).collect();
        DataFrame {
            x: xn,
            y: yn,
            n_cat: self.n_cat,
        }
    }
    pub fn y(&self) -> &[usize] {
        &self.y
    }
}

pub struct DecisionSlice {
    values: Vec<usize>,
    n_cat: usize,
    summary: Votes,
}

impl DecisionSlice {
    fn new(input: &DataFrame, mask: &Mask) -> Self {
        let n_cat = input.n_cat;
        let mut summary = Votes::new(n_cat);
        let values = mask
            .iter()
            .map(|&e| input.y[e])
            .inspect(|&v| summary.ingest_vote(v))
            .collect();
        Self {
            values,
            n_cat,
            summary,
        }
    }
}

impl crate::DecisionSlice<usize> for DecisionSlice {
    fn is_pure(&self) -> bool {
        self.summary.is_pure()
    }
    fn condense(&self, rng: &mut RfRng) -> usize {
        self.summary.collapse(rng)
    }
}

#[derive(Clone)]
pub struct Votes(pub Vec<usize>);

impl Votes {
    pub fn is_pure(&self) -> bool {
        self.0.iter().filter(|&&x| x > 0).count() <= 1
    }
    pub fn new(n_cat: usize) -> Self {
        Self(std::iter::repeat_n(0, n_cat as usize).collect())
    }
    // pub fn n_cat(&self) -> usize {
    //     self.0.len()
    // }
    pub fn collapse(&self, rng: &mut RfRng) -> usize {
        self.0
            .iter()
            .enumerate()
            .fold(FairBest::new(), |mut fair_best, (cls, count)| {
                fair_best.ingest(count, cls, rng);
                fair_best
            })
            .consume()
            .map(|(_score, cls)| cls)
            .unwrap()
    }
}

impl VoteAggregator<DataFrame> for Votes {
    fn new(input: &DataFrame) -> Self {
        Votes::new(input.n_cat)
    }
    fn ingest_vote(&mut self, v: usize) {
        self.0[v as usize] += 1;
    }
    fn merge(&mut self, other: &Self) {
        self.0
            .iter_mut()
            .zip(other.0.iter())
            .for_each(|(t, s)| *t += s);
    }
}

fn scan(x: &[f64], ys: &DecisionSlice, mask: &Mask) -> Option<(f64, f64)> {
    let mut bound: Vec<(f64, usize)> = mask
        .iter()
        .zip(ys.values.iter())
        .map(|(&xe, &y)| (x[xe], y))
        .collect();
    bound.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

    let n = bound.len();
    let mut left = Votes::new(ys.n_cat);
    let mut scanned = 0_usize;
    bound
        .windows(2)
        .map(|x| (x[0].0, x[1].0, x[0].1))
        .fold(None, |acc: Option<(f64, f64)>, (x, next_x, y)| {
            scanned += 1;
            left.ingest_vote(y);
            if x.total_cmp(&next_x).is_ne() {
                let score: f64 = ys
                    .summary
                    .0
                    .iter()
                    .zip(left.0.iter())
                    .map(|(&all, &left)| {
                        let ahead = (n - scanned) as f64;
                        let scanned = scanned as f64;
                        let n = n as f64;
                        let right = (all - left) as f64;
                        let left = left as f64;
                        (left / scanned) * (left / n) + (right / ahead) * (right / n)
                    })
                    .sum();
                if score > acc.map(|x| x.1).unwrap_or(f64::NEG_INFINITY) {
                    return Some((0.5 * (x + next_x), score));
                }
            }
            acc
        })
        .map(|(thresh, score)| (thresh, score))
}

pub struct DaAggregator {
    direct: Vec<Option<usize>>,
    drops: HashMap<usize, isize>,
    n: usize,
    true_decision: Vec<usize>,
}

impl AccuracyDecreaseAggregator<DataFrame> for DaAggregator {
    fn new(input: &DataFrame, on: &Mask, n: usize) -> Self {
        Self {
            direct: vec![None; n],
            drops: HashMap::new(),
            n: on.len(),
            true_decision: input.y.clone(),
        }
    }
    fn ingest(&mut self, permuted: Option<usize>, mask: &Mask, vote: &usize) {
        if let Some(permuted) = permuted {
            let diff: isize = mask
                .iter()
                .map(|&e| {
                    let oob_vote = self.direct.get(e).unwrap().unwrap();
                    if !oob_vote.eq(vote) {
                        let truth = self.true_decision[e];
                        match (truth.eq(vote), truth.eq(&oob_vote)) {
                            (true, true) => unreachable!("Logic error"),
                            (true, false) => -1,
                            (false, true) => 1,

                            (false, false) => 0,
                        }
                    } else {
                        0
                    }
                })
                .sum();
            *self.drops.entry(permuted).or_insert(0) += diff;
        } else {
            for &e in mask.iter() {
                self.direct[e] = Some(*vote);
            }
        }
    }
    fn get_direct_vote(&self, e: usize) -> usize {
        self.direct.get(e).unwrap().unwrap()
    }
    fn mda_iter(&self) -> impl Iterator<Item = (<DataFrame as RfInput>::FeatureId, f64)> {
        self.drops
            .iter()
            .map(|(a, b)| (*a, (*b as f64) / (self.n as f64)))
    }
}
