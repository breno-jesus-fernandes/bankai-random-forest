use super::simple_cls::DataFrame;
use std::iter;

// Identity-like many-class problem; solution is an integer division, so this should
//  more-less generate a balanced tree
pub fn generate_ident(n_cat: usize, n: usize) -> DataFrame {
    assert!(n % n_cat == 0);
    let x: Vec<Vec<f64>> = vec![(0..n).map(|x| x as f64).collect()];
    let y: Vec<usize> = (0..n_cat)
        .flat_map(|e| iter::repeat_n(e, n / n_cat))
        .collect();
    DataFrame::new(x, y, n_cat)
}
