//! Similarity metrics for vector comparison.
//!
//! When compiled with the `simd` feature, uses SIMD-accelerated implementations
//! for up to 200x faster similarity computations on supported hardware.

use crate::vector::Vector;

/// Available similarity metrics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    /// Cosine similarity (most common, scale-invariant)
    Cosine,
    /// Dot product (raw inner product)
    Dot,
    /// Euclidean distance (converted to similarity)
    Euclidean,
    /// Manhattan distance (converted to similarity)
    Manhattan,
    /// Hamming similarity (agreement rate)
    Hamming,
    /// Overlap similarity (non-zero agreement)
    Overlap,
    /// Agreement similarity: (agreements - disagreements) / dimension
    Agreement,
    /// Chebyshev (L∞) distance converted to similarity
    Chebyshev,
}

/// Similarity computation for vectors.
///
/// With the `simd` feature enabled, cosine, dot, and euclidean use
/// hardware-accelerated SIMD implementations.
pub struct Similarity;

impl Similarity {
    /// Compute similarity using the specified metric.
    pub fn compute(a: &Vector, b: &Vector, metric: Metric) -> f64 {
        match metric {
            Metric::Cosine => Self::cosine(a, b),
            Metric::Dot => Self::dot(a, b),
            Metric::Euclidean => Self::euclidean(a, b),
            Metric::Manhattan => Self::manhattan(a, b),
            Metric::Hamming => Self::hamming(a, b),
            Metric::Overlap => Self::overlap(a, b),
            Metric::Agreement => Self::agreement(a, b),
            Metric::Chebyshev => Self::chebyshev(a, b),
        }
    }

    /// Cosine similarity: dot(a, b) / (||a|| * ||b||)
    ///
    /// Returns a value in [-1, 1] where:
    /// - 1 means identical (parallel)
    /// - 0 means orthogonal (unrelated)
    /// - -1 means opposite (anti-parallel)
    #[cfg(feature = "simd")]
    pub fn cosine(a: &Vector, b: &Vector) -> f64 {
        use simsimd::SpatialSimilarity;
        // Use SIMD dot products to compute cosine similarity
        // cosine = dot(a,b) / sqrt(dot(a,a) * dot(b,b))
        let dot_ab = i8::dot(a.data(), b.data()).unwrap_or(0.0);
        let dot_aa = i8::dot(a.data(), a.data()).unwrap_or(0.0);
        let dot_bb = i8::dot(b.data(), b.data()).unwrap_or(0.0);

        let norm_product = (dot_aa * dot_bb).sqrt();
        if norm_product < 1e-10 {
            return 0.0;
        }

        dot_ab / norm_product
    }

    #[cfg(not(feature = "simd"))]
    pub fn cosine(a: &Vector, b: &Vector) -> f64 {
        let dot = Self::dot_raw(a, b);
        let norm_a = a.norm();
        let norm_b = b.norm();

        if norm_a < 1e-10 || norm_b < 1e-10 {
            return 0.0;
        }

        dot / (norm_a * norm_b)
    }

    /// Raw dot product.
    #[cfg(feature = "simd")]
    pub fn dot(a: &Vector, b: &Vector) -> f64 {
        use simsimd::SpatialSimilarity;
        i8::dot(a.data(), b.data()).unwrap_or(0.0)
    }

    #[cfg(not(feature = "simd"))]
    pub fn dot(a: &Vector, b: &Vector) -> f64 {
        Self::dot_raw(a, b)
    }

    #[allow(dead_code)]
    fn dot_raw(a: &Vector, b: &Vector) -> f64 {
        assert_eq!(
            a.dimensions(),
            b.dimensions(),
            "Dimension mismatch in dot product"
        );

        a.data()
            .iter()
            .zip(b.data().iter())
            .map(|(&x, &y)| (x as i64) * (y as i64))
            .sum::<i64>() as f64
    }

    /// Euclidean distance (converted to similarity).
    ///
    /// Returns 1 / (1 + distance) so higher values mean more similar.
    pub fn euclidean(a: &Vector, b: &Vector) -> f64 {
        let distance = Self::euclidean_distance(a, b);
        1.0 / (1.0 + distance)
    }

    /// Raw Euclidean distance.
    pub fn euclidean_distance(a: &Vector, b: &Vector) -> f64 {
        assert_eq!(
            a.dimensions(),
            b.dimensions(),
            "Dimension mismatch in euclidean"
        );

        let sum_sq: i64 = a
            .data()
            .iter()
            .zip(b.data().iter())
            .map(|(&x, &y)| {
                let diff = (x as i64) - (y as i64);
                diff * diff
            })
            .sum();

        (sum_sq as f64).sqrt()
    }

    /// Manhattan distance (converted to similarity).
    ///
    /// Returns 1 / (1 + distance) so higher values mean more similar.
    pub fn manhattan(a: &Vector, b: &Vector) -> f64 {
        let distance = Self::manhattan_distance(a, b);
        1.0 / (1.0 + distance)
    }

    /// Raw Manhattan distance.
    pub fn manhattan_distance(a: &Vector, b: &Vector) -> f64 {
        assert_eq!(
            a.dimensions(),
            b.dimensions(),
            "Dimension mismatch in manhattan"
        );

        a.data()
            .iter()
            .zip(b.data().iter())
            .map(|(&x, &y)| ((x as i32) - (y as i32)).abs())
            .sum::<i32>() as f64
    }

    /// Hamming similarity: fraction of positions that agree.
    ///
    /// Returns a value in [0, 1].
    pub fn hamming(a: &Vector, b: &Vector) -> f64 {
        assert_eq!(
            a.dimensions(),
            b.dimensions(),
            "Dimension mismatch in hamming"
        );

        let matching = a
            .data()
            .iter()
            .zip(b.data().iter())
            .filter(|(&x, &y)| x == y)
            .count();

        matching as f64 / a.dimensions() as f64
    }

    /// Overlap similarity: agreement among non-zero positions.
    ///
    /// Useful when sparsity matters.
    pub fn overlap(a: &Vector, b: &Vector) -> f64 {
        assert_eq!(
            a.dimensions(),
            b.dimensions(),
            "Dimension mismatch in overlap"
        );

        let mut agree = 0;
        let mut active = 0;

        for (&x, &y) in a.data().iter().zip(b.data().iter()) {
            if x != 0 || y != 0 {
                active += 1;
                if x == y {
                    agree += 1;
                }
            }
        }

        if active == 0 {
            return 0.0;
        }

        agree as f64 / active as f64
    }

    /// Agreement similarity: (agreements - disagreements) / dimension.
    ///
    /// Returns a value in [-1, 1] where:
    /// - 1 means perfect agreement
    /// - 0 means balanced (equal agreements and disagreements)
    /// - -1 means total disagreement
    pub fn agreement(a: &Vector, b: &Vector) -> f64 {
        assert_eq!(
            a.dimensions(),
            b.dimensions(),
            "Dimension mismatch in agreement"
        );

        let mut agree: i64 = 0;
        let mut disagree: i64 = 0;

        for (&x, &y) in a.data().iter().zip(b.data().iter()) {
            let product = (x as i16) * (y as i16);
            if product > 0 {
                agree += 1;
            } else if product < 0 {
                disagree += 1;
            }
        }

        (agree - disagree) as f64 / a.dimensions() as f64
    }

    /// Chebyshev (L∞) distance converted to similarity.
    ///
    /// Returns 1 - (max_diff / 2) for bipolar vectors, clamped to [0, 1].
    pub fn chebyshev(a: &Vector, b: &Vector) -> f64 {
        let dist = Self::chebyshev_distance(a, b);
        // For bipolar vectors, max possible distance is 2
        if dist <= 2.0 {
            1.0 - dist / 2.0
        } else {
            0.0
        }
    }

    /// Raw Chebyshev (L∞) distance: maximum absolute difference.
    pub fn chebyshev_distance(a: &Vector, b: &Vector) -> f64 {
        assert_eq!(
            a.dimensions(),
            b.dimensions(),
            "Dimension mismatch in chebyshev"
        );

        a.data()
            .iter()
            .zip(b.data().iter())
            .map(|(&x, &y)| ((x as i32) - (y as i32)).unsigned_abs())
            .max()
            .unwrap_or(0) as f64
    }

    /// Minkowski (Lp) distance: generalized distance metric.
    ///
    /// Special cases: p=1 is Manhattan, p=2 is Euclidean.
    /// Returns raw distance (lower = more similar).
    pub fn minkowski_distance(a: &Vector, b: &Vector, p: f64) -> f64 {
        assert_eq!(
            a.dimensions(),
            b.dimensions(),
            "Dimension mismatch in minkowski"
        );
        assert!(p >= 1.0, "Minkowski p must be >= 1.0");

        let sum: f64 = a
            .data()
            .iter()
            .zip(b.data().iter())
            .map(|(&x, &y)| ((x as f64) - (y as f64)).abs().powf(p))
            .sum();

        sum.powf(1.0 / p)
    }

    /// Minkowski distance converted to similarity: 1 / (1 + distance).
    pub fn minkowski(a: &Vector, b: &Vector, p: f64) -> f64 {
        let dist = Self::minkowski_distance(a, b, p);
        1.0 / (1.0 + dist)
    }

    /// Weighted cosine similarity: per-dimension importance weighting.
    ///
    /// Applies `weights` element-wise before computing cosine similarity.
    /// Returns a value in [-1, 1].
    pub fn weighted_cosine(a: &Vector, b: &Vector, weights: &[f64]) -> f64 {
        assert_eq!(
            a.dimensions(),
            b.dimensions(),
            "Dimension mismatch in weighted_cosine"
        );
        assert_eq!(
            a.dimensions(),
            weights.len(),
            "Weight vector dimension mismatch"
        );

        let mut dot = 0.0_f64;
        let mut norm_a = 0.0_f64;
        let mut norm_b = 0.0_f64;

        for ((&x, &y), &w) in a.data().iter().zip(b.data().iter()).zip(weights.iter()) {
            let wa = (x as f64) * w;
            let wb = (y as f64) * w;
            dot += wa * wb;
            norm_a += wa * wa;
            norm_b += wb * wb;
        }

        let norm_product = (norm_a * norm_b).sqrt();
        if norm_product < 1e-10 {
            return 0.0;
        }

        dot / norm_product
    }

    /// Weighted Euclidean distance: per-dimension importance weighting.
    ///
    /// Applies sqrt(weights) to differences before computing L2 norm.
    /// Returns raw distance (lower = more similar).
    pub fn weighted_euclidean_distance(a: &Vector, b: &Vector, weights: &[f64]) -> f64 {
        assert_eq!(
            a.dimensions(),
            b.dimensions(),
            "Dimension mismatch in weighted_euclidean"
        );
        assert_eq!(
            a.dimensions(),
            weights.len(),
            "Weight vector dimension mismatch"
        );

        let sum: f64 = a
            .data()
            .iter()
            .zip(b.data().iter())
            .zip(weights.iter())
            .map(|((&x, &y), &w)| {
                let diff = (x as f64) - (y as f64);
                diff * diff * w
            })
            .sum();

        sum.sqrt()
    }

    /// Weighted Euclidean distance converted to similarity: 1 / (1 + distance).
    pub fn weighted_euclidean(a: &Vector, b: &Vector, weights: &[f64]) -> f64 {
        let dist = Self::weighted_euclidean_distance(a, b, weights);
        1.0 / (1.0 + dist)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_identical() {
        let v = Vector::from_data(vec![1, -1, 1, -1]);
        let sim = Similarity::cosine(&v, &v);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_opposite() {
        let a = Vector::from_data(vec![1, -1, 1, -1]);
        let b = Vector::from_data(vec![-1, 1, -1, 1]);
        let sim = Similarity::cosine(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_hamming() {
        let a = Vector::from_data(vec![1, -1, 1, -1]);
        let b = Vector::from_data(vec![1, 1, 1, -1]);
        let sim = Similarity::hamming(&a, &b);
        assert!((sim - 0.75).abs() < 1e-10); // 3 out of 4 match
    }

    #[test]
    fn test_euclidean() {
        let a = Vector::from_data(vec![1, 1, 1, 1]);
        let b = Vector::from_data(vec![1, 1, 1, 1]);
        let sim = Similarity::euclidean(&a, &b);
        assert!((sim - 1.0).abs() < 1e-10); // Distance 0 → similarity 1
    }

    #[test]
    fn test_agreement_identical() {
        let a = Vector::from_data(vec![1, -1, 1, -1]);
        let sim = Similarity::agreement(&a, &a);
        // All non-zero dims agree, none disagree: (4 - 0) / 4 = 1.0
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_agreement_opposite() {
        let a = Vector::from_data(vec![1, -1, 1, -1]);
        let b = Vector::from_data(vec![-1, 1, -1, 1]);
        let sim = Similarity::agreement(&a, &b);
        // All disagree: (0 - 4) / 4 = -1.0
        assert!((sim - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_agreement_mixed() {
        let a = Vector::from_data(vec![1, -1, 1, -1]);
        let b = Vector::from_data(vec![1, 1, 1, -1]);
        // agree: dims 0, 2, 3 (3); disagree: dim 1 (1)
        let sim = Similarity::agreement(&a, &b);
        assert!((sim - 0.5).abs() < 1e-10); // (3 - 1) / 4 = 0.5
    }

    #[test]
    fn test_agreement_with_zeros() {
        let a = Vector::from_data(vec![1, 0, 1, -1]);
        let b = Vector::from_data(vec![1, 0, -1, -1]);
        // products: 1*1=1(agree), 0*0=0(neither), 1*-1=-1(disagree), -1*-1=1(agree)
        let sim = Similarity::agreement(&a, &b);
        assert!((sim - 0.25).abs() < 1e-10); // (2 - 1) / 4 = 0.25
    }

    #[test]
    fn test_chebyshev_identical() {
        let a = Vector::from_data(vec![1, -1, 1, -1]);
        let sim = Similarity::chebyshev(&a, &a);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_chebyshev_opposite() {
        let a = Vector::from_data(vec![1, -1, 1, -1]);
        let b = Vector::from_data(vec![-1, 1, -1, 1]);
        let sim = Similarity::chebyshev(&a, &b);
        // Max diff is |1 - (-1)| = 2, similarity = 1 - 2/2 = 0.0
        assert!(sim.abs() < 1e-10);
    }

    #[test]
    fn test_chebyshev_distance_raw() {
        let a = Vector::from_data(vec![1, -1, 0, 1]);
        let b = Vector::from_data(vec![1, 1, 0, 1]);
        let dist = Similarity::chebyshev_distance(&a, &b);
        // Only dim 1 differs: |-1 - 1| = 2
        assert!((dist - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_minkowski_p1_is_manhattan() {
        let a = Vector::from_data(vec![1, -1, 1, -1]);
        let b = Vector::from_data(vec![-1, 1, 0, -1]);
        let mink = Similarity::minkowski_distance(&a, &b, 1.0);
        let manh = Similarity::manhattan_distance(&a, &b);
        assert!((mink - manh).abs() < 1e-10);
    }

    #[test]
    fn test_minkowski_p2_is_euclidean() {
        let a = Vector::from_data(vec![1, -1, 1, -1]);
        let b = Vector::from_data(vec![-1, 1, 0, -1]);
        let mink = Similarity::minkowski_distance(&a, &b, 2.0);
        let eucl = Similarity::euclidean_distance(&a, &b);
        assert!((mink - eucl).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_cosine_uniform_weights() {
        let a = Vector::from_data(vec![1, -1, 1, -1]);
        let b = Vector::from_data(vec![1, -1, 1, -1]);
        let weights = vec![1.0; 4];
        let sim = Similarity::weighted_cosine(&a, &b, &weights);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_cosine_zeroed_dims() {
        let a = Vector::from_data(vec![1, -1, 1, -1]);
        let b = Vector::from_data(vec![1, 1, 1, -1]);
        // Zero out the disagreeing dimension (dim 1)
        let weights = vec![1.0, 0.0, 1.0, 1.0];
        let sim = Similarity::weighted_cosine(&a, &b, &weights);
        // After zeroing dim 1, remaining dims all agree
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_euclidean_uniform() {
        let a = Vector::from_data(vec![1, -1, 1, -1]);
        let b = Vector::from_data(vec![1, -1, 1, -1]);
        let weights = vec![1.0; 4];
        let dist = Similarity::weighted_euclidean_distance(&a, &b, &weights);
        assert!(dist.abs() < 1e-10);
    }

    #[test]
    fn test_weighted_euclidean_emphasis() {
        let a = Vector::from_data(vec![1, 0, 0, 0]);
        let b = Vector::from_data(vec![-1, 0, 0, 0]);
        // High weight on dim 0 should amplify the difference
        let weights_high = vec![10.0, 1.0, 1.0, 1.0];
        let weights_low = vec![0.1, 1.0, 1.0, 1.0];
        let dist_high = Similarity::weighted_euclidean_distance(&a, &b, &weights_high);
        let dist_low = Similarity::weighted_euclidean_distance(&a, &b, &weights_low);
        assert!(dist_high > dist_low);
    }

    #[test]
    fn test_compute_dispatches_agreement() {
        let a = Vector::from_data(vec![1, -1, 1, -1]);
        let direct = Similarity::agreement(&a, &a);
        let via_compute = Similarity::compute(&a, &a, Metric::Agreement);
        assert!((direct - via_compute).abs() < 1e-10);
    }

    #[test]
    fn test_compute_dispatches_chebyshev() {
        let a = Vector::from_data(vec![1, -1, 1, -1]);
        let b = Vector::from_data(vec![-1, 1, -1, 1]);
        let direct = Similarity::chebyshev(&a, &b);
        let via_compute = Similarity::compute(&a, &b, Metric::Chebyshev);
        assert!((direct - via_compute).abs() < 1e-10);
    }
}
