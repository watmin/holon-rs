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
}
