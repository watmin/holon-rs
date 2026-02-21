//! Accumulator: Frequency-preserving streaming operations.
//!
//! Accumulators are essential for building baselines from streaming data
//! where frequency information matters. Unlike simple bundling, accumulators
//! preserve the fact that a pattern seen 100 times should contribute 100x
//! more than a pattern seen once.
//!
//! # Key Insight
//!
//! The critical difference between accumulator and bundle:
//! - `bundle([a, a, a, a, a])` = `a` (idempotent)
//! - `accumulator.add(a) * 5` preserves that `a` was seen 5 times
//!
//! This matters for anomaly detection: common patterns should dominate
//! the baseline, so rare anomalies have low similarity.

use super::vector::Vector;

/// A frequency-preserving accumulator for streaming data.
///
/// Unlike bundling, accumulation preserves frequency information:
/// patterns seen many times contribute proportionally more.
#[derive(Clone, Debug)]
pub struct Accumulator {
    /// Running sum of all vectors (not thresholded)
    sums: Vec<f64>,
    /// Number of examples accumulated
    count: usize,
}

impl Accumulator {
    /// Create a new empty accumulator.
    pub fn new(dimensions: usize) -> Self {
        Self {
            sums: vec![0.0; dimensions],
            count: 0,
        }
    }

    /// Get the dimensionality.
    pub fn dimensions(&self) -> usize {
        self.sums.len()
    }

    /// Get the number of accumulated examples.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Add an example to the accumulator.
    ///
    /// Unlike bundling, this does NOT threshold after each addition,
    /// preserving frequency information.
    pub fn add(&mut self, example: &Vector) {
        assert_eq!(
            self.dimensions(),
            example.dimensions(),
            "Dimension mismatch in accumulator"
        );

        for (i, &v) in example.data().iter().enumerate() {
            self.sums[i] += v as f64;
        }
        self.count += 1;
    }

    /// Add an example with a specific weight.
    pub fn add_weighted(&mut self, example: &Vector, weight: f64) {
        assert_eq!(
            self.dimensions(),
            example.dimensions(),
            "Dimension mismatch in accumulator"
        );

        for (i, &v) in example.data().iter().enumerate() {
            self.sums[i] += (v as f64) * weight;
        }
        self.count += 1;
    }

    /// Normalize the accumulator to a unit vector (for similarity).
    ///
    /// Returns a continuous f64 vector suitable for cosine similarity.
    pub fn normalize(&self) -> Vector {
        let norm: f64 = self.sums.iter().map(|&x| x * x).sum::<f64>().sqrt();

        if norm < 1e-10 {
            return Vector::zeros(self.dimensions());
        }

        // Return as bipolar by thresholding the normalized values
        let data: Vec<i8> = self
            .sums
            .iter()
            .map(|&v| {
                let normalized = v / norm;
                if normalized > 0.0 {
                    1
                } else if normalized < 0.0 {
                    -1
                } else {
                    0
                }
            })
            .collect();

        Vector::from_data(data)
    }

    /// Get the raw f64 sums (for advanced use).
    pub fn raw_sums(&self) -> &[f64] {
        &self.sums
    }

    /// Normalize as f64 vector (preserving continuous values).
    pub fn normalize_f64(&self) -> Vec<f64> {
        let norm: f64 = self.sums.iter().map(|&x| x * x).sum::<f64>().sqrt();

        if norm < 1e-10 {
            return vec![0.0; self.dimensions()];
        }

        self.sums.iter().map(|&v| v / norm).collect()
    }

    /// Threshold the accumulator to bipolar {-1, 0, 1}.
    ///
    /// This loses frequency information but produces a clean bipolar vector.
    pub fn threshold(&self) -> Vector {
        let data: Vec<i8> = self
            .sums
            .iter()
            .map(|&v| {
                if v > 0.0 {
                    1
                } else if v < 0.0 {
                    -1
                } else {
                    0
                }
            })
            .collect();

        Vector::from_data(data)
    }

    /// Merge another accumulator into this one.
    ///
    /// Useful for parallel processing: accumulate in separate threads,
    /// then merge results.
    pub fn merge(&mut self, other: &Accumulator) {
        assert_eq!(
            self.dimensions(),
            other.dimensions(),
            "Dimension mismatch in accumulator merge"
        );

        for (i, &v) in other.sums.iter().enumerate() {
            self.sums[i] += v;
        }
        self.count += other.count;
    }

    /// Clear the accumulator to start fresh.
    pub fn clear(&mut self) {
        self.sums.fill(0.0);
        self.count = 0;
    }

    /// Apply exponential decay to the accumulator.
    ///
    /// Useful for time-weighted baselines where recent patterns
    /// should have more influence.
    pub fn decay(&mut self, factor: f64) {
        for v in &mut self.sums {
            *v *= factor;
        }
    }

    /// Estimate how close the accumulator is to saturation.
    ///
    /// Based on the capacity bound: N ≤ d / (2 * ln(k)) for d-dimensional
    /// vectors with k codebook items. Returns remaining capacity as a
    /// fraction in [0.0, 1.0].
    ///
    /// - 1.0 = completely empty
    /// - 0.0 = fully saturated
    pub fn capacity(&self, codebook_size: usize) -> f64 {
        let d = self.dimensions() as f64;
        if codebook_size < 2 {
            return 1.0;
        }

        let max_items = d / (2.0 * (codebook_size as f64).ln());

        let l2_sq: f64 = self.sums.iter().map(|&x| x * x).sum();
        let estimated_items = l2_sq / d;

        (1.0 - estimated_items / max_items).clamp(0.0, 1.0)
    }

    /// Quantum-inspired purity measure: how concentrated is the accumulator?
    ///
    /// For a single bipolar vector: purity ≈ 1.0.
    /// For N dissimilar vectors accumulated: purity ≈ 1/N.
    ///
    /// Analogous to Tr(ρ²) from quantum mechanics.
    pub fn purity(&self) -> f64 {
        let d = self.dimensions() as f64;
        if d == 0.0 {
            return 0.0;
        }

        let l2_sq: f64 = self.sums.iter().map(|&x| x * x).sum();
        if l2_sq < 1e-10 {
            return 0.0;
        }

        (d / l2_sq).min(1.0)
    }

    /// Participation ratio: effective number of active dimensions.
    ///
    /// PR = (sum v_i²)² / sum(v_i⁴)
    ///
    /// For a single bipolar vector of dimension d: PR = d.
    /// As structure concentrates into fewer dimensions, PR decreases.
    pub fn participation_ratio(&self) -> f64 {
        let l2_sq: f64 = self.sums.iter().map(|&x| x * x).sum();
        if l2_sq < 1e-10 {
            return 0.0;
        }

        let l4_sum: f64 = self.sums.iter().map(|&x| x * x * x * x).sum();
        if l4_sum < 1e-10 {
            return 0.0;
        }

        (l2_sq * l2_sq) / l4_sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accumulator_basic() {
        let mut acc = Accumulator::new(4);
        assert_eq!(acc.count(), 0);

        let v = Vector::from_data(vec![1, -1, 1, -1]);
        acc.add(&v);

        assert_eq!(acc.count(), 1);
        assert_eq!(acc.raw_sums(), &[1.0, -1.0, 1.0, -1.0]);
    }

    #[test]
    fn test_frequency_preservation() {
        let mut acc = Accumulator::new(4);

        let common = Vector::from_data(vec![1, 1, 1, 1]);
        let rare = Vector::from_data(vec![-1, -1, -1, -1]);

        // Add common 10 times
        for _ in 0..10 {
            acc.add(&common);
        }
        // Add rare once
        acc.add(&rare);

        // Sums should reflect frequency
        // Each dimension: 10 * 1 + 1 * (-1) = 9
        assert_eq!(acc.raw_sums(), &[9.0, 9.0, 9.0, 9.0]);
    }

    #[test]
    fn test_threshold() {
        let mut acc = Accumulator::new(4);
        acc.add(&Vector::from_data(vec![1, -1, 1, -1]));
        acc.add(&Vector::from_data(vec![1, 1, 1, -1]));

        // Sums: [2, 0, 2, -2]
        let thresholded = acc.threshold();
        assert_eq!(thresholded.data(), &[1, 0, 1, -1]);
    }

    #[test]
    fn test_merge() {
        let mut acc1 = Accumulator::new(4);
        let mut acc2 = Accumulator::new(4);

        acc1.add(&Vector::from_data(vec![1, 0, 0, 0]));
        acc2.add(&Vector::from_data(vec![0, 1, 0, 0]));

        acc1.merge(&acc2);

        assert_eq!(acc1.count(), 2);
        assert_eq!(acc1.raw_sums(), &[1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_decay() {
        let mut acc = Accumulator::new(4);
        acc.add(&Vector::from_data(vec![1, -1, 1, -1]));
        acc.add(&Vector::from_data(vec![1, -1, 1, -1]));

        // Sums: [2, -2, 2, -2]
        acc.decay(0.5);

        // After decay: [1, -1, 1, -1]
        assert_eq!(acc.raw_sums(), &[1.0, -1.0, 1.0, -1.0]);
    }

    // =========================================================================
    // Capacity, Purity, Participation Ratio Tests
    // =========================================================================

    fn make_bipolar(n: usize, seed: u64) -> Vector {
        use rand::prelude::*;
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let data: Vec<i8> = (0..n)
            .map(|_| if rng.gen_bool(0.5) { 1 } else { -1 })
            .collect();
        Vector::from_data(data)
    }

    #[test]
    fn test_capacity_empty() {
        let acc = Accumulator::new(1024);
        assert_eq!(acc.capacity(100), 1.0);
    }

    #[test]
    fn test_capacity_decreases() {
        let mut acc = Accumulator::new(1024);
        let first_cap;
        acc.add(&make_bipolar(1024, 0));
        first_cap = acc.capacity(100);

        for i in 1..50 {
            acc.add(&make_bipolar(1024, i));
        }
        let last_cap = acc.capacity(100);
        assert!(last_cap < first_cap);
    }

    #[test]
    fn test_purity_single_vector() {
        let mut acc = Accumulator::new(1024);
        acc.add(&make_bipolar(1024, 0));
        let p = acc.purity();
        assert!(p > 0.95, "Single vector purity should be ~1.0: {}", p);
    }

    #[test]
    fn test_purity_many_random() {
        let mut acc = Accumulator::new(1024);
        for i in 0..50 {
            acc.add(&make_bipolar(1024, i));
        }
        let p = acc.purity();
        assert!(p < 0.5, "50 random vectors should have low purity: {}", p);
    }

    #[test]
    fn test_purity_empty() {
        let acc = Accumulator::new(1024);
        assert_eq!(acc.purity(), 0.0);
    }

    #[test]
    fn test_participation_ratio_single() {
        let mut acc = Accumulator::new(1024);
        acc.add(&make_bipolar(1024, 0));
        let pr = acc.participation_ratio();
        assert!(pr > 900.0, "Single bipolar should have high PR: {}", pr);
    }

    #[test]
    fn test_participation_ratio_empty() {
        let acc = Accumulator::new(1024);
        assert_eq!(acc.participation_ratio(), 0.0);
    }

    #[test]
    fn test_participation_ratio_concentrated() {
        let mut acc = Accumulator::new(1024);
        // Manually set: energy in only 10 dims
        for i in 0..10 {
            acc.sums[i] = 100.0;
        }
        let pr = acc.participation_ratio();
        assert!(pr < 20.0, "Concentrated should have low PR: {}", pr);
    }
}
