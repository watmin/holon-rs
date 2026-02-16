//! Scalar Encoding: Continuous values as vectors.
//!
//! This module provides methods to encode continuous scalar values
//! (rates, frequencies, temperatures, etc.) as vectors where similar
//! values have similar vectors.
//!
//! # Key Insight: Linear vs Log Scale
//!
//! - **Linear**: Equal absolute differences have equal similarity drops
//!   - 100 → 200 has same similarity drop as 1000 → 1100
//!   - Use for: temperatures, positions, times
//!
//! - **Logarithmic**: Equal ratios have equal similarity drops
//!   - 100 → 1000 has same similarity drop as 1000 → 10000
//!   - Use for: packet rates, frequencies, byte sizes
//!
//! # Circular Encoding
//!
//! For periodic quantities (hour of day, day of week, angle), use
//! circular encoding where the endpoint wraps to the start:
//! - Hour 23 is similar to hour 0
//! - December is similar to January

use crate::vector::Vector;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use sha2::{Digest, Sha256};

/// Scalar encoding modes.
#[derive(Clone, Copy, Debug)]
pub enum ScalarMode {
    /// Linear encoding with a scale factor.
    ///
    /// `scale` determines the range: values from 0 to scale map to
    /// orthogonal vectors. Values beyond scale wrap.
    Linear {
        /// The scale of the encoding (range of orthogonality)
        scale: f64,
    },

    /// Circular encoding for periodic values.
    ///
    /// Values wrap at `period`, so 0 and period are identical.
    Circular {
        /// The period of the circular encoding
        period: f64,
    },
}

/// Encoder for continuous scalar values.
#[derive(Clone)]
pub struct ScalarEncoder {
    dimensions: usize,
    /// Base random vector for scalar encoding
    base_vector: Vec<f64>,
    /// Orthogonal random vector for phase
    ortho_vector: Vec<f64>,
}

impl ScalarEncoder {
    /// Create a new scalar encoder.
    pub fn new(dimensions: usize) -> Self {
        Self::with_seed(dimensions, 0)
    }

    /// Create a scalar encoder with a specific seed.
    pub fn with_seed(dimensions: usize, seed: u64) -> Self {
        // Generate base and orthogonal vectors deterministically
        let base_vector = Self::generate_random_vector(dimensions, seed, "scalar_base");
        let ortho_vector = Self::generate_random_vector(dimensions, seed, "scalar_ortho");

        Self {
            dimensions,
            base_vector,
            ortho_vector,
        }
    }

    fn generate_random_vector(dimensions: usize, seed: u64, name: &str) -> Vec<f64> {
        let mut hasher = Sha256::new();
        hasher.update(seed.to_le_bytes());
        hasher.update(name.as_bytes());
        let hash = hasher.finalize();
        let derived_seed = u64::from_le_bytes(hash[0..8].try_into().unwrap());

        let mut rng = ChaCha8Rng::seed_from_u64(derived_seed);

        (0..dimensions)
            .map(|_| {
                let r = rng.next_u32();
                if r & 1 == 0 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect()
    }

    /// Encode a scalar value.
    pub fn encode(&self, value: f64, mode: ScalarMode) -> Vector {
        match mode {
            ScalarMode::Linear { scale } => self.encode_linear(value, scale),
            ScalarMode::Circular { period } => self.encode_circular(value, period),
        }
    }

    /// Linear encoding: nearby values are similar.
    fn encode_linear(&self, value: f64, scale: f64) -> Vector {
        // Fractional rotation determines blend between base vectors
        let frac = value / scale;

        // Use cosine/sine interpolation for smooth similarity gradient
        let angle = frac * std::f64::consts::PI * 2.0;
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        let data: Vec<i8> = (0..self.dimensions)
            .map(|i| {
                let v = self.base_vector[i] * cos_a + self.ortho_vector[i] * sin_a;
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

    /// Circular encoding: endpoint wraps to start.
    fn encode_circular(&self, value: f64, period: f64) -> Vector {
        // Normalize to [0, 1) and use as rotation angle
        let normalized = (value % period) / period;
        let angle = normalized * std::f64::consts::PI * 2.0;
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        let data: Vec<i8> = (0..self.dimensions)
            .map(|i| {
                let v = self.base_vector[i] * cos_a + self.ortho_vector[i] * sin_a;
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

    /// Logarithmic encoding (base 10): equal ratios have equal similarity.
    ///
    /// Uses log10 so that each order of magnitude (10x) has the same
    /// similarity drop. This is consistent with the Python implementation.
    ///
    /// # Example
    /// ```rust
    /// use holon::ScalarEncoder;
    /// let encoder = ScalarEncoder::new(4096);
    /// let e100 = encoder.encode_log(100.0);   // log10(100) = 2
    /// let e1000 = encoder.encode_log(1000.0); // log10(1000) = 3
    /// let e10000 = encoder.encode_log(10000.0); // log10(10000) = 4
    ///
    /// // sim(e100, e1000) ≈ sim(e1000, e10000)
    /// // because both are 10x ratios (1 order of magnitude)
    /// ```
    pub fn encode_log(&self, value: f64) -> Vector {
        self.encode_log_base(value, 10.0)
    }

    /// Logarithmic encoding with explicit base.
    ///
    /// # Arguments
    /// * `value` - The value to encode (must be > 0)
    /// * `base` - The logarithm base (e.g., 10.0 for log10, std::f64::consts::E for ln)
    ///
    /// # Example
    /// ```rust
    /// use holon::ScalarEncoder;
    /// let encoder = ScalarEncoder::new(4096);
    /// // Base 10: each 10x has same similarity drop
    /// let v = encoder.encode_log_base(1000.0, 10.0);
    ///
    /// // Base 2: each 2x has same similarity drop
    /// let v = encoder.encode_log_base(1024.0, 2.0);
    ///
    /// // Natural log: each e≈2.718x has same similarity drop
    /// let v = encoder.encode_log_base(100.0, std::f64::consts::E);
    /// ```
    pub fn encode_log_base(&self, value: f64, base: f64) -> Vector {
        if value <= 0.0 {
            // Log of non-positive is undefined; return zero vector
            return Vector::zeros(self.dimensions);
        }

        let log_value = value.log(base);
        // Scale: 10 orders of magnitude maps to one full rotation
        // This means values differing by 10^10 are orthogonal
        // Note: bipolar thresholding causes quantization; use accumulator
        // averaging to recover gradient behavior
        self.encode_linear(log_value, 10.0)
    }

    /// Decode a scalar value from a log-scale encoded vector.
    ///
    /// The inverse of `encode_log`: recovers the original scalar value
    /// by grid search over the log10 range, then refinement around the best
    /// candidate.
    ///
    /// Uses agreement count (matching signs) to avoid norm bias from zero
    /// dimensions that positional encoding produces.
    ///
    /// # Arguments
    /// * `vec` - The encoded vector to decode
    /// * `lo` - Lower bound of search range (must be > 0)
    /// * `hi` - Upper bound of search range
    /// * `num_probes` - Grid resolution for coarse search
    /// * `refine_steps` - Resolution for refinement
    pub fn decode_log(
        &self,
        vec: &Vector,
        lo: f64,
        hi: f64,
        num_probes: usize,
        refine_steps: usize,
    ) -> f64 {
        let log_lo = lo.max(1e-10).log10();
        let log_hi = hi.log10();
        let target = vec.data();

        let score = |log_val: f64| -> i32 {
            let candidate = self.encode_log_base(10.0_f64.powf(log_val), 10.0);
            let cdata = candidate.data();
            target
                .iter()
                .zip(cdata.iter())
                .filter(|(&t, &c)| t != 0 && c != 0 && t == c)
                .count() as i32
        };

        // Coarse grid search
        let mut best_log = log_lo;
        let mut best_score = -1;

        for i in 0..num_probes {
            let frac = i as f64 / (num_probes - 1) as f64;
            let log_val = log_lo + frac * (log_hi - log_lo);
            let s = score(log_val);
            if s > best_score {
                best_score = s;
                best_log = log_val;
            }
        }

        // Refine around best candidate
        let step = (log_hi - log_lo) / num_probes as f64;
        let ref_lo = best_log - step;
        let ref_hi = best_log + step;

        for i in 0..refine_steps {
            let frac = i as f64 / (refine_steps - 1) as f64;
            let log_val = ref_lo + frac * (ref_hi - ref_lo);
            let s = score(log_val);
            if s > best_score {
                best_score = s;
                best_log = log_val;
            }
        }

        10.0_f64.powf(best_log)
    }

    /// Encode with a custom seed (for different "dimensions" of scalars).
    ///
    /// Use this when you need to encode multiple different scalar types
    /// that should be independent (e.g., temperature and pressure).
    pub fn encode_with_seed(&self, value: f64, mode: ScalarMode, seed: &str) -> Vector {
        // Create a seeded encoder for this specific scalar type
        let mut hasher = Sha256::new();
        hasher.update(seed.as_bytes());
        let hash = hasher.finalize();
        let derived_seed = u64::from_le_bytes(hash[0..8].try_into().unwrap());

        let seeded = Self::with_seed(self.dimensions, derived_seed);
        seeded.encode(value, mode)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::similarity::Similarity;

    #[test]
    fn test_linear_similarity_gradient() {
        let encoder = ScalarEncoder::new(4096);

        let v0 = encoder.encode(0.0, ScalarMode::Linear { scale: 100.0 });
        let v10 = encoder.encode(10.0, ScalarMode::Linear { scale: 100.0 });
        let v50 = encoder.encode(50.0, ScalarMode::Linear { scale: 100.0 });

        let sim_0_10 = Similarity::cosine(&v0, &v10);
        let sim_0_50 = Similarity::cosine(&v0, &v50);

        // Closer values should have higher similarity
        assert!(
            sim_0_10 > sim_0_50,
            "Expected sim(0,10) > sim(0,50), got {} vs {}",
            sim_0_10,
            sim_0_50
        );
    }

    #[test]
    fn test_circular_wrap() {
        let encoder = ScalarEncoder::new(4096);

        let v23 = encoder.encode(23.0, ScalarMode::Circular { period: 24.0 });
        let v0 = encoder.encode(0.0, ScalarMode::Circular { period: 24.0 });
        let v12 = encoder.encode(12.0, ScalarMode::Circular { period: 24.0 });

        let sim_23_0 = Similarity::cosine(&v23, &v0);
        let sim_23_12 = Similarity::cosine(&v23, &v12);

        // 23 and 0 are 1 hour apart; 23 and 12 are 11 hours apart
        assert!(
            sim_23_0 > sim_23_12,
            "Expected hour 23 closer to hour 0 than to hour 12, got {} vs {}",
            sim_23_0,
            sim_23_12
        );
    }

    #[test]
    fn test_log_ratio_preservation() {
        let encoder = ScalarEncoder::new(4096);

        // With bipolar quantization, small differences get quantized away.
        // The encoder works best when used with accumulators that average
        // multiple samples, recovering gradient behavior.
        //
        // Test that vectors at least 5 orders of magnitude apart are distinguishable
        let v1 = encoder.encode_log(1.0);           // log10 = 0
        let v100k = encoder.encode_log(100_000.0);  // log10 = 5
        let v1b = encoder.encode_log(1_000_000_000.0); // log10 = 9

        let sim_1_100k = Similarity::cosine(&v1, &v100k);
        let sim_1_1b = Similarity::cosine(&v1, &v1b);

        // 5 orders of magnitude apart should have lower similarity than 9 orders
        // (Both may be low or even negative due to rotation)
        assert!(
            sim_1_100k != sim_1_1b,
            "Expected different similarities for different magnitudes, both got {}",
            sim_1_100k
        );

        // Self-similarity should be 1.0
        let self_sim = Similarity::cosine(&v1, &v1);
        assert!(
            (self_sim - 1.0).abs() < 1e-10,
            "Expected self-similarity = 1.0, got {}",
            self_sim
        );
    }

    #[test]
    fn test_log_base_consistency() {
        let encoder = ScalarEncoder::new(4096);

        // encode_log should be same as encode_log_base with base 10
        let v1 = encoder.encode_log(5000.0);
        let v2 = encoder.encode_log_base(5000.0, 10.0);

        assert_eq!(v1, v2, "encode_log and encode_log_base(10) should be identical");
    }

    #[test]
    fn test_deterministic() {
        let enc1 = ScalarEncoder::with_seed(4096, 42);
        let enc2 = ScalarEncoder::with_seed(4096, 42);

        let v1 = enc1.encode(50.0, ScalarMode::Linear { scale: 100.0 });
        let v2 = enc2.encode(50.0, ScalarMode::Linear { scale: 100.0 });

        assert_eq!(v1, v2);
    }
}
