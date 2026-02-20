//! Online subspace learning via CCIPCA (Candid Covariance-free Incremental PCA).
//!
//! Learns the low-dimensional manifold occupied by "familiar" vectors, then
//! scores new vectors by their residual distance from that manifold.
//!
//! # Algorithm: Weng et al., 2003
//!
//! Tracks k unnormalized eigenvectors whose L2 norms approximate eigenvalues.
//! Each update deflates the input through the learned components, so each
//! component captures a successively smaller slice of variance.
//!
//! # Usage
//!
//! ```rust
//! use holon::memory::OnlineSubspace;
//!
//! let mut sub = OnlineSubspace::new(4096, 32);
//!
//! // Training phase: update with normal observations
//! for _ in 0..200 {
//!     let x: Vec<f64> = vec![1.0; 4096]; // placeholder
//!     sub.update(&x);
//! }
//!
//! // Scoring: residual > threshold → anomalous
//! let probe: Vec<f64> = vec![0.0; 4096];
//! if sub.residual(&probe) > sub.threshold() {
//!     println!("anomaly");
//! }
//! ```

use serde::{Deserialize, Serialize};

/// Serializable snapshot of an OnlineSubspace for persistence and distribution.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubspaceSnapshot {
    pub dim: usize,
    pub k: usize,
    pub amnesia: f64,
    pub ema_alpha: f64,
    pub sigma_mult: f64,
    pub reorth_interval: usize,
    pub n: usize,
    pub mean: Vec<f64>,
    /// Flat row-major: k rows × dim columns.
    pub components: Vec<f64>,
    pub res_ema: f64,
    pub res_var_ema: f64,
}

/// Online subspace learner using CCIPCA.
///
/// Inputs must be `&[f64]`. Callers convert bipolar [`Vector`] with
/// `vec.to_f64()` before passing in — this keeps the memory layer
/// decoupled from the VSA vector type.
///
/// [`Vector`]: crate::vector::Vector
#[derive(Clone, Debug)]
pub struct OnlineSubspace {
    dim: usize,
    k: usize,
    amnesia: f64,
    ema_alpha: f64,
    sigma_mult: f64,
    reorth_interval: usize,

    mean: Vec<f64>,
    /// Row-major flat storage: components[i * dim .. (i+1) * dim] = component i.
    components: Vec<f64>,
    n: usize,

    res_ema: f64,
    res_var_ema: f64,
    initialized: bool,
}

impl OnlineSubspace {
    /// Create with default parameters.
    ///
    /// Defaults match Python: `amnesia=2.0, ema_alpha=0.01, sigma_mult=3.5, reorth_interval=500`.
    pub fn new(dim: usize, k: usize) -> Self {
        Self::with_params(dim, k, 2.0, 0.01, 3.5, 500)
    }

    /// Create with explicit parameters.
    ///
    /// - `amnesia`: forgetting exponent (>1 forgets old data faster; 2.0 = moderate)
    /// - `ema_alpha`: EMA decay for threshold tracking (0.01 = slow, 0.1 = fast)
    /// - `sigma_mult`: standard deviations above EMA for adaptive threshold
    /// - `reorth_interval`: re-orthogonalize every N updates (0 = never)
    pub fn with_params(
        dim: usize,
        k: usize,
        amnesia: f64,
        ema_alpha: f64,
        sigma_mult: f64,
        reorth_interval: usize,
    ) -> Self {
        let k = k.min(dim);
        Self {
            dim,
            k,
            amnesia,
            ema_alpha,
            sigma_mult,
            reorth_interval,
            mean: vec![0.0; dim],
            components: vec![0.0; k * dim],
            n: 0,
            res_ema: 0.0,
            res_var_ema: 0.0,
            initialized: false,
        }
    }

    // --- Accessors ---

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn n(&self) -> usize {
        self.n
    }

    /// Adaptive anomaly threshold: EMA(residual) + sigma_mult * sqrt(variance).
    ///
    /// Returns `f64::INFINITY` until at least 2 observations have been seen.
    pub fn threshold(&self) -> f64 {
        if self.n < 2 {
            return f64::INFINITY;
        }
        self.res_ema + self.sigma_mult * self.res_var_ema.max(0.0).sqrt()
    }

    /// Approximate eigenvalues: L2 norms of the unnormalized component vectors.
    pub fn eigenvalues(&self) -> Vec<f64> {
        (0..self.k)
            .map(|i| self.component_norm(i))
            .collect()
    }

    /// Fraction of variance explained by the subspace (0.0–1.0).
    ///
    /// Estimated from recent residuals. Only meaningful after ≥10 updates.
    pub fn explained_ratio(&self) -> f64 {
        if self.n < 10 {
            return 0.0;
        }
        let total_var = self.dim as f64;
        let unexplained = self.res_ema * self.res_ema;
        (1.0 - unexplained / total_var).max(0.0)
    }

    // --- Core methods ---

    /// Update subspace with a new observation and return its residual.
    ///
    /// The residual is computed *before* the CCIPCA update so it matches
    /// what `residual()` would return — ensuring the adaptive threshold is
    /// calibrated against the same distribution as test-time scoring.
    ///
    /// # Panics
    /// Panics if `x.len() != self.dim`.
    pub fn update(&mut self, x: &[f64]) -> f64 {
        assert_eq!(
            x.len(),
            self.dim,
            "Expected dim={}, got {}",
            self.dim,
            x.len()
        );

        // Compute residual BEFORE updating (matches test-time residual())
        let res = if self.initialized {
            self.residual(x)
        } else {
            norm(x)
        };

        self.n += 1;
        let n = self.n as f64;
        let amn = self.amnesia;

        // Update running mean
        let n_minus_1_over_n = (n - 1.0) / n;
        let one_over_n = 1.0 / n;
        for (i, m) in self.mean.iter_mut().enumerate().take(self.dim) {
            *m = n_minus_1_over_n * *m + one_over_n * x[i];
        }

        // Center
        let mut x_c: Vec<f64> = x.iter().zip(self.mean.iter()).map(|(xi, mi)| xi - mi).collect();

        if !self.initialized && self.n == 1 {
            // Seed first component with the first centered observation
            self.components[..self.dim].copy_from_slice(&x_c[..self.dim]);
            self.initialized = true;
            // Update threshold EMA with this initial residual
            self.update_threshold_ema(res);
            return res;
        }

        // CCIPCA update for each component
        for i in 0..self.k {
            let v_norm = self.component_norm(i);

            if v_norm < 1e-10 {
                if norm(&x_c) > 1e-10 {
                    let scale = (1.0 + amn) / n;
                    let base = i * self.dim;
                    for (j, c) in self.components[base..base + self.dim].iter_mut().enumerate() {
                        *c = x_c[j] * scale;
                    }
                }
            } else {
                // u = v / v_norm
                let base = i * self.dim;
                let x_c_proj: f64 = (0..self.dim)
                    .map(|j| x_c[j] * self.components[base + j] / v_norm)
                    .sum();

                // Weng et al. CCIPCA update
                let decay = (n - 1.0 - amn) / n;
                let grow = (1.0 + amn) / n * x_c_proj;
                for (j, c) in self.components[base..base + self.dim].iter_mut().enumerate() {
                    *c = decay * *c + grow * x_c[j];
                }
            }

            // Deflate x_c for next component
            let v_new_norm = self.component_norm(i);
            if v_new_norm > 1e-10 {
                let base = i * self.dim;
                let proj: f64 = (0..self.dim)
                    .map(|j| x_c[j] * self.components[base + j] / v_new_norm)
                    .sum();
                for (j, xc) in x_c.iter_mut().enumerate().take(self.dim) {
                    *xc -= proj * self.components[base + j] / v_new_norm;
                }
            }
        }

        // Update adaptive threshold via EMA
        self.update_threshold_ema(res);

        // Periodic re-orthogonalization
        if self.reorth_interval > 0 && self.n % self.reorth_interval == 0 {
            self.reorthogonalize();
        }

        res
    }

    /// Score a vector without updating the subspace.
    ///
    /// Returns the residual norm (anomaly score). Higher = more anomalous.
    pub fn residual(&self, x: &[f64]) -> f64 {
        assert_eq!(
            x.len(),
            self.dim,
            "Expected dim={}, got {}",
            self.dim,
            x.len()
        );

        let mut x_c: Vec<f64> = x.iter().zip(self.mean.iter()).map(|(xi, mi)| xi - mi).collect();

        for i in 0..self.k {
            let v_norm = self.component_norm(i);
            if v_norm < 1e-10 {
                continue;
            }
            let base = i * self.dim;
            let proj: f64 = (0..self.dim)
                .map(|j| x_c[j] * self.components[base + j] / v_norm)
                .sum();
            for (j, xc) in x_c.iter_mut().enumerate().take(self.dim) {
                *xc -= proj * self.components[base + j] / v_norm;
            }
        }

        norm(&x_c)
    }

    /// Project onto learned subspace, returning k coefficients.
    pub fn project(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.dim);
        let x_c: Vec<f64> = x.iter().zip(self.mean.iter()).map(|(xi, mi)| xi - mi).collect();
        (0..self.k)
            .map(|i| {
                let v_norm = self.component_norm(i);
                if v_norm < 1e-10 {
                    return 0.0;
                }
                let base = i * self.dim;
                (0..self.dim)
                    .map(|j| x_c[j] * self.components[base + j] / v_norm)
                    .sum()
            })
            .collect()
    }

    /// Reconstruct vector from its subspace projection.
    pub fn reconstruct(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.dim);
        let coeffs = self.project(x);
        let mut result = self.mean.clone();
        for (i, &coeff) in coeffs.iter().enumerate().take(self.k) {
            let v_norm = self.component_norm(i);
            if v_norm < 1e-10 {
                continue;
            }
            let base = i * self.dim;
            for (j, r) in result.iter_mut().enumerate().take(self.dim) {
                *r += coeff * self.components[base + j] / v_norm;
            }
        }
        result
    }

    /// Extract the anomalous (out-of-subspace) component: `x - reconstruct(x)`.
    pub fn anomalous_component(&self, x: &[f64]) -> Vec<f64> {
        let rec = self.reconstruct(x);
        x.iter().zip(rec.iter()).map(|(xi, ri)| xi - ri).collect()
    }

    /// Batch update, returns a Vec of residuals.
    pub fn update_batch(&mut self, vectors: &[Vec<f64>]) -> Vec<f64> {
        vectors.iter().map(|v| self.update(v)).collect()
    }

    /// Export state for persistence or shipping.
    pub fn snapshot(&self) -> SubspaceSnapshot {
        SubspaceSnapshot {
            dim: self.dim,
            k: self.k,
            amnesia: self.amnesia,
            ema_alpha: self.ema_alpha,
            sigma_mult: self.sigma_mult,
            reorth_interval: self.reorth_interval,
            n: self.n,
            mean: self.mean.clone(),
            components: self.components.clone(),
            res_ema: self.res_ema,
            res_var_ema: self.res_var_ema,
        }
    }

    /// Restore from a snapshot.
    pub fn from_snapshot(snap: SubspaceSnapshot) -> Self {
        Self {
            dim: snap.dim,
            k: snap.k,
            amnesia: snap.amnesia,
            ema_alpha: snap.ema_alpha,
            sigma_mult: snap.sigma_mult,
            reorth_interval: snap.reorth_interval,
            n: snap.n,
            mean: snap.mean,
            components: snap.components,
            res_ema: snap.res_ema,
            res_var_ema: snap.res_var_ema,
            initialized: snap.n > 0,
        }
    }

    // --- Private helpers ---

    #[inline]
    fn component_norm(&self, i: usize) -> f64 {
        let base = i * self.dim;
        norm(&self.components[base..base + self.dim])
    }

    fn update_threshold_ema(&mut self, res: f64) {
        let mut alpha = self.ema_alpha;
        if self.n as f64 <= 1.0 / alpha {
            alpha = 1.0 / self.n as f64; // simple average during warmup
        }
        let delta = res - self.res_ema;
        self.res_ema += alpha * delta;
        self.res_var_ema = (1.0 - alpha) * self.res_var_ema + alpha * delta * delta;
    }

    /// Modified Gram-Schmidt re-orthogonalization preserving component norms.
    fn reorthogonalize(&mut self) {
        let norms: Vec<f64> = (0..self.k).map(|i| self.component_norm(i)).collect();

        for i in 0..self.k {
            if norms[i] < 1e-10 {
                continue;
            }
            for (j, &nj) in norms.iter().enumerate().take(i) {
                if nj < 1e-10 {
                    continue;
                }
                // proj = dot(component_i, unit_j)
                let base_i = i * self.dim;
                let base_j = j * self.dim;
                let proj: f64 = (0..self.dim)
                    .map(|d| self.components[base_i + d] * self.components[base_j + d] / nj)
                    .sum();
                // component_i -= proj * unit_j
                for d in 0..self.dim {
                    let sub = proj * self.components[base_j + d] / nj;
                    self.components[base_i + d] -= sub;
                }
            }

            // Restore original norm (eigenvalue estimate)
            let new_norm = self.component_norm(i);
            if new_norm > 1e-10 {
                let scale = norms[i] / new_norm;
                let base = i * self.dim;
                for d in 0..self.dim {
                    self.components[base + d] *= scale;
                }
            }
        }
    }
}

#[inline]
fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate vectors from a low-rank distribution (span of 3 basis vectors).
    fn low_rank_sample(rng_state: &mut u64, dim: usize) -> Vec<f64> {
        // Simple LCG for deterministic but varied samples
        let basis: Vec<Vec<f64>> = (0..3)
            .map(|b| {
                (0..dim)
                    .map(|i| {
                        if i % 3 == b {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect();
        let coeffs: Vec<f64> = (0..3)
            .map(|_| {
                *rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((*rng_state >> 33) as f64) / (u32::MAX as f64) * 2.0 - 1.0
            })
            .collect();
        let mut result = vec![0.0f64; dim];
        for (c, b) in coeffs.iter().zip(basis.iter()) {
            for i in 0..dim {
                result[i] += c * b[i];
            }
        }
        result
    }

    fn random_sample(rng_state: &mut u64, dim: usize) -> Vec<f64> {
        (0..dim)
            .map(|_| {
                *rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((*rng_state >> 33) as f64) / (u32::MAX as f64) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn test_threshold_infinite_before_training() {
        let sub = OnlineSubspace::new(64, 4);
        assert_eq!(sub.threshold(), f64::INFINITY);
    }

    #[test]
    fn test_threshold_finite_after_training() {
        let mut sub = OnlineSubspace::new(64, 4);
        let mut rng = 42u64;
        for _ in 0..50 {
            let v = low_rank_sample(&mut rng, 64);
            sub.update(&v);
        }
        assert!(sub.threshold().is_finite(), "threshold should be finite after training");
    }

    #[test]
    fn test_residual_in_distribution_below_threshold() {
        let dim = 256;
        let mut sub = OnlineSubspace::with_params(dim, 8, 2.0, 0.01, 2.5, 500);
        let mut rng = 42u64;

        // Train on 200 low-rank samples
        for _ in 0..200 {
            let v = low_rank_sample(&mut rng, dim);
            sub.update(&v);
        }

        // In-distribution: should mostly be below threshold
        let mut above = 0;
        for _ in 0..20 {
            let v = low_rank_sample(&mut rng, dim);
            if sub.residual(&v) > sub.threshold() {
                above += 1;
            }
        }
        assert!(above <= 5, "Expected at most 5/20 in-distribution samples above threshold, got {}", above);
    }

    #[test]
    fn test_residual_out_of_distribution_above_threshold() {
        let dim = 256;
        let mut sub = OnlineSubspace::with_params(dim, 8, 2.0, 0.01, 2.5, 500);
        let mut rng = 42u64;

        // Train on low-rank samples (only 3 dimensions active)
        for _ in 0..300 {
            let v = low_rank_sample(&mut rng, dim);
            sub.update(&v);
        }

        // Out-of-distribution: random full-rank vectors should exceed threshold
        let mut above = 0;
        let mut rng2 = 999u64;
        for _ in 0..10 {
            let v = random_sample(&mut rng2, dim);
            if sub.residual(&v) > sub.threshold() {
                above += 1;
            }
        }
        assert!(above >= 7, "Expected most OOD samples above threshold, got {}/10", above);
    }

    #[test]
    fn test_snapshot_round_trip() {
        let dim = 128;
        let mut sub = OnlineSubspace::new(dim, 8);
        let mut rng = 42u64;

        for _ in 0..100 {
            let v = low_rank_sample(&mut rng, dim);
            sub.update(&v);
        }

        let snap = sub.snapshot();
        let restored = OnlineSubspace::from_snapshot(snap);

        // Both should produce identical residuals
        let mut rng2 = 1234u64;
        for _ in 0..10 {
            let v = low_rank_sample(&mut rng2, dim);
            let r1 = sub.residual(&v);
            let r2 = restored.residual(&v);
            assert!(
                (r1 - r2).abs() < 1e-10,
                "Residuals differ after snapshot round-trip: {} vs {}",
                r1,
                r2
            );
        }
    }

    #[test]
    fn test_project_reconstruct_anomalous_component() {
        let dim = 128;
        let mut sub = OnlineSubspace::new(dim, 8);
        let mut rng = 42u64;

        for _ in 0..200 {
            let v = low_rank_sample(&mut rng, dim);
            sub.update(&v);
        }

        let v = low_rank_sample(&mut rng, dim);
        let rec = sub.reconstruct(&v);
        let anom = sub.anomalous_component(&v);

        // x = reconstruct(x) + anomalous_component(x)
        for i in 0..dim {
            assert!(
                (v[i] - rec[i] - anom[i]).abs() < 1e-10,
                "reconstruct + anomalous_component should equal original at dim {}", i
            );
        }
    }

    #[test]
    fn test_explained_ratio_increases_with_training() {
        let dim = 64;
        let mut sub = OnlineSubspace::with_params(dim, 4, 2.0, 0.01, 3.5, 0);
        let mut rng = 42u64;

        // After minimal training, explained ratio is 0
        assert_eq!(sub.explained_ratio(), 0.0);

        for _ in 0..300 {
            let v = low_rank_sample(&mut rng, dim);
            sub.update(&v);
        }

        // After substantial training on low-rank data, should explain most variance
        assert!(
            sub.explained_ratio() > 0.5,
            "Expected explained_ratio > 0.5 after training, got {}",
            sub.explained_ratio()
        );
    }
}
