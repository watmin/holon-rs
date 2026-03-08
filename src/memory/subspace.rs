//! Online subspace learning via CCIPCA (Candid Covariance-free Incremental PCA).
//!
//! Uses `ndarray` for vectorized dot products and element-wise operations,
//! giving significant speedups on the per-observation scoring and update steps.
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

use ndarray::{Array1, Array2, ArrayView1};
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
/// Internally stores the component matrix as `ndarray::Array2<f64>` (k × dim)
/// so that dot products, projections, and deflations compile down to
/// vectorized (SIMD) loops.
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

    mean: Array1<f64>,
    /// Row-major: row i = component i (shape k × dim).
    components: Array2<f64>,
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
            mean: Array1::zeros(dim),
            components: Array2::zeros((k, dim)),
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

        let x_view = ArrayView1::from(x);

        // Compute residual BEFORE updating (matches test-time residual())
        let res = if self.initialized {
            self.residual(x)
        } else {
            norm(x)
        };

        self.n += 1;
        let n = self.n as f64;
        let amn = self.amnesia;

        // Update running mean: mean = ((n-1)/n) * mean + (1/n) * x
        self.mean *= (n - 1.0) / n;
        self.mean.scaled_add(1.0 / n, &x_view);

        // Center
        let mut x_c = &x_view - &self.mean;

        if !self.initialized && self.n == 1 {
            self.components.row_mut(0).assign(&x_c);
            self.initialized = true;
            self.update_threshold_ema(res);
            return res;
        }

        // CCIPCA update for each component
        for i in 0..self.k {
            let v_norm = self.component_norm(i);

            if v_norm < 1e-10 {
                if x_c.dot(&x_c).sqrt() > 1e-10 {
                    let scale = (1.0 + amn) / n;
                    self.components.row_mut(i).assign(&(&x_c * scale));
                }
            } else {
                let x_c_proj: f64 = x_c.dot(&self.components.row(i)) / v_norm;

                let decay = (n - 1.0 - amn) / n;
                let grow = (1.0 + amn) / n * x_c_proj;
                self.components
                    .row_mut(i)
                    .zip_mut_with(&x_c, |c, &xc| {
                        *c = decay * *c + grow * xc;
                    });
            }

            // Deflate x_c for next component
            let v_new_norm = self.component_norm(i);
            if v_new_norm > 1e-10 {
                let proj: f64 = x_c.dot(&self.components.row(i)) / v_new_norm;
                x_c.scaled_add(-proj / v_new_norm, &self.components.row(i));
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

        let x_view = ArrayView1::from(x);
        let mut x_c = &x_view - &self.mean;

        for i in 0..self.k {
            let v_norm = self.component_norm(i);
            if v_norm < 1e-10 {
                continue;
            }
            let proj: f64 = x_c.dot(&self.components.row(i)) / v_norm;
            x_c.scaled_add(-proj / v_norm, &self.components.row(i));
        }

        x_c.dot(&x_c).sqrt()
    }

    /// Project onto learned subspace, returning k coefficients.
    pub fn project(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.dim);
        let x_view = ArrayView1::from(x);
        let x_c = &x_view - &self.mean;
        (0..self.k)
            .map(|i| {
                let v_norm = self.component_norm(i);
                if v_norm < 1e-10 {
                    return 0.0;
                }
                x_c.dot(&self.components.row(i)) / v_norm
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
            result.scaled_add(coeff / v_norm, &self.components.row(i));
        }
        result.to_vec()
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
            mean: self.mean.to_vec(),
            components: self
                .components
                .as_slice()
                .expect("components is contiguous")
                .to_vec(),
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
            mean: Array1::from_vec(snap.mean),
            components: Array2::from_shape_vec((snap.k, snap.dim), snap.components)
                .expect("components length must equal k × dim"),
            res_ema: snap.res_ema,
            res_var_ema: snap.res_var_ema,
            initialized: snap.n > 0,
        }
    }

    /// Measure directional alignment between two subspaces.
    ///
    /// Computes cosines of principal angles via SVD of the basis inner
    /// product matrix. Focuses on the top principal angles — the best-
    /// aligned directions — since minor components are typically noise.
    ///
    /// # Arguments
    /// * `other` — subspace to compare against
    /// * `top_angles` — number of top principal angles to average;
    ///   0 = use `max(3, min(k_a, k_b) / 4)`
    ///
    /// Returns a value in \[0, 1\]: 1.0 = same directions, 0.0 = orthogonal.
    ///
    /// Cost: O(k · dim) for the basis product, O(k²) for the SVD.
    pub fn subspace_alignment(&self, other: &OnlineSubspace, top_angles: usize) -> f64 {
        let u = self.active_basis();
        let v = other.active_basis();

        if u.nrows() == 0 || v.nrows() == 0 {
            return 0.0;
        }

        // M = U · Vᵀ  (active_u × active_v)
        let m = u.dot(&v.t());

        // Singular values from eigenvalues of MᵀM
        let mtm = m.t().dot(&m);
        let eigenvalues = jacobi_eigenvalues(&mtm);

        let mut cos_angles: Vec<f64> = eigenvalues
            .iter()
            .map(|&e| e.max(0.0).sqrt().min(1.0))
            .collect();
        cos_angles.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let n = cos_angles.len();
        if n == 0 {
            return 0.0;
        }

        let top = if top_angles == 0 {
            3usize.max(n / 4).min(n)
        } else {
            top_angles.min(n)
        };

        cos_angles[..top].iter().sum::<f64>() / top as f64
    }

    // --- Private helpers ---

    /// Unit-normalized basis vectors for active (non-zero-norm) components.
    fn active_basis(&self) -> Array2<f64> {
        let mut rows: Vec<Array1<f64>> = Vec::new();
        for i in 0..self.k {
            let v_norm = self.component_norm(i);
            if v_norm > 1e-10 {
                rows.push(self.components.row(i).to_owned() / v_norm);
            }
        }
        if rows.is_empty() {
            return Array2::zeros((0, self.dim));
        }
        let n = rows.len();
        let mut basis = Array2::zeros((n, self.dim));
        for (i, row) in rows.iter().enumerate() {
            basis.row_mut(i).assign(row);
        }
        basis
    }

    #[inline]
    fn component_norm(&self, i: usize) -> f64 {
        let row = self.components.row(i);
        row.dot(&row).sqrt()
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
            for j in 0..i {
                if norms[j] < 1e-10 {
                    continue;
                }
                // Clone row j to avoid simultaneous borrow of rows i and j.
                let unit_j = self.components.row(j).to_owned() / norms[j];
                let proj: f64 = self.components.row(i).dot(&unit_j);
                self.components.row_mut(i).scaled_add(-proj, &unit_j);
            }

            // Restore original norm (eigenvalue estimate)
            let new_norm = self.component_norm(i);
            if new_norm > 1e-10 {
                let scale = norms[i] / new_norm;
                self.components
                    .row_mut(i)
                    .mapv_inplace(|v| v * scale);
            }
        }
    }
}

// =============================================================================
// StripedSubspace — N independent subspaces for crosstalk-free attribution
// =============================================================================

/// Serializable snapshot of a StripedSubspace.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StripedSubspaceSnapshot {
    pub stripes: Vec<SubspaceSnapshot>,
}

/// N independent OnlineSubspaces, one per stripe.
///
/// Each stripe learns and scores its own portion of the encoded data.
/// Aggregate residual is the root-sum-of-squares (RSS) of per-stripe
/// residuals, preserving the geometric interpretation of anomaly magnitude.
///
/// Used with [`Encoder::encode_walkable_striped`] which distributes leaf
/// bindings across stripes via FQDN path hashing.
///
/// [`Encoder::encode_walkable_striped`]: crate::kernel::Encoder::encode_walkable_striped
#[derive(Clone, Debug)]
pub struct StripedSubspace {
    stripes: Vec<OnlineSubspace>,
}

impl StripedSubspace {
    /// Create N independent subspaces, each with the given dimensionality and components.
    pub fn new(dim: usize, k: usize, n_stripes: usize) -> Self {
        Self {
            stripes: (0..n_stripes)
                .map(|_| OnlineSubspace::new(dim, k))
                .collect(),
        }
    }

    /// Create with explicit subspace parameters.
    pub fn with_params(
        dim: usize,
        k: usize,
        n_stripes: usize,
        amnesia: f64,
        ema_alpha: f64,
        sigma_mult: f64,
        reorth_interval: usize,
    ) -> Self {
        Self {
            stripes: (0..n_stripes)
                .map(|_| OnlineSubspace::with_params(dim, k, amnesia, ema_alpha, sigma_mult, reorth_interval))
                .collect(),
        }
    }

    pub fn n_stripes(&self) -> usize {
        self.stripes.len()
    }

    pub fn dim(&self) -> usize {
        self.stripes.first().map_or(0, |s| s.dim())
    }

    pub fn k(&self) -> usize {
        self.stripes.first().map_or(0, |s| s.k())
    }

    /// Total observations fed to the stripes (uses stripe 0 as reference).
    pub fn n(&self) -> usize {
        self.stripes.first().map_or(0, |s| s.n())
    }

    /// Update all stripes with their corresponding vectors.
    ///
    /// Returns the RSS aggregate residual.
    ///
    /// # Panics
    /// Panics if `stripe_vecs.len() != n_stripes` or any vec has wrong dim.
    pub fn update(&mut self, stripe_vecs: &[Vec<f64>]) -> f64 {
        assert_eq!(stripe_vecs.len(), self.stripes.len());
        let mut sum_sq = 0.0;
        for (sub, vec) in self.stripes.iter_mut().zip(stripe_vecs.iter()) {
            let r = sub.update(vec);
            sum_sq += r * r;
        }
        sum_sq.sqrt()
    }

    /// Per-stripe residual profile: the N-dim vector of individual stripe
    /// residuals.  This is the directional signal — the *pattern* of which
    /// stripes are anomalous — complementing the scalar magnitude (RSS).
    pub fn residual_profile(&self, stripe_vecs: &[Vec<f64>]) -> Vec<f64> {
        assert_eq!(stripe_vecs.len(), self.stripes.len());
        self.stripes.iter().zip(stripe_vecs.iter())
            .map(|(sub, vec)| sub.residual(vec))
            .collect()
    }

    /// RSS aggregate residual across all stripes.
    pub fn residual(&self, stripe_vecs: &[Vec<f64>]) -> f64 {
        let profile = self.residual_profile(stripe_vecs);
        profile.iter().map(|r| r * r).sum::<f64>().sqrt()
    }

    /// RSS aggregate threshold across all stripes.
    pub fn threshold(&self) -> f64 {
        let mut sum_sq = 0.0;
        for sub in &self.stripes {
            let t = sub.threshold();
            if t.is_infinite() {
                return f64::INFINITY;
            }
            sum_sq += t * t;
        }
        sum_sq.sqrt()
    }

    /// Anomalous component for a single stripe (for drilldown).
    pub fn anomalous_component(&self, stripe_vecs: &[Vec<f64>], stripe_idx: usize) -> Vec<f64> {
        self.stripes[stripe_idx].anomalous_component(&stripe_vecs[stripe_idx])
    }

    /// Per-stripe residual (for diagnostics).
    pub fn stripe_residual(&self, stripe_vecs: &[Vec<f64>], stripe_idx: usize) -> f64 {
        self.stripes[stripe_idx].residual(&stripe_vecs[stripe_idx])
    }

    /// Per-stripe threshold (for diagnostics).
    pub fn stripe_threshold(&self, stripe_idx: usize) -> f64 {
        self.stripes[stripe_idx].threshold()
    }

    /// Access an individual stripe subspace.
    pub fn stripe(&self, idx: usize) -> &OnlineSubspace {
        &self.stripes[idx]
    }

    /// Export state for persistence.
    pub fn snapshot(&self) -> StripedSubspaceSnapshot {
        StripedSubspaceSnapshot {
            stripes: self.stripes.iter().map(|s| s.snapshot()).collect(),
        }
    }

    /// Restore from a snapshot.
    pub fn from_snapshot(snap: StripedSubspaceSnapshot) -> Self {
        Self {
            stripes: snap.stripes.into_iter().map(OnlineSubspace::from_snapshot).collect(),
        }
    }
}

#[inline]
fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Eigenvalues of a small symmetric matrix via Jacobi rotations.
///
/// Returns eigenvalues sorted descending. Intended for k×k matrices
/// where k is the number of active subspace components (typically 3–128),
/// so O(k³) convergence is fine.
fn jacobi_eigenvalues(a: &Array2<f64>) -> Vec<f64> {
    let n = a.nrows();
    debug_assert_eq!(n, a.ncols());
    if n == 0 {
        return vec![];
    }

    let mut s = a.clone();
    let max_iter = 100 * n * n;
    let tol = 1e-12;

    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_val = 0.0f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let v = s[[i, j]].abs();
                if v > max_val {
                    max_val = v;
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < tol {
            break;
        }

        // Jacobi rotation angle
        let diff = s[[q, q]] - s[[p, p]];
        let t = if diff.abs() < 1e-15 {
            1.0f64
        } else {
            let tau = diff / (2.0 * s[[p, q]]);
            let sign = if tau >= 0.0 { 1.0 } else { -1.0 };
            sign / (tau.abs() + (1.0 + tau * tau).sqrt())
        };

        let c = 1.0 / (1.0 + t * t).sqrt();
        let sv = t * c;

        // Apply rotation to rows/columns p, q
        let s_pp = s[[p, p]];
        let s_qq = s[[q, q]];
        let s_pq = s[[p, q]];

        s[[p, p]] = c * c * s_pp - 2.0 * sv * c * s_pq + sv * sv * s_qq;
        s[[q, q]] = sv * sv * s_pp + 2.0 * sv * c * s_pq + c * c * s_qq;
        s[[p, q]] = 0.0;
        s[[q, p]] = 0.0;

        for i in 0..n {
            if i != p && i != q {
                let s_ip = s[[i, p]];
                let s_iq = s[[i, q]];
                s[[i, p]] = c * s_ip - sv * s_iq;
                s[[p, i]] = s[[i, p]];
                s[[i, q]] = sv * s_ip + c * s_iq;
                s[[q, i]] = s[[i, q]];
            }
        }
    }

    let mut eigenvalues: Vec<f64> = (0..n).map(|i| s[[i, i]]).collect();
    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    eigenvalues
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate vectors from a low-rank distribution (span of 3 basis vectors).
    fn low_rank_sample(rng_state: &mut u64, dim: usize) -> Vec<f64> {
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

        for _ in 0..200 {
            let v = low_rank_sample(&mut rng, dim);
            sub.update(&v);
        }

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

        for _ in 0..300 {
            let v = low_rank_sample(&mut rng, dim);
            sub.update(&v);
        }

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

        for i in 0..dim {
            assert!(
                (v[i] - rec[i] - anom[i]).abs() < 1e-10,
                "reconstruct + anomalous_component should equal original at dim {}", i
            );
        }
    }

    #[test]
    fn test_subspace_alignment_same_distribution() {
        let dim = 256;
        let mut sub_a = OnlineSubspace::new(dim, 8);
        let mut sub_b = OnlineSubspace::new(dim, 8);
        let mut rng = 42u64;

        for _ in 0..200 {
            let v = low_rank_sample(&mut rng, dim);
            sub_a.update(&v);
            sub_b.update(&v);
        }

        let alignment = sub_a.subspace_alignment(&sub_b, 0);
        assert!(
            alignment > 0.9,
            "Same-distribution subspaces should be well-aligned, got {}",
            alignment
        );
    }

    #[test]
    fn test_subspace_alignment_orthogonal_distributions() {
        let dim = 256;
        let mut sub_a = OnlineSubspace::new(dim, 8);
        let mut sub_b = OnlineSubspace::new(dim, 8);

        // Distribution A: signal only in even dimensions
        let mut rng_a = 42u64;
        for _ in 0..200 {
            rng_a = rng_a
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let coeff = (rng_a >> 33) as f64 / u32::MAX as f64 * 2.0 - 1.0;
            let v: Vec<f64> = (0..dim)
                .map(|i| if i < dim / 2 { coeff * ((i + 1) as f64) } else { 0.0 })
                .collect();
            sub_a.update(&v);
        }

        // Distribution B: signal only in odd dimensions
        let mut rng_b = 999u64;
        for _ in 0..200 {
            rng_b = rng_b
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let coeff = (rng_b >> 33) as f64 / u32::MAX as f64 * 2.0 - 1.0;
            let v: Vec<f64> = (0..dim)
                .map(|i| if i >= dim / 2 { coeff * ((i + 1) as f64) } else { 0.0 })
                .collect();
            sub_b.update(&v);
        }

        let alignment = sub_a.subspace_alignment(&sub_b, 0);
        assert!(
            alignment < 0.3,
            "Orthogonal subspaces should have low alignment, got {}",
            alignment
        );
    }

    #[test]
    fn test_subspace_alignment_empty() {
        let sub_a = OnlineSubspace::new(64, 4);
        let sub_b = OnlineSubspace::new(64, 4);
        assert_eq!(sub_a.subspace_alignment(&sub_b, 0), 0.0);
    }

    #[test]
    fn test_subspace_alignment_with_explicit_top_angles() {
        let dim = 128;
        let mut sub_a = OnlineSubspace::new(dim, 8);
        let mut sub_b = OnlineSubspace::new(dim, 8);
        let mut rng = 42u64;

        for _ in 0..200 {
            let v = low_rank_sample(&mut rng, dim);
            sub_a.update(&v);
            sub_b.update(&v);
        }

        let align_default = sub_a.subspace_alignment(&sub_b, 0);
        let align_top1 = sub_a.subspace_alignment(&sub_b, 1);

        // Both should be high for same-distribution
        assert!(align_default > 0.8, "default alignment: {}", align_default);
        assert!(align_top1 > 0.8, "top-1 alignment: {}", align_top1);
    }

    #[test]
    fn test_explained_ratio_increases_with_training() {
        let dim = 64;
        let mut sub = OnlineSubspace::with_params(dim, 4, 2.0, 0.01, 3.5, 0);
        let mut rng = 42u64;

        assert_eq!(sub.explained_ratio(), 0.0);

        for _ in 0..300 {
            let v = low_rank_sample(&mut rng, dim);
            sub.update(&v);
        }

        assert!(
            sub.explained_ratio() > 0.5,
            "Expected explained_ratio > 0.5 after training, got {}",
            sub.explained_ratio()
        );
    }

    // =========================================================================
    // StripedSubspace Tests
    // =========================================================================

    #[test]
    fn test_striped_threshold_infinite_before_training() {
        let striped = StripedSubspace::new(64, 4, 8);
        assert_eq!(striped.threshold(), f64::INFINITY);
        assert_eq!(striped.n_stripes(), 8);
    }

    #[test]
    fn test_striped_update_returns_rss_residual() {
        let dim = 128;
        let n = 4;
        let mut striped = StripedSubspace::new(dim, 8, n);
        let mut rng = 42u64;

        for _ in 0..100 {
            let vecs: Vec<Vec<f64>> = (0..n)
                .map(|_| low_rank_sample(&mut rng, dim))
                .collect();
            let res = striped.update(&vecs);
            assert!(res >= 0.0, "RSS residual should be non-negative");
        }
        assert!(striped.threshold().is_finite());
    }

    #[test]
    fn test_striped_in_distribution_below_threshold() {
        let dim = 256;
        let n = 4;
        let mut striped = StripedSubspace::with_params(dim, 8, n, 2.0, 0.01, 2.5, 500);
        let mut rng = 42u64;

        for _ in 0..200 {
            let vecs: Vec<Vec<f64>> = (0..n)
                .map(|_| low_rank_sample(&mut rng, dim))
                .collect();
            striped.update(&vecs);
        }

        let mut above = 0;
        for _ in 0..20 {
            let vecs: Vec<Vec<f64>> = (0..n)
                .map(|_| low_rank_sample(&mut rng, dim))
                .collect();
            if striped.residual(&vecs) > striped.threshold() {
                above += 1;
            }
        }
        assert!(above <= 5, "Expected at most 5/20 in-dist above threshold, got {}", above);
    }

    #[test]
    fn test_striped_ood_above_threshold() {
        let dim = 256;
        let n = 4;
        let mut striped = StripedSubspace::with_params(dim, 8, n, 2.0, 0.01, 2.5, 500);
        let mut rng = 42u64;

        for _ in 0..300 {
            let vecs: Vec<Vec<f64>> = (0..n)
                .map(|_| low_rank_sample(&mut rng, dim))
                .collect();
            striped.update(&vecs);
        }

        let mut above = 0;
        let mut rng2 = 999u64;
        for _ in 0..10 {
            let vecs: Vec<Vec<f64>> = (0..n)
                .map(|_| random_sample(&mut rng2, dim))
                .collect();
            if striped.residual(&vecs) > striped.threshold() {
                above += 1;
            }
        }
        assert!(above >= 7, "Expected most OOD samples above threshold, got {}/10", above);
    }

    #[test]
    fn test_striped_snapshot_round_trip() {
        let dim = 128;
        let n = 4;
        let mut striped = StripedSubspace::new(dim, 8, n);
        let mut rng = 42u64;

        for _ in 0..100 {
            let vecs: Vec<Vec<f64>> = (0..n)
                .map(|_| low_rank_sample(&mut rng, dim))
                .collect();
            striped.update(&vecs);
        }

        let snap = striped.snapshot();
        let restored = StripedSubspace::from_snapshot(snap);

        let mut rng2 = 1234u64;
        for _ in 0..10 {
            let vecs: Vec<Vec<f64>> = (0..n)
                .map(|_| low_rank_sample(&mut rng2, dim))
                .collect();
            let r1 = striped.residual(&vecs);
            let r2 = restored.residual(&vecs);
            assert!(
                (r1 - r2).abs() < 1e-10,
                "Striped residuals differ after round-trip: {} vs {}", r1, r2
            );
        }
    }

    #[test]
    fn test_striped_anomalous_component() {
        let dim = 128;
        let n = 4;
        let mut striped = StripedSubspace::new(dim, 8, n);
        let mut rng = 42u64;

        for _ in 0..200 {
            let vecs: Vec<Vec<f64>> = (0..n)
                .map(|_| low_rank_sample(&mut rng, dim))
                .collect();
            striped.update(&vecs);
        }

        let probe: Vec<Vec<f64>> = (0..n)
            .map(|_| random_sample(&mut rng, dim))
            .collect();

        for stripe_idx in 0..n {
            let anom = striped.anomalous_component(&probe, stripe_idx);
            assert_eq!(anom.len(), dim);
            let norm: f64 = anom.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(norm > 0.0, "Anomalous component should be non-zero for OOD");
        }
    }
}
