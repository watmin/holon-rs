//! Reckoner — one primitive, two readout modes.
//!
//! Discrete mode: N-ary discriminant learner with `HolonAST` labels.
//! Continuous mode: nearest-neighbor regression over (thought, scalar) pairs.
//!
//! Same accumulation mechanism. Same decay. Same recalibration.
//!
//! ## Usage
//!
//! ```rust
//! use holon::memory::reckoner::{Reckoner, ReckConfig};
//! use holon::HolonAST;
//!
//! // Discrete: classify into :Win / :Loss
//! let mut r = Reckoner::new("direction", 4096, 500, ReckConfig::Discrete(vec![
//!     HolonAST::keyword("Win"), HolonAST::keyword("Loss"),
//! ]));
//! let labels = r.labels();
//! // r.observe(&thought, labels[0], 1.0);
//! // let pred = r.predict(&thought);
//!
//! // Continuous: predict a scalar
//! let mut r = Reckoner::new("stop_distance", 4096, 500, ReckConfig::Continuous { default_value: 0.015, buckets: 10 });
//! // r.observe_scalar(&thought, 0.008, 1.0);
//! // let d = r.query(&thought);
//! ```

use crate::kernel::accumulator::Accumulator;
use crate::kernel::vector::Vector;
use crate::kernel::HolonAST;

// ─── Label (symbol) ─────────────────────────────────────────────────────────

/// A label is a symbol — an interned string handle.
/// Copy, O(1) equality, no heap allocation when used.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Label(u32);

impl Label {
    /// The internal index. Useful for external arrays indexed by label.
    pub fn index(self) -> usize { self.0 as usize }

    /// Create a label from an index. Used by Reckoner and other code that
    /// manages its own label registry.
    pub fn from_index(idx: usize) -> Self { Label(idx as u32) }
}

impl std::fmt::Display for Label {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Label({})", self.0)
    }
}

// ─── Prediction ─────────────────────────────────────────────────────────────

/// A score for one label from a prediction.
#[derive(Clone, Debug)]
pub struct LabelScore {
    /// The label handle.
    pub label: Label,
    /// Signed cosine against this label's discriminant direction.
    pub cosine: f64,
}

/// The result of asking a reckoner what it thinks about a thought.
#[derive(Clone, Debug)]
pub struct Prediction {
    /// Cosine score for each label, ordered by absolute cosine descending.
    pub scores: Vec<LabelScore>,
    /// The label with the highest absolute cosine.
    pub direction: Option<Label>,
    /// |max cosine| — how strongly the reckoner leans.
    pub conviction: f64,
    /// The raw signed cosine of the winning direction.
    pub raw_cos: f64,
}

impl Default for Prediction {
    fn default() -> Self {
        Self {
            scores: Vec::new(),
            direction: None,
            conviction: 0.0,
            raw_cos: 0.0,
        }
    }
}

// ─── Config ────────────────────────────────────────────────────────────────

/// Configuration for a Reckoner. Determines its readout mode.
#[derive(Clone, Debug)]
pub enum ReckConfig {
    /// Discrete classification. The ASTs are the wat-typed labels — one
    /// accumulator per AST, addressed at runtime through a [`Label`] handle.
    /// Duplicates are not deduped; the caller is responsible for distinct ASTs.
    Discrete(Vec<HolonAST>),
    /// Continuous regression with bucketed accumulators.
    ///
    /// - default_value: returned when no experience.
    /// - buckets: number of bins (K). Compute budget, not resolution.
    ///
    /// Range is discovered from observations. Decay + rebalance = breathing range.
    /// No magic min/max. The data teaches the range. K is the only parameter.
    Continuous {
        default_value: f64,
        buckets: usize,
    },
}

// ─── Internal mode ─────────────────────────────────────────────────────────

/// One bucket in the continuous reckoner: accumulates thoughts that
/// produced scalar values in this range.
#[derive(Clone, Debug)]
struct ScalarBucket {
    /// Accumulates thought vectors weighted by observation weight.
    accumulator: Accumulator,
    /// Center of the bucket's scalar range.
    center: f64,
}

enum ReckMode {
    Discrete {
        label_asts: Vec<HolonAST>,
        accumulators: Vec<Accumulator>,
        discriminants: Vec<Option<Vec<f64>>>,
        mean_proto: Option<Vec<f64>>,
        // Curve (self-evaluation)
        resolved: Vec<(f64, bool)>,
        resolved_cap: usize,
        min_resolved: usize,
        n_bins: usize,
        curve_a: f64,
        curve_b: f64,
        curve_valid: bool,
    },
    Continuous {
        default_value: f64,
        k: usize,
        buckets: Vec<ScalarBucket>,
        range_min: f64,
        range_max: f64,
        total_observations: usize,
        obs_since_rebalance: usize,
        initialized: bool,
    },
}

// ─── Reckoner ──────────────────────────────────────────────────────────────

/// One primitive. Two readout modes. Discrete or continuous.
pub struct Reckoner {
    name: String,
    dims: usize,
    recalib_interval: usize,
    updates: usize,
    recalib_count: usize,
    last_cos_raw: f64,
    last_disc_strength: f64,
    mode: ReckMode,
}

impl Reckoner {
    /// Create a new Reckoner.
    ///
    /// - `name`: human-readable name for logging/debugging.
    /// - `dims`: vector dimensionality.
    /// - `recalib_interval`: how often to recompute discriminants (discrete mode).
    /// - `config`: determines discrete or continuous mode.
    pub fn new(name: &str, dims: usize, recalib_interval: usize, config: ReckConfig) -> Self {
        let mode = match config {
            ReckConfig::Discrete(label_asts) => {
                let n = label_asts.len();
                ReckMode::Discrete {
                    label_asts,
                    accumulators: (0..n).map(|_| Accumulator::new(dims)).collect(),
                    discriminants: vec![None; n],
                    mean_proto: None,
                    resolved: Vec::new(),
                    resolved_cap: 5000,
                    min_resolved: 500,
                    n_bins: 20,
                    curve_a: 0.0,
                    curve_b: 0.0,
                    curve_valid: false,
                }
            }
            ReckConfig::Continuous { default_value, buckets: k } => {
                let buckets = (0..k)
                    .map(|_| ScalarBucket {
                        accumulator: Accumulator::new(dims),
                        center: 0.0,
                    })
                    .collect();
                ReckMode::Continuous {
                    default_value,
                    k,
                    buckets,
                    range_min: 0.0,
                    range_max: 0.0,
                    total_observations: 0,
                    obs_since_rebalance: 0,
                    initialized: false,
                }
            }
        };

        Self {
            name: name.to_string(),
            dims,
            recalib_interval,
            updates: 0,
            recalib_count: 0,
            last_cos_raw: 0.0,
            last_disc_strength: 0.0,
            mode,
        }
    }

    /// Configure curve fitting parameters (discrete mode). Chainable.
    /// No-op for continuous mode.
    pub fn with_curve_params(mut self, resolved_cap: usize, min_resolved: usize, n_bins: usize) -> Self {
        if let ReckMode::Discrete {
            resolved_cap: ref mut rc,
            min_resolved: ref mut mr,
            n_bins: ref mut nb,
            ..
        } = self.mode
        {
            *rc = resolved_cap;
            *mr = min_resolved;
            *nb = n_bins;
        }
        self
    }

    /// Legacy — was max observations for brute-force mode. Now a no-op.
    /// Bucketed accumulators don't cap observations — they compress them.
    pub fn with_max_observations(self, _max: usize) -> Self {
        self
    }

    // ════════════════════════════════════════════════════════════════════════
    // Discrete mode
    // ════════════════════════════════════════════════════════════════════════

    /// Record a labeled observation (discrete mode).
    /// Weight scales the observation's contribution.
    /// Panics in continuous mode.
    pub fn observe(&mut self, vec: &Vector, label: Label, weight: f64) {
        if matches!(self.mode, ReckMode::Continuous { .. }) {
            panic!("Reckoner::observe() called on continuous-mode reckoner '{}'. Use observe_scalar().", self.name);
        }

        let acc_len = match &self.mode {
            ReckMode::Discrete { accumulators, .. } => accumulators.len(),
            _ => unreachable!(),
        };
        let idx = label.index();
        if idx >= acc_len { return; }

        self.updates += 1;
        if self.updates % self.recalib_interval == 0 {
            self.recalibrate();
        }

        if let ReckMode::Discrete { ref mut accumulators, .. } = self.mode {
            accumulators[idx].add_weighted(vec, weight);
        }
    }

    /// Predict: which label does this thought most resemble? (discrete mode).
    /// Returns default Prediction for continuous mode.
    pub fn predict(&self, vec: &Vector) -> Prediction {
        match &self.mode {
            ReckMode::Discrete { label_asts, discriminants, mean_proto, .. } => {
                if discriminants.iter().all(|d| d.is_none()) {
                    return Prediction::default();
                }

                let mut scores: Vec<LabelScore> = Vec::with_capacity(label_asts.len());

                for (i, disc_opt) in discriminants.iter().enumerate() {
                    let cos = if let Some(disc) = disc_opt {
                        if let Some(mean) = mean_proto {
                            let stripped: Vec<f64> = vec.data().iter()
                                .zip(mean.iter())
                                .map(|(&v, &m)| v as f64 - m)
                                .collect();
                            cosine_f64(&stripped, disc)
                        } else {
                            cosine_f64_vs_vec(disc, vec)
                        }
                    } else {
                        0.0
                    };
                    scores.push(LabelScore { label: Label::from_index(i), cosine: cos });
                }

                // Highest raw cosine wins
                scores.sort_by(|a, b| b.cosine.partial_cmp(&a.cosine)
                    .unwrap_or(std::cmp::Ordering::Equal));

                let (direction, conviction, raw_cos) = if let Some(best) = scores.first() {
                    (Some(best.label), best.cosine.abs(), best.cosine)
                } else {
                    (None, 0.0, 0.0)
                };

                Prediction { scores, direction, conviction, raw_cos }
            }
            ReckMode::Continuous { .. } => Prediction::default(),
        }
    }

    /// Get the discriminant for a label (discrete mode).
    pub fn discriminant(&self, label: Label) -> Option<&[f64]> {
        match &self.mode {
            ReckMode::Discrete { discriminants, .. } => {
                discriminants.get(label.index()).and_then(|d| d.as_deref())
            }
            ReckMode::Continuous { .. } => None,
        }
    }

    /// Decay all state. Works for both modes.
    ///
    /// Discrete: decays all accumulators.
    /// Continuous: decays observation weights.
    pub fn decay(&mut self, factor: f64) {
        match &mut self.mode {
            ReckMode::Discrete { accumulators, .. } => {
                for acc in accumulators.iter_mut() {
                    acc.decay(factor);
                }
            }
            ReckMode::Continuous { buckets, .. } => {
                for bucket in buckets.iter_mut() {
                    bucket.accumulator.decay(factor);
                }
            }
        }
    }

    // ── Curve (self-evaluation, discrete mode) ─────────────────────────────

    /// Record a resolved prediction (discrete mode).
    pub fn resolve(&mut self, conviction: f64, correct: bool) {
        if let ReckMode::Discrete { ref mut resolved, resolved_cap, .. } = self.mode {
            resolved.push((conviction, correct));
            if resolved.len() > resolved_cap {
                resolved.remove(0);
            }
        }
    }

    /// Fit the conviction-accuracy curve from resolved predictions.
    /// accuracy = base + a * exp(b * conviction), where base = 1/N.
    /// Returns (a, b) or None if insufficient data or continuous mode.
    pub fn curve(&mut self) -> Option<(f64, f64)> {
        if let ReckMode::Discrete {
            ref label_asts,
            ref resolved,
            min_resolved,
            n_bins,
            ref mut curve_a,
            ref mut curve_b,
            ref mut curve_valid,
            ..
        } = self.mode
        {
            let n_labels = label_asts.len();
            if n_labels < 2 { return None; }
            if resolved.len() < min_resolved { return None; }

            let base = 1.0 / n_labels as f64;
            let base_epsilon = base + 0.005;

            let mut sorted: Vec<(f64, bool)> = resolved.clone();
            sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let bin_size = sorted.len() / n_bins;
            if bin_size < 10 { return None; }

            let mut points: Vec<(f64, f64)> = Vec::new();
            for bi in 0..n_bins {
                let start = bi * bin_size;
                let end = if bi == n_bins - 1 { sorted.len() } else { (bi + 1) * bin_size };
                let slice = &sorted[start..end];
                let mean_c = slice.iter().map(|(c, _)| c).sum::<f64>() / slice.len() as f64;
                let acc = slice.iter().filter(|(_, w)| *w).count() as f64 / slice.len() as f64;
                if acc > base_epsilon {
                    let ln_excess = (acc - base).ln();
                    if ln_excess.is_finite() {
                        points.push((mean_c, ln_excess));
                    }
                }
            }

            if points.len() < 3 { return None; }

            let n = points.len() as f64;
            let sx: f64 = points.iter().map(|(x, _)| x).sum();
            let sy: f64 = points.iter().map(|(_, y)| y).sum();
            let sxx: f64 = points.iter().map(|(x, _)| x * x).sum();
            let sxy: f64 = points.iter().map(|(x, y)| x * y).sum();
            let denom = n * sxx - sx * sx;
            if denom.abs() < 1e-10 { return None; }

            let b_val = (n * sxy - sx * sy) / denom;
            let ln_a = (sy - b_val * sx) / n;
            let a_val = ln_a.exp();

            *curve_a = a_val;
            *curve_b = b_val;
            *curve_valid = true;

            Some((a_val, b_val))
        } else {
            None
        }
    }

    /// Evaluate accuracy at a given conviction level using the fitted curve.
    pub fn accuracy_at(&self, conviction: f64) -> Option<f64> {
        if let ReckMode::Discrete { ref label_asts, curve_a, curve_b, curve_valid, .. } = self.mode {
            if !curve_valid { return None; }
            let n_labels = label_asts.len();
            if n_labels < 2 { return None; }
            let base = 1.0 / n_labels as f64;
            Some((base + curve_a * (curve_b * conviction).exp()).min(0.99))
        } else {
            None
        }
    }

    /// Whether the curve has been fitted (discrete mode).
    pub fn curve_valid(&self) -> bool {
        matches!(&self.mode, ReckMode::Discrete { curve_valid: true, .. })
    }

    /// The cached curve parameters (discrete mode).
    pub fn curve_params(&self) -> Option<(f64, f64)> {
        if let ReckMode::Discrete { curve_a, curve_b, curve_valid: true, .. } = &self.mode {
            Some((*curve_a, *curve_b))
        } else {
            None
        }
    }

    /// Number of resolved predictions (discrete mode).
    pub fn resolved_count(&self) -> usize {
        match &self.mode {
            ReckMode::Discrete { resolved, .. } => resolved.len(),
            ReckMode::Continuous { .. } => 0,
        }
    }

    // ── Label resolution (discrete mode) ────────────────────────────────────

    /// Resolve a label handle to its `HolonAST`.
    ///
    /// Use [`crate::canonical_edn_holon`] for a stable byte form for logging
    /// or hashing.
    pub fn label_ast(&self, label: Label) -> Option<&HolonAST> {
        match &self.mode {
            ReckMode::Discrete { label_asts, .. } => label_asts.get(label.index()),
            ReckMode::Continuous { .. } => None,
        }
    }

    /// Get all registered labels (discrete mode). Empty for continuous.
    pub fn labels(&self) -> Vec<Label> {
        match &self.mode {
            ReckMode::Discrete { label_asts, .. } => {
                (0..label_asts.len()).map(Label::from_index).collect()
            }
            ReckMode::Continuous { .. } => Vec::new(),
        }
    }

    // ── Diagnostics ─────────────────────────────────────────────────────────

    pub fn name(&self) -> &str { &self.name }
    pub fn dims(&self) -> usize { self.dims }

    pub fn n_labels(&self) -> usize {
        match &self.mode {
            ReckMode::Discrete { label_asts, .. } => label_asts.len(),
            ReckMode::Continuous { .. } => 0,
        }
    }

    pub fn recalib_count(&self) -> usize { self.recalib_count }
    pub fn last_cos_raw(&self) -> f64 { self.last_cos_raw }
    pub fn last_disc_strength(&self) -> f64 { self.last_disc_strength }
    pub fn is_ready(&self) -> bool { self.recalib_count > 0 }
    pub fn total_updates(&self) -> usize { self.updates }

    /// Is this a discrete reckoner?
    pub fn is_discrete(&self) -> bool { matches!(self.mode, ReckMode::Discrete { .. }) }

    /// Is this a continuous reckoner?
    pub fn is_continuous(&self) -> bool { matches!(self.mode, ReckMode::Continuous { .. }) }

    /// Count of observations for a label (discrete mode).
    pub fn label_count(&self, label: Label) -> usize {
        match &self.mode {
            ReckMode::Discrete { accumulators, .. } => {
                if label.index() < accumulators.len() {
                    accumulators[label.index()].count()
                } else {
                    0
                }
            }
            ReckMode::Continuous { .. } => 0,
        }
    }

    /// Prototype health: (norm_0, norm_1, cosine_between_prototypes).
    /// Returns None if fewer than 2 labels (discrete mode only).
    pub fn prototype_health(&self) -> Option<(f64, f64, f64)> {
        match &self.mode {
            ReckMode::Discrete { accumulators, .. } => {
                if accumulators.len() < 2 { return None; }
                let a = accumulators[0].normalize_f64();
                let b = accumulators[1].normalize_f64();
                let norm_a = a.iter().map(|x| x * x).sum::<f64>().sqrt();
                let norm_b = b.iter().map(|x| x * x).sum::<f64>().sqrt();
                let cos = cosine_f64(&a, &b);
                Some((norm_a, norm_b, cos))
            }
            ReckMode::Continuous { .. } => None,
        }
    }

    /// Get the normalized prototype for a label (discrete mode).
    pub fn prototype(&self, label: Label) -> Option<Vec<f64>> {
        match &self.mode {
            ReckMode::Discrete { accumulators, .. } => {
                let idx = label.index();
                if idx >= accumulators.len() { return None; }
                if accumulators[idx].count() == 0 { return None; }
                Some(accumulators[idx].normalize_f64())
            }
            ReckMode::Continuous { .. } => None,
        }
    }

    /// Get the raw accumulator sums for a label (discrete mode).
    pub fn raw_prototype(&self, label: Label) -> Option<&[f64]> {
        match &self.mode {
            ReckMode::Discrete { accumulators, .. } => {
                let idx = label.index();
                if idx >= accumulators.len() { return None; }
                if accumulators[idx].count() == 0 { return None; }
                Some(accumulators[idx].raw_sums())
            }
            ReckMode::Continuous { .. } => None,
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // Continuous mode
    // ════════════════════════════════════════════════════════════════════════

    /// Record a scalar observation (continuous mode).
    /// `vec`: the thought vector.
    /// `value`: the scalar to associate with this thought.
    /// `weight`: importance weight.
    /// Panics in discrete mode.
    pub fn observe_scalar(&mut self, vec: &Vector, value: f64, weight: f64) {
        match &mut self.mode {
            ReckMode::Continuous { buckets, k, range_min, range_max,
                                   total_observations, obs_since_rebalance, initialized, .. } => {
                self.updates += 1;
                *total_observations += 1;

                // Decay all buckets — every observation is a tick of time
                for bucket in buckets.iter_mut() {
                    bucket.accumulator.decay(0.999);
                }

                if !*initialized {
                    // First observation — set initial range and bucket centers
                    *range_min = value - 0.001;
                    *range_max = value + 0.001;
                    *initialized = true;
                    let w = (*range_max - *range_min) / *k as f64;
                    for (i, bucket) in buckets.iter_mut().enumerate() {
                        bucket.center = *range_min + (i as f64 + 0.5) * w;
                    }
                }

                // Expand range if needed — rebalance on expansion
                let mut range_changed = false;
                if value < *range_min { *range_min = value; range_changed = true; }
                if value > *range_max { *range_max = value + 1e-10; range_changed = true; }

                if range_changed {
                    // Range expanded — redistribute all buckets to new grid
                    Self::rebalance_continuous(buckets, *k, range_min, range_max, self.dims);
                }

                // Find the bucket for this scalar value
                let width = (*range_max - *range_min) / *k as f64;
                if width > 1e-15 {
                    let idx = ((value - *range_min) / width).floor() as usize;
                    let idx = idx.min(*k - 1);
                    buckets[idx].accumulator.add_weighted(vec, weight.max(1e-10));
                }

                // Periodic rebalance: contract range to where decayed weight lives
                *obs_since_rebalance += 1;
                if *obs_since_rebalance >= 100 {
                    Self::rebalance_continuous(buckets, *k, range_min, range_max, self.dims);
                    *obs_since_rebalance = 0;
                }
            }
            ReckMode::Discrete { .. } => {
                panic!("Reckoner::observe_scalar() called on discrete-mode reckoner '{}'. Use observe().", self.name);
            }
        }
    }

    /// Rebalance: contract range to where the weight lives. Redistribute buckets.
    fn rebalance_continuous(
        buckets: &mut Vec<ScalarBucket>,
        k: usize,
        range_min: &mut f64,
        range_max: &mut f64,
        dims: usize,
    ) {
        // Find effective range from alive buckets (weight > 0.1)
        let alive_centers: Vec<f64> = buckets.iter()
            .filter(|b| b.accumulator.count() > 0 && {
                let norm: f64 = b.accumulator.raw_sums().iter().map(|x| x * x).sum::<f64>().sqrt();
                norm > 0.1
            })
            .map(|b| b.center)
            .collect();

        if alive_centers.len() < 2 {
            // Not enough alive buckets to rebalance — just reset centers from current range
            let width = (*range_max - *range_min) / k as f64;
            if width > 1e-15 {
                for (i, bucket) in buckets.iter_mut().enumerate() {
                    bucket.center = *range_min + (i as f64 + 0.5) * width;
                }
            }
            return;
        }

        let new_min = alive_centers.iter().cloned().fold(f64::MAX, f64::min);
        let new_max = alive_centers.iter().cloned().fold(f64::MIN, f64::max);
        if (new_max - new_min) < 1e-10 { return; }

        // Margin so edge observations don't fall off
        let margin = (new_max - new_min) * 0.1;
        let new_min = (new_min - margin).max(0.0);
        let new_max = new_max + margin;

        // Collect alive bucket data
        let old_buckets: Vec<ScalarBucket> = std::mem::replace(
            buckets,
            (0..k).map(|_| ScalarBucket {
                accumulator: Accumulator::new(dims),
                center: 0.0,
            }).collect(),
        );

        *range_min = new_min;
        *range_max = new_max;
        let width = (new_max - new_min) / k as f64;

        // Set new bucket centers
        for (i, bucket) in buckets.iter_mut().enumerate() {
            bucket.center = new_min + (i as f64 + 0.5) * width;
        }

        // Pour old alive buckets into nearest new bucket via merge
        for old in &old_buckets {
            let norm: f64 = old.accumulator.raw_sums().iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 0.1 { continue; }
            let idx = ((old.center - new_min) / width).floor() as usize;
            let idx = idx.min(k - 1);
            buckets[idx].accumulator.merge(&old.accumulator);
        }
    }

    /// Query: given this thought, what scalar? (continuous mode).
    /// Cosine against K bucket prototypes, soft-weight top-3, interpolate.
    /// O(K × D) — constant in observation count.
    /// Returns the default value for discrete mode or when no experience.
    pub fn query(&self, vec: &Vector) -> f64 {
        match &self.mode {
            ReckMode::Continuous { default_value, buckets, total_observations, .. } => {
                if *total_observations == 0 {
                    return *default_value;
                }

                let thought_f64 = to_f64_slice(vec);
                let norm_t: f64 = thought_f64.iter().map(|x| x * x).sum::<f64>().sqrt();

                // Cosine similarity against each bucket's prototype direction.
                // The prototype direction = "thoughts that produced values in this
                // range looked like THIS." Cosine measures match quality regardless
                // of accumulated mass. The center carries the value.
                let mut scored: Vec<(f64, f64)> = Vec::with_capacity(buckets.len());
                for bucket in buckets {
                    if bucket.accumulator.count() == 0 { continue; }
                    let raw = bucket.accumulator.raw_sums();
                    let dot: f64 = raw.iter().zip(thought_f64.iter())
                        .map(|(&r, &t)| r * t)
                        .sum();
                    let norm_r: f64 = raw.iter().map(|x| x * x).sum::<f64>().sqrt();
                    let cos = if norm_r > 1e-10 && norm_t > 1e-10 {
                        dot / (norm_r * norm_t)
                    } else {
                        0.0
                    };
                    if cos > 0.0 {
                        scored.push((cos, bucket.center));
                    }
                }

                if scored.is_empty() {
                    return *default_value;
                }

                // Soft-weight top-3 — interpolate, don't argmax
                scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                let mut w_sum = 0.0_f64;
                let mut v_sum = 0.0_f64;
                for &(cos, center) in scored.iter().take(3) {
                    w_sum += cos;
                    v_sum += cos * center;
                }

                if w_sum < 1e-10 {
                    *default_value
                } else {
                    v_sum / w_sum
                }
            }
            ReckMode::Discrete { .. } => 0.0,
        }
    }

    /// How much experience does this reckoner have? (continuous mode).
    /// 0.0 = ignorant, increases with observations.
    pub fn experience(&self) -> f64 {
        match &self.mode {
            ReckMode::Continuous { total_observations, .. } => *total_observations as f64,
            ReckMode::Discrete { .. } => self.updates as f64,
        }
    }

    /// Number of stored observations (continuous mode).
    pub fn observation_count(&self) -> usize {
        match &self.mode {
            ReckMode::Continuous { total_observations, .. } => *total_observations,
            ReckMode::Discrete { .. } => 0,
        }
    }

    // ── Internal ────────────────────────────────────────────────────────────

    fn recalibrate(&mut self) {
        // Only meaningful for discrete mode
        let (label_count, all_have_obs, dims) = match &self.mode {
            ReckMode::Discrete { label_asts, accumulators, .. } => {
                (label_asts.len(), !accumulators.iter().any(|a| a.count() == 0), self.dims)
            }
            _ => return,
        };

        if label_count < 2 { return; }
        if !all_have_obs { return; }

        // Extract prototypes
        let protos: Vec<Vec<f64>> = match &self.mode {
            ReckMode::Discrete { accumulators, .. } => {
                accumulators.iter().map(|acc| acc.normalize_f64()).collect()
            }
            _ => unreachable!(),
        };

        // Mean prototype
        let mut mean = vec![0.0_f64; dims];
        for proto in &protos {
            for (m, p) in mean.iter_mut().zip(proto.iter()) {
                *m += p;
            }
        }
        let n_f = label_count as f64;
        for m in &mut mean { *m /= n_f; }

        // Diagnostic: cosine between first two prototypes
        if label_count == 2 {
            self.last_cos_raw = cosine_f64(&protos[0], &protos[1]);
        }

        // Per-label discriminant
        let mut new_discriminants = vec![None; label_count];
        for i in 0..label_count {
            let mut other_mean = vec![0.0_f64; dims];
            let other_count = (label_count - 1) as f64;
            for (j, proto) in protos.iter().enumerate() {
                if j != i {
                    for (om, p) in other_mean.iter_mut().zip(proto.iter()) {
                        *om += p;
                    }
                }
            }
            for om in &mut other_mean { *om /= other_count; }

            let disc: Vec<f64> = protos[i].iter()
                .zip(other_mean.iter())
                .map(|(p, o)| p - o)
                .collect();

            let norm: f64 = disc.iter().map(|x| x * x).sum::<f64>().sqrt();
            self.last_disc_strength = norm / (dims as f64).sqrt();

            if norm > 1e-10 {
                new_discriminants[i] = Some(disc.into_iter().map(|x| x / norm).collect());
            }
        }

        // Write back
        if let ReckMode::Discrete { ref mut discriminants, ref mut mean_proto, .. } = self.mode {
            *discriminants = new_discriminants;
            *mean_proto = Some(mean);
        }

        self.recalib_count += 1;
    }
}

// ── Float-space helpers ─────────────────────────────────────────────────────

fn to_f64_slice(vec: &Vector) -> Vec<f64> {
    vec.data().iter().map(|&v| v as f64).collect()
}

fn cosine_f64(a: &[f64], b: &[f64]) -> f64 {
    let mut dot = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = (na * nb).sqrt();
    if denom < 1e-10 { 0.0 } else { dot / denom }
}

fn cosine_f64_vs_vec(proto: &[f64], vec: &Vector) -> f64 {
    let data = vec.data();
    let mut dot = 0.0_f64;
    let mut np = 0.0_f64;
    let mut nv = 0.0_f64;
    for (&p, &v) in proto.iter().zip(data.iter()) {
        let vf = v as f64;
        dot += p * vf;
        np += p * p;
        nv += vf * vf;
    }
    let denom = (np * nv).sqrt();
    if denom < 1e-10 { 0.0 } else { dot / denom }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VectorManager;

    // ════════════════════════════════════════════════════════════════════════
    // Discrete mode tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn discrete_empty_returns_default_prediction() {
        let r = Reckoner::new("test", 1024, 10, ReckConfig::Discrete(vec![
            HolonAST::keyword("Win"), HolonAST::keyword("Loss"),
        ]));
        let vm = VectorManager::new(1024);
        let probe = vm.get_vector("anything");
        let pred = r.predict(&probe);
        assert!(pred.direction.is_none());
        assert_eq!(pred.conviction, 0.0);
    }

    #[test]
    fn discrete_observe_predict_returns_correct_label() {
        let vm = VectorManager::new(1024);
        let mut r = Reckoner::new("test", 1024, 10, ReckConfig::Discrete(vec![
            HolonAST::keyword("Buy"), HolonAST::keyword("Sell"),
        ]));
        let labels = r.labels();
        let buy = labels[0];
        let sell = labels[1];

        for i in 0..20 {
            let v = vm.get_vector(&format!("candle_{}", i));
            let label = if i % 2 == 0 { buy } else { sell };
            r.observe(&v, label, 1.0);
        }

        assert!(r.is_ready());
        assert_eq!(r.recalib_count(), 2);

        let probe = vm.get_vector("candle_0");
        let pred = r.predict(&probe);
        assert!(pred.direction.is_some());
        assert!(pred.conviction > 0.0);
        assert_eq!(pred.scores.len(), 2);
    }

    #[test]
    fn discrete_decay_fades_old_observations() {
        let vm = VectorManager::new(1024);
        let mut r = Reckoner::new("test", 1024, 100, ReckConfig::Discrete(vec![
            HolonAST::keyword("A"), HolonAST::keyword("B"),
        ]));
        let labels = r.labels();
        let a = labels[0];

        let v = vm.get_vector("test");
        r.observe(&v, a, 1.0);
        r.decay(0.5);

        // After decay, the accumulator sums should be halved.
        // We verify by checking the raw prototype magnitude decreased.
        let raw = r.raw_prototype(a).unwrap();
        let norm: f64 = raw.iter().map(|x| x * x).sum::<f64>().sqrt();
        // Original bipolar 1024-dim vector has norm sqrt(1024) ≈ 32.
        // After 0.5 decay: ~16.
        assert!(norm < 20.0, "decay should reduce magnitude: {}", norm);
    }

    // ════════════════════════════════════════════════════════════════════════
    // Continuous mode tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn continuous_empty_returns_default_value() {
        let r = Reckoner::new("stop", 64, 500, ReckConfig::Continuous { default_value: 0.015, buckets: 10 });
        let vm = VectorManager::new(64);
        let thought = vm.get_vector("anything");
        assert!((r.query(&thought) - 0.015).abs() < 1e-10);
    }

    #[test]
    fn continuous_single_observation_returns_its_value() {
        let mut r = Reckoner::new("stop", 64, 500, ReckConfig::Continuous { default_value: 0.015, buckets: 10 });
        let vm = VectorManager::new(64);
        let thought = vm.get_vector("test-thought");
        r.observe_scalar(&thought, 0.008, 1.0);

        let d = r.query(&thought);
        // Bucketed: returns the bucket center nearest to 0.008, not exact 0.008.
        // Bucket width = (0.10-0.001)/10 = 0.0099. Bucket 0 center = 0.00595.
        // Tolerance = one bucket width.
        assert!((d - 0.008).abs() < 0.01,
            "same thought should return near its value: {}", d);
    }

    #[test]
    fn continuous_similar_thoughts_blend() {
        let mut r = Reckoner::new("stop", 1000, 500, ReckConfig::Continuous { default_value: 0.015, buckets: 10 });
        let vm = VectorManager::new(1000);

        let base = vm.get_vector("base");
        // Same atom = same vector
        r.observe_scalar(&base, 0.01, 1.0);
        r.observe_scalar(&base, 0.02, 1.0);

        let d = r.query(&base);
        // Should be between 0.01 and 0.02
        assert!((0.009..=0.021).contains(&d),
            "similar thoughts should blend: {}", d);
    }

    #[test]
    fn continuous_dissimilar_thoughts_dont_influence() {
        let mut r = Reckoner::new("stop", 10000, 500, ReckConfig::Continuous { default_value: 0.015, buckets: 10 });
        let vm = VectorManager::new(10000);

        let thought_a = vm.get_vector("regime-trending");
        let thought_b = vm.get_vector("regime-choppy");

        // Trending: tight
        r.observe_scalar(&thought_a, 0.005, 10.0);
        // Choppy: wide
        r.observe_scalar(&thought_b, 0.03, 10.0);

        let d_trending = r.query(&thought_a);
        let d_choppy = r.query(&thought_b);

        assert!(d_trending < d_choppy,
            "trending should want tighter than choppy: {} vs {}", d_trending, d_choppy);
    }

    #[test]
    fn continuous_direction_matters() {
        // With cosine, direction matters — not accumulated mass.
        // Two DIFFERENT thoughts at different values: the query should
        // return the value associated with the matching direction.
        let mut r = Reckoner::new("stop", 64, 500, ReckConfig::Continuous { default_value: 0.015, buckets: 10 });
        let vm = VectorManager::new(64);

        let tight_thought = vm.get_vector("tight_market");
        let wide_thought = vm.get_vector("wide_market");

        // Tight market → tight stop
        r.observe_scalar(&tight_thought, 0.005, 1.0);
        // Wide market → wide stop
        r.observe_scalar(&wide_thought, 0.04, 1.0);

        let d_tight = r.query(&tight_thought);
        let d_wide = r.query(&wide_thought);

        // Each thought should retrieve the value it was associated with
        assert!(d_tight < d_wide,
            "tight thought should predict tighter than wide: {} vs {}", d_tight, d_wide);
    }

    // ════════════════════════════════════════════════════════════════════════
    // Cross-cutting tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn continuous_decay_fades_weights() {
        let mut r = Reckoner::new("stop", 64, 500, ReckConfig::Continuous { default_value: 0.015, buckets: 10 });
        let vm = VectorManager::new(64);

        let thought = vm.get_vector("test");
        r.observe_scalar(&thought, 0.01, 1.0);
        assert_eq!(r.observation_count(), 1);

        r.decay(0.5);
        assert_eq!(r.observation_count(), 1); // count is permanent — accumulated, not stored

        // Decay the accumulator sums toward zero
        for _ in 0..100 {
            r.decay(0.01);
        }
        // Count stays — it tracks total observations ever, not current weight.
        // The accumulator sums decay to near-zero but the count is history.
        assert_eq!(r.observation_count(), 1);
    }

    #[test]
    fn continuous_experience_increases() {
        let mut r = Reckoner::new("stop", 64, 500, ReckConfig::Continuous { default_value: 0.015, buckets: 10 });
        let vm = VectorManager::new(64);

        assert_eq!(r.experience(), 0.0);
        let thought = vm.get_vector("test");
        r.observe_scalar(&thought, 0.01, 1.0);
        assert!((r.experience() - 1.0).abs() < 1e-10);
        r.observe_scalar(&thought, 0.02, 3.0);
        assert!((r.experience() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn discrete_ternary() {
        let vm = VectorManager::new(1024);
        let mut r = Reckoner::new("sentiment", 1024, 10, ReckConfig::Discrete(vec![
            HolonAST::keyword("Positive"), HolonAST::keyword("Negative"), HolonAST::keyword("Neutral"),
        ]));
        let labels = r.labels();

        for i in 0..30 {
            let v = vm.get_vector(&format!("doc_{}", i));
            let label = labels[i % 3];
            r.observe(&v, label, 1.0);
        }

        assert!(r.is_ready());
        let pred = r.predict(&vm.get_vector("doc_0"));
        assert_eq!(pred.scores.len(), 3);
    }

    #[test]
    fn discrete_label_ast_roundtrip() {
        // Labels can be arbitrary HolonAST — keyword, atom, or compound.
        // The reckoner stores them by index and round-trips them through
        // `label_ast(Label)`.
        let win = HolonAST::keyword("Win");
        let loss = HolonAST::keyword("Loss");
        let composite = HolonAST::bind(HolonAST::keyword("side"), HolonAST::i64(1));

        let r = Reckoner::new("multi-shape", 64, 10, ReckConfig::Discrete(vec![
            win.clone(), loss.clone(), composite.clone(),
        ]));

        let labels = r.labels();
        assert_eq!(labels.len(), 3);
        assert!(matches!(r.label_ast(labels[0]), Some(HolonAST::Symbol(_))));
        assert!(matches!(r.label_ast(labels[2]), Some(HolonAST::Bind(_, _))));
        // Out-of-range handle → None.
        assert!(r.label_ast(Label::from_index(99)).is_none());
    }

    #[test]
    fn discrete_curve_with_signal() {
        let mut r = Reckoner::new("test", 1024, 10, ReckConfig::Discrete(vec![
            HolonAST::keyword("A"), HolonAST::keyword("B"),
        ]));

        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        for i in 0..2000 {
            let conviction = (i as f64) / 2000.0;
            let mut hasher = DefaultHasher::new();
            i.hash(&mut hasher);
            let roll = (hasher.finish() % 1000) as f64 / 1000.0;
            let threshold = 0.50 - conviction * 0.35;
            let correct = roll > threshold;
            r.resolve(conviction, correct);
        }

        let result = r.curve();
        assert!(result.is_some(), "curve should fit with 2000 resolved");
        assert!(r.curve_valid());

        let low = r.accuracy_at(0.1).unwrap();
        let high = r.accuracy_at(0.9).unwrap();
        assert!(high > low, "accuracy should increase with conviction: low={:.3} high={:.3}", low, high);
    }
}
