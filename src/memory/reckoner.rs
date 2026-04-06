//! Reckoner — one primitive, two readout modes.
//!
//! Discrete mode: N-ary discriminant learner. Backward-compatible with Journal.
//! Continuous mode: nearest-neighbor regression over (thought, scalar) pairs.
//!
//! Same accumulation mechanism. Same decay. Same recalibration.
//!
//! ## Usage
//!
//! ```rust
//! use holon::memory::reckoner::{Reckoner, ReckConfig};
//!
//! // Discrete: classify into Win/Loss
//! let mut r = Reckoner::new("direction", 4096, 500, ReckConfig::Discrete(vec![
//!     "Win".into(), "Loss".into(),
//! ]));
//! let labels = r.labels();
//! // r.observe(&thought, labels[0], 1.0);
//! // let pred = r.predict(&thought);
//!
//! // Continuous: predict a scalar
//! let mut r = Reckoner::new("stop_distance", 4096, 500, ReckConfig::Continuous(0.015));
//! // r.observe_scalar(&thought, 0.008, 1.0);
//! // let d = r.query(&thought);
//! ```

use crate::kernel::accumulator::Accumulator;
use crate::kernel::vector::Vector;

// Re-export Label and Prediction — they are used externally and stay stable.
pub use crate::memory::journal::{Label, LabelScore, Prediction};

// ─── Config ────────────────────────────────────────────────────────────────

/// Configuration for a Reckoner. Determines its readout mode.
#[derive(Clone, Debug)]
pub enum ReckConfig {
    /// Discrete classification. The strings are label names.
    Discrete(Vec<String>),
    /// Continuous regression. The f64 is the default value when no experience.
    Continuous(f64),
}

// ─── Internal mode ─────────────────────────────────────────────────────────

/// One continuous observation: (thought as f64 slice, value, weight).
#[derive(Clone, Debug)]
struct ContinuousObs {
    thought: Vec<f64>,
    value: f64,
    weight: f64,
}

enum ReckMode {
    Discrete {
        label_names: Vec<String>,
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
        observations: Vec<ContinuousObs>,
        max_observations: usize,
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
            ReckConfig::Discrete(label_names) => {
                let n = label_names.len();
                ReckMode::Discrete {
                    label_names,
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
            ReckConfig::Continuous(default_value) => {
                ReckMode::Continuous {
                    default_value,
                    observations: Vec::new(),
                    max_observations: 5000,
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

    /// Configure max observations (continuous mode). Chainable.
    /// No-op for discrete mode.
    pub fn with_max_observations(mut self, max: usize) -> Self {
        if let ReckMode::Continuous { ref mut max_observations, .. } = self.mode {
            *max_observations = max;
        }
        self
    }

    // ════════════════════════════════════════════════════════════════════════
    // Discrete mode — backward-compatible with Journal
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
            ReckMode::Discrete { label_names, discriminants, mean_proto, .. } => {
                if discriminants.iter().all(|d| d.is_none()) {
                    return Prediction::default();
                }

                let mut scores: Vec<LabelScore> = Vec::with_capacity(label_names.len());

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
            ReckMode::Continuous { observations, .. } => {
                for obs in observations.iter_mut() {
                    obs.weight *= factor;
                }
                // Prune negligible observations
                observations.retain(|obs| obs.weight > 1e-10);
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
            ref label_names,
            ref resolved,
            min_resolved,
            n_bins,
            ref mut curve_a,
            ref mut curve_b,
            ref mut curve_valid,
            ..
        } = self.mode
        {
            let n_labels = label_names.len();
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
        if let ReckMode::Discrete { ref label_names, curve_a, curve_b, curve_valid, .. } = self.mode {
            if !curve_valid { return None; }
            let n_labels = label_names.len();
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

    /// Resolve a label handle to its name.
    pub fn label_name(&self, label: Label) -> Option<&str> {
        match &self.mode {
            ReckMode::Discrete { label_names, .. } => {
                label_names.get(label.index()).map(|s| s.as_str())
            }
            ReckMode::Continuous { .. } => None,
        }
    }

    /// Get all registered labels (discrete mode). Empty for continuous.
    pub fn labels(&self) -> Vec<Label> {
        match &self.mode {
            ReckMode::Discrete { label_names, .. } => {
                (0..label_names.len()).map(Label::from_index).collect()
            }
            ReckMode::Continuous { .. } => Vec::new(),
        }
    }

    // ── Diagnostics ─────────────────────────────────────────────────────────

    pub fn name(&self) -> &str { &self.name }
    pub fn dims(&self) -> usize { self.dims }

    pub fn n_labels(&self) -> usize {
        match &self.mode {
            ReckMode::Discrete { label_names, .. } => label_names.len(),
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
            ReckMode::Continuous { observations, max_observations, .. } => {
                self.updates += 1;
                let thought: Vec<f64> = vec.data().iter().map(|&v| v as f64).collect();
                observations.push(ContinuousObs {
                    thought,
                    value,
                    weight: weight.max(1e-10),
                });
                // Cap observations — oldest evicted
                if observations.len() > *max_observations {
                    observations.remove(0);
                }
            }
            ReckMode::Discrete { .. } => {
                panic!("Reckoner::observe_scalar() called on discrete-mode reckoner '{}'. Use observe().", self.name);
            }
        }
    }

    /// Query: given this thought, what scalar? (continuous mode).
    /// Returns the default value for discrete mode or when no experience.
    pub fn query(&self, vec: &Vector) -> f64 {
        match &self.mode {
            ReckMode::Continuous { default_value, observations, .. } => {
                if observations.is_empty() {
                    return *default_value;
                }

                let thought: Vec<f64> = vec.data().iter().map(|&v| v as f64).collect();
                let mut weighted_sum = 0.0_f64;
                let mut weight_total = 0.0_f64;

                for obs in observations {
                    let cos = cosine_f64(&thought, &obs.thought);
                    if cos <= 0.0 { continue; }
                    let w = cos * obs.weight;
                    weighted_sum += obs.value * w;
                    weight_total += w;
                }

                if weight_total < 1e-10 {
                    *default_value
                } else {
                    weighted_sum / weight_total
                }
            }
            ReckMode::Discrete { .. } => 0.0,
        }
    }

    /// How much experience does this reckoner have? (continuous mode).
    /// 0.0 = ignorant, increases with observations.
    pub fn experience(&self) -> f64 {
        match &self.mode {
            ReckMode::Continuous { observations, .. } => {
                observations.iter().map(|obs| obs.weight).sum()
            }
            ReckMode::Discrete { .. } => self.updates as f64,
        }
    }

    /// Number of stored observations (continuous mode).
    pub fn observation_count(&self) -> usize {
        match &self.mode {
            ReckMode::Continuous { observations, .. } => observations.len(),
            ReckMode::Discrete { .. } => 0,
        }
    }

    // ── Internal ────────────────────────────────────────────────────────────

    fn recalibrate(&mut self) {
        // Only meaningful for discrete mode
        let (label_count, all_have_obs, dims) = match &self.mode {
            ReckMode::Discrete { label_names, accumulators, .. } => {
                (label_names.len(), !accumulators.iter().any(|a| a.count() == 0), self.dims)
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

// ── Float-space cosine helpers ──────────────────────────────────────────────

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
            "Win".into(), "Loss".into(),
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
            "Buy".into(), "Sell".into(),
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
            "A".into(), "B".into(),
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
        let r = Reckoner::new("stop", 64, 500, ReckConfig::Continuous(0.015));
        let vm = VectorManager::new(64);
        let thought = vm.get_vector("anything");
        assert!((r.query(&thought) - 0.015).abs() < 1e-10);
    }

    #[test]
    fn continuous_single_observation_returns_its_value() {
        let mut r = Reckoner::new("stop", 64, 500, ReckConfig::Continuous(0.015));
        let vm = VectorManager::new(64);
        let thought = vm.get_vector("test-thought");
        r.observe_scalar(&thought, 0.008, 1.0);

        let d = r.query(&thought);
        assert!((d - 0.008).abs() < 0.001,
            "same thought should return its value: {}", d);
    }

    #[test]
    fn continuous_similar_thoughts_blend() {
        let mut r = Reckoner::new("stop", 1000, 500, ReckConfig::Continuous(0.015));
        let vm = VectorManager::new(1000);

        let base = vm.get_vector("base");
        // Same atom = same vector
        r.observe_scalar(&base, 0.01, 1.0);
        r.observe_scalar(&base, 0.02, 1.0);

        let d = r.query(&base);
        // Should be between 0.01 and 0.02
        assert!(d >= 0.009 && d <= 0.021,
            "similar thoughts should blend: {}", d);
    }

    #[test]
    fn continuous_dissimilar_thoughts_dont_influence() {
        let mut r = Reckoner::new("stop", 10000, 500, ReckConfig::Continuous(0.015));
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
    fn continuous_weight_matters() {
        let mut r = Reckoner::new("stop", 64, 500, ReckConfig::Continuous(0.015));
        let vm = VectorManager::new(64);

        let thought = vm.get_vector("same");
        // Heavy weight on tight stop
        r.observe_scalar(&thought, 0.005, 100.0);
        // Light weight on wide stop
        r.observe_scalar(&thought, 0.04, 1.0);

        let d = r.query(&thought);
        assert!(d < 0.01, "heavy weight should dominate: {}", d);
    }

    // ════════════════════════════════════════════════════════════════════════
    // Cross-cutting tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn continuous_decay_fades_weights() {
        let mut r = Reckoner::new("stop", 64, 500, ReckConfig::Continuous(0.015));
        let vm = VectorManager::new(64);

        let thought = vm.get_vector("test");
        r.observe_scalar(&thought, 0.01, 1.0);
        assert_eq!(r.observation_count(), 1);

        r.decay(0.5);
        assert_eq!(r.observation_count(), 1); // still there, just decayed

        // Decay to oblivion
        for _ in 0..100 {
            r.decay(0.01);
        }
        assert_eq!(r.observation_count(), 0, "negligible observations should be pruned");
    }

    #[test]
    fn continuous_experience_increases() {
        let mut r = Reckoner::new("stop", 64, 500, ReckConfig::Continuous(0.015));
        let vm = VectorManager::new(64);

        assert_eq!(r.experience(), 0.0);
        let thought = vm.get_vector("test");
        r.observe_scalar(&thought, 0.01, 5.0);
        assert!((r.experience() - 5.0).abs() < 1e-10);
        r.observe_scalar(&thought, 0.02, 3.0);
        assert!((r.experience() - 8.0).abs() < 1e-10);
    }

    #[test]
    fn discrete_ternary() {
        let vm = VectorManager::new(1024);
        let mut r = Reckoner::new("sentiment", 1024, 10, ReckConfig::Discrete(vec![
            "Positive".into(), "Negative".into(), "Neutral".into(),
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
    fn discrete_curve_with_signal() {
        let mut r = Reckoner::new("test", 1024, 10, ReckConfig::Discrete(vec![
            "A".into(), "B".into(),
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
