//! Journal — the learning coalgebra.
//!
//! N-ary discriminant learner. Accumulates labeled observations into
//! N class accumulators (one per label). The discriminant is the direction
//! that maximally separates the class centroids.
//!
//! Opaque state. The accumulators are private. observe/predict/decay
//! are the only way to interact with journal state.
//!
//! Labels are symbols — created from strings once, used as cheap integer
//! handles forever. Like Ruby's `:buy` or Clojure's `:buy`. Copy, O(1)
//! equality, no heap allocation per call.
//!
//! ## Usage
//!
//! ```rust
//! use holon::memory::Journal;
//!
//! let mut journal = Journal::new("direction", 4096, 500);
//! let buy  = journal.register("Buy");
//! let sell = journal.register("Sell");
//!
//! // observe and predict use Label handles, not strings
//! // journal.observe(&thought_vec, buy, 1.0);
//! // let pred = journal.predict(&thought_vec);
//! // pred.direction == Some(buy)
//! ```

use crate::kernel::accumulator::Accumulator;
use crate::kernel::vector::Vector;

// ─── Label (symbol) ─────────────────────────────────────────────────────────

/// A label is a symbol — an interned string handle.
/// Copy, O(1) equality, no heap allocation when used.
/// Created via `journal.register("name")`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Label(u32);

impl Label {
    /// The internal index. Useful for external arrays indexed by label.
    pub fn index(self) -> usize { self.0 as usize }
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

/// The result of asking a journal what it thinks about a thought.
#[derive(Clone, Debug)]
pub struct Prediction {
    /// Cosine score for each label, ordered by absolute cosine descending.
    pub scores: Vec<LabelScore>,
    /// The label with the highest absolute cosine.
    pub direction: Option<Label>,
    /// |max cosine| — how strongly the journal leans.
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

// ─── Journal ────────────────────────────────────────────────────────────────

/// N-ary discriminant learner with symbol labels.
///
/// Opaque state. The accumulators are private — `observe`, `predict`,
/// and `decay` are the only way to interact with journal internals.
pub struct Journal {
    /// Human-readable name for logging/debugging.
    name: String,
    /// Label names, in registration order. Index matches Label.0.
    label_names: Vec<String>,
    /// One accumulator per label. PRIVATE — the coalgebra requires opacity.
    accumulators: Vec<Accumulator>,
    /// Vector dimensionality.
    dims: usize,
    /// Total observations since creation.
    updates: usize,
    /// How often to recompute the discriminant.
    recalib_interval: usize,
    /// Per-label discriminant in float space.
    /// discriminants[i] = normalize(proto_i - mean(proto_j for j != i))
    discriminants: Vec<Option<Vec<f64>>>,
    /// Mean prototype across all classes (for input stripping).
    mean_proto: Option<Vec<f64>>,

    // ── Curve (self-evaluation) ────────────────────────────────────
    /// Resolved predictions: (conviction, correct). The raw data for curve fitting.
    resolved: Vec<(f64, bool)>,
    /// Maximum resolved predictions to retain. Configurable.
    resolved_cap: usize,
    /// Minimum resolved predictions before curve fitting is attempted.
    min_resolved: usize,
    /// Number of bins for curve fitting.
    n_bins: usize,
    /// Cached curve parameters from last fit.
    curve_a: f64,
    curve_b: f64,
    curve_valid: bool,

    // ── Diagnostics ─────────────────────────────────────────────────
    recalib_count: usize,
    last_cos_raw: f64,
    last_disc_strength: f64,
}

impl Journal {
    /// Create an empty journal. Register labels with `register()`.
    pub fn new(name: &str, dims: usize, recalib_interval: usize) -> Self {
        Self {
            name: name.to_string(),
            label_names: Vec::new(),
            accumulators: Vec::new(),
            dims,
            updates: 0,
            recalib_interval,
            discriminants: Vec::new(),
            mean_proto: None,
            resolved: Vec::new(),
            resolved_cap: 5000,
            min_resolved: 500,
            n_bins: 20,
            curve_a: 0.0,
            curve_b: 0.0,
            curve_valid: false,
            recalib_count: 0,
            last_cos_raw: 0.0,
            last_disc_strength: 0.0,
        }
    }

    /// Configure curve fitting parameters. Chainable.
    pub fn with_curve_params(mut self, resolved_cap: usize, min_resolved: usize, n_bins: usize) -> Self {
        self.resolved_cap = resolved_cap;
        self.min_resolved = min_resolved;
        self.n_bins = n_bins;
        self
    }

    /// Register a label and get its symbol handle.
    /// If the label already exists, returns the existing handle.
    pub fn register(&mut self, name: &str) -> Label {
        if let Some(idx) = self.label_names.iter().position(|n| n == name) {
            return Label(idx as u32);
        }
        let idx = self.label_names.len();
        self.label_names.push(name.to_string());
        self.accumulators.push(Accumulator::new(self.dims));
        self.discriminants.push(None);
        Label(idx as u32)
    }

    /// Record a labeled observation.
    /// Weight scales the observation's contribution.
    pub fn observe(&mut self, vec: &Vector, label: Label, weight: f64) {
        let idx = label.index();
        if idx >= self.accumulators.len() { return; }
        self.updates += 1;
        if self.updates % self.recalib_interval == 0 {
            self.recalibrate();
        }
        self.accumulators[idx].add_weighted(vec, weight);
    }

    /// Predict: which label does this thought most resemble?
    /// Returns scores for all labels. Consumer decides top-1/top-k/full.
    pub fn predict(&self, vec: &Vector) -> Prediction {
        if self.discriminants.iter().all(|d| d.is_none()) {
            return Prediction::default();
        }

        let mut scores: Vec<LabelScore> = Vec::with_capacity(self.label_names.len());

        for i in 0..self.label_names.len() {
            let cos = if let Some(disc) = &self.discriminants[i] {
                if let Some(mean) = &self.mean_proto {
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
            scores.push(LabelScore { label: Label(i as u32), cosine: cos });
        }

        // Highest raw cosine wins — the label whose discriminant the input
        // aligns with most positively. Positive = "looks like this class."
        // Negative = "does NOT look like this class." Abs sort confuses the two.
        scores.sort_by(|a, b| b.cosine.partial_cmp(&a.cosine)
            .unwrap_or(std::cmp::Ordering::Equal));

        let (direction, conviction, raw_cos) = if let Some(best) = scores.first() {
            (Some(best.label), best.cosine.abs(), best.cosine)
        } else {
            (None, 0.0, 0.0)
        };

        Prediction { scores, direction, conviction, raw_cos }
    }

    /// Decay all accumulators. Older observations fade.
    pub fn decay(&mut self, factor: f64) {
        for acc in &mut self.accumulators {
            acc.decay(factor);
        }
    }

    // ── Curve (self-evaluation) ────────────────────────────────────

    /// Record a resolved prediction: "I predicted with this conviction, and I was correct/wrong."
    /// The journal accumulates these to fit its conviction-accuracy curve.
    pub fn resolve(&mut self, conviction: f64, correct: bool) {
        self.resolved.push((conviction, correct));
        if self.resolved.len() > self.resolved_cap {
            self.resolved.remove(0);
        }
    }

    /// Fit the conviction-accuracy curve from resolved predictions.
    /// accuracy = base + a × exp(b × conviction)
    /// where base = 1/N (random chance for N labels).
    /// Returns (a, b) or None if insufficient data.
    pub fn curve(&mut self) -> Option<(f64, f64)> {
        let n_labels = self.label_names.len();
        if n_labels < 2 { return None; }
        if self.resolved.len() < self.min_resolved { return None; }

        let base = 1.0 / n_labels as f64;  // random chance: 0.50 for binary, 0.33 for ternary, etc.
        let base_epsilon = base + 0.005;     // just above random chance

        // Bin resolved predictions by conviction
        let mut sorted: Vec<(f64, bool)> = self.resolved.clone();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let bin_size = sorted.len() / self.n_bins;
        if bin_size < 10 { return None; }

        // Log-linear regression: ln(acc - base) = ln(a) + b × conviction
        let mut points: Vec<(f64, f64)> = Vec::new();
        for bi in 0..self.n_bins {
            let start = bi * bin_size;
            let end = if bi == self.n_bins - 1 { sorted.len() } else { (bi + 1) * bin_size };
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

        let b = (n * sxy - sx * sy) / denom;
        let ln_a = (sy - b * sx) / n;
        let a = ln_a.exp();

        self.curve_a = a;
        self.curve_b = b;
        self.curve_valid = true;

        Some((a, b))
    }

    /// Evaluate accuracy at a given conviction level using the fitted curve.
    /// accuracy = base + a × exp(b × conviction), where base = 1/N.
    /// Returns None if curve has not been fitted yet.
    pub fn accuracy_at(&self, conviction: f64) -> Option<f64> {
        if !self.curve_valid { return None; }
        let n_labels = self.label_names.len();
        if n_labels < 2 { return None; }
        let base = 1.0 / n_labels as f64;
        Some((base + self.curve_a * (self.curve_b * conviction).exp()).min(0.99))
    }

    /// Whether the curve has been fitted.
    pub fn curve_valid(&self) -> bool { self.curve_valid }

    /// The cached curve parameters.
    pub fn curve_params(&self) -> Option<(f64, f64)> {
        if self.curve_valid { Some((self.curve_a, self.curve_b)) } else { None }
    }

    /// Number of resolved predictions.
    pub fn resolved_count(&self) -> usize { self.resolved.len() }

    // ── Label resolution ────────────────────────────────────────────

    /// Resolve a label handle to its name.
    pub fn label_name(&self, label: Label) -> Option<&str> {
        self.label_names.get(label.index()).map(|s| s.as_str())
    }

    /// Get all registered labels.
    pub fn labels(&self) -> Vec<Label> {
        (0..self.label_names.len()).map(|i| Label(i as u32)).collect()
    }

    // ── Diagnostics ─────────────────────────────────────────────────

    pub fn name(&self) -> &str { &self.name }
    pub fn n_labels(&self) -> usize { self.label_names.len() }
    pub fn recalib_count(&self) -> usize { self.recalib_count }
    pub fn last_cos_raw(&self) -> f64 { self.last_cos_raw }
    pub fn last_disc_strength(&self) -> f64 { self.last_disc_strength }
    pub fn is_ready(&self) -> bool { self.recalib_count > 0 }
    pub fn total_updates(&self) -> usize { self.updates }

    /// Count of observations for a label.
    pub fn label_count(&self, label: Label) -> usize {
        if label.index() < self.accumulators.len() {
            self.accumulators[label.index()].count()
        } else { 0 }
    }

    /// Prototype health: (buy_norm, sell_norm, cosine_between_prototypes).
    /// Returns None if fewer than 2 labels registered.
    pub fn prototype_health(&self) -> Option<(f64, f64, f64)> {
        if self.accumulators.len() < 2 { return None; }
        let a = self.accumulators[0].normalize_f64();
        let b = self.accumulators[1].normalize_f64();
        let norm_a = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        let cos = cosine_f64(&a, &b);
        Some((norm_a, norm_b, cos))
    }

    /// Get the discriminant for a label (for decode/analysis).
    pub fn discriminant(&self, label: Label) -> Option<&[f64]> {
        self.discriminants.get(label.index())
            .and_then(|d| d.as_deref())
    }

    // ── Internal ────────────────────────────────────────────────────

    fn recalibrate(&mut self) {
        let n = self.label_names.len();
        if n < 2 { return; }

        let protos: Vec<Vec<f64>> = self.accumulators.iter()
            .map(|acc| acc.normalize_f64())
            .collect();

        if self.accumulators.iter().any(|a| a.count() == 0) { return; }

        // Mean prototype
        let mut mean = vec![0.0f64; self.dims];
        for proto in &protos {
            for (m, p) in mean.iter_mut().zip(proto.iter()) {
                *m += p;
            }
        }
        let n_f = n as f64;
        for m in &mut mean { *m /= n_f; }
        self.mean_proto = Some(mean);

        // Diagnostic: cosine between first two prototypes
        if n == 2 {
            self.last_cos_raw = cosine_f64(&protos[0], &protos[1]);
        }

        // Per-label discriminant
        for i in 0..n {
            let mut other_mean = vec![0.0f64; self.dims];
            let other_count = (n - 1) as f64;
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
            self.last_disc_strength = norm / (self.dims as f64).sqrt();

            if norm > 1e-10 {
                self.discriminants[i] = Some(disc.into_iter().map(|x| x / norm).collect());
            }
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

    #[test]
    fn test_binary_journal() {
        let vm = VectorManager::new(1024);
        let mut j = Journal::new("test", 1024, 10);
        let buy = j.register("Buy");
        let sell = j.register("Sell");

        for i in 0..20 {
            let v = vm.get_vector(&format!("candle_{}", i));
            let label = if i % 2 == 0 { buy } else { sell };
            j.observe(&v, label, 1.0);
        }

        assert!(j.is_ready());
        assert_eq!(j.recalib_count(), 2);

        let probe = vm.get_vector("candle_0");
        let pred = j.predict(&probe);
        assert!(pred.direction.is_some());
        assert!(pred.conviction > 0.0);
        assert_eq!(pred.scores.len(), 2);
    }

    #[test]
    fn test_ternary_journal() {
        let vm = VectorManager::new(1024);
        let mut j = Journal::new("sentiment", 1024, 10);
        let pos = j.register("Positive");
        let neg = j.register("Negative");
        let neu = j.register("Neutral");

        for i in 0..30 {
            let v = vm.get_vector(&format!("doc_{}", i));
            let label = match i % 3 { 0 => pos, 1 => neg, _ => neu };
            j.observe(&v, label, 1.0);
        }

        assert!(j.is_ready());
        let pred = j.predict(&vm.get_vector("doc_0"));
        assert_eq!(pred.scores.len(), 3);
    }

    #[test]
    fn test_label_is_symbol() {
        let mut j = Journal::new("test", 1024, 10);
        let a = j.register("Buy");
        let b = j.register("Sell");
        let a2 = j.register("Buy"); // re-register same name

        // Same name → same label
        assert_eq!(a, a2);
        // Different name → different label
        assert_ne!(a, b);
        // Label is Copy — no allocation
        let c = a;
        assert_eq!(a, c);
        // Resolve back to name
        assert_eq!(j.label_name(a), Some("Buy"));
        assert_eq!(j.label_name(b), Some("Sell"));
    }

    #[test]
    fn test_runtime_label_creation() {
        let mut j = Journal::new("dynamic", 1024, 10);

        // Labels can be created at runtime from any string
        let labels: Vec<Label> = vec!["BTC", "ETH", "GOLD", "SOL"]
            .into_iter()
            .map(|name| j.register(name))
            .collect();

        assert_eq!(j.n_labels(), 4);
        assert_eq!(labels[0].index(), 0);
        assert_eq!(labels[3].index(), 3);
        assert_eq!(j.label_name(labels[2]), Some("GOLD"));
    }

    #[test]
    fn test_decay() {
        let vm = VectorManager::new(1024);
        let mut j = Journal::new("test", 1024, 100);
        let a = j.register("A");
        j.register("B");

        let v = vm.get_vector("test");
        j.observe(&v, a, 1.0);
        j.decay(0.5);
    }

    #[test]
    fn test_curve_insufficient_data() {
        let mut j = Journal::new("test", 1024, 10);
        j.register("A");
        j.register("B");

        // Not enough resolved predictions
        for i in 0..100 {
            j.resolve(i as f64 * 0.01, i % 2 == 0);
        }
        assert!(j.curve().is_none());
        assert!(!j.curve_valid());
        assert!(j.accuracy_at(0.5).is_none());
    }

    #[test]
    fn test_curve_with_signal() {
        let mut j = Journal::new("test", 1024, 10);
        j.register("A");
        j.register("B");

        // Simulate: higher conviction → higher accuracy
        // Low conviction: ~50% correct (noise). High conviction: ~80% correct (signal).
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        for i in 0..2000 {
            let conviction = (i as f64) / 2000.0;  // 0.0 to 1.0
            // Probability of correct increases with conviction
            let mut hasher = DefaultHasher::new();
            i.hash(&mut hasher);
            let roll = (hasher.finish() % 1000) as f64 / 1000.0;
            let threshold = 0.50 - conviction * 0.35;  // high conviction → lower threshold → more correct
            let correct = roll > threshold;
            j.resolve(conviction, correct);
        }

        let result = j.curve();
        assert!(result.is_some(), "curve should fit with 2000 resolved");
        assert!(j.curve_valid());

        // Higher conviction should give higher accuracy
        let low = j.accuracy_at(0.1).unwrap();
        let high = j.accuracy_at(0.9).unwrap();
        assert!(high > low, "accuracy should increase with conviction: low={:.3} high={:.3}", low, high);
    }

    #[test]
    fn test_resolve_cap() {
        let mut j = Journal::new("test", 1024, 10);
        j.register("A");
        j.register("B");

        for i in 0..10000 {
            j.resolve(0.5, i % 2 == 0);
        }
        assert!(j.resolved_count() <= 5000, "resolved should be capped");
    }
}
