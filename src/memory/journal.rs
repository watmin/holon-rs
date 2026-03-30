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
            recalib_count: 0,
            last_cos_raw: 0.0,
            last_disc_strength: 0.0,
        }
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

        scores.sort_by(|a, b| b.cosine.abs().partial_cmp(&a.cosine.abs())
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
}
