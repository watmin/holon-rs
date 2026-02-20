//! Engram: a named, serializable snapshot of a trained [`OnlineSubspace`].
//!
//! An engram represents a learned pattern ("what normal looks like" for a
//! given class of input). [`EngramLibrary`] holds many engrams and provides
//! fast two-tier matching: eigenvalue pre-filter followed by full residual
//! scoring.
//!
//! # Example
//!
//! ```rust
//! use holon::memory::{OnlineSubspace, EngramLibrary};
//! use std::collections::HashMap;
//!
//! // Train two distinct patterns
//! let mut sub_a = OnlineSubspace::new(256, 16);
//! let mut sub_b = OnlineSubspace::new(256, 16);
//!
//! let mut rng = 42u64;
//! for _ in 0..100 {
//!     rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
//!     let v_a: Vec<f64> = (0..256).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
//!     sub_a.update(&v_a);
//!     let v_b: Vec<f64> = (0..256).map(|i| if i % 3 == 0 { 1.0 } else { 0.0 }).collect();
//!     sub_b.update(&v_b);
//! }
//!
//! // Store as engrams
//! let mut lib = EngramLibrary::new(256);
//! lib.add("pattern_a", &sub_a, None, Default::default());
//! lib.add("pattern_b", &sub_b, None, Default::default());
//!
//! // Match a probe — returns (name, residual) sorted ascending (lower = better)
//! let probe: Vec<f64> = (0..256).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
//! let matches = lib.match_vec(&probe, 2, 10);
//! assert_eq!(matches[0].0, "pattern_a"); // closest match first
//! ```

use super::subspace::{OnlineSubspace, SubspaceSnapshot};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io;

// =============================================================================
// Engram
// =============================================================================

/// A named, serializable snapshot of a trained subspace.
///
/// Engrams are immutable once created — they record what a specific pattern
/// "looks like" at the moment of training. Use [`EngramLibrary`] to collect
/// and query multiple engrams.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Engram {
    name: String,
    snapshot: SubspaceSnapshot,
    /// L2-normalised eigenvalue vector (used as fast pre-filter fingerprint).
    eigenvalue_signature: Vec<f64>,
    /// Per-feature surprise scores captured at mint time (optional).
    surprise_profile: HashMap<String, f64>,
    /// Arbitrary metadata for downstream use.
    metadata: HashMap<String, serde_json::Value>,
    /// Lazily reconstructed subspace — not serialized.
    #[serde(skip)]
    subspace_cache: Option<Box<OnlineSubspace>>,
}

impl Engram {
    fn new(
        name: String,
        snapshot: SubspaceSnapshot,
        eigenvalue_signature: Vec<f64>,
        surprise_profile: HashMap<String, f64>,
        metadata: HashMap<String, serde_json::Value>,
    ) -> Self {
        Self {
            name,
            snapshot,
            eigenvalue_signature,
            surprise_profile,
            metadata,
            subspace_cache: None,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn eigenvalue_signature(&self) -> &[f64] {
        &self.eigenvalue_signature
    }

    pub fn surprise_profile(&self) -> &HashMap<String, f64> {
        &self.surprise_profile
    }

    pub fn metadata(&self) -> &HashMap<String, serde_json::Value> {
        &self.metadata
    }

    /// Number of training observations seen when this engram was minted.
    pub fn n(&self) -> usize {
        self.snapshot.n
    }

    /// Lazily reconstruct the [`OnlineSubspace`] from the stored snapshot.
    pub fn subspace(&mut self) -> &OnlineSubspace {
        if self.subspace_cache.is_none() {
            self.subspace_cache = Some(Box::new(OnlineSubspace::from_snapshot(
                self.snapshot.clone(),
            )));
        }
        self.subspace_cache.as_ref().unwrap()
    }

    /// Score how well a vector fits this engram's manifold.
    ///
    /// Lower residual = better fit (vector is more "in-distribution" for
    /// the pattern this engram represents).
    pub fn residual(&mut self, x: &[f64]) -> f64 {
        self.subspace().residual(x)
    }
}

// =============================================================================
// EngramLibrary
// =============================================================================

/// A collection of named engrams with efficient matching.
///
/// Matching uses a two-tier strategy to avoid scoring every engram against
/// every probe:
///
/// 1. **Eigenvalue pre-filter** (O(k·n)): rank engrams by eigenvalue energy.
/// 2. **Full residual** (O(k·dim)): score top-`prefilter_k` candidates.
///
/// For small libraries (≤ `prefilter_k`), step 1 is skipped.
#[derive(Debug, Serialize, Deserialize)]
pub struct EngramLibrary {
    dim: usize,
    engrams: HashMap<String, Engram>,
}

impl EngramLibrary {
    /// Create a new, empty library for the given vector dimensionality.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            engrams: HashMap::new(),
        }
    }

    /// Mint and store an engram from a trained subspace.
    ///
    /// The eigenvalue signature is L2-normalised so that engrams of different
    /// sizes can be compared on a level playing field.
    pub fn add(
        &mut self,
        name: &str,
        subspace: &OnlineSubspace,
        surprise_profile: Option<HashMap<String, f64>>,
        metadata: HashMap<String, serde_json::Value>,
    ) -> &Engram {
        let eig = subspace.eigenvalues();
        let eig_norm: f64 = eig.iter().map(|e| e * e).sum::<f64>().sqrt();
        let sig: Vec<f64> = if eig_norm > 1e-10 {
            eig.iter().map(|e| e / eig_norm).collect()
        } else {
            eig
        };

        let engram = Engram::new(
            name.to_string(),
            subspace.snapshot(),
            sig,
            surprise_profile.unwrap_or_default(),
            metadata,
        );
        self.engrams.insert(name.to_string(), engram);
        self.engrams.get(name).unwrap()
    }

    /// Two-tier matching: eigenvalue pre-filter → full residual scoring.
    ///
    /// Returns `Vec<(name, residual)>` sorted ascending (lower = better fit).
    ///
    /// - `top_k`: how many results to return.
    /// - `prefilter_k`: how many candidates to consider for full residual
    ///   scoring after the eigenvalue pre-filter. Set equal to `self.len()`
    ///   to always score all engrams.
    pub fn match_vec(
        &mut self,
        x: &[f64],
        top_k: usize,
        prefilter_k: usize,
    ) -> Vec<(String, f64)> {
        if self.engrams.is_empty() {
            return vec![];
        }

        // If library is small enough, skip pre-filter entirely
        let candidate_names: Vec<String> = if self.engrams.len() > prefilter_k {
            // Rank by eigenvalue energy (sum of squared signature values = 1.0
            // for all normalised engrams, so we use raw norms instead)
            let mut scored: Vec<(f64, String)> = self
                .engrams
                .iter()
                .map(|(name, eng)| {
                    let energy: f64 = eng.eigenvalue_signature.iter().map(|e| e * e).sum();
                    (energy, name.clone())
                })
                .collect();
            // Sort descending by energy — more "spread" subspaces first
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            scored
                .into_iter()
                .take(prefilter_k)
                .map(|(_, name)| name)
                .collect()
        } else {
            self.engrams.keys().cloned().collect()
        };

        // Full residual scoring on candidates
        let mut results: Vec<(String, f64)> = candidate_names
            .into_iter()
            .map(|name| {
                let engram = self.engrams.get_mut(&name).unwrap();
                let res = engram.residual(x);
                (name, res)
            })
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    /// Fast eigenvalue-only matching (no full residual computation).
    ///
    /// Computes cosine similarity between the probe's eigenvalue spectrum and
    /// each stored engram's eigenvalue signature. Use for rough pre-screening.
    ///
    /// Returns `Vec<(name, cosine_similarity)>` sorted descending (higher = better match).
    pub fn match_spectrum(&self, eigenvalues: &[f64], top_k: usize) -> Vec<(String, f64)> {
        let probe_norm: f64 = eigenvalues.iter().map(|e| e * e).sum::<f64>().sqrt();
        if probe_norm < 1e-10 {
            return vec![];
        }

        let mut results: Vec<(String, f64)> = self
            .engrams
            .iter()
            .map(|(name, eng)| {
                let sig = &eng.eigenvalue_signature;
                let len = eigenvalues.len().min(sig.len());
                let dot: f64 = (0..len).map(|i| eigenvalues[i] * sig[i]).sum::<f64>();
                let sim = dot / probe_norm; // sig is already L2-normalised
                (name.clone(), sim)
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    /// Remove an engram by name. Returns `true` if it existed.
    pub fn remove(&mut self, name: &str) -> bool {
        self.engrams.remove(name).is_some()
    }

    pub fn names(&self) -> Vec<&str> {
        self.engrams.keys().map(|s| s.as_str()).collect()
    }

    pub fn get(&self, name: &str) -> Option<&Engram> {
        self.engrams.get(name)
    }

    pub fn get_mut(&mut self, name: &str) -> Option<&mut Engram> {
        self.engrams.get_mut(name)
    }

    pub fn len(&self) -> usize {
        self.engrams.len()
    }

    pub fn is_empty(&self) -> bool {
        self.engrams.is_empty()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.engrams.contains_key(name)
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Persist to a JSON file.
    pub fn save(&self, path: &str) -> io::Result<()> {
        let json = serde_json::to_string(self)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    /// Load from a JSON file.
    pub fn load(path: &str) -> io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::subspace::OnlineSubspace;

    /// Train a subspace on varied low-rank samples.
    /// The pattern is defined by which dimensions are "active" (determined by pattern_offset),
    /// with random coefficients so the mean is non-trivial and components actually learn.
    fn train_subspace(dim: usize, k: usize, pattern_offset: usize, n: usize) -> OnlineSubspace {
        let mut sub = OnlineSubspace::new(dim, k);
        let mut rng = (pattern_offset as u64).wrapping_add(42);
        for _ in 0..n {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let coeff = (rng >> 33) as f64 / u32::MAX as f64; // [0, 1)
            let v: Vec<f64> = (0..dim)
                .map(|i| {
                    if i % (pattern_offset + 2) == 0 {
                        coeff
                    } else {
                        0.0
                    }
                })
                .collect();
            sub.update(&v);
        }
        sub
    }

    #[test]
    fn test_add_and_match() {
        let dim = 256;
        let sub_a = train_subspace(dim, 8, 0, 200);
        let sub_b = train_subspace(dim, 8, 3, 200);

        let mut lib = EngramLibrary::new(dim);
        lib.add("a", &sub_a, None, Default::default());
        lib.add("b", &sub_b, None, Default::default());

        // Probe from distribution A
        let probe_a: Vec<f64> = (0..dim).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
        let matches = lib.match_vec(&probe_a, 2, 10);

        assert_eq!(matches.len(), 2);
        assert_eq!(
            matches[0].0, "a",
            "Probe from distribution A should match engram 'a' best"
        );
        assert!(
            matches[0].1 <= matches[1].1,
            "Results should be sorted ascending by residual"
        );
    }

    #[test]
    fn test_remove_and_names() {
        let dim = 64;
        let sub = train_subspace(dim, 4, 0, 50);
        let mut lib = EngramLibrary::new(dim);
        lib.add("x", &sub, None, Default::default());
        lib.add("y", &sub, None, Default::default());
        lib.add("z", &sub, None, Default::default());

        assert_eq!(lib.len(), 3);

        assert!(lib.remove("y"));
        assert!(!lib.remove("y")); // already gone

        assert_eq!(lib.len(), 2);
        let mut names = lib.names();
        names.sort();
        assert_eq!(names, vec!["x", "z"]);
    }

    #[test]
    fn test_contains_and_empty() {
        let dim = 64;
        let mut lib = EngramLibrary::new(dim);
        assert!(lib.is_empty());

        let sub = train_subspace(dim, 4, 0, 50);
        lib.add("foo", &sub, None, Default::default());

        assert!(!lib.is_empty());
        assert!(lib.contains("foo"));
        assert!(!lib.contains("bar"));
    }

    #[test]
    fn test_serialization_round_trip() {
        let dim = 128;
        let sub_a = train_subspace(dim, 8, 0, 150);
        let sub_b = train_subspace(dim, 8, 3, 150);

        let mut lib = EngramLibrary::new(dim);
        lib.add("a", &sub_a, None, Default::default());
        lib.add("b", &sub_b, None, Default::default());

        let path = "/tmp/holon_test_engram_library.json";
        lib.save(path).expect("save failed");

        let mut lib2 = EngramLibrary::load(path).expect("load failed");

        // Matches should be reproducible after round-trip
        let probe: Vec<f64> = (0..dim).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
        let m1 = lib.match_vec(&probe, 2, 10);
        let m2 = lib2.match_vec(&probe, 2, 10);

        assert_eq!(m1.len(), m2.len());
        for ((n1, r1), (n2, r2)) in m1.iter().zip(m2.iter()) {
            assert_eq!(n1, n2, "Names should match after round-trip");
            assert!(
                (r1 - r2).abs() < 1e-10,
                "Residuals should match after round-trip: {} vs {}",
                r1,
                r2
            );
        }

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_match_spectrum() {
        let dim = 128;
        let sub_a = train_subspace(dim, 8, 0, 150);
        let sub_b = train_subspace(dim, 8, 3, 150);

        let mut lib = EngramLibrary::new(dim);
        lib.add("a", &sub_a, None, Default::default());
        lib.add("b", &sub_b, None, Default::default());

        // Match using sub_a's eigenvalue spectrum
        let eig_a = sub_a.eigenvalues();
        let spectrum_matches = lib.match_spectrum(&eig_a, 2);

        // Should return both engrams (we asked for top_k=2 from a library of 2)
        assert_eq!(spectrum_matches.len(), 2, "Should return 2 results");

        // Similarities should be sorted descending
        assert!(
            spectrum_matches[0].1 >= spectrum_matches[1].1,
            "Results should be sorted descending by similarity: {} vs {}",
            spectrum_matches[0].1,
            spectrum_matches[1].1
        );

        // Results should contain both engrams
        let names: Vec<&str> = spectrum_matches.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"a"), "Results should contain engram 'a'");
        assert!(names.contains(&"b"), "Results should contain engram 'b'");
    }
}
