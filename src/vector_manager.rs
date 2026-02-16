//! Vector Manager: Deterministic atom → vector mapping.
//!
//! This module provides the core guarantee of Holon:
//! The same atomic value ALWAYS produces the SAME vector.

use crate::vector::Vector;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::RwLock;

/// Manages the mapping from atoms (strings) to vectors.
///
/// Uses deterministic hash-based seeding to ensure reproducibility.
#[derive(Clone)]
pub struct VectorManager {
    dimensions: usize,
    global_seed: u64,
    /// Cache of computed atom vectors
    cache: std::sync::Arc<RwLock<HashMap<String, Vector>>>,
    /// Cache of computed position vectors
    position_cache: std::sync::Arc<RwLock<HashMap<i64, Vector>>>,
}

impl VectorManager {
    /// Create a new VectorManager with default seed.
    pub fn new(dimensions: usize) -> Self {
        Self::with_seed(dimensions, 0)
    }

    /// Create a new VectorManager with a specific global seed.
    pub fn with_seed(dimensions: usize, global_seed: u64) -> Self {
        Self {
            dimensions,
            global_seed,
            cache: std::sync::Arc::new(RwLock::new(HashMap::new())),
            position_cache: std::sync::Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get the dimensions.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Get the vector for an atomic value.
    ///
    /// If the vector has been computed before, returns it from cache.
    /// Otherwise, computes it deterministically and caches it.
    pub fn get_vector(&self, atom: &str) -> Vector {
        // Check cache first
        {
            let cache = self.cache.read().unwrap();
            if let Some(vec) = cache.get(atom) {
                return vec.clone();
            }
        }

        // Compute the vector
        let vec = self.compute_vector(atom);

        // Cache it
        {
            let mut cache = self.cache.write().unwrap();
            cache.insert(atom.to_string(), vec.clone());
        }

        vec
    }

    /// Compute a deterministic vector for an atom.
    ///
    /// Uses SHA-256 hash of (global_seed || atom) to seed a ChaCha8 RNG,
    /// then generates random bipolar values in {-1, 0, 1}.
    fn compute_vector(&self, atom: &str) -> Vector {
        // Hash the atom with global seed
        let mut hasher = Sha256::new();
        hasher.update(self.global_seed.to_le_bytes());
        hasher.update(atom.as_bytes());
        let hash = hasher.finalize();

        // Use first 8 bytes of hash as seed
        let seed = u64::from_le_bytes(hash[0..8].try_into().unwrap());

        // Create RNG from seed
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        // Generate random bipolar vector with {-1, 0, 1}
        // Match Python's distribution: 1/3 probability each
        let mut data = vec![0i8; self.dimensions];
        for i in 0..self.dimensions {
            let r = rng.next_u32() % 3;
            data[i] = match r {
                0 => -1,
                1 => 0,
                2 => 1,
                _ => unreachable!(),
            };
        }

        Vector::from_data(data)
    }

    /// Get a deterministic position vector for sequence encoding.
    ///
    /// Position vectors are seeded with `__pos__{position}` to match
    /// the Python implementation's `_position_to_seed()`.
    pub fn get_position_vector(&self, position: i64) -> Vector {
        // Check cache first
        {
            let cache = self.position_cache.read().unwrap();
            if let Some(vec) = cache.get(&position) {
                return vec.clone();
            }
        }

        // Compute using the __pos__ prefix (matches Python)
        let atom = format!("__pos__{}", position);
        let vec = self.compute_vector(&atom);

        // Cache it
        {
            let mut cache = self.position_cache.write().unwrap();
            cache.insert(position, vec.clone());
        }

        vec
    }

    /// Export the atom codebook as a serializable map.
    ///
    /// Returns atom names mapped to their raw i8 vector data.
    /// Useful for persisting the vector cache across process restarts
    /// or sharing identical mappings between distributed nodes.
    pub fn export_codebook(&self) -> HashMap<String, Vec<i8>> {
        let cache = self.cache.read().unwrap();
        cache
            .iter()
            .map(|(atom, vec)| (atom.clone(), vec.data().to_vec()))
            .collect()
    }

    /// Import a previously exported codebook.
    ///
    /// Restores cached vectors from the format produced by `export_codebook()`.
    /// Existing cache entries are preserved; imported entries overwrite on conflict.
    pub fn import_codebook(&self, codebook: HashMap<String, Vec<i8>>) {
        let mut cache = self.cache.write().unwrap();
        for (atom, data) in codebook {
            cache.insert(atom, Vector::from_data(data));
        }
    }

    /// Clear the vector cache.
    pub fn clear_cache(&self) {
        let mut cache = self.cache.write().unwrap();
        cache.clear();
        let mut pos_cache = self.position_cache.write().unwrap();
        pos_cache.clear();
    }

    /// Get the number of cached atom vectors.
    pub fn cache_size(&self) -> usize {
        let cache = self.cache.read().unwrap();
        cache.len()
    }

    /// Get the number of cached position vectors.
    pub fn position_cache_size(&self) -> usize {
        let cache = self.position_cache.read().unwrap();
        cache.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic() {
        let vm1 = VectorManager::with_seed(1000, 42);
        let vm2 = VectorManager::with_seed(1000, 42);

        let v1 = vm1.get_vector("hello");
        let v2 = vm2.get_vector("hello");

        assert_eq!(v1, v2);
    }

    #[test]
    fn test_different_seeds() {
        let vm1 = VectorManager::with_seed(1000, 42);
        let vm2 = VectorManager::with_seed(1000, 43);

        let v1 = vm1.get_vector("hello");
        let v2 = vm2.get_vector("hello");

        // Different seeds should produce different vectors
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_different_atoms() {
        let vm = VectorManager::new(1000);

        let v1 = vm.get_vector("hello");
        let v2 = vm.get_vector("world");

        assert_ne!(v1, v2);
    }

    #[test]
    fn test_caching() {
        let vm = VectorManager::new(1000);

        assert_eq!(vm.cache_size(), 0);

        vm.get_vector("hello");
        assert_eq!(vm.cache_size(), 1);

        vm.get_vector("hello"); // Should use cache
        assert_eq!(vm.cache_size(), 1);

        vm.get_vector("world");
        assert_eq!(vm.cache_size(), 2);
    }

    #[test]
    fn test_position_vector_deterministic() {
        let vm1 = VectorManager::with_seed(1000, 42);
        let vm2 = VectorManager::with_seed(1000, 42);

        let p0_a = vm1.get_position_vector(0);
        let p0_b = vm2.get_position_vector(0);
        assert_eq!(p0_a, p0_b);

        let p1 = vm1.get_position_vector(1);
        assert_ne!(p0_a, p1); // Different positions → different vectors
    }

    #[test]
    fn test_position_vector_caching() {
        let vm = VectorManager::new(1000);

        assert_eq!(vm.position_cache_size(), 0);

        vm.get_position_vector(0);
        assert_eq!(vm.position_cache_size(), 1);

        vm.get_position_vector(0); // Should use cache
        assert_eq!(vm.position_cache_size(), 1);

        vm.get_position_vector(5);
        assert_eq!(vm.position_cache_size(), 2);
    }

    #[test]
    fn test_position_vector_different_from_atom() {
        let vm = VectorManager::new(1000);

        // Position vector for 0 should differ from atom "0"
        let pos = vm.get_position_vector(0);
        let atom = vm.get_vector("0");
        assert_ne!(pos, atom);
    }

    #[test]
    fn test_export_import_codebook() {
        let vm1 = VectorManager::with_seed(1000, 42);

        // Populate some vectors
        let hello = vm1.get_vector("hello");
        let world = vm1.get_vector("world");

        // Export
        let codebook = vm1.export_codebook();
        assert_eq!(codebook.len(), 2);

        // Import into a fresh manager
        let vm2 = VectorManager::with_seed(1000, 99); // Different seed
        vm2.import_codebook(codebook);

        // Should get identical vectors from cache (not recomputed)
        let hello2 = vm2.get_vector("hello");
        let world2 = vm2.get_vector("world");

        assert_eq!(hello, hello2);
        assert_eq!(world, world2);
    }

    #[test]
    fn test_clear_cache_clears_both() {
        let vm = VectorManager::new(1000);

        vm.get_vector("hello");
        vm.get_position_vector(0);

        assert_eq!(vm.cache_size(), 1);
        assert_eq!(vm.position_cache_size(), 1);

        vm.clear_cache();

        assert_eq!(vm.cache_size(), 0);
        assert_eq!(vm.position_cache_size(), 0);
    }

    #[test]
    fn test_verify_determinism() {
        // Same seed, different VectorManager instances → same vectors
        let atoms = vec!["alpha", "beta", "gamma", "delta", "epsilon"];
        let seed = 12345;

        for _ in 0..3 {
            let vm = VectorManager::with_seed(4096, seed);
            let vecs: Vec<Vector> = atoms.iter().map(|a| vm.get_vector(a)).collect();

            // Verify against a fresh instance
            let vm2 = VectorManager::with_seed(4096, seed);
            for (i, atom) in atoms.iter().enumerate() {
                assert_eq!(vecs[i], vm2.get_vector(atom));
            }
        }
    }
}
