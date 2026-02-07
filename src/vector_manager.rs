//! Vector Manager: Deterministic atom → vector mapping.
//!
//! This module provides the core guarantee of Holon:
//! The same atomic value ALWAYS produces the SAME vector.

use crate::vector::Vector;
use rand::SeedableRng;
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
    /// Cache of computed vectors
    cache: std::sync::Arc<RwLock<HashMap<String, Vector>>>,
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
    /// then generates random bipolar values.
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

        // Generate random bipolar vector
        let mut data = vec![0i8; self.dimensions];
        for i in 0..self.dimensions {
            // Use next u32 to determine sign
            let r = rng.next_u32();
            data[i] = if r & 1 == 0 { 1 } else { -1 };
        }

        Vector::from_data(data)
    }

    /// Clear the vector cache.
    pub fn clear_cache(&self) {
        let mut cache = self.cache.write().unwrap();
        cache.clear();
    }

    /// Get the number of cached vectors.
    pub fn cache_size(&self) -> usize {
        let cache = self.cache.read().unwrap();
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
}
