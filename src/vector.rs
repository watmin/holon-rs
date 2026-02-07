//! Vector type for Holon.
//!
//! Holon uses bipolar vectors with elements in {-1, 0, 1}.
//! Internally stored as i8 for memory efficiency.

use std::ops::{Index, IndexMut};

/// A high-dimensional vector with bipolar elements {-1, 0, 1}.
///
/// This is the core data structure for all VSA operations.
#[derive(Clone, Debug)]
pub struct Vector {
    /// The actual vector data
    data: Vec<i8>,
}

impl Vector {
    /// Create a new zero vector of given dimensionality.
    pub fn zeros(dimensions: usize) -> Self {
        Self {
            data: vec![0; dimensions],
        }
    }

    /// Create a vector from raw data.
    pub fn from_data(data: Vec<i8>) -> Self {
        Self { data }
    }

    /// Create a vector from f64 values (thresholds to bipolar).
    pub fn from_f64(values: &[f64]) -> Self {
        let data: Vec<i8> = values
            .iter()
            .map(|&v| {
                if v > 0.0 {
                    1
                } else if v < 0.0 {
                    -1
                } else {
                    0
                }
            })
            .collect();
        Self { data }
    }

    /// Get the dimensionality.
    pub fn dimensions(&self) -> usize {
        self.data.len()
    }

    /// Get the raw data as a slice.
    pub fn data(&self) -> &[i8] {
        &self.data
    }

    /// Get mutable access to the raw data.
    pub fn data_mut(&mut self) -> &mut [i8] {
        &mut self.data
    }

    /// Convert to f64 vector.
    pub fn to_f64(&self) -> Vec<f64> {
        self.data.iter().map(|&v| v as f64).collect()
    }

    /// Compute the L2 norm.
    pub fn norm(&self) -> f64 {
        let sum_sq: i64 = self.data.iter().map(|&v| (v as i64) * (v as i64)).sum();
        (sum_sq as f64).sqrt()
    }

    /// Return a unit-normalized vector (as f64).
    pub fn normalized(&self) -> Vec<f64> {
        let norm = self.norm();
        if norm < 1e-10 {
            return vec![0.0; self.dimensions()];
        }
        self.data.iter().map(|&v| (v as f64) / norm).collect()
    }

    /// Count non-zero elements.
    pub fn nnz(&self) -> usize {
        self.data.iter().filter(|&&v| v != 0).count()
    }
}

impl Index<usize> for Vector {
    type Output = i8;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for Vector {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl PartialEq for Vector {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl Eq for Vector {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let v = Vector::zeros(100);
        assert_eq!(v.dimensions(), 100);
        assert!(v.data().iter().all(|&x| x == 0));
    }

    #[test]
    fn test_from_f64() {
        let v = Vector::from_f64(&[1.0, -0.5, 0.0, 2.3, -10.0]);
        assert_eq!(v.data(), &[1, -1, 0, 1, -1]);
    }

    #[test]
    fn test_norm() {
        let v = Vector::from_data(vec![1, -1, 1, -1]);
        assert!((v.norm() - 2.0).abs() < 1e-10);
    }
}
