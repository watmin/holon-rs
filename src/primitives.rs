//! VSA Primitives: Core vector operations.
//!
//! These are the fundamental building blocks of Vector Symbolic Architectures:
//! - **bind**: Create associations (AND-like)
//! - **bundle**: Create superpositions (OR-like)
//! - **negate**: Remove from superposition (NOT)
//! - **permute**: Encode sequence positions

use crate::vector::Vector;

/// Collection of VSA primitive operations.
pub struct Primitives;

impl Primitives {
    /// Bind two vectors (element-wise multiplication for bipolar).
    ///
    /// Creates an association between two concepts. The result is
    /// dissimilar to both inputs, but `unbind(bind(A, B), A) ≈ B`.
    ///
    /// For bipolar vectors, this is element-wise multiplication.
    pub fn bind(a: &Vector, b: &Vector) -> Vector {
        assert_eq!(
            a.dimensions(),
            b.dimensions(),
            "Dimension mismatch in bind"
        );

        let data: Vec<i8> = a
            .data()
            .iter()
            .zip(b.data().iter())
            .map(|(&x, &y)| x * y)
            .collect();

        Vector::from_data(data)
    }

    /// Bundle multiple vectors (element-wise majority vote).
    ///
    /// Creates a superposition that is similar to ALL input vectors.
    /// The result contains information about all inputs.
    pub fn bundle(vectors: &[&Vector]) -> Vector {
        if vectors.is_empty() {
            panic!("Cannot bundle empty vector list");
        }

        let dimensions = vectors[0].dimensions();
        let mut sums = vec![0i32; dimensions];

        for vec in vectors {
            for (i, &v) in vec.data().iter().enumerate() {
                sums[i] += v as i32;
            }
        }

        // Threshold: positive → 1, negative → -1, zero → 0
        let data: Vec<i8> = sums
            .iter()
            .map(|&s| {
                if s > 0 {
                    1
                } else if s < 0 {
                    -1
                } else {
                    0
                }
            })
            .collect();

        Vector::from_data(data)
    }

    /// Bundle vectors with explicit weights.
    ///
    /// Allows weighting certain vectors more heavily in the superposition.
    pub fn weighted_bundle(vectors: &[&Vector], weights: &[f64]) -> Vector {
        assert_eq!(
            vectors.len(),
            weights.len(),
            "Vectors and weights must have same length"
        );
        if vectors.is_empty() {
            panic!("Cannot bundle empty vector list");
        }

        let dimensions = vectors[0].dimensions();
        let mut sums = vec![0.0; dimensions];

        for (vec, &weight) in vectors.iter().zip(weights.iter()) {
            for (i, &v) in vec.data().iter().enumerate() {
                sums[i] += (v as f64) * weight;
            }
        }

        // Threshold to bipolar
        let data: Vec<i8> = sums
            .iter()
            .map(|&s| {
                if s > 0.0 {
                    1
                } else if s < 0.0 {
                    -1
                } else {
                    0
                }
            })
            .collect();

        Vector::from_data(data)
    }

    /// Negate a component from a superposition.
    ///
    /// Removes or weakens a component's influence from a bundled vector.
    /// The default method is "subtract".
    pub fn negate(superposition: &Vector, component: &Vector) -> Vector {
        Self::negate_with_method(superposition, component, NegateMethod::Subtract)
    }

    /// Negate with a specific method.
    pub fn negate_with_method(
        superposition: &Vector,
        component: &Vector,
        method: NegateMethod,
    ) -> Vector {
        assert_eq!(
            superposition.dimensions(),
            component.dimensions(),
            "Dimension mismatch in negate"
        );

        match method {
            NegateMethod::Subtract => {
                let data: Vec<i8> = superposition
                    .data()
                    .iter()
                    .zip(component.data().iter())
                    .map(|(&s, &c)| {
                        let result = (s as i32) - (c as i32);
                        result.clamp(-1, 1) as i8
                    })
                    .collect();
                Vector::from_data(data)
            }
            NegateMethod::Zero => {
                // Zero out positions where component is non-zero
                let data: Vec<i8> = superposition
                    .data()
                    .iter()
                    .zip(component.data().iter())
                    .map(|(&s, &c)| if c != 0 { 0 } else { s })
                    .collect();
                Vector::from_data(data)
            }
            NegateMethod::Invert => {
                // Invert the component and bundle
                let inverted: Vec<i8> = component.data().iter().map(|&c| -c).collect();
                let data: Vec<i8> = superposition
                    .data()
                    .iter()
                    .zip(inverted.iter())
                    .map(|(&s, &i)| {
                        let result = (s as i32) + (i as i32);
                        result.clamp(-1, 1) as i8
                    })
                    .collect();
                Vector::from_data(data)
            }
        }
    }

    /// Amplify a component's presence in a superposition.
    ///
    /// Strengthens the signal from a specific component.
    pub fn amplify(superposition: &Vector, component: &Vector, strength: f64) -> Vector {
        assert_eq!(
            superposition.dimensions(),
            component.dimensions(),
            "Dimension mismatch in amplify"
        );

        let data: Vec<i8> = superposition
            .data()
            .iter()
            .zip(component.data().iter())
            .map(|(&s, &c)| {
                let result = (s as f64) + (c as f64) * strength;
                if result > 0.0 {
                    1
                } else if result < 0.0 {
                    -1
                } else {
                    0
                }
            })
            .collect();

        Vector::from_data(data)
    }

    /// Extract common pattern from multiple vectors.
    ///
    /// Returns a vector containing only dimensions where at least
    /// `threshold` fraction of vectors agree.
    pub fn prototype(vectors: &[&Vector], threshold: f64) -> Vector {
        if vectors.is_empty() {
            panic!("Cannot create prototype from empty vector list");
        }

        let dimensions = vectors[0].dimensions();
        let n = vectors.len() as f64;
        let min_votes = (n * threshold).ceil() as i32;

        let mut sums = vec![0i32; dimensions];

        for vec in vectors {
            for (i, &v) in vec.data().iter().enumerate() {
                sums[i] += v as i32;
            }
        }

        // Only keep dimensions with sufficient agreement
        let data: Vec<i8> = sums
            .iter()
            .map(|&s| {
                if s >= min_votes {
                    1
                } else if s <= -min_votes {
                    -1
                } else {
                    0
                }
            })
            .collect();

        Vector::from_data(data)
    }

    /// Incremental prototype update.
    ///
    /// Updates an existing prototype with a new example.
    /// `count` is the number of examples already in the prototype.
    pub fn prototype_add(prototype: &Vector, example: &Vector, count: usize) -> Vector {
        assert_eq!(
            prototype.dimensions(),
            example.dimensions(),
            "Dimension mismatch in prototype_add"
        );

        let weight = 1.0 / (count as f64 + 1.0);

        let data: Vec<i8> = prototype
            .data()
            .iter()
            .zip(example.data().iter())
            .map(|(&p, &e)| {
                let result = (p as f64) * (1.0 - weight) + (e as f64) * weight;
                if result > 0.0 {
                    1
                } else if result < 0.0 {
                    -1
                } else {
                    0
                }
            })
            .collect();

        Vector::from_data(data)
    }

    /// Compute what changed between two states.
    ///
    /// Positive values indicate additions, negative indicate removals.
    pub fn difference(before: &Vector, after: &Vector) -> Vector {
        assert_eq!(
            before.dimensions(),
            after.dimensions(),
            "Dimension mismatch in difference"
        );

        let data: Vec<i8> = before
            .data()
            .iter()
            .zip(after.data().iter())
            .map(|(&b, &a)| {
                let diff = (a as i32) - (b as i32);
                diff.clamp(-1, 1) as i8
            })
            .collect();

        Vector::from_data(data)
    }

    /// Weighted interpolation between two vectors.
    ///
    /// `alpha = 0.0` returns vec1, `alpha = 1.0` returns vec2.
    pub fn blend(vec1: &Vector, vec2: &Vector, alpha: f64) -> Vector {
        assert_eq!(
            vec1.dimensions(),
            vec2.dimensions(),
            "Dimension mismatch in blend"
        );

        let data: Vec<i8> = vec1
            .data()
            .iter()
            .zip(vec2.data().iter())
            .map(|(&v1, &v2)| {
                let result = (v1 as f64) * (1.0 - alpha) + (v2 as f64) * alpha;
                if result > 0.0 {
                    1
                } else if result < 0.0 {
                    -1
                } else {
                    0
                }
            })
            .collect();

        Vector::from_data(data)
    }

    /// Extract the part of vec that resonates with reference.
    ///
    /// Keeps only dimensions where both vectors agree.
    pub fn resonance(vec: &Vector, reference: &Vector) -> Vector {
        assert_eq!(
            vec.dimensions(),
            reference.dimensions(),
            "Dimension mismatch in resonance"
        );

        let data: Vec<i8> = vec
            .data()
            .iter()
            .zip(reference.data().iter())
            .map(|(&v, &r)| if v == r { v } else { 0 })
            .collect();

        Vector::from_data(data)
    }

    /// Circular shift (permutation) of vector dimensions.
    ///
    /// Used for encoding sequential position.
    pub fn permute(vec: &Vector, k: i32) -> Vector {
        let n = vec.dimensions() as i32;
        let shift = ((k % n) + n) % n; // Handle negative shifts

        let mut data = vec![0i8; vec.dimensions()];

        for (i, &v) in vec.data().iter().enumerate() {
            let new_pos = ((i as i32 + shift) % n) as usize;
            data[new_pos] = v;
        }

        Vector::from_data(data)
    }

    /// Cleanup: find the closest vector in a codebook.
    ///
    /// Returns the index and similarity of the best match.
    pub fn cleanup(noisy: &Vector, codebook: &[Vector]) -> Option<(usize, f64)> {
        if codebook.is_empty() {
            return None;
        }

        let mut best_idx = 0;
        let mut best_sim = f64::NEG_INFINITY;

        for (i, vec) in codebook.iter().enumerate() {
            let sim = crate::similarity::Similarity::cosine(noisy, vec);
            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
        }

        Some((best_idx, best_sim))
    }
}

/// Method for negation operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NegateMethod {
    /// Subtract component (most common)
    Subtract,
    /// Zero out matching positions
    Zero,
    /// Invert and add
    Invert,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bind_self_inverse() {
        let a = Vector::from_data(vec![1, -1, 1, -1, 1, -1]);
        let b = Vector::from_data(vec![1, 1, -1, -1, 1, 1]);

        let ab = Primitives::bind(&a, &b);
        let b_recovered = Primitives::bind(&ab, &a);

        assert_eq!(b, b_recovered);
    }

    #[test]
    fn test_bundle_majority() {
        let a = Vector::from_data(vec![1, 1, 1, -1]);
        let b = Vector::from_data(vec![1, 1, -1, -1]);
        let c = Vector::from_data(vec![1, -1, -1, -1]);

        let abc = Primitives::bundle(&[&a, &b, &c]);

        // First: 3 votes for 1 → 1
        // Second: 2 votes for 1, 1 for -1 → 1
        // Third: 1 vote for 1, 2 for -1 → -1
        // Fourth: 3 votes for -1 → -1
        assert_eq!(abc.data(), &[1, 1, -1, -1]);
    }

    #[test]
    fn test_permute_circular() {
        let v = Vector::from_data(vec![1, 2, 3, 4, 5]);

        let shifted = Primitives::permute(&v, 2);
        assert_eq!(shifted.data(), &[4, 5, 1, 2, 3]);

        // Inverse shift
        let restored = Primitives::permute(&shifted, -2);
        assert_eq!(restored.data(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_negate() {
        let a = Vector::from_data(vec![1, 1, 1, 1]);
        let b = Vector::from_data(vec![1, 1, -1, -1]);

        let ab = Primitives::bundle(&[&a, &b]);
        let without_b = Primitives::negate(&ab, &b);

        // After negating B, should be less similar to B
        let sim_before = crate::similarity::Similarity::cosine(&ab, &b);
        let sim_after = crate::similarity::Similarity::cosine(&without_b, &b);

        assert!(sim_after < sim_before);
    }

    #[test]
    fn test_difference() {
        let before = Vector::from_data(vec![1, -1, 0, 1]);
        let after = Vector::from_data(vec![1, 1, 0, -1]);

        let diff = Primitives::difference(&before, &after);

        // Dimension 0: 1 - 1 = 0
        // Dimension 1: 1 - (-1) = 2 → clamped to 1
        // Dimension 2: 0 - 0 = 0
        // Dimension 3: -1 - 1 = -2 → clamped to -1
        assert_eq!(diff.data(), &[0, 1, 0, -1]);
    }

    #[test]
    fn test_resonance() {
        let a = Vector::from_data(vec![1, -1, 1, -1]);
        let b = Vector::from_data(vec![1, 1, 1, -1]);

        let r = Primitives::resonance(&a, &b);

        // Only positions 0, 2, 3 agree
        assert_eq!(r.data(), &[1, 0, 1, -1]);
    }
}
