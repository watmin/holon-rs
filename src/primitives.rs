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

    // =========================================================================
    // Extended Algebra
    // =========================================================================

    /// Unbind a key from a bound vector.
    ///
    /// For bipolar vectors, unbind is identical to bind (self-inverse property):
    /// `unbind(bind(A, B), A) = B`
    ///
    /// This is an explicit alias to help users discover the algebra.
    pub fn unbind(bound: &Vector, key: &Vector) -> Vector {
        Self::bind(bound, key)
    }

    /// Return similarity as a VECTOR, not a scalar.
    ///
    /// Preserves dimension-wise agreement pattern.
    /// +1 where both agree, -1 where they disagree, 0 where either is zero.
    pub fn similarity_profile(vec_a: &Vector, vec_b: &Vector) -> Vector {
        assert_eq!(
            vec_a.dimensions(),
            vec_b.dimensions(),
            "Dimension mismatch in similarity_profile"
        );

        let data: Vec<i8> = vec_a
            .data()
            .iter()
            .zip(vec_b.data().iter())
            .map(|(&a, &b)| a * b)
            .collect();

        Vector::from_data(data)
    }

    /// Weighted resonance - soft attention in VSA algebra.
    ///
    /// Modes:
    /// - Hard: Binary resonance (same as resonance)
    /// - Soft: Smooth weighting based on agreement
    /// - Amplify: Boost agreeing dimensions
    pub fn attend(query: &Vector, memory: &Vector, strength: f64, mode: AttendMode) -> Vector {
        assert_eq!(
            query.dimensions(),
            memory.dimensions(),
            "Dimension mismatch in attend"
        );

        match mode {
            AttendMode::Hard => Self::resonance(query, memory),
            AttendMode::Soft => {
                // Weighted by agreement
                let q_norm = query.norm();
                let m_norm = memory.norm();

                let data: Vec<i8> = query
                    .data()
                    .iter()
                    .zip(memory.data().iter())
                    .map(|(&q, &m)| {
                        let q_n = if q_norm > 1e-10 { q as f64 / q_norm } else { 0.0 };
                        let m_n = if m_norm > 1e-10 { m as f64 / m_norm } else { 0.0 };
                        let agreement = q_n * m_n;
                        let weight = (1.0 + (strength * agreement).tanh()) / 2.0;
                        let result = (m as f64) * weight;
                        if result > 0.5 {
                            1
                        } else if result < -0.5 {
                            -1
                        } else {
                            0
                        }
                    })
                    .collect();

                Vector::from_data(data)
            }
            AttendMode::Amplify => {
                let data: Vec<i8> = query
                    .data()
                    .iter()
                    .zip(memory.data().iter())
                    .map(|(&q, &m)| {
                        // Agree if same sign and both non-zero
                        let agree = q != 0 && m != 0 && (q > 0) == (m > 0);
                        let result = if agree {
                            (m as f64) * (1.0 + strength)
                        } else {
                            m as f64
                        };
                        if result > 0.5 {
                            1
                        } else if result < -0.5 {
                            -1
                        } else {
                            0
                        }
                    })
                    .collect();

                Vector::from_data(data)
            }
        }
    }

    /// Relational transfer: A is to B as C is to ?
    ///
    /// Computes: C + difference(B, A)
    pub fn analogy(a: &Vector, b: &Vector, c: &Vector) -> Vector {
        let delta = Self::difference(a, b);
        let data: Vec<i8> = c
            .data()
            .iter()
            .zip(delta.data().iter())
            .map(|(&cv, &dv)| {
                let result = (cv as i32) + (dv as i32);
                result.clamp(-1, 1) as i8
            })
            .collect();

        Vector::from_data(data)
    }

    /// Project vector onto subspace defined by exemplars.
    ///
    /// Returns the component of `vec` that lies in the subspace spanned
    /// by the given exemplar vectors.
    pub fn project(vec: &Vector, subspace: &[&Vector], orthogonalize: bool) -> Vector {
        if subspace.is_empty() {
            return Vector::zeros(vec.dimensions());
        }

        let v: Vec<f64> = vec.data().iter().map(|&x| x as f64).collect();
        let mut basis: Vec<Vec<f64>> = subspace
            .iter()
            .map(|u| u.data().iter().map(|&x| x as f64).collect())
            .collect();

        // Gram-Schmidt orthogonalization
        if orthogonalize && basis.len() > 1 {
            let mut ortho_basis: Vec<Vec<f64>> = Vec::new();
            for u in basis.iter() {
                let mut u_new = u.clone();
                for prev in ortho_basis.iter() {
                    let prev_norm_sq: f64 = prev.iter().map(|&x| x * x).sum();
                    if prev_norm_sq > 1e-10 {
                        let dot: f64 = u_new.iter().zip(prev.iter()).map(|(&a, &b)| a * b).sum();
                        let coeff = dot / prev_norm_sq;
                        for (i, p) in prev.iter().enumerate() {
                            u_new[i] -= coeff * p;
                        }
                    }
                }
                let norm: f64 = u_new.iter().map(|&x| x * x).sum::<f64>().sqrt();
                if norm > 1e-10 {
                    ortho_basis.push(u_new);
                }
            }
            basis = ortho_basis;
        }

        // Project onto each basis vector and sum
        let mut projection = vec![0.0; vec.dimensions()];
        for u in basis.iter() {
            let norm_sq: f64 = u.iter().map(|&x| x * x).sum();
            if norm_sq > 1e-10 {
                let dot: f64 = v.iter().zip(u.iter()).map(|(&a, &b)| a * b).sum();
                let coeff = dot / norm_sq;
                for (i, &ui) in u.iter().enumerate() {
                    projection[i] += coeff * ui;
                }
            }
        }

        // Threshold to bipolar
        let data: Vec<i8> = projection
            .iter()
            .map(|&x| {
                if x > 0.0 {
                    1
                } else if x < 0.0 {
                    -1
                } else {
                    0
                }
            })
            .collect();

        Vector::from_data(data)
    }

    /// Bind only where condition is met (gated binding).
    pub fn conditional_bind(
        vec_a: &Vector,
        vec_b: &Vector,
        gate: &Vector,
        mode: GateMode,
    ) -> Vector {
        assert_eq!(
            vec_a.dimensions(),
            vec_b.dimensions(),
            "Dimension mismatch in conditional_bind"
        );
        assert_eq!(
            vec_a.dimensions(),
            gate.dimensions(),
            "Dimension mismatch in conditional_bind"
        );

        let data: Vec<i8> = vec_a
            .data()
            .iter()
            .zip(vec_b.data().iter())
            .zip(gate.data().iter())
            .map(|((&a, &b), &g)| {
                let pass = match mode {
                    GateMode::Positive => g > 0,
                    GateMode::Negative => g < 0,
                    GateMode::NonZero => g != 0,
                };
                if pass { a * b } else { 0 }
            })
            .collect();

        Vector::from_data(data)
    }

    /// Measure the "complexity" or "mixedness" of a vector.
    ///
    /// Returns a value between 0.0 (minimal) and 1.0 (maximal).
    pub fn complexity(vec: &Vector) -> f64 {
        let total = vec.dimensions() as f64;
        if total == 0.0 {
            return 0.0;
        }

        let nnz = vec.nnz() as f64;
        let density = nnz / total;

        let pos = vec.data().iter().filter(|&&x| x > 0).count() as f64;
        let neg = vec.data().iter().filter(|&&x| x < 0).count() as f64;
        let total_active = pos + neg;

        let balance = if total_active == 0.0 {
            0.0
        } else {
            let ratio = pos.min(neg) / total_active;
            ratio * 2.0
        };

        // Combined metric
        let complexity_score = 0.5 * density + 0.5 * balance;
        complexity_score.clamp(0.0, 1.0)
    }

    /// Reconstruct components from a vector using a codebook.
    ///
    /// Returns a list of (index, similarity) tuples for components
    /// that exceed the threshold, sorted by similarity descending.
    pub fn invert(
        vec: &Vector,
        codebook: &[Vector],
        top_k: usize,
        threshold: f64,
    ) -> Vec<(usize, f64)> {
        let mut results: Vec<(usize, f64)> = codebook
            .iter()
            .enumerate()
            .map(|(i, code_vec)| {
                let sim = crate::similarity::Similarity::cosine(vec, code_vec);
                (i, sim)
            })
            .filter(|(_, sim)| *sim >= threshold)
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    /// Find structural breakpoints in a vector stream.
    ///
    /// Returns indices where segments begin (including 0).
    pub fn segment(
        stream: &[Vector],
        window: usize,
        threshold: f64,
        method: SegmentMethod,
    ) -> Vec<usize> {
        if stream.len() < 2 {
            return if stream.is_empty() { vec![] } else { vec![0] };
        }

        let mut breakpoints = vec![0];

        match method {
            SegmentMethod::Diff => {
                for i in 1..stream.len() {
                    let sim = crate::similarity::Similarity::cosine(&stream[i], &stream[i - 1]);
                    if sim < threshold {
                        breakpoints.push(i);
                    }
                }
            }
            SegmentMethod::Prototype => {
                for i in 1..stream.len() {
                    let start = if i > window { i - window } else { 0 };
                    let window_vecs: Vec<&Vector> = stream[start..i].iter().collect();
                    let baseline = Self::prototype(&window_vecs, 0.5);
                    let sim = crate::similarity::Similarity::cosine(&stream[i], &baseline);
                    if sim < threshold {
                        breakpoints.push(i);
                    }
                }
            }
        }

        breakpoints
    }
}

/// Mode for attention operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttendMode {
    /// Binary resonance
    Hard,
    /// Smooth weighting
    Soft,
    /// Boost agreeing dimensions
    Amplify,
}

/// Mode for gated binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateMode {
    /// Bind where gate > 0
    Positive,
    /// Bind where gate < 0
    Negative,
    /// Bind where gate != 0
    NonZero,
}

/// Method for segmentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentMethod {
    /// Compare consecutive vectors
    Diff,
    /// Compare to running prototype
    Prototype,
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

    // =========================================================================
    // Extended Primitives Tests
    // =========================================================================

    #[test]
    fn test_unbind_recovers_value() {
        let a = Vector::from_data(vec![1, -1, 1, -1, 1, -1]);
        let b = Vector::from_data(vec![1, 1, -1, -1, 1, 1]);

        let ab = Primitives::bind(&a, &b);
        let recovered = Primitives::unbind(&ab, &a);

        assert_eq!(b, recovered);
    }

    #[test]
    fn test_similarity_profile() {
        let a = Vector::from_data(vec![1, -1, 1, -1]);
        let b = Vector::from_data(vec![1, 1, 1, -1]);

        let profile = Primitives::similarity_profile(&a, &b);

        // Position 0: 1*1 = 1, Position 1: -1*1 = -1
        // Position 2: 1*1 = 1, Position 3: -1*-1 = 1
        assert_eq!(profile.data(), &[1, -1, 1, 1]);
    }

    #[test]
    fn test_attend_hard() {
        let query = Vector::from_data(vec![1, -1, 1, -1]);
        let memory = Vector::from_data(vec![1, 1, 1, -1]);

        let attended = Primitives::attend(&query, &memory, 1.0, AttendMode::Hard);

        // Same as resonance
        let resonance = Primitives::resonance(&query, &memory);
        assert_eq!(attended, resonance);
    }

    #[test]
    fn test_analogy() {
        let a = Vector::from_data(vec![1, 1, 1, 1]);
        let b = Vector::from_data(vec![1, 1, 1, 1]); // Same as A
        let c = Vector::from_data(vec![-1, -1, 1, 1]);

        // If A == B, result should be ~C
        let result = Primitives::analogy(&a, &b, &c);
        assert_eq!(result, c);
    }

    #[test]
    fn test_project_onto_self() {
        let a = Vector::from_data(vec![1, -1, 1, -1]);

        let projected = Primitives::project(&a, &[&a], true);

        // Projecting onto self should return ~self
        assert_eq!(projected, a);
    }

    #[test]
    fn test_conditional_bind() {
        let a = Vector::from_data(vec![1, -1, 1, -1]);
        let b = Vector::from_data(vec![1, 1, -1, -1]);
        let gate = Vector::from_data(vec![1, -1, 1, -1]); // Positive at 0, 2

        let gated = Primitives::conditional_bind(&a, &b, &gate, GateMode::Positive);

        // Only positions 0 and 2 should have binding
        // Position 0: 1*1 = 1, Position 2: 1*-1 = -1
        assert_eq!(gated.data(), &[1, 0, -1, 0]);
    }

    #[test]
    fn test_complexity() {
        let zeros = Vector::zeros(4);
        let full = Vector::from_data(vec![1, -1, 1, -1]);

        let c_zeros = Primitives::complexity(&zeros);
        let c_full = Primitives::complexity(&full);

        assert_eq!(c_zeros, 0.0);
        assert!(c_full > 0.0);
    }

    #[test]
    fn test_segment_detects_change() {
        // Create stream with clear change
        let pattern1 = Vector::from_data(vec![1, 1, 1, 1]);
        let pattern2 = Vector::from_data(vec![-1, -1, -1, -1]);

        let stream: Vec<Vector> = (0..10)
            .map(|i| if i < 5 { pattern1.clone() } else { pattern2.clone() })
            .collect();

        let breakpoints = Primitives::segment(&stream, 3, 0.5, SegmentMethod::Diff);

        // Should detect change at index 5
        assert!(breakpoints.contains(&5), "Should detect change at 5: {:?}", breakpoints);
    }

    #[test]
    fn test_invert_finds_components() {
        let a = Vector::from_data(vec![1, 1, 1, 1]);
        let b = Vector::from_data(vec![-1, -1, -1, -1]);
        let c = Vector::from_data(vec![1, -1, 1, -1]);

        let codebook = vec![a.clone(), b.clone(), c.clone()];

        // Query with 'a' should find 'a' as top match
        let results = Primitives::invert(&a, &codebook, 3, 0.0);

        assert!(!results.is_empty());
        assert_eq!(results[0].0, 0, "Should find 'a' (index 0) as top match");
        assert!(results[0].1 > 0.9, "Should have high similarity");
    }
}
