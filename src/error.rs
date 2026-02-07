//! Error types for Holon.

use thiserror::Error;

/// Holon error types.
#[derive(Error, Debug)]
pub enum HolonError {
    /// JSON parsing error
    #[error("JSON parse error: {0}")]
    JsonParse(#[from] serde_json::Error),

    /// Invalid vector dimensions
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// Invalid scalar value
    #[error("Invalid scalar value: {0}")]
    InvalidScalar(String),

    /// Empty input where non-empty was required
    #[error("Empty input: {0}")]
    EmptyInput(String),
}

/// Result type alias for Holon operations.
pub type Result<T> = std::result::Result<T, HolonError>;
