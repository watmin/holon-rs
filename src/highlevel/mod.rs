//! High-level convenience API.
//!
//! This layer provides [`Holon`], an ergonomic wrapper that owns an
//! [`Encoder`](crate::kernel::Encoder), [`VectorManager`](crate::kernel::VectorManager),
//! and [`ScalarEncoder`](crate::kernel::ScalarEncoder) and delegates to the
//! [`kernel`](crate::kernel) and [`memory`](crate::memory) layers.
//!
//! For production or library code, prefer importing from [`kernel`](crate::kernel)
//! and [`memory`](crate::memory) directly.

pub mod client;

pub use client::Holon;
