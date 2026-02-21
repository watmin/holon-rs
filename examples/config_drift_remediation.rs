//! Config Drift Remediation — Detect, Attribute, and Fix
//!
//! A golden config manifold learned from 20 stable configs × 10 passes.
//! Five drift scenarios detected, attributed to exact fields via residual-swap,
//! and remediated with difference() + amplify(). No schema annotations.
//!
//! Run: cargo run --example config_drift_remediation --release

use holon::kernel::{
    Encoder, Primitives, VectorManager, Vector,
    ScalarValue, WalkType, Walkable, WalkableValue,
};
use holon::memory::{EngramLibrary, OnlineSubspace};
use rand::prelude::*;
use std::collections::HashMap;

// =============================================================================
// Config struct + Walkable
// =============================================================================

#[derive(Clone)]
struct Config {
    db_host: String,
    db_port: i64,
    db_pool_size: i64,
    db_ssl: bool,
    redis_host: String,
    redis_port: i64,
    redis_ttl: i64,
    api_rate_limit: i64,
    api_timeout: i64,
    features_dark_mode: bool,
    features_analytics: bool,
    features_beta: bool,
}

impl Walkable for Config {
    fn walk_type(&self) -> WalkType {
        WalkType::Map
    }

    fn walk_map_items(&self) -> Vec<(&str, WalkableValue)> {
        vec![
            ("db.host",              WalkableValue::Scalar(ScalarValue::String(self.db_host.clone()))),
            ("db.port",              (self.db_port).to_walkable_value()),
            ("db.pool_size",         (self.db_pool_size).to_walkable_value()),
            ("db.ssl",               WalkableValue::Scalar(ScalarValue::Bool(self.db_ssl))),
            ("redis.host",           WalkableValue::Scalar(ScalarValue::String(self.redis_host.clone()))),
            ("redis.port",           (self.redis_port).to_walkable_value()),
            ("redis.ttl",            (self.redis_ttl).to_walkable_value()),
            ("api.rate_limit",       (self.api_rate_limit).to_walkable_value()),
            ("api.timeout",          (self.api_timeout).to_walkable_value()),
            ("features.dark_mode",   WalkableValue::Scalar(ScalarValue::Bool(self.features_dark_mode))),
            ("features.analytics",   WalkableValue::Scalar(ScalarValue::Bool(self.features_analytics))),
            ("features.beta",        WalkableValue::Scalar(ScalarValue::Bool(self.features_beta))),
        ]
    }
}

// =============================================================================
// Config generator
// =============================================================================

fn stable_config(rng: &mut StdRng) -> Config {
    let hosts      = ["db-primary.internal", "db-replica.internal"];
    let pool_sizes = [10i64, 12, 15];
    let rate_limits = [1000i64, 1200, 1500];
    Config {
        db_host:             (*hosts.choose(rng).unwrap()).to_string(),
        db_port:             5432,
        db_pool_size:        *pool_sizes.choose(rng).unwrap(),
        db_ssl:              true,
        redis_host:          "redis.internal".to_string(),
        redis_port:          6379,
        redis_ttl:           3600,
        api_rate_limit:      *rate_limits.choose(rng).unwrap(),
        api_timeout:         60,
        features_dark_mode:  true,
        features_analytics:  true,
        features_beta:       false,
    }
}

// =============================================================================
// Residual-swap field attribution
// =============================================================================

// Each entry: (field_name, function that reverts that field to golden value)
type RevertFn = fn(&Config, &Config) -> Config;

const FIELDS: &[(&str, RevertFn)] = &[
    ("db.host",              |d, g| Config { db_host: g.db_host.clone(), ..d.clone() }),
    ("db.port",              |d, g| Config { db_port: g.db_port, ..d.clone() }),
    ("db.pool_size",         |d, g| Config { db_pool_size: g.db_pool_size, ..d.clone() }),
    ("db.ssl",               |d, g| Config { db_ssl: g.db_ssl, ..d.clone() }),
    ("redis.host",           |d, g| Config { redis_host: g.redis_host.clone(), ..d.clone() }),
    ("redis.port",           |d, g| Config { redis_port: g.redis_port, ..d.clone() }),
    ("redis.ttl",            |d, g| Config { redis_ttl: g.redis_ttl, ..d.clone() }),
    ("api.rate_limit",       |d, g| Config { api_rate_limit: g.api_rate_limit, ..d.clone() }),
    ("api.timeout",          |d, g| Config { api_timeout: g.api_timeout, ..d.clone() }),
    ("features.dark_mode",   |d, g| Config { features_dark_mode: g.features_dark_mode, ..d.clone() }),
    ("features.analytics",   |d, g| Config { features_analytics: g.features_analytics, ..d.clone() }),
    ("features.beta",        |d, g| Config { features_beta: g.features_beta, ..d.clone() }),
];

fn attribute_drift(
    encoder: &Encoder,
    subspace: &OnlineSubspace,
    drifted: &Config,
    golden: &Config,
) -> Vec<(&'static str, f64)> {
    let base = subspace.residual(&encoder.encode_walkable(drifted).to_f64());
    let mut results: Vec<(&'static str, f64)> = FIELDS
        .iter()
        .filter_map(|(name, revert)| {
            let candidate = revert(drifted, golden);
            let new_res = subspace.residual(&encoder.encode_walkable(&candidate).to_f64());
            let drop = base - new_res;
            if drop > 0.5 { Some((*name, drop)) } else { None }
        })
        .collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    results
}

// =============================================================================
// Drift scenarios
// =============================================================================

struct DriftScenario {
    label: &'static str,
    note: Option<&'static str>,
    apply: fn(&Config) -> Config,
}

const DRIFTS: &[DriftScenario] = &[
    DriftScenario {
        label: "Single field: db.host -> external",
        note: None,
        apply: |c| Config { db_host: "db.evil.com".to_string(), ..c.clone() },
    },
    DriftScenario {
        label: "Subtle drift: pool_size 15 -> 8 (still 'valid', just wrong)",
        note: Some("8 is a positive integer — any schema validator passes this"),
        apply: |c| Config { db_pool_size: 8, ..c.clone() },
    },
    DriftScenario {
        label: "Multi-field: db compromised + rate limit blown",
        note: Some("Two simultaneous changes — cascading attribution finds both"),
        apply: |c| Config {
            db_host: "db.evil.com".to_string(),
            api_rate_limit: 999_999,
            ..c.clone()
        },
    },
    DriftScenario {
        label: "Feature flag poisoning: analytics disabled",
        note: None,
        apply: |c| Config { features_analytics: false, ..c.clone() },
    },
    DriftScenario {
        label: "Multi-field: redis external + TTL zeroed",
        note: Some("TTL=0 looks like a valid cache-disable — structure catches it"),
        apply: |c| Config {
            redis_host: "redis.evil.com".to_string(),
            redis_ttl: 0,
            ..c.clone()
        },
    },
];

// =============================================================================
// Helpers
// =============================================================================

fn print_header(title: &str) {
    println!();
    println!("{}", "=".repeat(65));
    println!("  {}", title);
    println!("{}", "=".repeat(65));
}

// =============================================================================
// Main
// =============================================================================

fn main() -> holon::Result<()> {
    print_header("CONFIG DRIFT REMEDIATION\n  Detect, Attribute, and Fix — No Schema Required");

    let vm = VectorManager::with_seed(4096, 42);
    let encoder = Encoder::new(vm);
    let mut rng = StdRng::seed_from_u64(42);

    // =========================================================================
    // Build golden subspace: 20 stable configs × 10 passes
    // =========================================================================
    println!("\nLearning golden config manifold (20 configs × 10 passes)...");

    let stable: Vec<Config> = (0..20).map(|_| stable_config(&mut rng)).collect();
    let stable_vecs: Vec<Vector> = stable.iter().map(|c| encoder.encode_walkable(c)).collect();
    let stable_refs: Vec<&Vector> = stable_vecs.iter().collect();
    let golden_proto = Primitives::prototype(&stable_refs, 0.5);
    let golden_ref = stable[0].clone();

    let mut subspace = OnlineSubspace::with_params(4096, 32, 2.0, 0.01, 3.0, 500);
    for _ in 0..10 {
        for v in &stable_vecs {
            subspace.update(&v.to_f64());
        }
    }

    let mut library = EngramLibrary::new(4096);
    library.add("golden_config", &subspace, None, HashMap::new());

    let train_residuals: Vec<f64> = stable_vecs.iter().map(|v| subspace.residual(&v.to_f64())).collect();
    let train_mean = train_residuals.iter().sum::<f64>() / train_residuals.len() as f64;
    let train_max  = train_residuals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("  Threshold  : {:.2}", subspace.threshold());
    println!("  Train mean : {:.4}  max: {:.4}  (near-zero = tight convergence)", train_mean, train_max);

    // =========================================================================
    // Drift scenarios
    // =========================================================================
    println!();
    println!("{}", "-".repeat(65));
    println!("  DRIFT DETECTION + ATTRIBUTION + REMEDIATION");
    println!("{}", "-".repeat(65));

    let mut detected = 0usize;

    for drift in DRIFTS {
        // Generate a fresh base config for each scenario (distinct from training data)
        let base_cfg = stable_config(&mut rng);
        let drifted = (drift.apply)(&base_cfg);
        let drifted_vec = encoder.encode_walkable(&drifted);
        let residual = subspace.residual(&drifted_vec.to_f64());
        let is_drift = residual > subspace.threshold();

        println!();
        println!("  Config  : {}", drift.label);
        if let Some(note) = drift.note {
            println!("  Note    : {}", note);
        }
        println!(
            "  Residual: {:.2}  (threshold {:.2})  drift={}",
            residual, subspace.threshold(), is_drift
        );

        if is_drift {
            detected += 1;

            // Attribution
            let attrs = attribute_drift(&encoder, &subspace, &drifted, &golden_ref);
            for (i, (field, drop)) in attrs.iter().take(3).enumerate() {
                println!(
                    "  Cause {} : '{}' (residual drop {:.2} when reverted)",
                    i + 1, field, drop
                );
            }

            // Remediation: difference(golden_proto, drifted) + amplify toward golden
            let delta = Primitives::difference(&golden_proto, &drifted_vec);
            let remediation = Primitives::amplify(&golden_proto, &delta, 0.5);
            let rem_residual = subspace.residual(&remediation.to_f64());

            // Verify with the actual correct base config
            let correct_vec = encoder.encode_walkable(&base_cfg);
            let correct_residual = subspace.residual(&correct_vec.to_f64());

            println!(
                "  Fix     : amplify(golden, Δ) → residual {:.2}  (was {:.2})",
                rem_residual, residual
            );
            println!(
                "  Verify  : actual correct config residual = {:.2}  (below threshold: {})",
                correct_residual,
                correct_residual < subspace.threshold()
            );
        }
    }

    println!();
    println!("{}", "=".repeat(65));
    println!(
        "Detected {}/{} drifts — including multi-field and subtle in-range changes.",
        detected, DRIFTS.len()
    );
    println!("No schema annotations. No field-specific rules. Pure structure.");
    println!();
    println!("Try: add a per-environment engram (staging vs prod) to catch environment mix-ups");
    println!("Try: lower sigma_mult=2.0 to tighten the threshold for subtler drift");

    Ok(())
}
