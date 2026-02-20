//! Streaming Changepoint Detection — System Health Without Labels
//!
//! 125 structured metric observations across 4 unlabeled phases: healthy →
//! degraded → incident → recovery. OnlineSubspace learns "healthy" from the
//! first 50. segment() finds transitions, difference() fingerprints the outage,
//! invert() decomposes it against phase prototypes.
//!
//! Run: cargo run --example streaming_changepoint --release

use holon::memory::OnlineSubspace;
use holon::primitives::{Primitives, SegmentMethod};
use holon::{Holon, ScalarValue, WalkType, Walkable, WalkableValue};
use rand::prelude::*;
use std::collections::HashMap;

// =============================================================================
// Metrics struct + Walkable
// =============================================================================

struct Metrics {
    latency_ms: i64,
    error_rate: i64,
    cpu_pct: i64,
    mem_pct: i64,
    req_per_sec: i64,
    status: &'static str,
    db_pool: i64,
}

impl Walkable for Metrics {
    fn walk_type(&self) -> WalkType {
        WalkType::Map
    }

    fn walk_map_items(&self) -> Vec<(&str, WalkableValue)> {
        vec![
            ("latency_ms",  (self.latency_ms).to_walkable_value()),
            ("error_rate",  (self.error_rate).to_walkable_value()),
            ("cpu_pct",     (self.cpu_pct).to_walkable_value()),
            ("mem_pct",     (self.mem_pct).to_walkable_value()),
            ("req_per_sec", (self.req_per_sec).to_walkable_value()),
            ("status",      WalkableValue::Scalar(ScalarValue::String(self.status.to_string()))),
            ("db_pool",     (self.db_pool).to_walkable_value()),
        ]
    }
}

// =============================================================================
// Phase generators (ported from Python, seed-for-seed compatible)
// =============================================================================

fn healthy_metrics(rng: &mut StdRng) -> Metrics {
    let lat   = *[22i64, 23, 24, 25, 26].choose(rng).unwrap();
    let err   = *[0i64, 0, 0, 1].choose(rng).unwrap();
    let cpu   = *[20i64, 21, 22, 23, 24].choose(rng).unwrap();
    let mem   = *[44i64, 45, 46, 47, 48].choose(rng).unwrap();
    let req   = *[990i64, 995, 1000, 1005, 1010].choose(rng).unwrap();
    let pool  = *[20i64, 21, 22, 23].choose(rng).unwrap();
    Metrics { latency_ms: lat, error_rate: err, cpu_pct: cpu, mem_pct: mem,
              req_per_sec: req, status: "ok", db_pool: pool }
}

fn degraded_metrics(rng: &mut StdRng, step: usize) -> Metrics {
    let f = (step as f64 / 15.0).min(1.0);
    let lat_lo = (80.0 + f * 200.0) as i64;
    let lat_hi = (150.0 + f * 250.0) as i64;
    let err_lo = (8.0 + f * 20.0) as i64;
    let err_hi = (15.0 + f * 30.0) as i64;
    let cpu_lo = (52.0 + f * 22.0) as i64;
    let cpu_hi = (65.0 + f * 18.0) as i64;
    let mem_lo = (62.0 + f * 16.0) as i64;
    let mem_hi = (74.0 + f * 13.0) as i64;
    let req_lo = (1300.0 + f * 700.0) as i64;
    let req_hi = (1700.0 + f * 900.0) as i64;
    let pool_lo = ((16.0 - f * 13.0).max(3.0)) as i64;
    let pool_hi = ((19.0 - f * 11.0).max(6.0)) as i64;
    Metrics {
        latency_ms:  rng.gen_range(lat_lo..=lat_hi),
        error_rate:  rng.gen_range(err_lo..=err_hi),
        cpu_pct:     rng.gen_range(cpu_lo..=cpu_hi),
        mem_pct:     rng.gen_range(mem_lo..=mem_hi),
        req_per_sec: rng.gen_range(req_lo..=req_hi),
        status:      "ok",
        db_pool:     rng.gen_range(pool_lo..=pool_hi),
    }
}

fn incident_metrics(rng: &mut StdRng) -> Metrics {
    Metrics {
        latency_ms:  rng.gen_range(2000..=5000),
        error_rate:  rng.gen_range(150..=350),
        cpu_pct:     rng.gen_range(92..=99),
        mem_pct:     rng.gen_range(92..=99),
        req_per_sec: rng.gen_range(6000..=10000),
        status:      "degraded",
        db_pool:     rng.gen_range(0..=1),
    }
}

fn recovery_metrics(rng: &mut StdRng, step: usize) -> Metrics {
    let f = (1.0_f64 - step as f64 / 18.0).max(0.0);
    let lat_lo = (28.0 + f * 120.0) as i64;
    let lat_hi = (50.0 + f * 100.0) as i64;
    let err_lo = (1.0 + f * 18.0) as i64;
    let err_hi = (4.0 + f * 18.0) as i64;
    let cpu_lo = (22.0 + f * 28.0) as i64;
    let cpu_hi = (35.0 + f * 28.0) as i64;
    let mem_lo = (46.0 + f * 22.0) as i64;
    let mem_hi = (56.0 + f * 22.0) as i64;
    Metrics {
        latency_ms:  rng.gen_range(lat_lo..=lat_hi),
        error_rate:  rng.gen_range(err_lo..=err_hi),
        cpu_pct:     rng.gen_range(cpu_lo..=cpu_hi),
        mem_pct:     rng.gen_range(mem_lo..=mem_hi),
        req_per_sec: rng.gen_range(950..=1050),
        status:      "ok",
        db_pool:     rng.gen_range(12..=22),
    }
}

// =============================================================================
// Helpers
// =============================================================================

fn print_header(title: &str) {
    println!();
    println!("{}", "=".repeat(65));
    println!("  {}", title);
    println!("{}", "=".repeat(65));
}

fn print_subheader(title: &str) {
    println!();
    println!("{}", "-".repeat(65));
    println!("  {}", title);
    println!("{}", "-".repeat(65));
}

// =============================================================================
// Main
// =============================================================================

fn main() -> holon::Result<()> {
    print_header("STREAMING CHANGEPOINT DETECTION\n  System Health Without Labels");

    let holon = Holon::with_seed(4096, 42);
    let mut rng = StdRng::seed_from_u64(42);

    let n_healthy  = 50usize;
    let n_degraded = 25usize;
    let n_incident = 20usize;
    let n_recovery = 30usize;
    let b1 = n_healthy;
    let b2 = b1 + n_degraded;
    let b3 = b2 + n_incident;
    let total = b3 + n_recovery;

    // Build the stream
    let mut stream_vecs: Vec<holon::Vector> = Vec::with_capacity(total);
    let mut phase_labels: Vec<&'static str> = Vec::with_capacity(total);

    for _ in 0..n_healthy {
        stream_vecs.push(holon.encode_walkable(&healthy_metrics(&mut rng)));
        phase_labels.push("healthy");
    }
    for i in 0..n_degraded {
        stream_vecs.push(holon.encode_walkable(&degraded_metrics(&mut rng, i)));
        phase_labels.push("degraded");
    }
    for _ in 0..n_incident {
        stream_vecs.push(holon.encode_walkable(&incident_metrics(&mut rng)));
        phase_labels.push("incident");
    }
    for i in 0..n_recovery {
        stream_vecs.push(holon.encode_walkable(&recovery_metrics(&mut rng, i)));
        phase_labels.push("recovery");
    }

    println!();
    println!(
        "Stream: {} healthy → {} degraded → {} incident → {} recovery",
        n_healthy, n_degraded, n_incident, n_recovery
    );
    println!(
        "Total : {} observations  |  True boundaries: [{}, {}, {}]",
        total, b1, b2, b3
    );
    println!("Holon sees an unlabeled stream — no phase column, no field metadata");

    // =========================================================================
    // Learn healthy baseline from first 50 observations
    // =========================================================================
    println!("\nLearning healthy baseline from first {} observations...", n_healthy);
    // sigma_mult=1.5 — tighter threshold so degraded/incident are clearly flagged
    let mut subspace = OnlineSubspace::with_params(4096, 32, 2.0, 0.01, 1.5, 500);
    for v in &stream_vecs[..n_healthy] {
        subspace.update(&v.to_f64());
    }

    let mut library = holon.create_engram_library();
    library.add("healthy", &subspace, None, HashMap::new());

    let train_residuals: Vec<f64> = stream_vecs[..n_healthy]
        .iter()
        .map(|v| subspace.residual(&v.to_f64()))
        .collect();
    let train_max = train_residuals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("  Threshold : {:.2}  (1.5σ above healthy EMA)", subspace.threshold());
    println!("  Train max : {:.2}  (healthy data is highly consistent)", train_max);

    // =========================================================================
    // Residual timeline
    // =========================================================================
    let residuals: Vec<f64> = stream_vecs
        .iter()
        .map(|v| subspace.residual(&v.to_f64()))
        .collect();

    print_subheader("RESIDUAL TIMELINE  (structural anomaly score vs healthy baseline)");
    println!(
        "  threshold={:.1}  |  bar scale: each █ ≈ 3 residual units",
        subspace.threshold()
    );
    println!();

    let blocks: &[(&str, usize, usize)] = &[
        ("Healthy",  0,  b1),
        ("Degraded", b1, b2),
        ("Incident", b2, b3),
        ("Recovery", b3, total),
    ];

    let threshold = subspace.threshold();
    for (name, start, end) in blocks {
        let block = &residuals[*start..*end];
        let mean_r = block.iter().sum::<f64>() / block.len() as f64;
        let flagged = block.iter().filter(|&&r| r > threshold).count();
        let bar_len = (mean_r / 3.0).min(42.0) as usize;
        let bar = "█".repeat(bar_len);
        println!(
            "  {:<9}  [{}-{}]{}  mean={:5.1}  flagged={:2}/{:2}  {}",
            name,
            start,
            end - 1,
            if end - 1 < 100 { "   " } else { "  " },
            mean_r,
            flagged,
            end - start,
            bar
        );
    }

    println!();
    println!(
        "  Threshold line at ~{:.1} separates healthy from anomalous.",
        threshold
    );
    println!("  No per-metric rules wrote this — the subspace inferred it from structure.");

    // =========================================================================
    // Changepoint detection
    // =========================================================================
    print_subheader("CHANGEPOINT DETECTION  (segment() — finds transitions in the stream)");

    let breakpoints = Primitives::segment(&stream_vecs, 20, 0.70, SegmentMethod::Prototype);

    // Consolidate: keep first in each cluster where gap < 12
    let mut consolidated: Vec<usize> = Vec::new();
    for bp in &breakpoints {
        if *bp == 0 { continue; }
        if consolidated.is_empty() || bp - consolidated.last().unwrap() > 12 {
            consolidated.push(*bp);
        }
    }

    let true_boundaries: &[(usize, &str)] = &[
        (b1, "degraded"),
        (b2, "incident"),
        (b3, "recovery"),
    ];

    // Match consolidated breakpoints to true boundaries within ±12 steps
    let mut matched: Vec<(usize, usize, &str)> = Vec::new(); // (bp, nearest_true, phase_name)
    for &bp in &consolidated {
        for &(true_bp, phase) in true_boundaries {
            if (bp as isize - true_bp as isize).unsigned_abs() <= 12 {
                matched.push((bp, true_bp, phase));
                break;
            }
        }
    }

    println!();
    println!(
        "  All consolidated breakpoints: {:?}",
        consolidated
    );
    println!(
        "  True phase boundaries      : {:?}",
        true_boundaries.iter().map(|(b, _)| b).collect::<Vec<_>>()
    );
    println!(
        "  Matched (within 12 steps)  : {:?}",
        matched.iter().map(|(bp, _, _)| bp).collect::<Vec<_>>()
    );
    println!();

    for (bp, true_bp, expected_phase) in &matched {
        let actual_phase = phase_labels[*bp];
        let drift = (*bp as isize - *true_bp as isize).unsigned_abs();
        println!(
            "  → index {:3}: entering '{}' territory  (expected '{}' at {}, drift={} steps)",
            bp, actual_phase, expected_phase, true_bp, drift
        );
    }

    // =========================================================================
    // Change analysis — difference + invert
    // =========================================================================
    print_subheader("CHANGE ANALYSIS  (difference + invert)");

    let healthy_refs: Vec<&holon::Vector>  = stream_vecs[..b1].iter().collect();
    let incident_refs: Vec<&holon::Vector> = stream_vecs[b2..b3].iter().collect();
    let recovery_refs: Vec<&holon::Vector> = stream_vecs[b3..].iter().collect();

    let healthy_proto  = Primitives::prototype(&healthy_refs,  0.5);
    let incident_proto = Primitives::prototype(&incident_refs, 0.5);
    let recovery_proto = Primitives::prototype(&recovery_refs, 0.5);

    let delta = Primitives::difference(&healthy_proto, &incident_proto);
    let delta_density = delta.data().iter().filter(|&&x| x != 0).count() as f64
        / delta.dimensions() as f64;

    println!();
    println!("  difference(healthy, incident):");
    println!("    {:.1}% of vector dimensions changed", delta_density * 100.0);
    println!("    This delta IS the outage fingerprint — storable and algebraically composable");

    // invert — structural overlap of incident_proto against phase codebook
    let codebook = vec![
        healthy_proto.clone(),
        incident_proto.clone(),
        recovery_proto.clone(),
    ];
    let codebook_names = ["healthy", "incident", "recovery"];
    let components = Primitives::invert(&incident_proto, &codebook, 3, 0.0);

    println!();
    println!("  invert(incident_proto, codebook) — structural overlap per phase:");
    for (idx, sim) in &components {
        let bar_len = (sim * 30.0) as usize;
        let bar = "█".repeat(bar_len);
        println!("    '{}':  {:.3}  {}", codebook_names[*idx], sim, bar);
    }

    // EngramLibrary: does recovery match back to healthy?
    let matches = library.match_vec(&recovery_proto.to_f64(), 1, 10);
    let (match_name, match_res) = &matches[0];
    let flagged = *match_res > threshold;
    println!();
    println!("  EngramLibrary.match_vec(recovery_proto) → '{}'", match_name);
    println!("    residual={:.2}  threshold={:.2}  anomalous={}", match_res, threshold, flagged);
    let msg = if !flagged { "structurally stabilised" } else { "still elevated" };
    println!("    Recovery {} relative to learned healthy baseline", msg);

    // =========================================================================
    // Summary
    // =========================================================================
    let healthy_mean  = residuals[..b1].iter().sum::<f64>() / b1 as f64;
    let incident_mean = residuals[b2..b3].iter().sum::<f64>() / n_incident as f64;
    println!();
    println!("{}", "=".repeat(65));
    println!(
        "Residual rises {:.0}x from healthy ({:.1}) to incident ({:.1}).",
        incident_mean / healthy_mean, healthy_mean, incident_mean
    );
    println!(
        "segment() identified {}/{} phase transitions within 12 steps.",
        // count unique true boundaries matched
        {
            let mut seen = std::collections::HashSet::new();
            matched.iter().filter(|(_, tb, _)| seen.insert(*tb)).count()
        },
        true_boundaries.len()
    );
    println!("Zero per-metric thresholds. Zero label columns.");
    println!();
    println!("Try: sigma_mult=1.0 for stricter threshold (flags more of recovery)");
    println!("Try: add a second engram for 'incident' and use match_vec to classify windows");

    Ok(())
}
