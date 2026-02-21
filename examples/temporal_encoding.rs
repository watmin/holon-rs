//! Temporal Encoding: TimeFloat for time-aware vector representations.
//!
//! Demonstrates ScalarValue::TimeFloat and TimeResolution for encoding Unix
//! timestamps into vectors that preserve temporal structure:
//!
//! - Circular similarity: same hour of day across different weeks → high similarity
//! - Positional discrimination: nearby timestamps → similar vectors
//! - Resolution control: Second vs Hour vs Day changes granularity
//! - Walkable integration: compose TimeFloat with other struct fields
//! - Anomaly detection: off-hours log entries flagged by OnlineSubspace
//!
//! Key insight: A 3am Saturday log entry that looks structurally identical to
//! a 2pm Monday entry can still be flagged as anomalous purely from its
//! temporal signature — no hard-coded business-hours logic needed.
//!
//! Run: cargo run --example temporal_encoding --release

use holon::memory::OnlineSubspace;
use holon::highlevel::Holon;
use holon::kernel::{
    ScalarValue, Similarity, TimeResolution, WalkType, Walkable, WalkableRef, WalkableValue,
};
use rand::prelude::*;

// =============================================================================
// LogEntry struct with TimeFloat in Walkable
// =============================================================================

struct LogEntry {
    service: String,
    level: String,
    timestamp: f64, // Unix timestamp (seconds)
}

impl Walkable for LogEntry {
    fn walk_type(&self) -> WalkType {
        WalkType::Map
    }

    fn walk_map_items(&self) -> Vec<(&str, WalkableValue)> {
        vec![
            (
                "service",
                WalkableValue::Scalar(ScalarValue::String(self.service.clone())),
            ),
            (
                "level",
                WalkableValue::Scalar(ScalarValue::String(self.level.clone())),
            ),
            (
                "timestamp",
                WalkableValue::Scalar(ScalarValue::time(self.timestamp)),
            ),
        ]
    }

    fn has_fast_visitor(&self) -> bool {
        true
    }

    fn walk_map_visitor(&self, visitor: &mut dyn FnMut(&str, WalkableRef<'_>)) {
        visitor("service", WalkableRef::string(&self.service));
        visitor("level", WalkableRef::string(&self.level));
        visitor("timestamp", WalkableRef::time(self.timestamp));
    }
}

// =============================================================================
// Helpers
// =============================================================================

fn print_header(title: &str) {
    println!();
    println!("{}", "=".repeat(70));
    println!("  {}", title);
    println!("{}", "=".repeat(70));
}

fn print_similarity_matrix(labels: &[&str], vecs: &[holon::Vector], _holon: &Holon) {
    // Header
    print!("{:>18}", "");
    for label in labels {
        print!("{:>12}", label);
    }
    println!();
    // Rows
    for (i, row_label) in labels.iter().enumerate() {
        print!("{:>18}", row_label);
        for j in 0..vecs.len() {
            let sim = Similarity::cosine(&vecs[i], &vecs[j]);
            print!("{:>12.3}", sim);
        }
        println!();
    }
}

/// Format a Unix timestamp as a human-readable label.
fn ts_label(ts: f64) -> String {
    let secs = ts as i64;
    // Day of week: Jan 1 1970 = Thursday (4), so (days + 4) % 7 gives 0=Mon offset
    let days = if secs >= 0 { secs / 86400 } else { (secs - 86399) / 86400 };
    let sec_of_day = ((secs % 86400) + 86400) % 86400;
    let hour = sec_of_day / 3600;
    let dow = ((days % 7 + 3) % 7 + 7) % 7; // 0=Mon .. 6=Sun (Thu Jan 1 1970 + 3 = Mon=0)
    let dow_name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][dow as usize];
    format!("{} {:02}:00", dow_name, hour)
}

fn main() {
    let holon = Holon::with_seed(4096, 42);

    // Reference epoch: 2024-01-01 00:00:00 UTC (Mon midnight)
    // 2024-01-01 = day 19723 from epoch
    let epoch_2024: f64 = 19723.0 * 86400.0;

    // =========================================================================
    // SECTION 1 — BASIC TIME ENCODING
    // =========================================================================
    print_header("SECTION 1: Basic Time Encoding");

    println!("  Encoding five timestamps spaced 6 hours apart.");
    println!("  Closer timestamps should have higher cosine similarity.\n");

    let base = epoch_2024 + 9.0 * 3600.0; // Mon 09:00
    let timestamps: Vec<f64> = (0..5).map(|i| base + i as f64 * 6.0 * 3600.0).collect();
    let labels: Vec<String> = timestamps.iter().map(|&ts| ts_label(ts)).collect();
    let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

    let vecs: Vec<holon::Vector> = timestamps
        .iter()
        .map(|&ts| {
            holon.encode_walkable_value(&WalkableValue::Scalar(ScalarValue::time(ts)))
        })
        .collect();

    print_similarity_matrix(&label_refs, &vecs, &holon);

    println!();
    // Adjacent (6h) vs near-same-time-next-day (30h=24+6h apart)
    // The circular hour component dominates: 9am and 15am have different hour encoding,
    // but 9am and 9am-next-day share hour-of-day → the matrix shows both effects.
    let sim_adjacent = Similarity::cosine(&vecs[0], &vecs[1]);
    let sim_far = Similarity::cosine(&vecs[0], &vecs[4]);
    println!(
        "  {} ↔ {} (6h apart):   {:.3}",
        label_refs[0], label_refs[1], sim_adjacent
    );
    println!(
        "  {} ↔ {} (24h apart):  {:.3}",
        label_refs[0], label_refs[4], sim_far
    );
    println!("  Note: same-hour-of-day similarity can exceed adjacent-hour similarity");
    println!("  because the circular hour component wraps — 9am clusters across days.");

    // =========================================================================
    // SECTION 2 — CIRCULAR PERIODICITY
    // =========================================================================
    print_header("SECTION 2: Circular Periodicity");

    println!("  TimeFloat encodes hour-of-day, day-of-week, and month circularly.");
    println!("  Same time of day across different weeks → high similarity.\n");

    // Mon 09:00 in weeks 1, 2, and 3; plus Mon 03:00 (same dow, diff hour)
    let mon_9am_w1 = epoch_2024 + 9.0 * 3600.0;
    let mon_9am_w2 = mon_9am_w1 + 7.0 * 86400.0;
    let mon_9am_w3 = mon_9am_w1 + 14.0 * 86400.0;
    let tue_9am_w1 = mon_9am_w1 + 86400.0;
    let mon_3am_w1 = epoch_2024 + 3.0 * 3600.0;

    let circ_ts = [mon_9am_w1, mon_9am_w2, mon_9am_w3, tue_9am_w1, mon_3am_w1];
    let circ_labels = ["Mon 09:00 W1", "Mon 09:00 W2", "Mon 09:00 W3", "Tue 09:00 W1", "Mon 03:00 W1"];

    let circ_vecs: Vec<holon::Vector> = circ_ts
        .iter()
        .map(|&ts| {
            holon.encode_walkable_value(&WalkableValue::Scalar(ScalarValue::time(ts)))
        })
        .collect();

    print_similarity_matrix(&circ_labels, &circ_vecs, &holon);

    println!();
    let sim_same_time_diff_week = Similarity::cosine(&circ_vecs[0], &circ_vecs[1]);
    let sim_same_time_next_day = Similarity::cosine(&circ_vecs[0], &circ_vecs[3]);
    let sim_same_day_diff_hour = Similarity::cosine(&circ_vecs[0], &circ_vecs[4]);

    println!("  Mon 09:00 W1 ↔ Mon 09:00 W2 (same time, +1 week): {:.3}", sim_same_time_diff_week);
    println!("  Mon 09:00 W1 ↔ Tue 09:00 W1 (same time, +1 day):  {:.3}", sim_same_time_next_day);
    println!("  Mon 09:00 W1 ↔ Mon 03:00 W1 (same day, -6 hours):  {:.3}", sim_same_day_diff_hour);
    println!();
    println!("  Same-time-of-day across weeks should cluster (high similarity).");
    println!("  Different hours should be more distant.");

    // =========================================================================
    // SECTION 3 — RESOLUTION COMPARISON
    // =========================================================================
    print_header("SECTION 3: Resolution Comparison");

    println!("  TimeResolution controls the positional component granularity.");
    println!("  Finer resolution → more discrimination for small time differences.\n");

    let t0 = epoch_2024 + 12.0 * 3600.0; // Mon 12:00
    let resolutions = [
        ("Second", TimeResolution::Second),
        ("Minute", TimeResolution::Minute),
        ("Hour", TimeResolution::Hour),
        ("Day", TimeResolution::Day),
    ];

    // Compare t0 vs t0 + 1 hour, t0 + 6 hours, t0 + 24 hours
    let offsets = [3600.0f64, 6.0 * 3600.0, 86400.0];
    let offset_labels = ["  +1 hour", " +6 hours", "+24 hours"];

    // Header
    print!("{:>10}", "Resolution");
    for lbl in &offset_labels {
        print!("{:>12}", lbl);
    }
    println!();
    println!("{}", "-".repeat(46));

    for (res_name, res) in &resolutions {
        let v0 = holon.encode_walkable_value(&WalkableValue::Scalar(ScalarValue::TimeFloat {
            value: t0,
            resolution: *res,
        }));
        print!("{:>10}", res_name);
        for offset in &offsets {
            let v1 = holon.encode_walkable_value(&WalkableValue::Scalar(ScalarValue::TimeFloat {
                value: t0 + offset,
                resolution: *res,
            }));
            let sim = Similarity::cosine(&v0, &v1);
            print!("{:>12.3}", sim);
        }
        println!();
    }

    println!();
    println!("  Finer resolutions (Second) discriminate small offsets more strongly.");
    println!("  Coarser resolutions (Day) cluster within-day events together.");

    // =========================================================================
    // SECTION 4 — WALKABLE INTEGRATION
    // =========================================================================
    print_header("SECTION 4: Walkable Integration");

    println!("  TimeFloat composes naturally with other fields in a Walkable struct.");
    println!("  LogEntry {{ service, level, timestamp }} encodes all fields together.\n");

    let entries = vec![
        LogEntry { service: "auth".into(), level: "INFO".into(),  timestamp: t0 },
        LogEntry { service: "auth".into(), level: "INFO".into(),  timestamp: t0 + 300.0 }, // 5 min later
        LogEntry { service: "auth".into(), level: "ERROR".into(), timestamp: t0 },
        LogEntry { service: "db".into(),   level: "INFO".into(),  timestamp: t0 },
        LogEntry { service: "auth".into(), level: "INFO".into(),  timestamp: t0 + 86400.0 }, // 1 day later
    ];

    let entry_labels = [
        "auth/INFO  t+0",
        "auth/INFO  t+5m",
        "auth/ERROR t+0",
        "db/INFO    t+0",
        "auth/INFO  t+1d",
    ];

    let entry_vecs: Vec<holon::Vector> = entries
        .iter()
        .map(|e| holon.encode_walkable(e))
        .collect();

    print_similarity_matrix(&entry_labels, &entry_vecs, &holon);

    println!();
    println!("  auth/INFO t+0 ↔ auth/INFO t+5m (same fields, nearby time):");
    println!("    sim = {:.3}  (high — almost identical)", Similarity::cosine(&entry_vecs[0], &entry_vecs[1]));
    println!("  auth/INFO t+0 ↔ auth/ERROR t+0 (diff level, same time):");
    println!("    sim = {:.3}  (lower — level field differs)", Similarity::cosine(&entry_vecs[0], &entry_vecs[2]));
    println!("  auth/INFO t+0 ↔ db/INFO t+0 (diff service, same time):");
    println!("    sim = {:.3}  (lower — service field differs)", Similarity::cosine(&entry_vecs[0], &entry_vecs[3]));
    println!("  auth/INFO t+0 ↔ auth/INFO t+1d (same fields, 1 day later):");
    println!("    sim = {:.3}  (moderate — temporal distance)", Similarity::cosine(&entry_vecs[0], &entry_vecs[4]));

    // =========================================================================
    // SECTION 5 — TIME-SERIES ANOMALY DETECTION
    // =========================================================================
    print_header("SECTION 5: Off-Hours Anomaly Detection");

    println!("  Train OnlineSubspace on 100 business-hours entries (Mon-Fri 9am-5pm).");
    println!("  Score off-hours entries — residual should be higher.\n");

    let mut rng = StdRng::seed_from_u64(42);
    // sigma_mult=2.0 gives a tighter threshold (2σ above EMA vs default 3.5σ)
    let mut time_subspace = OnlineSubspace::with_params(4096, 16, 2.0, 0.02, 2.0, 500);

    // Business hours: Mon-Fri (days 0-4 of week), 9am-5pm
    // epoch_2024 = Monday, so week offsets 0-4 = Mon-Fri.
    // Train on 300 samples across multiple weeks for a well-calibrated threshold.
    let n_business = 300;
    for i in 0..n_business {
        let week = (i / 40) as u64; // spread across ~7 weeks
        let weekday = rng.gen_range(0u64..5); // 0=Mon..4=Fri
        let hour: f64 = rng.gen_range(9.0..17.0);
        let ts = epoch_2024 + week as f64 * 7.0 * 86400.0
            + weekday as f64 * 86400.0
            + hour * 3600.0;
        let entry = LogEntry {
            service: "api".into(),
            level: "INFO".into(),
            timestamp: ts,
        };
        let vec = holon.encode_walkable(&entry);
        time_subspace.update(&vec.to_f64());
    }

    let threshold = time_subspace.threshold();
    println!("  Trained on {} business-hours entries.", n_business);
    println!("  Adaptive threshold (2σ): {:.3}\n", threshold);

    // Score business-hours probes (use week 4 to stay in training window)
    println!("  Business-hours probes (should be BELOW threshold):");
    let fixed_weekdays = [0u64, 1, 2, 3, 4]; // Mon..Fri
    for &weekday in &fixed_weekdays {
        let hour: f64 = rng.gen_range(10.0..16.0);
        let ts = epoch_2024 + 4.0 * 7.0 * 86400.0 + weekday as f64 * 86400.0 + hour * 3600.0;
        let entry = LogEntry {
            service: "api".into(),
            level: "INFO".into(),
            timestamp: ts,
        };
        let vec = holon.encode_walkable(&entry);
        let residual = time_subspace.residual(&vec.to_f64());
        let flag = if residual > threshold { "⚠ ANOMALY" } else { "✓ normal" };
        println!("    {}  residual={:.3}  {}", ts_label(ts), residual, flag);
    }

    println!();
    println!("  Off-hours probes (should be ABOVE threshold):");
    let off_hours = [
        // 3am Saturday
        (epoch_2024 + 5.0 * 86400.0 + 3.0 * 3600.0, "Sat 03:00"),
        // Midnight Sunday
        (epoch_2024 + 6.0 * 86400.0, "Sun 00:00"),
        // 2am Monday (before business hours)
        (epoch_2024 + 2.0 * 3600.0, "Mon 02:00"),
        // 11pm Friday
        (epoch_2024 + 4.0 * 86400.0 + 23.0 * 3600.0, "Fri 23:00"),
        // 6am Wednesday (before open)
        (epoch_2024 + 2.0 * 86400.0 + 6.0 * 3600.0, "Wed 06:00"),
    ];

    let mut off_hours_detected = 0;
    for (ts, label) in &off_hours {
        let entry = LogEntry {
            service: "api".into(),
            level: "INFO".into(),
            timestamp: *ts,
        };
        let vec = holon.encode_walkable(&entry);
        let residual = time_subspace.residual(&vec.to_f64());
        let is_anomaly = residual > threshold;
        if is_anomaly {
            off_hours_detected += 1;
        }
        let flag = if is_anomaly { "⚠ ANOMALY" } else { "✓ normal" };
        println!("    {}  residual={:.3}  {}", label, residual, flag);
    }

    println!();
    println!("  Off-hours detection: {}/{} flagged", off_hours_detected, off_hours.len());
    println!();
    println!("  Note: The subspace learns the temporal distribution of normal activity.");
    println!("  It flags off-hours entries based on their timestamp encoding alone —");
    println!("  no if/else business-hours logic, no hard-coded time ranges.");

    // =========================================================================
    // SUMMARY
    // =========================================================================
    print_header("Summary");
    println!("  TimeFloat encodes Unix timestamps into four composable components:");
    println!("    • Hour-of-day  (circular, period=24)");
    println!("    • Day-of-week  (circular, period=7)");
    println!("    • Month        (circular, period=12)");
    println!("    • Position     (transformer sin/cos, resolution-dependent)");
    println!();
    println!("  Combined via role-filler binding, the result captures both");
    println!("  periodic structure (same time next week ≈ same vector) and");
    println!("  absolute position (distinguishes 9am from 3pm).");
    println!();
    println!("  Composes naturally with other Walkable fields — service, level,");
    println!("  payload size — so temporal anomalies can be detected alongside");
    println!("  structural ones in a single unified representation.");
    println!();
}
