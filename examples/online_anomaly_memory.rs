//! Online Anomaly Memory: OnlineSubspace + EngramLibrary for traffic analysis.
//!
//! Demonstrates the memory layer for streaming anomaly detection:
//!
//! 1. Learn normal traffic manifold with OnlineSubspace (CCIPCA)
//! 2. Score new packets by residual distance — anomalies stand out
//! 3. Mint learned patterns as named Engrams in an EngramLibrary
//! 4. Recall the best-matching pattern for any probe vector
//! 5. Snapshot/restore the subspace and library — identical results
//!
//! Key insight: No hard-coded thresholds or signatures. The subspace learns
//! what "normal" looks like and flags anything that doesn't fit — including
//! attacks it has never seen before.
//!
//! Run: cargo run --example online_anomaly_memory --release

use holon::memory::{EngramLibrary, OnlineSubspace};
use holon::highlevel::Holon;
use holon::kernel::{ScalarRef, ScalarValue, WalkType, Walkable, WalkableRef, WalkableValue};
use rand::prelude::*;
use std::collections::HashMap;

// =============================================================================
// Packet struct with Walkable implementation
// =============================================================================

struct Packet {
    protocol: String,
    dst_port: u16,
    payload_size: u32,
    rate_pps: f64, // packets per second — log-encoded
}

impl Walkable for Packet {
    fn walk_type(&self) -> WalkType {
        WalkType::Map
    }

    fn walk_map_items(&self) -> Vec<(&str, WalkableValue)> {
        vec![
            (
                "protocol",
                WalkableValue::Scalar(ScalarValue::String(self.protocol.clone())),
            ),
            ("dst_port", (self.dst_port as i64).to_walkable_value()),
            ("payload_size", (self.payload_size as i64).to_walkable_value()),
            ("rate", WalkableValue::Scalar(ScalarValue::log(self.rate_pps))),
        ]
    }

    fn has_fast_visitor(&self) -> bool {
        true
    }

    fn walk_map_visitor(&self, visitor: &mut dyn FnMut(&str, WalkableRef<'_>)) {
        visitor("protocol", WalkableRef::string(&self.protocol));
        visitor("dst_port", WalkableRef::int(self.dst_port as i64));
        visitor("payload_size", WalkableRef::int(self.payload_size as i64));
        visitor("rate", WalkableRef::Scalar(ScalarRef::log(self.rate_pps)));
    }
}

// =============================================================================
// Traffic generators
// =============================================================================

/// Normal web traffic: HTTPS (443), HTTP (80), DNS (53)
fn normal_packet(rng: &mut StdRng) -> Packet {
    let r: f64 = rng.gen();
    if r < 0.5 {
        Packet {
            protocol: "TCP".into(),
            dst_port: 443,
            payload_size: rng.gen_range(200..1500),
            rate_pps: rng.gen_range(50.0..200.0),
        }
    } else if r < 0.8 {
        Packet {
            protocol: "TCP".into(),
            dst_port: 80,
            payload_size: rng.gen_range(100..800),
            rate_pps: rng.gen_range(20.0..150.0),
        }
    } else {
        Packet {
            protocol: "UDP".into(),
            dst_port: 53,
            payload_size: rng.gen_range(40..512),
            rate_pps: rng.gen_range(10.0..100.0),
        }
    }
}

/// IoT traffic: MQTT (1883), CoAP (5683), small payloads, low rates
fn iot_packet(rng: &mut StdRng) -> Packet {
    let r: f64 = rng.gen();
    if r < 0.6 {
        Packet {
            protocol: "TCP".into(),
            dst_port: 1883,
            payload_size: rng.gen_range(20..200),
            rate_pps: rng.gen_range(1.0..20.0),
        }
    } else {
        Packet {
            protocol: "UDP".into(),
            dst_port: 5683,
            payload_size: rng.gen_range(10..100),
            rate_pps: rng.gen_range(0.5..10.0),
        }
    }
}

/// Attack traffic: DDoS amplification, SYN floods, port scans
fn attack_packet(rng: &mut StdRng) -> Packet {
    let attack_type = rng.gen_range(0u8..4);
    match attack_type {
        0 => Packet {
            // DNS amplification: huge UDP responses
            protocol: "UDP".into(),
            dst_port: rng.gen_range(1024..65535),
            payload_size: rng.gen_range(4000..9000),
            rate_pps: rng.gen_range(10000.0..100000.0),
        },
        1 => Packet {
            // SYN flood: tiny TCP packets, insane rate
            protocol: "TCP".into(),
            dst_port: 80,
            payload_size: rng.gen_range(0..60),
            rate_pps: rng.gen_range(50000.0..500000.0),
        },
        2 => Packet {
            // Port scan: sequential unusual ports, low payload
            protocol: "TCP".into(),
            dst_port: rng.gen_range(1..1024),
            payload_size: rng.gen_range(0..40),
            rate_pps: rng.gen_range(100.0..5000.0),
        },
        _ => Packet {
            // NTP amplification: UDP, unusual port, large response
            protocol: "UDP".into(),
            dst_port: 123,
            payload_size: rng.gen_range(400..1200),
            rate_pps: rng.gen_range(5000.0..50000.0),
        },
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

fn print_subheader(title: &str) {
    println!();
    println!("  --- {} ---", title);
}

fn main() {
    let holon = Holon::with_seed(4096, 42);
    let mut rng = StdRng::seed_from_u64(1234);

    // =========================================================================
    // PHASE 1 — BASELINE LEARNING
    // =========================================================================
    print_header("PHASE 1: Baseline Learning (OnlineSubspace / CCIPCA)");

    println!("  Training on 200 normal web traffic packets...");
    println!("  OnlineSubspace(dim=4096, k=32) — learns the traffic manifold online.");
    println!();

    let mut subspace = holon.create_subspace(32);
    let n_train = 200;

    for i in 0..n_train {
        let pkt = normal_packet(&mut rng);
        let vec = holon.encode_walkable(&pkt);
        let residual = subspace.update(&vec.to_f64());

        // Print convergence snapshots
        if i == 0 || (i + 1) % 50 == 0 {
            println!(
                "  [{:>3}/{}]  residual={:.3}  threshold={:.3}  explained={:.1}%",
                i + 1,
                n_train,
                residual,
                subspace.threshold(),
                subspace.explained_ratio() * 100.0
            );
        }
    }

    println!();
    let eigs = subspace.eigenvalues();
    let top5: Vec<f64> = eigs.iter().take(5).cloned().collect();
    println!("  Subspace converged.");
    println!("  Threshold: {:.3}", subspace.threshold());
    println!("  Explained ratio: {:.1}%", subspace.explained_ratio() * 100.0);
    println!(
        "  Top-5 eigenvalues: {}",
        top5.iter()
            .map(|e| format!("{:.2}", e))
            .collect::<Vec<_>>()
            .join(", ")
    );

    // =========================================================================
    // PHASE 2 — ANOMALY DETECTION
    // =========================================================================
    print_header("PHASE 2: Anomaly Detection");

    let threshold = subspace.threshold();
    println!("  Adaptive threshold: {:.3}", threshold);
    println!();

    print_subheader("Normal traffic (should be BELOW threshold)");
    let mut normal_detected = 0;
    for i in 0..10 {
        let pkt = normal_packet(&mut rng);
        let proto = pkt.protocol.clone();
        let port = pkt.dst_port;
        let rate = pkt.rate_pps;
        let vec = holon.encode_walkable(&pkt);
        let residual = subspace.residual(&vec.to_f64());
        let flag = if residual > threshold { "⚠ ANOMALY" } else { "✓ normal" };
        if residual > threshold {
            normal_detected += 1;
        }
        println!(
            "  [{:>2}] {:<3} :{:<5}  rate={:>8.1} pps  residual={:.3}  {}",
            i + 1,
            proto,
            port,
            rate,
            residual,
            flag
        );
    }

    print_subheader("Attack traffic (should be ABOVE threshold)");
    let mut attacks_detected = 0;
    for i in 0..10 {
        let pkt = attack_packet(&mut rng);
        let proto = pkt.protocol.clone();
        let port = pkt.dst_port;
        let rate = pkt.rate_pps;
        let vec = holon.encode_walkable(&pkt);
        let residual = subspace.residual(&vec.to_f64());
        let flag = if residual > threshold {
            attacks_detected += 1;
            "⚠ ANOMALY"
        } else {
            "✓ normal"
        };
        println!(
            "  [{:>2}] {:<3} :{:<5}  rate={:>8.1} pps  residual={:.3}  {}",
            i + 1,
            proto,
            port,
            rate,
            residual,
            flag
        );
    }

    println!();
    println!("  Detection results:");
    println!(
        "    Normal packets correctly passed:   {}/10",
        10 - normal_detected
    );
    println!("    Attacks correctly flagged:         {}/10", attacks_detected);

    // =========================================================================
    // PHASE 3 — ENGRAM MINTING
    // =========================================================================
    print_header("PHASE 3: Engram Minting (Named Pattern Library)");

    println!("  Minting normal web traffic pattern as engram 'web_traffic'...");
    let mut library = holon.create_engram_library();
    library.add("web_traffic", &subspace, None, HashMap::new());

    println!("  Training second subspace on IoT traffic (MQTT/CoAP)...");
    let mut iot_subspace = holon.create_subspace(32);
    let mut iot_rng = StdRng::seed_from_u64(9999);
    for _ in 0..200 {
        let pkt = iot_packet(&mut iot_rng);
        let vec = holon.encode_walkable(&pkt);
        iot_subspace.update(&vec.to_f64());
    }
    library.add("iot_traffic", &iot_subspace, None, HashMap::new());

    println!();
    println!("  Library contains {} engrams:", library.len());
    let mut names = library.names();
    names.sort();
    for name in &names {
        println!("    - {}", name);
    }

    // =========================================================================
    // PHASE 4 — PATTERN RECALL
    // =========================================================================
    print_header("PHASE 4: Pattern Recall (match_vec)");

    println!("  Probing with a web traffic packet → expect 'web_traffic' ranked first.");
    let web_probe_pkt = normal_packet(&mut rng);
    let web_probe = holon.encode_walkable(&web_probe_pkt);
    let web_matches = library.match_vec(&web_probe.to_f64(), 2, 10);
    println!("  Match results (lower residual = better fit):");
    for (name, residual) in &web_matches {
        let marker = if name == "web_traffic" { " ← best match" } else { "" };
        println!("    {:<15}  residual={:.3}{}", name, residual, marker);
    }
    let web_top = &web_matches[0].0;
    println!(
        "\n  Result: ranked '{}' first — {}",
        web_top,
        if web_top == "web_traffic" { "CORRECT ✓" } else { "unexpected" }
    );

    println!();
    println!("  Probing with an IoT packet → expect 'iot_traffic' ranked first.");
    let iot_probe_pkt = iot_packet(&mut iot_rng);
    let iot_probe = holon.encode_walkable(&iot_probe_pkt);
    let iot_matches = library.match_vec(&iot_probe.to_f64(), 2, 10);
    println!("  Match results (lower residual = better fit):");
    for (name, residual) in &iot_matches {
        let marker = if name == "iot_traffic" { " ← best match" } else { "" };
        println!("    {:<15}  residual={:.3}{}", name, residual, marker);
    }
    let iot_top = &iot_matches[0].0;
    println!(
        "\n  Result: ranked '{}' first — {}",
        iot_top,
        if iot_top == "iot_traffic" { "CORRECT ✓" } else { "unexpected" }
    );

    // =========================================================================
    // PHASE 5 — SNAPSHOT PERSISTENCE
    // =========================================================================
    print_header("PHASE 5: Snapshot & Persistence");

    print_subheader("Subspace snapshot round-trip");
    let snap = subspace.snapshot();
    let restored = OnlineSubspace::from_snapshot(snap);

    let test_pkt = normal_packet(&mut rng);
    let test_vec = holon.encode_walkable(&test_pkt);
    let test_f64 = test_vec.to_f64();
    let res_original = subspace.residual(&test_f64);
    let res_restored = restored.residual(&test_f64);
    println!("  Original residual:  {:.6}", res_original);
    println!("  Restored residual:  {:.6}", res_restored);
    println!(
        "  Difference:         {:.2e}  {}",
        (res_original - res_restored).abs(),
        if (res_original - res_restored).abs() < 1e-10 {
            "✓ exact match"
        } else {
            "✗ mismatch"
        }
    );

    print_subheader("EngramLibrary JSON persistence");
    let path = "/tmp/holon_engram_demo.json";
    library.save(path).expect("save failed");
    println!("  Saved library to {}", path);

    let mut library2 = EngramLibrary::load(path).expect("load failed");
    println!("  Loaded library: {} engrams", library2.len());

    let matches_before = library.match_vec(&web_probe.to_f64(), 2, 10);
    let matches_after = library2.match_vec(&web_probe.to_f64(), 2, 10);

    println!("  Comparing match results before/after round-trip:");
    let mut all_match = true;
    for ((name1, r1), (name2, r2)) in matches_before.iter().zip(matches_after.iter()) {
        let same = name1 == name2 && (r1 - r2).abs() < 1e-10;
        if !same {
            all_match = false;
        }
        println!(
            "    {:<15}  {:.6} → {:.6}  {}",
            name1,
            r1,
            r2,
            if same { "✓" } else { "✗" }
        );
    }
    println!(
        "  Round-trip fidelity: {}",
        if all_match { "PERFECT ✓" } else { "DEGRADED ✗" }
    );

    let _ = std::fs::remove_file(path);
    println!("  Cleaned up {}", path);

    // =========================================================================
    // SUMMARY
    // =========================================================================
    print_header("Summary");
    println!("  OnlineSubspace (CCIPCA) learned normal traffic in 200 updates.");
    println!("  Adaptive threshold flags anomalies without any hard-coded rules.");
    println!("  EngramLibrary stores named patterns and retrieves the best match.");
    println!("  Snapshot/JSON persistence produces byte-exact round-trips.");
    println!();
    println!("  This is the foundation for the XDP/eBPF pipeline:");
    println!("  userspace Rust learns the manifold → pushes thresholds to kernel.");
    println!();
}
