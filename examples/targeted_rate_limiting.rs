//! Targeted Rate Limiting with similarity_profile()
//!
//! Uses dimension-level analysis to create wider separation between
//! legitimate and attack traffic, enabling more precise rate limiting.
//!
//! Run: cargo run --example targeted_rate_limiting

use holon::highlevel::Holon;
use holon::kernel::{Primitives, Vector};
use rand::prelude::*;
use std::collections::HashMap;

/// Simulated network packet
#[derive(Clone)]
struct Packet {
    src_port: u16,
    dst_port: u16,
    protocol: String,
    payload_size: usize,
    label: String,
}

fn generate_normal_traffic(count: usize, seed: u64, include_dns: bool) -> Vec<Packet> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut packets = Vec::with_capacity(count);

    for _ in 0..count {
        let r: f64 = rng.gen();
        if r < 0.35 {
            packets.push(Packet {
                src_port: rng.gen_range(49152..65535),
                dst_port: 443,
                protocol: "TCP".to_string(),
                payload_size: (rng.gen::<f64>() * 500.0) as usize,
                label: "normal".to_string(),
            });
        } else if r < 0.60 {
            packets.push(Packet {
                src_port: rng.gen_range(49152..65535),
                dst_port: 80,
                protocol: "TCP".to_string(),
                payload_size: (rng.gen::<f64>() * 800.0) as usize,
                label: "normal".to_string(),
            });
        } else if r < 0.80 && include_dns {
            // Legitimate DNS query: src=ephemeral, dst=53
            packets.push(Packet {
                src_port: rng.gen_range(49152..65535),
                dst_port: 53,
                protocol: "UDP".to_string(),
                payload_size: (rng.gen::<f64>() * 80.0) as usize,
                label: "normal".to_string(),
            });
        } else {
            packets.push(Packet {
                src_port: rng.gen_range(49152..65535),
                dst_port: rng.gen_range(1024..49151),
                protocol: "TCP".to_string(),
                payload_size: (rng.gen::<f64>() * 300.0) as usize,
                label: "normal".to_string(),
            });
        }
    }
    packets
}

fn generate_dns_reflection(count: usize, seed: u64) -> Vec<Packet> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| Packet {
            src_port: 53, // Spoofed source
            dst_port: rng.gen_range(49152..65535),
            protocol: "UDP".to_string(),
            payload_size: (rng.gen::<f64>() * 4000.0) as usize,
            label: "dns_reflection".to_string(),
        })
        .collect()
}

fn generate_legit_dns(count: usize, seed: u64) -> Vec<Packet> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| Packet {
            src_port: rng.gen_range(49152..65535), // Ephemeral source
            dst_port: 53, // Going TO DNS server
            protocol: "UDP".to_string(),
            payload_size: (rng.gen::<f64>() * 80.0) as usize,
            label: "legitimate_dns".to_string(),
        })
        .collect()
}

fn encode_packet(holon: &Holon, pkt: &Packet) -> Vector {
    let src_port_band = match pkt.src_port {
        53 => "dns",
        123 => "ntp",
        p if p < 1024 => "wellknown",
        _ => "ephemeral",
    };

    let dst_port_band = match pkt.dst_port {
        80 | 8080 => "http",
        443 => "https",
        53 => "dns",
        123 => "ntp",
        p if p < 1024 => "wellknown",
        _ => "ephemeral",
    };

    let direction = if pkt.src_port < 1024 && pkt.dst_port >= 1024 {
        "amplified"
    } else {
        "normal"
    };

    let mut data = HashMap::new();
    data.insert("src_port_band".to_string(), serde_json::Value::String(src_port_band.to_string()));
    data.insert("dst_port_band".to_string(), serde_json::Value::String(dst_port_band.to_string()));
    data.insert("protocol".to_string(), serde_json::Value::String(pkt.protocol.clone()));
    data.insert("size_class".to_string(), serde_json::Value::String(
        if pkt.payload_size < 100 { "tiny" }
        else if pkt.payload_size < 500 { "small" }
        else if pkt.payload_size < 2000 { "medium" }
        else { "large" }.to_string()
    ));
    data.insert("direction".to_string(), serde_json::Value::String(direction.to_string()));

    holon.encode_value(&serde_json::Value::Object(data.into_iter().collect()))
}

/// Old approach: scalar similarity
fn rate_factor_old(holon: &Holon, vec: &Vector, baseline: &Vector) -> f64 {
    let sim = holon.similarity(vec, baseline);
    ((sim + 1.0) / 2.0).clamp(0.0, 1.0) // [-1,1] → [0,1]
}

/// Targeted approach: scale by anomalous ratio
fn rate_factor_targeted(vec: &Vector, baseline: &Vector) -> (f64, f64) {
    use holon::Similarity;

    let profile = Primitives::similarity_profile(vec, baseline);
    let baseline_sim = Similarity::cosine(vec, baseline);

    let active_dims = profile.data().iter().filter(|&&v| v.abs() > 0).count();
    let disagreeing = profile.data().iter().filter(|&&v| v < 0).count();
    let anomalous_ratio = if active_dims > 0 { disagreeing as f64 / active_dims as f64 } else { 0.0 };

    let rate = baseline_sim * (1.0 - anomalous_ratio);
    (rate.clamp(0.0, 1.0), anomalous_ratio)
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║          TARGETED RATE LIMITING (Rust)                               ║");
    println!("║                                                                      ║");
    println!("║  Goal: Create wider separation between legitimate and attack traffic ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    let holon = Holon::new(4096);

    // =========================================================================
    // PHASE 1: Learn Baseline (includes legitimate DNS)
    // =========================================================================

    println!("======================================================================");
    println!("PHASE 1: LEARNING BASELINE");
    println!("======================================================================");

    let normal_train = generate_normal_traffic(500, 1, true);
    let normal_vecs: Vec<Vector> = normal_train.iter().map(|p| encode_packet(&holon, p)).collect();
    let baseline_proto = holon.prototype(&normal_vecs.iter().collect::<Vec<_>>(), 0.5);

    println!("Learned baseline from {} packets (includes legitimate DNS)", normal_train.len());

    // =========================================================================
    // PHASE 2: Generate Test Traffic
    // =========================================================================

    println!("\n======================================================================");
    println!("PHASE 2: COMPARING RATE LIMITING APPROACHES");
    println!("======================================================================");

    let normal_test = generate_normal_traffic(100, 10, true);
    let legit_dns = generate_legit_dns(100, 30);
    let attack_test = generate_dns_reflection(100, 20);

    let test_sets: Vec<(&str, Vec<Packet>)> = vec![
        ("Normal HTTPS/HTTP", normal_test),
        ("Legitimate DNS queries", legit_dns),
        ("DNS Reflection Attack", attack_test),
    ];

    println!("\n┌─────────────────────────┬───────────────────┬───────────────────┐");
    println!("│ Traffic Type            │ Old Rate Factor   │ Targeted Factor   │");
    println!("├─────────────────────────┼───────────────────┼───────────────────┤");

    let mut legit_dns_old = Vec::new();
    let mut legit_dns_targeted = Vec::new();
    let mut attack_old = Vec::new();
    let mut attack_targeted = Vec::new();

    for (name, packets) in &test_sets {
        let mut old_factors = Vec::new();
        let mut targeted_factors = Vec::new();
        let mut anomalous_ratios = Vec::new();

        for pkt in packets {
            let vec = encode_packet(&holon, pkt);
            let old_factor = rate_factor_old(&holon, &vec, &baseline_proto);
            let (targeted_factor, anomalous_ratio) = rate_factor_targeted(&vec, &baseline_proto);

            old_factors.push(old_factor);
            targeted_factors.push(targeted_factor);
            anomalous_ratios.push(anomalous_ratio);

            // Store for later analysis
            if *name == "Legitimate DNS queries" {
                legit_dns_old.push(old_factor);
                legit_dns_targeted.push(targeted_factor);
            } else if *name == "DNS Reflection Attack" {
                attack_old.push(old_factor);
                attack_targeted.push(targeted_factor);
            }
        }

        let avg_old: f64 = old_factors.iter().sum::<f64>() / old_factors.len() as f64;
        let avg_targeted: f64 = targeted_factors.iter().sum::<f64>() / targeted_factors.len() as f64;
        let avg_anomalous: f64 = anomalous_ratios.iter().sum::<f64>() / anomalous_ratios.len() as f64;

        println!("│ {:23} │ {:.3}             │ {:.3} ({:.0}% anom) │",
            name, avg_old, avg_targeted, avg_anomalous * 100.0);
    }

    println!("└─────────────────────────┴───────────────────┴───────────────────┘");

    // =========================================================================
    // PHASE 3: Separation Analysis
    // =========================================================================

    println!("\n======================================================================");
    println!("PHASE 3: SEPARATION ANALYSIS");
    println!("======================================================================");

    let min_legit_old = legit_dns_old.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_attack_old = attack_old.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let gap_old = min_legit_old - max_attack_old;

    let min_legit_targeted = legit_dns_targeted.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_attack_targeted = attack_targeted.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let gap_targeted = min_legit_targeted - max_attack_targeted;

    println!("\nSeparation analysis (can we find a threshold that works?):");
    if gap_old > 0.0 {
        println!("    Old approach: min_legit={:.3}, max_attack={:.3}, GAP={:.3} ✓", min_legit_old, max_attack_old, gap_old);
    } else {
        println!("    Old approach: min_legit={:.3}, max_attack={:.3}, OVERLAP ✗", min_legit_old, max_attack_old);
    }

    if gap_targeted > 0.0 {
        println!("    Targeted:     min_legit={:.3}, max_attack={:.3}, GAP={:.3} ✓", min_legit_targeted, max_attack_targeted, gap_targeted);
    } else {
        println!("    Targeted:     min_legit={:.3}, max_attack={:.3}, OVERLAP ✗", min_legit_targeted, max_attack_targeted);
    }

    // =========================================================================
    // PHASE 4: Attack Blocking
    // =========================================================================

    println!("\n======================================================================");
    println!("PHASE 4: ATTACK MITIGATION EFFECTIVENESS");
    println!("======================================================================");

    let old_blocked = attack_old.iter().filter(|&&f| f < 0.3).count();
    let targeted_blocked = attack_targeted.iter().filter(|&&f| f < 0.3).count();

    println!("\nAttack packets blocked (rate < 0.3):");
    println!("    Old approach:      {}/100 ({}%)", old_blocked, old_blocked);
    println!("    Targeted approach: {}/100 ({}%)", targeted_blocked, targeted_blocked);

    // =========================================================================
    // SUMMARY
    // =========================================================================

    println!("\n======================================================================");
    println!("SUMMARY: TARGETED vs OLD RATE LIMITING");
    println!("======================================================================");

    let avg_legit_old: f64 = legit_dns_old.iter().sum::<f64>() / legit_dns_old.len() as f64;
    let avg_attack_old: f64 = attack_old.iter().sum::<f64>() / attack_old.len() as f64;
    let avg_legit_targeted: f64 = legit_dns_targeted.iter().sum::<f64>() / legit_dns_targeted.len() as f64;
    let avg_attack_targeted: f64 = attack_targeted.iter().sum::<f64>() / attack_targeted.len() as f64;

    let gap_avg_old = avg_legit_old - avg_attack_old;
    let gap_avg_targeted = avg_legit_targeted - avg_attack_targeted;
    let improvement = if gap_avg_old > 0.0 { gap_avg_targeted / gap_avg_old } else { 0.0 };

    println!(r#"
    ┌───────────────────────────────────────────────────────────────────┐
    │  Metric                     │  Old Approach  │  Targeted Approach │
    ├─────────────────────────────┼────────────────┼────────────────────┤
    │  Legit DNS rate factor      │     {:.3}       │       {:.3}          │
    │  Attack DNS rate factor     │     {:.3}       │       {:.3}          │
    │  Gap (Legit - Attack)       │     {:.3}       │       {:.3}          │
    │  Gap improvement            │       -        │       {:.1}x           │
    └─────────────────────────────┴────────────────┴────────────────────┘
"#, avg_legit_old, avg_legit_targeted, avg_attack_old, avg_attack_targeted,
    gap_avg_old, gap_avg_targeted, improvement);

    if gap_avg_targeted > gap_avg_old {
        println!("✓ Targeted approach creates {:.1}x wider separation!", improvement);
        println!("  This enables more precise rate limiting with less collateral damage.");
    } else {
        println!("⚠ Targeted approach did not improve separation");
    }
}
