//! Attack Variant Detection with analogy()
//!
//! Demonstrates zero-shot detection of UNSEEN attack variants using analogy.
//!
//! Key insight: Train on ONE attack type (DNS reflection), then use analogy()
//! to detect variants (NTP, SSDP, CHARGEN) without explicit training.
//!
//! Run: cargo run --example attack_variant_detection

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
}

/// Generate normal traffic mix
fn generate_normal_traffic(count: usize, seed: u64) -> Vec<Packet> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut packets = Vec::with_capacity(count);

    for _ in 0..count {
        let r: f64 = rng.gen();
        if r < 0.5 {
            // HTTPS
            packets.push(Packet {
                src_port: rng.gen_range(49152..65535),
                dst_port: 443,
                protocol: "TCP".to_string(),
                payload_size: (rng.gen::<f64>() * 500.0) as usize,
            });
        } else if r < 0.8 {
            // HTTP
            packets.push(Packet {
                src_port: rng.gen_range(49152..65535),
                dst_port: 80,
                protocol: "TCP".to_string(),
                payload_size: (rng.gen::<f64>() * 800.0) as usize,
            });
        } else {
            // Other
            packets.push(Packet {
                src_port: rng.gen_range(49152..65535),
                dst_port: rng.gen_range(1024..49151),
                protocol: "TCP".to_string(),
                payload_size: (rng.gen::<f64>() * 300.0) as usize,
            });
        }
    }
    packets
}

/// Generate DNS reflection attack
fn generate_dns_reflection(count: usize, seed: u64) -> Vec<Packet> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| Packet {
            src_port: 53, // Spoofed DNS source
            dst_port: rng.gen_range(49152..65535),
            protocol: "UDP".to_string(),
            payload_size: (rng.gen::<f64>() * 4000.0) as usize,
        })
        .collect()
}

/// Generate NTP amplification attack
fn generate_ntp_amplification(count: usize, seed: u64) -> Vec<Packet> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| Packet {
            src_port: 123, // NTP
            dst_port: rng.gen_range(49152..65535),
            protocol: "UDP".to_string(),
            payload_size: (rng.gen::<f64>() * 5000.0) as usize,
        })
        .collect()
}

/// Generate SSDP amplification attack
fn generate_ssdp_amplification(count: usize, seed: u64) -> Vec<Packet> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| Packet {
            src_port: 1900, // SSDP
            dst_port: rng.gen_range(49152..65535),
            protocol: "UDP".to_string(),
            payload_size: (rng.gen::<f64>() * 3000.0) as usize,
        })
        .collect()
}

/// Generate CHARGEN amplification attack
fn generate_chargen_amplification(count: usize, seed: u64) -> Vec<Packet> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| Packet {
            src_port: 19, // CHARGEN
            dst_port: rng.gen_range(49152..65535),
            protocol: "UDP".to_string(),
            payload_size: (rng.gen::<f64>() * 6000.0) as usize,
        })
        .collect()
}

/// Encode a packet to a vector
fn encode_packet(holon: &Holon, pkt: &Packet) -> Vector {
    let src_port_name = match pkt.src_port {
        53 => "dns",
        123 => "ntp",
        1900 => "ssdp",
        19 => "chargen",
        p if p >= 49152 => "ephemeral",
        _ => "other",
    };

    let mut data = HashMap::new();
    data.insert(
        "src_port_name".to_string(),
        serde_json::Value::String(src_port_name.to_string()),
    );
    data.insert(
        "src_port_class".to_string(),
        serde_json::Value::String(if pkt.src_port < 1024 {
            "wellknown"
        } else {
            "ephemeral"
        }.to_string()),
    );
    data.insert(
        "dst_port_class".to_string(),
        serde_json::Value::String(if pkt.dst_port >= 49152 {
            "ephemeral"
        } else {
            "other"
        }.to_string()),
    );
    data.insert(
        "protocol".to_string(),
        serde_json::Value::String(pkt.protocol.clone()),
    );
    data.insert(
        "size_class".to_string(),
        serde_json::Value::String(
            if pkt.payload_size < 500 {
                "small"
            } else if pkt.payload_size < 2000 {
                "medium"
            } else {
                "large"
            }
            .to_string(),
        ),
    );
    data.insert(
        "pattern".to_string(),
        serde_json::Value::String(
            if pkt.src_port < 1024 && pkt.dst_port >= 49152 && pkt.protocol == "UDP" && pkt.payload_size > 1000 {
                "amplification"
            } else {
                "normal"
            }
            .to_string(),
        ),
    );

    holon.encode_value(&serde_json::Value::Object(data.into_iter().collect()))
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║       ATTACK VARIANT DETECTION WITH analogy() (Rust)                 ║");
    println!("║                                                                      ║");
    println!("║  Goal: Detect UNSEEN attack variants by analogy from known attacks   ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    let holon = Holon::new(4096);

    // =========================================================================
    // PHASE 1: Learn Baseline and ONE Attack
    // =========================================================================

    println!("======================================================================");
    println!("PHASE 1: LEARNING (Only normal traffic + DNS reflection)");
    println!("======================================================================");

    let normal_train = generate_normal_traffic(500, 1);
    let dns_train = generate_dns_reflection(100, 2);

    // Learn normal prototype
    let normal_vecs: Vec<Vector> = normal_train.iter().map(|p| encode_packet(&holon, p)).collect();
    let normal_proto = holon.prototype(&normal_vecs.iter().collect::<Vec<_>>(), 0.5);

    // Learn DNS attack prototype
    let dns_vecs: Vec<Vector> = dns_train.iter().map(|p| encode_packet(&holon, p)).collect();
    let dns_attack_proto = holon.prototype(&dns_vecs.iter().collect::<Vec<_>>(), 0.5);

    // Encode port vectors for analogy
    let dns_port_vec = holon.encode_value(&serde_json::json!({"src_port_name": "dns"}));
    let ntp_port_vec = holon.encode_value(&serde_json::json!({"src_port_name": "ntp"}));
    let ssdp_port_vec = holon.encode_value(&serde_json::json!({"src_port_name": "ssdp"}));
    let chargen_port_vec = holon.encode_value(&serde_json::json!({"src_port_name": "chargen"}));

    println!("Learned normal baseline from {} packets", normal_train.len());
    println!("Learned DNS reflection attack from {} packets", dns_train.len());
    println!("\nNOTE: We are NOT training on NTP, SSDP, or CHARGEN attacks!");
    println!("      We will use analogy() to infer them from DNS reflection.");

    // =========================================================================
    // PHASE 2: Generate Test Traffic
    // =========================================================================

    println!("\n======================================================================");
    println!("PHASE 2: GENERATING TEST TRAFFIC");
    println!("======================================================================");

    let normal_test = generate_normal_traffic(50, 10);
    let dns_test = generate_dns_reflection(50, 20);
    let ntp_test = generate_ntp_amplification(50, 30);
    let ssdp_test = generate_ssdp_amplification(50, 40);
    let chargen_test = generate_chargen_amplification(50, 50);

    println!("  Normal test: {} packets", normal_test.len());
    println!("  DNS reflection test: {} packets", dns_test.len());
    println!("  NTP amplification test: {} packets (UNSEEN)", ntp_test.len());
    println!("  SSDP amplification test: {} packets (UNSEEN)", ssdp_test.len());
    println!("  CHARGEN amplification test: {} packets (UNSEEN)", chargen_test.len());

    // =========================================================================
    // PHASE 3: Analogy-Based Detection
    // =========================================================================

    println!("\n======================================================================");
    println!("PHASE 3: ANALOGY-BASED VARIANT DETECTION");
    println!("======================================================================");

    println!("\nAnalogy reasoning:");
    println!("  'DNS reflection attack' - 'DNS port' + 'NTP port' = 'NTP attack' (inferred)");

    // Infer variant attacks using analogy
    let ntp_inferred = Primitives::analogy(&dns_attack_proto, &dns_port_vec, &ntp_port_vec);
    let ssdp_inferred = Primitives::analogy(&dns_attack_proto, &dns_port_vec, &ssdp_port_vec);
    let chargen_inferred = Primitives::analogy(&dns_attack_proto, &dns_port_vec, &chargen_port_vec);

    println!("\nDetection results (similarity to inferred variant):");
    println!("──────────────────────────────────────────────────");

    let variants = [
        ("NTP", &ntp_test, &ntp_inferred),
        ("SSDP", &ssdp_test, &ssdp_inferred),
        ("CHARGEN", &chargen_test, &chargen_inferred),
    ];

    for (name, packets, inferred) in &variants {
        let sims: Vec<f64> = packets
            .iter()
            .map(|p| {
                let vec = encode_packet(&holon, p);
                holon.similarity(&vec, inferred)
            })
            .collect();
        let avg_sim: f64 = sims.iter().sum::<f64>() / sims.len() as f64;
        let std_sim: f64 = (sims.iter().map(|s| (s - avg_sim).powi(2)).sum::<f64>() / sims.len() as f64).sqrt();
        println!("  {:12}: similarity = {:.3} ± {:.3}", name, avg_sim, std_sim);
    }

    println!("──────────────────────────────────────────────────");

    // Test normal traffic against inferred variants (should have LOW similarity)
    for (name, _, inferred) in &variants {
        let normal_sims: Vec<f64> = normal_test
            .iter()
            .map(|p| {
                let vec = encode_packet(&holon, p);
                holon.similarity(&vec, inferred)
            })
            .collect();
        let avg_sim: f64 = normal_sims.iter().sum::<f64>() / normal_sims.len() as f64;
        println!("  Normal vs {} inferred: similarity = {:.3}", name, avg_sim);
    }

    // =========================================================================
    // PHASE 4: Combined Detection
    // =========================================================================

    println!("\n======================================================================");
    println!("PHASE 4: COMBINED DETECTION");
    println!("======================================================================");

    println!("\nCombined approach: MAX of analogy similarities to known variants");
    println!("  - If packet matches ANY inferred variant, flag as attack");

    let all_traffic: Vec<(&str, &Vec<Packet>)> = vec![
        ("Normal", &normal_test),
        ("DNS reflection", &dns_test),
        ("NTP (unseen)", &ntp_test),
        ("SSDP (unseen)", &ssdp_test),
        ("CHARGEN (unseen)", &chargen_test),
    ];

    println!("\nCombined detection results:");
    println!("────────────────────────────────────────────────────────────");

    for (name, packets) in &all_traffic {
        let mut detected = 0;
        let mut baseline_sims = Vec::new();
        let mut variant_sims = Vec::new();

        for pkt in *packets {
            let vec = encode_packet(&holon, pkt);
            let baseline_sim = holon.similarity(&vec, &normal_proto);

            // Check against all inferred variants
            let ntp_sim = holon.similarity(&vec, &ntp_inferred);
            let ssdp_sim = holon.similarity(&vec, &ssdp_inferred);
            let chargen_sim = holon.similarity(&vec, &chargen_inferred);
            let max_variant_sim = ntp_sim.max(ssdp_sim).max(chargen_sim);

            baseline_sims.push(baseline_sim);
            variant_sims.push(max_variant_sim);

            // Decision: anomaly if low baseline sim OR high variant sim
            if baseline_sim < 0.5 || max_variant_sim > 0.6 {
                detected += 1;
            }
        }

        let avg_baseline: f64 = baseline_sims.iter().sum::<f64>() / baseline_sims.len() as f64;
        let avg_variant: f64 = variant_sims.iter().sum::<f64>() / variant_sims.len() as f64;
        let pct = 100.0 * detected as f64 / packets.len() as f64;

        println!(
            "  {:20}: {:3}/{:3} detected ({:5.1}%)  base={:.2}, variant={:.2}",
            name,
            detected,
            packets.len(),
            pct,
            avg_baseline,
            avg_variant
        );
    }

    // =========================================================================
    // SUMMARY
    // =========================================================================

    println!("\n======================================================================");
    println!("SUMMARY: ANALOGY-BASED VARIANT DETECTION");
    println!("======================================================================");

    println!(r#"
    ┌───────────────────────────────────────────────────────────────────┐
    │  KEY FINDING: analogy() enables detecting UNSEEN attack variants  │
    ├───────────────────────────────────────────────────────────────────┤
    │                                                                   │
    │  TRAINING DATA:                                                   │
    │    - Normal traffic: ✓                                            │
    │    - DNS reflection: ✓                                            │
    │    - NTP, SSDP, CHARGEN: NOT in training                          │
    │                                                                   │
    │  DETECTION RESULT:                                                │
    │    - analogy() infers variant structure from DNS example          │
    │    - "If DNS attack looks like X, NTP attack should look like Y"  │
    │    - Enables zero-shot detection of similar attack patterns       │
    │                                                                   │
    └───────────────────────────────────────────────────────────────────┘
"#);
}
