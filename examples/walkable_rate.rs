//! Walkable Rate Detection - Zero Serialization
//!
//! Same as pure_vector_rate.rs but using the Walkable trait instead of JSON.
//! Demonstrates zero-serialization encoding with type-safe packet structs.
//!
//! Run: cargo run --example walkable_rate

use holon::highlevel::Holon;
use holon::kernel::{WalkType, Walkable, WalkableValue};
use std::collections::VecDeque;

// =============================================================================
// CONFIGURATION
// =============================================================================

const DIMENSIONS: usize = 4096;
const DECAY: f64 = 0.98;
const WARMUP_PACKETS: usize = 400;
const WINDOW_SIZE: usize = 100;

// =============================================================================
// PACKET TYPE WITH WALKABLE IMPLEMENTATION
// =============================================================================

#[derive(Clone, Debug)]
struct Packet {
    protocol: String,
    src_port: u16,
    dst_port: u16,
    size: u32,
    timestamp: f64,
}

impl Packet {
    fn new(protocol: &str, src_port: u16, dst_port: u16, size: u32, timestamp: f64) -> Self {
        Self {
            protocol: protocol.to_string(),
            src_port,
            dst_port,
            size,
            timestamp,
        }
    }
}

/// Implement Walkable for Packet - zero serialization path!
impl Walkable for Packet {
    fn walk_type(&self) -> WalkType {
        WalkType::Map
    }

    fn walk_map_items(&self) -> Vec<(&str, WalkableValue)> {
        vec![
            ("protocol", self.protocol.to_walkable_value()),
            ("src_port", (self.src_port as i64).to_walkable_value()),
            ("dst_port", (self.dst_port as i64).to_walkable_value()),
            ("size", (self.size as i64).to_walkable_value()),
        ]
    }
}

// =============================================================================
// RATE CALCULATOR
// =============================================================================

struct RateCalculator {
    timestamps: VecDeque<f64>,
    window_size: usize,
    last_rate: f64,
    smoothing: f64,
}

impl RateCalculator {
    fn new(window_size: usize) -> Self {
        Self {
            timestamps: VecDeque::with_capacity(window_size),
            window_size,
            last_rate: 100.0,
            smoothing: 0.1,
        }
    }

    fn add(&mut self, timestamp: f64) -> f64 {
        let instant_rate = if let Some(&last_ts) = self.timestamps.back() {
            let delta = timestamp - last_ts;
            if delta > 1e-9 {
                1.0 / delta
            } else {
                1_000_000.0
            }
        } else {
            100.0
        };

        self.timestamps.push_back(timestamp);
        if self.timestamps.len() > self.window_size {
            self.timestamps.pop_front();
        }

        self.last_rate = self.smoothing * instant_rate + (1.0 - self.smoothing) * self.last_rate;

        let window_rate = self.window_rate();
        window_rate.max(self.last_rate)
    }

    fn window_rate(&self) -> f64 {
        if self.timestamps.len() < 2 {
            return self.last_rate;
        }
        let duration = self.timestamps.back().unwrap() - self.timestamps.front().unwrap();
        if duration < 1e-9 {
            return 1_000_000.0;
        }
        (self.timestamps.len() - 1) as f64 / duration
    }
}

// =============================================================================
// WALKABLE DETECTOR (uses encode_walkable instead of encode_json!)
// =============================================================================

struct WalkableDetector {
    holon: Holon,
    pattern_accum: holon::Accumulator,
    rate_accum: holon::Accumulator,
    packet_count: usize,
    warmup_complete: bool,
    decay: f64,
}

impl WalkableDetector {
    fn new(dimensions: usize, decay: f64) -> Self {
        let holon = Holon::new(dimensions);
        let pattern_accum = holon.create_accumulator();
        let rate_accum = holon.create_accumulator();

        Self {
            holon,
            pattern_accum,
            rate_accum,
            packet_count: 0,
            warmup_complete: false,
            decay,
        }
    }

    fn process(&mut self, packet: &Packet, rate_pps: f64) -> Option<(f64, f64, bool)> {
        // =====================================================================
        // KEY DIFFERENCE: encode_walkable instead of encode_json!
        // No string serialization, no JSON parsing - direct struct encoding
        // =====================================================================
        let pattern_vec = self.holon.encode_walkable(packet);

        // Rate encoding (unchanged)
        let rate_vec = self.holon.encode_scalar_log(rate_pps);

        self.packet_count += 1;

        if !self.warmup_complete {
            self.holon.accumulate(&mut self.pattern_accum, &pattern_vec);
            self.holon.accumulate(&mut self.rate_accum, &rate_vec);

            if self.packet_count >= WARMUP_PACKETS {
                self.warmup_complete = true;
                println!("\n✓ Warmup complete after {} packets", self.packet_count);
            }
            return None;
        }

        let pattern_baseline = self.holon.normalize_accumulator(&self.pattern_accum);
        let rate_baseline = self.holon.normalize_accumulator(&self.rate_accum);

        let pattern_sim = self.holon.similarity(&pattern_vec, &pattern_baseline);
        let rate_sim = self.holon.similarity(&rate_vec, &rate_baseline);

        let pattern_anomaly = pattern_sim < 0.3;
        let rate_anomaly = rate_sim < 0.5;
        let is_anomaly = pattern_anomaly && rate_anomaly;

        if !is_anomaly {
            self.pattern_accum.decay(self.decay);
            self.rate_accum.decay(self.decay);
            self.holon.accumulate(&mut self.pattern_accum, &pattern_vec);
            self.holon.accumulate(&mut self.rate_accum, &rate_vec);
        }

        Some((pattern_sim, rate_sim, is_anomaly))
    }
}

// =============================================================================
// TRAFFIC GENERATORS
// =============================================================================

fn generate_normal_traffic(count: usize, start_time: f64) -> Vec<Packet> {
    let mut packets = Vec::with_capacity(count);
    let mut rng_state = 42u64;

    for i in 0..count {
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let r = (rng_state >> 16) as f32 / 65535.0;

        let (protocol, src_port, dst_port) = if r < 0.3 {
            ("TCP", 443, 50000 + (i as u16 % 1000))
        } else if r < 0.6 {
            ("TCP", 80, 50000 + (i as u16 % 1000))
        } else if r < 0.8 {
            ("UDP", 53, 50000 + (i as u16 % 1000))
        } else {
            ("TCP", 22, 50000 + (i as u16 % 1000))
        };

        let size = 100 + (rng_state % 1400) as u32;
        let timestamp = start_time + (i as f64 * 0.01);

        packets.push(Packet::new(protocol, src_port, dst_port, size, timestamp));
    }

    packets
}

fn generate_attack_traffic(count: usize, start_time: f64) -> Vec<Packet> {
    let mut packets = Vec::with_capacity(count);

    for i in 0..count {
        let packet = Packet::new(
            "UDP",
            53,
            49152 + (i as u16 % 100),
            4000,
            start_time + (i as f64 * 0.00001),
        );
        packets.push(packet);
    }

    packets
}

// =============================================================================
// MAIN
// =============================================================================

fn main() {
    println!("=================================================================");
    println!("WALKABLE RATE DETECTION - Zero Serialization Path");
    println!("=================================================================\n");

    println!("KEY DIFFERENCE FROM pure_vector_rate.rs:");
    println!("  - Packet struct implements Walkable trait");
    println!("  - Uses encode_walkable(&packet) instead of encode_json(&packet.to_json())");
    println!("  - No string building, no JSON parsing - direct struct traversal");
    println!("  - Type-safe: compiler catches field mismatches\n");

    let mut detector = WalkableDetector::new(DIMENSIONS, DECAY);
    let mut rate_calc = RateCalculator::new(WINDOW_SIZE);

    // Phase 1: Normal traffic
    println!("Phase 1: Normal traffic (learning baseline)...");
    let normal_traffic = generate_normal_traffic(600, 0.0);

    let mut normal_detections = 0;
    let mut normal_false_positives = 0;

    for packet in &normal_traffic {
        let rate = rate_calc.add(packet.timestamp);
        if let Some((pattern_sim, rate_sim, is_anomaly)) = detector.process(packet, rate) {
            normal_detections += 1;
            if is_anomaly {
                normal_false_positives += 1;
            }
            if normal_detections <= 5 || normal_detections % 50 == 0 {
                println!(
                    "  Packet {}: pattern_sim={:.3}, rate_sim={:.3}, rate={:.0} pps {}",
                    detector.packet_count,
                    pattern_sim,
                    rate_sim,
                    rate,
                    if is_anomaly { "⚠️ ANOMALY" } else { "✓" }
                );
            }
        }
    }

    println!(
        "\nNormal phase: {} detections, {} false positives ({:.1}% FPR)\n",
        normal_detections,
        normal_false_positives,
        100.0 * normal_false_positives as f64 / normal_detections.max(1) as f64
    );

    // Phase 2: Attack traffic
    println!("Phase 2: DNS Amplification Attack (100,000 pps)...");
    let attack_traffic = generate_attack_traffic(200, 10.0);

    let mut attack_detections = 0;
    let mut true_positives = 0;

    for packet in &attack_traffic {
        let rate = rate_calc.add(packet.timestamp);
        if let Some((pattern_sim, rate_sim, is_anomaly)) = detector.process(packet, rate) {
            attack_detections += 1;
            if is_anomaly {
                true_positives += 1;
            }
            if attack_detections <= 10 || attack_detections % 50 == 0 {
                println!(
                    "  Attack {}: pattern_sim={:.3}, rate_sim={:.3}, rate={:.0} pps {}",
                    attack_detections,
                    pattern_sim,
                    rate_sim,
                    rate,
                    if is_anomaly { "🚨 DETECTED" } else { "missed" }
                );
            }
        }
    }

    println!(
        "\nAttack phase: {} detections, {} true positives ({:.1}% detection rate)\n",
        attack_detections,
        true_positives,
        100.0 * true_positives as f64 / attack_detections.max(1) as f64
    );

    // Phase 3: Return to normal
    println!("Phase 3: Return to normal traffic...");
    let recovery_traffic = generate_normal_traffic(200, 20.0);

    let mut recovery_false_positives = 0;
    let mut recovery_count = 0;

    for packet in &recovery_traffic {
        let rate = rate_calc.add(packet.timestamp);
        if let Some((pattern_sim, rate_sim, is_anomaly)) = detector.process(packet, rate) {
            recovery_count += 1;
            if is_anomaly {
                recovery_false_positives += 1;
            }
            if recovery_count <= 5 || recovery_count % 50 == 0 {
                println!(
                    "  Recovery {}: pattern_sim={:.3}, rate_sim={:.3} {}",
                    recovery_count,
                    pattern_sim,
                    rate_sim,
                    if is_anomaly { "⚠️" } else { "✓" }
                );
            }
        }
    }

    println!(
        "\nRecovery phase: {} false positives ({:.1}% FPR)",
        recovery_false_positives,
        100.0 * recovery_false_positives as f64 / recovery_count.max(1) as f64
    );

    // Summary
    println!("\n=================================================================");
    println!("SUMMARY - WALKABLE TRAIT BENEFITS");
    println!("=================================================================");
    println!("✓ Zero string serialization (no JSON building)");
    println!("✓ Type-safe: struct fields are checked at compile time");
    println!("✓ Trait-based dispatch: zero runtime cost in Rust");
    println!("✓ Extensible: any struct can implement Walkable");
    println!(
        "✓ False positive rate: {:.1}%",
        100.0 * normal_false_positives as f64 / normal_detections.max(1) as f64
    );
    println!(
        "✓ Detection rate: {:.1}%",
        100.0 * true_positives as f64 / attack_detections.max(1) as f64
    );
}
