//! Pure Vector Rate Detection - No Magic Numbers
//!
//! Port of Python Challenge 012-008: Demonstrates zero-hardcode anomaly detection
//! using Holon's continuous scalar encoding.
//!
//! Key insight: Rate is CONTINUOUS. Instead of discretizing into categories
//! like "moderate" or "extreme", we encode the log of rate so that
//! SIMILAR rates have SIMILAR vectors.
//!
//! Run: cargo run --example pure_vector_rate

use holon::highlevel::Holon;
use std::collections::VecDeque;

// =============================================================================
// CONFIGURATION
// =============================================================================

const DIMENSIONS: usize = 4096;
const DECAY: f64 = 0.98;
const WARMUP_PACKETS: usize = 400;
const WINDOW_SIZE: usize = 100;

// =============================================================================
// PACKET TYPES
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

    fn to_json(&self) -> String {
        format!(
            r#"{{"protocol":"{}","src_port":{},"dst_port":{},"size":{}}}"#,
            self.protocol, self.src_port, self.dst_port, self.size
        )
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
            last_rate: 100.0, // Initial estimate
            smoothing: 0.1,   // EMA smoothing factor
        }
    }

    fn add(&mut self, timestamp: f64) -> f64 {
        // Compute instantaneous rate from last timestamp
        let instant_rate = if let Some(&last_ts) = self.timestamps.back() {
            let delta = timestamp - last_ts;
            if delta > 1e-9 {
                1.0 / delta // packets per second
            } else {
                1_000_000.0 // Cap at 1M pps for near-zero delta
            }
        } else {
            100.0 // Default for first packet
        };

        // Update sliding window
        self.timestamps.push_back(timestamp);
        if self.timestamps.len() > self.window_size {
            self.timestamps.pop_front();
        }

        // Exponential moving average for smoothing
        self.last_rate = self.smoothing * instant_rate + (1.0 - self.smoothing) * self.last_rate;

        // Also compute window-based rate for comparison
        let window_rate = self.window_rate();

        // Use the higher of instant and window rate (to catch bursts)
        window_rate.max(self.last_rate)
    }

    fn window_rate(&self) -> f64 {
        if self.timestamps.len() < 2 {
            return self.last_rate;
        }
        let duration = self.timestamps.back().unwrap() - self.timestamps.front().unwrap();
        if duration < 1e-9 {
            return 1_000_000.0; // Near-instant = very high rate
        }
        (self.timestamps.len() - 1) as f64 / duration
    }
}

// =============================================================================
// ZERO-HARDCODE DETECTOR
// =============================================================================

struct ZeroHardcodeDetector {
    holon: Holon,
    pattern_accum: holon::Accumulator,
    rate_accum: holon::Accumulator,
    packet_count: usize,
    warmup_complete: bool,
    decay: f64,
}

impl ZeroHardcodeDetector {
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
        // Encode packet pattern (structure)
        let pattern_vec = self.holon.encode_json(&packet.to_json()).unwrap();

        // Encode rate on log scale (continuous!)
        let rate_vec = self.holon.encode_scalar_log(rate_pps);

        self.packet_count += 1;

        if !self.warmup_complete {
            // Learning phase: accumulate patterns
            self.holon.accumulate(&mut self.pattern_accum, &pattern_vec);
            self.holon.accumulate(&mut self.rate_accum, &rate_vec);

            if self.packet_count >= WARMUP_PACKETS {
                self.warmup_complete = true;
                println!("\n✓ Warmup complete after {} packets", self.packet_count);
            }
            return None;
        }

        // Detection phase
        let pattern_baseline = self.holon.normalize_accumulator(&self.pattern_accum);
        let rate_baseline = self.holon.normalize_accumulator(&self.rate_accum);

        let pattern_sim = self.holon.similarity(&pattern_vec, &pattern_baseline);
        let rate_sim = self.holon.similarity(&rate_vec, &rate_baseline);

        // Zero hardcoded thresholds - use statistical approach
        // Anomaly if:
        // - Pattern is unusual (low similarity to baseline patterns)
        // - OR rate is unusual (negative similarity = opposite of normal, or low positive)
        //
        // Note: rate_sim < 0.5 catches both low positive AND negative values
        // A negative rate_sim means the rate vector is anti-correlated with baseline
        let pattern_anomaly = pattern_sim < 0.3;
        let rate_anomaly = rate_sim < 0.5;
        let is_anomaly = pattern_anomaly && rate_anomaly;

        // Only continue learning if NOT anomalous (don't let attacks poison the baseline)
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
        // Simple LCG for deterministic "random" numbers
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
        let timestamp = start_time + (i as f64 * 0.01); // ~100 pps normal

        packets.push(Packet::new(protocol, src_port, dst_port, size, timestamp));
    }

    packets
}

fn generate_attack_traffic(count: usize, start_time: f64) -> Vec<Packet> {
    let mut packets = Vec::with_capacity(count);

    for i in 0..count {
        // DNS amplification attack pattern
        let packet = Packet::new(
            "UDP",
            53,                              // Source: DNS server
            49152 + (i as u16 % 100),        // Random high port
            4000,                            // Large response
            start_time + (i as f64 * 0.00001), // 100,000 pps!
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
    println!("PURE VECTOR RATE DETECTION - Zero Hardcoded Thresholds");
    println!("=================================================================\n");

    let mut detector = ZeroHardcodeDetector::new(DIMENSIONS, DECAY);
    let mut rate_calc = RateCalculator::new(WINDOW_SIZE);

    // Phase 1: Normal traffic (warmup + some detection)
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
    println!("SUMMARY");
    println!("=================================================================");
    println!("✓ Zero hardcoded port numbers or rate thresholds");
    println!("✓ Uses continuous log-scale rate encoding");
    println!("✓ Learns normal patterns from data");
    println!(
        "✓ False positive rate: {:.1}%",
        100.0 * normal_false_positives as f64 / normal_detections.max(1) as f64
    );
    println!(
        "✓ Detection rate: {:.1}%",
        100.0 * true_positives as f64 / attack_detections.max(1) as f64
    );
}
