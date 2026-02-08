//! Walkable Zero-Hardcode Detection Demo
//!
//! Same as zero_hardcode_detection.rs but using the Walkable trait.
//! Demonstrates zero-serialization with typed packet structs.
//!
//! Run: cargo run --example walkable_detection

use holon::{Holon, ScalarValue, WalkType, Walkable, WalkableValue};
use std::collections::VecDeque;

// =============================================================================
// CONFIGURATION
// =============================================================================

const DIMENSIONS: usize = 4096;
const WARMUP_PACKETS: usize = 400;

// =============================================================================
// FROZEN Z-SCORE BASELINE
// =============================================================================

struct FrozenBaseline {
    samples: Vec<f64>,
    frozen: bool,
    mean: f64,
    std: f64,
}

impl FrozenBaseline {
    fn new() -> Self {
        Self {
            samples: Vec::new(),
            frozen: false,
            mean: 0.0,
            std: 1.0,
        }
    }

    fn observe(&mut self, value: f64) {
        if !self.frozen {
            self.samples.push(value);
        }
    }

    fn freeze(&mut self) {
        self.frozen = true;
        if !self.samples.is_empty() {
            let n = self.samples.len() as f64;
            self.mean = self.samples.iter().sum::<f64>() / n;

            if self.samples.len() > 1 {
                let variance: f64 = self
                    .samples
                    .iter()
                    .map(|x| (x - self.mean).powi(2))
                    .sum::<f64>()
                    / n;
                self.std = variance.sqrt().max(0.02);
            } else {
                self.std = 0.05;
            }
        }
    }

    fn z_score(&self, value: f64) -> f64 {
        (value - self.mean) / self.std
    }
}

// =============================================================================
// TYPED PACKET STRUCTS WITH WALKABLE
// =============================================================================

/// TCP Packet - type-safe, no optional fields for required data
#[derive(Clone, Debug)]
struct TcpPacket {
    src_port: u16,
    dst_port: u16,
    flags: String,
    payload_size: u32,
}

impl Walkable for TcpPacket {
    fn walk_type(&self) -> WalkType {
        WalkType::Map
    }

    fn walk_map_items(&self) -> Vec<(&str, WalkableValue)> {
        vec![
            (
                "protocol",
                WalkableValue::Scalar(ScalarValue::String("TCP".to_string())),
            ),
            ("src_port", (self.src_port as i64).to_walkable_value()),
            ("dst_port", (self.dst_port as i64).to_walkable_value()),
            ("flags", self.flags.to_walkable_value()),
            ("payload_size", (self.payload_size as i64).to_walkable_value()),
        ]
    }
}

/// UDP Packet
#[derive(Clone, Debug)]
struct UdpPacket {
    src_port: u16,
    dst_port: u16,
    payload_size: u32,
}

impl Walkable for UdpPacket {
    fn walk_type(&self) -> WalkType {
        WalkType::Map
    }

    fn walk_map_items(&self) -> Vec<(&str, WalkableValue)> {
        vec![
            (
                "protocol",
                WalkableValue::Scalar(ScalarValue::String("UDP".to_string())),
            ),
            ("src_port", (self.src_port as i64).to_walkable_value()),
            ("dst_port", (self.dst_port as i64).to_walkable_value()),
            ("payload_size", (self.payload_size as i64).to_walkable_value()),
        ]
    }
}

/// ICMP Packet
#[derive(Clone, Debug)]
struct IcmpPacket {
    icmp_type: u8,
    payload_size: u32,
}

impl Walkable for IcmpPacket {
    fn walk_type(&self) -> WalkType {
        WalkType::Map
    }

    fn walk_map_items(&self) -> Vec<(&str, WalkableValue)> {
        vec![
            (
                "protocol",
                WalkableValue::Scalar(ScalarValue::String("ICMP".to_string())),
            ),
            ("icmp_type", (self.icmp_type as i64).to_walkable_value()),
            ("payload_size", (self.payload_size as i64).to_walkable_value()),
        ]
    }
}

/// Enum for different packet types - demonstrates type-safe packet handling
#[derive(Clone, Debug)]
enum Packet {
    Tcp(TcpPacket),
    Udp(UdpPacket),
    Icmp(IcmpPacket),
}

impl Packet {
    fn field_summary(&self) -> String {
        match self {
            Packet::Tcp(p) => {
                format!(
                    "protocol=TCP, src_port={}, dst_port={}, flags={}",
                    p.src_port, p.dst_port, p.flags
                )
            }
            Packet::Udp(p) => {
                format!(
                    "protocol=UDP, src_port={}, dst_port={}",
                    p.src_port, p.dst_port
                )
            }
            Packet::Icmp(p) => {
                format!("protocol=ICMP, icmp_type={}", p.icmp_type)
            }
        }
    }
}

/// Implement Walkable for the Packet enum - dispatches to inner type
impl Walkable for Packet {
    fn walk_type(&self) -> WalkType {
        WalkType::Map
    }

    fn walk_map_items(&self) -> Vec<(&str, WalkableValue)> {
        match self {
            Packet::Tcp(p) => p.walk_map_items(),
            Packet::Udp(p) => p.walk_map_items(),
            Packet::Icmp(p) => p.walk_map_items(),
        }
    }
}

// =============================================================================
// WALKABLE DETECTOR
// =============================================================================

struct WalkableDetector {
    holon: Holon,
    warmup_packets: usize,
    packet_count: usize,
    warmup_complete: bool,

    rate_accum: holon::Accumulator,
    rate_norm: Option<holon::Vector>,
    rate_baseline: FrozenBaseline,

    pattern_accum: holon::Accumulator,
    pattern_norm: Option<holon::Vector>,
    pattern_baseline: FrozenBaseline,

    in_anomaly_state: bool,
    anomaly_window: VecDeque<u8>,
    consecutive_normal: usize,
}

struct DetectionResult {
    packet_num: usize,
    is_anomalous: bool,
    explanation: Option<String>,
}

impl WalkableDetector {
    fn new(warmup_packets: usize) -> Self {
        let holon = Holon::new(DIMENSIONS);
        let rate_accum = holon.create_accumulator();
        let pattern_accum = holon.create_accumulator();

        let mut anomaly_window = VecDeque::with_capacity(15);
        anomaly_window.resize(15, 0);

        Self {
            holon,
            warmup_packets,
            packet_count: 0,
            warmup_complete: false,
            rate_accum,
            rate_norm: None,
            rate_baseline: FrozenBaseline::new(),
            pattern_accum,
            pattern_norm: None,
            pattern_baseline: FrozenBaseline::new(),
            in_anomaly_state: false,
            anomaly_window,
            consecutive_normal: 0,
        }
    }

    fn process(&mut self, packet: &Packet, pps: f64) -> DetectionResult {
        self.packet_count += 1;
        let is_warmup = self.packet_count <= self.warmup_packets;

        // =====================================================================
        // KEY: encode_walkable instead of encode_json!
        // Zero serialization - direct struct traversal
        // =====================================================================
        let packet_vec = self.holon.encode_walkable(packet);
        let rate_vec = self.holon.encode_scalar_log(pps);

        if is_warmup {
            self.warmup_phase(&packet_vec, &rate_vec)
        } else {
            self.detection_phase(packet, &packet_vec, &rate_vec, pps)
        }
    }

    fn warmup_phase(
        &mut self,
        packet_vec: &holon::Vector,
        rate_vec: &holon::Vector,
    ) -> DetectionResult {
        self.holon.accumulate(&mut self.rate_accum, rate_vec);
        self.holon.accumulate(&mut self.pattern_accum, packet_vec);

        if self.packet_count > 100 {
            let temp_rate = self.holon.normalize_accumulator(&self.rate_accum);
            let temp_pattern = self.holon.normalize_accumulator(&self.pattern_accum);

            let rate_sim = self.holon.similarity(rate_vec, &temp_rate);
            let pattern_sim = self.holon.similarity(packet_vec, &temp_pattern);

            self.rate_baseline.observe(rate_sim);
            self.pattern_baseline.observe(pattern_sim);
        }

        if self.packet_count == self.warmup_packets {
            self.freeze_baselines();
        }

        DetectionResult {
            packet_num: self.packet_count,
            is_anomalous: false,
            explanation: None,
        }
    }

    fn freeze_baselines(&mut self) {
        self.warmup_complete = true;
        self.rate_norm = Some(self.holon.normalize_accumulator(&self.rate_accum));
        self.pattern_norm = Some(self.holon.normalize_accumulator(&self.pattern_accum));
        self.rate_baseline.freeze();
        self.pattern_baseline.freeze();
    }

    fn detection_phase(
        &mut self,
        packet: &Packet,
        packet_vec: &holon::Vector,
        rate_vec: &holon::Vector,
        pps: f64,
    ) -> DetectionResult {
        let rate_norm = self.rate_norm.as_ref().unwrap();
        let pattern_norm = self.pattern_norm.as_ref().unwrap();

        let rate_sim = self.holon.similarity(rate_vec, rate_norm);
        let pattern_sim = self.holon.similarity(packet_vec, pattern_norm);

        let rate_z = self.rate_baseline.z_score(rate_sim);
        let pattern_z = self.pattern_baseline.z_score(pattern_sim);

        let rate_anomalous = rate_z < -2.5;
        let pattern_anomalous = pattern_z < -2.0;
        let rate_confirms = rate_z < -0.5;

        let raw_anomaly = rate_anomalous || (pattern_anomalous && rate_confirms);

        if raw_anomaly {
            self.consecutive_normal = 0;
        } else {
            self.consecutive_normal += 1;
        }

        if self.consecutive_normal >= 5 {
            for _ in 0..3 {
                self.anomaly_window.push_back(0);
                self.anomaly_window.pop_front();
            }
        } else {
            self.anomaly_window.push_back(if raw_anomaly { 1 } else { 0 });
            self.anomaly_window.pop_front();
        }

        let sum: u8 = self.anomaly_window.iter().sum();
        let fraction = sum as f64 / self.anomaly_window.len() as f64;

        if !self.in_anomaly_state {
            if fraction > 0.5 {
                self.in_anomaly_state = true;
            }
        } else if fraction < 0.2 {
            self.in_anomaly_state = false;
        }

        let is_anomalous = self.in_anomaly_state;

        let explanation = if rate_anomalous || pattern_anomalous {
            let mut parts = Vec::new();
            if rate_anomalous {
                parts.push(format!(
                    "Rate ({:.0} pps) is {:.1} std below baseline",
                    pps,
                    rate_z.abs()
                ));
            }
            if pattern_anomalous {
                parts.push(format!(
                    "Pattern [{}] is {:.1} std below baseline",
                    packet.field_summary(),
                    pattern_z.abs()
                ));
            }
            Some(parts.join("; "))
        } else {
            Some("Traffic matches learned baseline".to_string())
        };

        DetectionResult {
            packet_num: self.packet_count,
            is_anomalous,
            explanation,
        }
    }
}

// =============================================================================
// TRAFFIC SIMULATION
// =============================================================================

#[derive(Clone, Copy, PartialEq)]
enum Phase {
    Calm,
    Attack,
}

struct TimePhase {
    name: &'static str,
    duration_seconds: u32,
    packets_per_second: u32,
    phase_type: Phase,
    attack_type: Option<&'static str>,
    attack_fraction: f64,
}

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    fn random(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn randint(&mut self, min: u64, max: u64) -> u64 {
        min + (self.next() % (max - min + 1))
    }

    fn choice<T: Clone>(&mut self, items: &[T]) -> T {
        let idx = self.randint(0, items.len() as u64 - 1) as usize;
        items[idx].clone()
    }
}

fn gen_normal(rng: &mut Rng) -> Packet {
    let r = rng.random();
    if r < 0.8 {
        Packet::Tcp(TcpPacket {
            src_port: rng.randint(49152, 65535) as u16,
            dst_port: rng.choice(&[80, 443, 8080, 22]),
            flags: rng.choice(&["PA", "A", "SA", "S"]).to_string(),
            payload_size: rng.randint(0, 1500) as u32,
        })
    } else if r < 0.98 {
        Packet::Udp(UdpPacket {
            src_port: rng.randint(49152, 65535) as u16,
            dst_port: rng.choice(&[53, 443, 123]),
            payload_size: rng.randint(20, 512) as u32,
        })
    } else {
        Packet::Icmp(IcmpPacket {
            icmp_type: rng.choice(&[0, 8]),
            payload_size: 64,
        })
    }
}

fn gen_attack(rng: &mut Rng, attack_type: &str) -> Packet {
    match attack_type {
        "dns_reflection" => Packet::Udp(UdpPacket {
            src_port: 53,
            dst_port: rng.randint(49152, 65535) as u16,
            payload_size: rng.randint(256, 4096) as u32,
        }),
        "syn_flood" => Packet::Tcp(TcpPacket {
            src_port: rng.randint(1, 65535) as u16,
            dst_port: 80,
            flags: "S".to_string(),
            payload_size: 0,
        }),
        "ntp_amplification" => Packet::Udp(UdpPacket {
            src_port: 123,
            dst_port: rng.randint(49152, 65535) as u16,
            payload_size: rng.randint(468, 482) as u32,
        }),
        "udp_flood" => Packet::Udp(UdpPacket {
            src_port: rng.randint(1, 65535) as u16,
            dst_port: rng.randint(1, 65535) as u16,
            payload_size: rng.randint(0, 1400) as u32,
        }),
        _ => gen_normal(rng),
    }
}

// =============================================================================
// MAIN
// =============================================================================

fn main() {
    println!("{}", "=".repeat(75));
    println!("WALKABLE ZERO-HARDCODE DETECTION (Rust)");
    println!("{}", "=".repeat(75));
    println!(
        r#"
    WALKABLE BENEFITS:
    1. Type-safe packet structs (TcpPacket, UdpPacket, IcmpPacket)
    2. No JSON string building or parsing
    3. Zero-cost trait dispatch in Rust
    4. Compiler catches field mismatches

    Same detection quality as JSON version - just cleaner code!
"#
    );

    // Demo the Walkable API
    println!("{}", "-".repeat(75));
    println!("WALKABLE PACKET DEMO");
    println!("{}", "-".repeat(75));

    let holon = Holon::new(DIMENSIONS);

    let tcp = TcpPacket {
        src_port: 443,
        dst_port: 8080,
        flags: "PA".to_string(),
        payload_size: 1200,
    };

    let udp = UdpPacket {
        src_port: 53,
        dst_port: 49152,
        payload_size: 512,
    };

    let tcp_vec = holon.encode_walkable(&tcp);
    let udp_vec = holon.encode_walkable(&udp);

    println!("\n  TcpPacket {{ src_port: 443, dst_port: 8080, flags: \"PA\", payload_size: 1200 }}");
    println!("  UdpPacket {{ src_port: 53, dst_port: 49152, payload_size: 512 }}");
    println!("\n  Similarity: {:.3}", holon.similarity(&tcp_vec, &udp_vec));
    println!("  (Different protocols, different structure)");

    // Define attack scenario
    let timeline = vec![
        TimePhase {
            name: "warmup",
            duration_seconds: 600,
            packets_per_second: 100,
            phase_type: Phase::Calm,
            attack_type: None,
            attack_fraction: 0.0,
        },
        TimePhase {
            name: "DNS Attack",
            duration_seconds: 30,
            packets_per_second: 100000,
            phase_type: Phase::Attack,
            attack_type: Some("dns_reflection"),
            attack_fraction: 0.95,
        },
        TimePhase {
            name: "recovery-1",
            duration_seconds: 300,
            packets_per_second: 100,
            phase_type: Phase::Calm,
            attack_type: None,
            attack_fraction: 0.0,
        },
        TimePhase {
            name: "SYN Flood",
            duration_seconds: 45,
            packets_per_second: 80000,
            phase_type: Phase::Attack,
            attack_type: Some("syn_flood"),
            attack_fraction: 0.95,
        },
        TimePhase {
            name: "recovery-2",
            duration_seconds: 300,
            packets_per_second: 100,
            phase_type: Phase::Calm,
            attack_type: None,
            attack_fraction: 0.0,
        },
        TimePhase {
            name: "NTP Attack",
            duration_seconds: 30,
            packets_per_second: 100000,
            phase_type: Phase::Attack,
            attack_type: Some("ntp_amplification"),
            attack_fraction: 0.95,
        },
        TimePhase {
            name: "final",
            duration_seconds: 300,
            packets_per_second: 100,
            phase_type: Phase::Calm,
            attack_type: None,
            attack_fraction: 0.0,
        },
    ];

    println!("\n{}", "-".repeat(75));
    println!("ATTACK SIMULATION");
    println!("{}", "-".repeat(75));

    let scale = 0.005;
    let mut rng = Rng::new(42);

    println!("\n  Timeline:");
    println!(
        "  {:15} {:>10} {:12} {:>10}",
        "Phase", "PPS", "Type", "Packets"
    );
    println!("  {}", "-".repeat(50));

    for phase in &timeline {
        let scaled =
            (phase.duration_seconds as f64 * phase.packets_per_second as f64 * scale) as u32;
        let ptype = phase.attack_type.unwrap_or("normal");
        println!(
            "  {:15} {:>10} {:12} {:>10}",
            phase.name, phase.packets_per_second, ptype, scaled
        );
    }

    // Run detection
    println!("\n{}", "-".repeat(75));
    println!("DETECTION RESULTS");
    println!("{}", "-".repeat(75));

    let first_calm_packets = (timeline[0].duration_seconds as f64
        * timeline[0].packets_per_second as f64
        * scale) as usize;
    let warmup_packets = (first_calm_packets - 10).min(WARMUP_PACKETS);

    let mut detector = WalkableDetector::new(warmup_packets);

    struct PhaseResult {
        name: String,
        phase_type: Phase,
        packets: usize,
        detections: usize,
        detection_rate: f64,
        status: String,
    }

    let mut results: Vec<PhaseResult> = Vec::new();
    let mut sample_alerts: Vec<(usize, String, String)> = Vec::new();

    for phase in &timeline {
        let scaled_packets =
            ((phase.duration_seconds as f64 * phase.packets_per_second as f64 * scale) as usize)
                .max(1);
        let mut phase_detections = 0;

        for _ in 0..scaled_packets {
            let packet = if phase.phase_type == Phase::Attack
                && phase.attack_type.is_some()
                && rng.random() < phase.attack_fraction
            {
                gen_attack(&mut rng, phase.attack_type.unwrap())
            } else {
                gen_normal(&mut rng)
            };

            let result = detector.process(&packet, phase.packets_per_second as f64);

            if detector.warmup_complete {
                if result.is_anomalous {
                    phase_detections += 1;

                    if sample_alerts.len() < 5 {
                        if let Some(ref expl) = result.explanation {
                            if expl != "Traffic matches learned baseline" {
                                sample_alerts.push((
                                    result.packet_num,
                                    phase.name.to_string(),
                                    expl.clone(),
                                ));
                            }
                        }
                    }
                }
            }
        }

        let detection_rate = if scaled_packets > 0 {
            phase_detections as f64 / scaled_packets as f64
        } else {
            0.0
        };

        let status = if phase.name == "warmup" {
            "LEARNING"
        } else if phase.phase_type == Phase::Attack {
            if detection_rate > 0.5 {
                "DETECTED"
            } else {
                "MISSED"
            }
        } else if detection_rate < 0.05 {
            "CLEAN"
        } else {
            "FP"
        };

        results.push(PhaseResult {
            name: phase.name.to_string(),
            phase_type: phase.phase_type,
            packets: scaled_packets,
            detections: phase_detections,
            detection_rate,
            status: status.to_string(),
        });
    }

    println!(
        "\n  {:15} {:>10} {:>10} {:>10} {:>12}",
        "Phase", "Packets", "Detected", "Rate", "Status"
    );
    println!("  {}", "-".repeat(60));

    for r in &results {
        let marker = match r.status.as_str() {
            "DETECTED" => "✓",
            "CLEAN" => "✓",
            "LEARNING" => "○",
            "FP" => "⚠",
            "MISSED" => "✗",
            _ => "?",
        };
        println!(
            "  {:15} {:>10} {:>10} {:>9.0}% {} {}",
            r.name,
            r.packets,
            r.detections,
            r.detection_rate * 100.0,
            marker,
            r.status
        );
    }

    // Metrics
    let attack_phases: Vec<_> = results
        .iter()
        .filter(|r| r.phase_type == Phase::Attack)
        .collect();
    let calm_phases: Vec<_> = results
        .iter()
        .filter(|r| r.phase_type == Phase::Calm && r.status != "LEARNING")
        .collect();

    let attack_detected: usize = attack_phases.iter().map(|r| r.detections).sum();
    let attack_total: usize = attack_phases.iter().map(|r| r.packets).sum();
    let attack_recall = if attack_total > 0 {
        attack_detected as f64 / attack_total as f64
    } else {
        0.0
    };

    let fp: usize = calm_phases.iter().map(|r| r.detections).sum();
    let fp_total: usize = calm_phases.iter().map(|r| r.packets).sum();
    let fp_rate = if fp_total > 0 {
        fp as f64 / fp_total as f64
    } else {
        0.0
    };

    println!("  {}", "-".repeat(60));
    println!("  {:37} {:>9.0}%", "ATTACK RECALL", attack_recall * 100.0);
    println!("  {:37} {:>9.0}%", "FALSE POSITIVE RATE", fp_rate * 100.0);

    // Sample alerts
    println!("\n{}", "-".repeat(75));
    println!("SAMPLE ALERTS");
    println!("{}", "-".repeat(75));

    for (packet_num, phase_name, explanation) in sample_alerts.iter().take(3) {
        println!("\n  Packet #{} ({}):", packet_num, phase_name);
        println!("    {}", explanation);
    }

    // Summary
    println!("\n{}", "=".repeat(75));
    println!("SUMMARY: WALKABLE ZERO-HARDCODE DETECTION");
    println!("{}", "=".repeat(75));
    println!(
        r#"
    RESULTS:
    - Attack Recall: {:.0}%
    - False Positive Rate: {:.0}%

    WALKABLE BENEFITS:
    1. Type-safe packets: TcpPacket, UdpPacket, IcmpPacket
    2. Zero serialization: no JSON building or parsing
    3. Zero-cost trait dispatch: Rust compiles to optimal code
    4. Compiler-checked: field mismatches caught at compile time
    5. Extensible: add new packet types by implementing Walkable

    SAME DETECTION QUALITY AS JSON VERSION:
    - All techniques work: frozen z-scores, gated detection, fast recovery
    - Just cleaner, safer, faster code!
"#,
        attack_recall * 100.0,
        fp_rate * 100.0
    );
}
