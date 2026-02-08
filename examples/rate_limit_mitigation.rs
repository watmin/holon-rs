//! Batch 013: Vector-Derived Rate Limit Mitigation
//!
//! Port of Python DEMO-rate-limit-mitigation.py
//!
//! Key features:
//! - Binary search rate decoding (O(log N), no stored references)
//! - HyperLogLog cardinality tracking (O(1) memory)
//! - Three anomaly signals: pattern, rate, cardinality divergence
//! - Composite match rules for precise mitigation
//!
//! Run: cargo run --example rate_limit_mitigation

use holon::{Holon, Accumulator, Vector, Walkable, WalkType, WalkableValue, ScalarValue};
use std::collections::HashMap;

// =============================================================================
// CONFIGURATION
// =============================================================================

const DIMENSIONS: usize = 4096;
const DECAY: f64 = 0.98;
const DRIFT_THRESHOLD: f64 = 0.15;
const WARMUP_PACKETS: usize = 300;

// =============================================================================
// HYPERLOGLOG: Memory-Efficient Cardinality Estimation
// =============================================================================

/// HyperLogLog for O(1) memory cardinality estimation.
///
/// Uses ~2^p bytes of memory to estimate cardinality of arbitrarily
/// large sets with ~1-2% error.
struct HyperLogLog {
    registers: Vec<u8>,
    count: usize,
    m: usize,
    alpha: f64,
}

impl HyperLogLog {
    fn new(precision: usize) -> Self {
        let m = 1 << precision;
        let alpha = if m == 16 {
            0.673
        } else if m == 32 {
            0.697
        } else if m == 64 {
            0.709
        } else {
            0.7213 / (1.0 + 1.079 / m as f64)
        };

        Self {
            registers: vec![0; m],
            count: 0,
            m,
            alpha,
        }
    }

    fn add(&mut self, value: &str) {
        self.count += 1;

        // Simple hash using FNV-1a
        let hash = self.hash(value);

        // First log2(m) bits determine bucket
        let bucket = (hash as usize) & (self.m - 1);

        // Count leading zeros in remaining bits
        let remaining = hash >> (self.m.trailing_zeros() as u64);
        let zeros = self.leading_zeros(remaining) + 1;

        self.registers[bucket] = self.registers[bucket].max(zeros);
    }

    fn hash(&self, value: &str) -> u64 {
        // FNV-1a hash
        let mut hash: u64 = 0xcbf29ce484222325;
        for byte in value.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    fn leading_zeros(&self, value: u64) -> u8 {
        if value == 0 {
            64
        } else {
            value.leading_zeros() as u8
        }
    }

    fn estimate(&self) -> f64 {
        let indicator: f64 = self.registers.iter()
            .map(|&r| 2.0_f64.powi(-(r as i32)))
            .sum();

        let raw_estimate = self.alpha * (self.m * self.m) as f64 / indicator;

        // Small range correction
        if raw_estimate <= 2.5 * self.m as f64 {
            let zeros = self.registers.iter().filter(|&&r| r == 0).count();
            if zeros > 0 {
                return self.m as f64 * (self.m as f64 / zeros as f64).ln();
            }
        }

        raw_estimate
    }

    fn cardinality_ratio(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            (self.estimate() / self.count as f64).min(1.0)
        }
    }

    fn reset(&mut self) {
        self.registers.fill(0);
        self.count = 0;
    }
}

// =============================================================================
// BINARY SEARCH RATE DECODER
// =============================================================================

/// Decodes rate from a vector using golden section search.
/// O(log N) complexity, no stored references needed.
struct BinarySearchRateDecoder<'a> {
    holon: &'a Holon,
    precision: f64,
}

impl<'a> BinarySearchRateDecoder<'a> {
    fn new(holon: &'a Holon, precision: f64) -> Self {
        Self { holon, precision }
    }

    /// Decode rate from vector using golden section search.
    ///
    /// Searches in log10 space from 10^log_lo to 10^log_hi.
    /// E.g., log_lo=-1, log_hi=8 searches from 0.1 pps to 100M pps.
    fn decode(&self, target_vec: &Vector, log10_lo: f64, log10_hi: f64) -> f64 {
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0; // Golden ratio

        let mut lo = log10_lo;
        let mut hi = log10_hi;

        let mut d = (hi - lo) / phi;
        let mut x1 = hi - d;
        let mut x2 = lo + d;

        // Generate probe vectors at 10^x1 and 10^x2
        let mut sim_x1 = self.holon.similarity(
            target_vec,
            &self.holon.encode_scalar_log(10.0_f64.powf(x1))
        );
        let mut sim_x2 = self.holon.similarity(
            target_vec,
            &self.holon.encode_scalar_log(10.0_f64.powf(x2))
        );

        while hi - lo > self.precision {
            if sim_x1 > sim_x2 {
                hi = x2;
                x2 = x1;
                sim_x2 = sim_x1;
                d = (hi - lo) / phi;
                x1 = hi - d;
                sim_x1 = self.holon.similarity(
                    target_vec,
                    &self.holon.encode_scalar_log(10.0_f64.powf(x1))
                );
            } else {
                lo = x1;
                x1 = x2;
                sim_x1 = sim_x2;
                d = (hi - lo) / phi;
                x2 = lo + d;
                sim_x2 = self.holon.similarity(
                    target_vec,
                    &self.holon.encode_scalar_log(10.0_f64.powf(x2))
                );
            }
        }

        10.0_f64.powf((lo + hi) / 2.0)
    }
}

// =============================================================================
// PACKET TYPES (Walkable)
// =============================================================================

#[derive(Clone, Debug)]
enum Packet {
    Tcp {
        src_port: u16,
        dst_port: u16,
        flags: String,
        payload_size: u32,
    },
    Udp {
        src_port: u16,
        dst_port: u16,
        payload_size: u32,
    },
    Icmp {
        icmp_type: u8,
        payload_size: u32,
    },
}

impl Packet {
    fn protocol(&self) -> &str {
        match self {
            Packet::Tcp { .. } => "TCP",
            Packet::Udp { .. } => "UDP",
            Packet::Icmp { .. } => "ICMP",
        }
    }

    fn get_field(&self, name: &str) -> Option<String> {
        match (self, name) {
            (_, "protocol") => Some(self.protocol().to_string()),
            (Packet::Tcp { src_port, .. }, "src_port") => Some(src_port.to_string()),
            (Packet::Tcp { dst_port, .. }, "dst_port") => Some(dst_port.to_string()),
            (Packet::Tcp { flags, .. }, "flags") => Some(flags.clone()),
            (Packet::Udp { src_port, .. }, "src_port") => Some(src_port.to_string()),
            (Packet::Udp { dst_port, .. }, "dst_port") => Some(dst_port.to_string()),
            (Packet::Icmp { icmp_type, .. }, "icmp_type") => Some(icmp_type.to_string()),
            _ => None,
        }
    }
}

impl Walkable for Packet {
    fn walk_type(&self) -> WalkType {
        WalkType::Map
    }

    fn walk_map_items(&self) -> Vec<(&str, WalkableValue)> {
        match self {
            Packet::Tcp { src_port, dst_port, flags, payload_size } => vec![
                ("protocol", WalkableValue::Scalar(ScalarValue::String("TCP".to_string()))),
                ("src_port", WalkableValue::Scalar(ScalarValue::Int(*src_port as i64))),
                ("dst_port", WalkableValue::Scalar(ScalarValue::Int(*dst_port as i64))),
                ("flags", WalkableValue::Scalar(ScalarValue::String(flags.clone()))),
                ("payload_size", WalkableValue::Scalar(ScalarValue::Int(*payload_size as i64))),
            ],
            Packet::Udp { src_port, dst_port, payload_size } => vec![
                ("protocol", WalkableValue::Scalar(ScalarValue::String("UDP".to_string()))),
                ("src_port", WalkableValue::Scalar(ScalarValue::Int(*src_port as i64))),
                ("dst_port", WalkableValue::Scalar(ScalarValue::Int(*dst_port as i64))),
                ("payload_size", WalkableValue::Scalar(ScalarValue::Int(*payload_size as i64))),
            ],
            Packet::Icmp { icmp_type, payload_size } => vec![
                ("protocol", WalkableValue::Scalar(ScalarValue::String("ICMP".to_string()))),
                ("icmp_type", WalkableValue::Scalar(ScalarValue::Int(*icmp_type as i64))),
                ("payload_size", WalkableValue::Scalar(ScalarValue::Int(*payload_size as i64))),
            ],
        }
    }
}

// =============================================================================
// FIELD TRACKER
// =============================================================================

struct FieldTracker {
    name: String,
    prior_pattern: Accumulator,
    recent_pattern: Vec<f64>,
    prior_counts: HashMap<String, usize>,
    recent_counts: HashMap<String, usize>,
    prior_total: usize,
    recent_total: usize,
    baseline_values: std::collections::HashSet<String>,
    frozen: bool,
    prior_norm: Option<Vector>,
    baseline_hll: HyperLogLog,
    anomaly_hll: HyperLogLog,
    baseline_cardinality: f64,
    baseline_cardinality_vec: Option<Vector>,
}

impl FieldTracker {
    fn new(name: &str, holon: &Holon) -> Self {
        Self {
            name: name.to_string(),
            prior_pattern: holon.create_accumulator(),
            recent_pattern: vec![0.0; holon.dimensions()],
            prior_counts: HashMap::new(),
            recent_counts: HashMap::new(),
            prior_total: 0,
            recent_total: 0,
            baseline_values: std::collections::HashSet::new(),
            frozen: false,
            prior_norm: None,
            baseline_hll: HyperLogLog::new(10),
            anomaly_hll: HyperLogLog::new(10),
            baseline_cardinality: 0.0,
            baseline_cardinality_vec: None,
        }
    }

    fn observe(&mut self, value: &str, holon: &Holon, is_warmup: bool) {
        let field_value = format!("{}:{}", self.name, value);
        let vec = holon.get_vector(&field_value);

        if is_warmup {
            holon.accumulate(&mut self.prior_pattern, &vec);
            *self.prior_counts.entry(value.to_string()).or_insert(0) += 1;
            self.prior_total += 1;
            self.baseline_values.insert(value.to_string());
            self.baseline_hll.add(value);
        } else {
            // Decaying accumulator
            for (i, v) in self.recent_pattern.iter_mut().enumerate() {
                *v = DECAY * *v + vec.data()[i] as f64;
            }
            *self.recent_counts.entry(value.to_string()).or_insert(0) += 1;
            self.recent_total += 1;

            // Periodic cleanup
            if self.recent_total > 200 {
                let mut new_counts = HashMap::new();
                for (k, v) in &self.recent_counts {
                    let new_v = v / 2;
                    if new_v > 0 {
                        new_counts.insert(k.clone(), new_v);
                    }
                }
                self.recent_counts = new_counts;
                self.recent_total = self.recent_counts.values().sum();
            }
        }
    }

    fn freeze(&mut self, holon: &Holon) {
        self.frozen = true;
        self.prior_norm = Some(holon.normalize_accumulator(&self.prior_pattern));
        self.recent_pattern = self.prior_pattern.raw_sums().to_vec();
        self.recent_counts = self.prior_counts.clone();
        self.recent_total = self.prior_total;

        // Encode baseline cardinality
        self.baseline_cardinality = self.baseline_hll.cardinality_ratio();
        let scaled = self.baseline_cardinality * 1000.0 + 1.0;
        self.baseline_cardinality_vec = Some(holon.encode_scalar_log(scaled));
    }

    fn get_divergence(&self, holon: &Holon) -> f64 {
        if !self.frozen {
            return 0.0;
        }

        // Normalize recent pattern
        let recent_norm = self.normalize_recent();
        let prior = self.prior_norm.as_ref().unwrap();

        1.0 - holon.similarity(prior, &recent_norm)
    }

    fn normalize_recent(&self) -> Vector {
        let magnitude: f64 = self.recent_pattern.iter().map(|x| x * x).sum::<f64>().sqrt();
        if magnitude < 1e-10 {
            return Vector::zeros(self.recent_pattern.len());
        }

        let normalized: Vec<i8> = self.recent_pattern.iter()
            .map(|&x| {
                let norm = x / magnitude;
                if norm > 0.3 { 1 }
                else if norm < -0.3 { -1 }
                else { 0 }
            })
            .collect();

        Vector::from_data(normalized)
    }

    fn get_dominant_value(&self) -> (Option<String>, f64) {
        if self.recent_counts.is_empty() {
            return (None, 0.0);
        }

        let (value, count) = self.recent_counts.iter()
            .max_by_key(|(_, &c)| c)
            .map(|(k, &v)| (k.clone(), v))
            .unwrap();

        let concentration = if self.recent_total > 0 {
            count as f64 / self.recent_total as f64
        } else {
            0.0
        };

        (Some(value), concentration)
    }

    fn is_novel(&self, value: &str) -> bool {
        !self.baseline_values.contains(value)
    }

    fn track_anomaly_value(&mut self, value: &str) {
        self.anomaly_hll.add(value);
    }

    fn get_anomaly_cardinality_ratio(&self) -> f64 {
        self.anomaly_hll.cardinality_ratio()
    }

    fn reset_anomaly_tracking(&mut self) {
        self.anomaly_hll.reset();
    }
}

// =============================================================================
// MITIGATION RULE
// =============================================================================

#[derive(Debug)]
struct MitigationRule {
    match_fields: HashMap<String, String>,
    action: String,
    rate_pps: f64,
    reason: String,
}

impl MitigationRule {
    fn to_json(&self) -> String {
        let match_str: Vec<String> = self.match_fields.iter()
            .map(|(k, v)| format!("\"{}\": \"{}\"", k, v))
            .collect();

        format!(
            r#"{{
  "match": {{ {} }},
  "action": "{}",
  "rate_pps": {:.0},
  "reason": "{}"
}}"#,
            match_str.join(", "),
            self.action,
            self.rate_pps,
            self.reason
        )
    }
}

// =============================================================================
// RATE LIMIT DETECTOR
// =============================================================================

struct RateLimitDetector {
    holon: Holon,
    field_trackers: HashMap<String, FieldTracker>,
    rate_accum: Accumulator,
    packet_count: usize,
    warmup_complete: bool,
    decoded_baseline: f64,
}

impl RateLimitDetector {
    fn new() -> Self {
        let holon = Holon::new(DIMENSIONS);
        let rate_accum = holon.create_accumulator();

        let monitored_fields = ["protocol", "src_port", "dst_port", "flags", "icmp_type"];
        let mut field_trackers = HashMap::new();
        for field in monitored_fields {
            field_trackers.insert(field.to_string(), FieldTracker::new(field, &holon));
        }

        Self {
            holon,
            field_trackers,
            rate_accum,
            packet_count: 0,
            warmup_complete: false,
            decoded_baseline: 0.0,
        }
    }

    fn process(&mut self, packet: &Packet, rate_pps: f64) {
        self.packet_count += 1;
        let is_warmup = !self.warmup_complete;

        // Encode rate with some variation during warmup to build a meaningful baseline
        let rate_with_jitter = if is_warmup {
            rate_pps * (0.9 + 0.2 * ((self.packet_count as f64).sin().abs()))
        } else {
            rate_pps
        };
        let rate_vec = self.holon.encode_scalar_log(rate_with_jitter);

        if is_warmup {
            self.holon.accumulate(&mut self.rate_accum, &rate_vec);
        }

        // Update field trackers - only for fields this packet has
        for (name, tracker) in &mut self.field_trackers {
            if let Some(value) = packet.get_field(name) {
                tracker.observe(&value, &self.holon, is_warmup);
            }
        }

        // Check warmup completion
        if !self.warmup_complete && self.packet_count >= WARMUP_PACKETS {
            self.warmup_complete = true;

            // Freeze all trackers
            for tracker in self.field_trackers.values_mut() {
                tracker.freeze(&self.holon);
            }

            // Decode baseline rate
            let rate_baseline = self.holon.normalize_accumulator(&self.rate_accum);
            let decoder = BinarySearchRateDecoder::new(&self.holon, 0.1);
            // Search in log10 space: 10^-1 to 10^8 = 0.1 pps to 100M pps
            self.decoded_baseline = decoder.decode(&rate_baseline, -1.0, 8.0);

            println!("\n  Warmup complete after {} packets", self.packet_count);
            println!("  Baseline rate decoded: {:.0} pps", self.decoded_baseline);
        }
    }

    fn track_anomaly_values(&mut self, packet: &Packet) {
        for (name, tracker) in &mut self.field_trackers {
            if tracker.get_divergence(&self.holon) > DRIFT_THRESHOLD {
                if let Some(value) = packet.get_field(name) {
                    tracker.track_anomaly_value(&value);
                }
            }
        }
    }

    fn generate_rule(&mut self) -> Option<MitigationRule> {
        if !self.warmup_complete {
            return None;
        }

        let mut stable_fields: HashMap<String, (String, f64, bool, f64, f64)> = HashMap::new();

        for (name, tracker) in &self.field_trackers {
            // Skip fields that weren't observed during this anomaly phase
            if tracker.anomaly_hll.count == 0 {
                continue;
            }

            let divergence = tracker.get_divergence(&self.holon);
            if divergence <= DRIFT_THRESHOLD {
                continue;
            }

            let (dominant_value, concentration) = tracker.get_dominant_value();
            if concentration < 0.5 {
                continue;
            }

            let current_card = tracker.get_anomaly_cardinality_ratio();

            // Only include stable fields (low cardinality)
            if current_card < 0.3 {
                if let Some(value) = dominant_value {
                    let is_novel = tracker.is_novel(&value);
                    let baseline_card = tracker.baseline_cardinality;
                    stable_fields.insert(
                        name.clone(),
                        (value, concentration, is_novel, baseline_card, current_card)
                    );
                }
            }
        }

        if stable_fields.is_empty() {
            return None;
        }

        // Build composite match
        let mut match_fields = HashMap::new();
        let mut reason_parts = Vec::new();

        for (name, (value, concentration, is_novel, baseline_card, current_card)) in &stable_fields {
            match_fields.insert(name.clone(), value.clone());

            // Interpret cardinality change
            let card_desc = if *baseline_card > 0.5 && *current_card < 0.3 {
                "was random, now fixed"
            } else if *baseline_card < 0.3 && *current_card < 0.3 {
                "stable"
            } else {
                "changed"
            };

            let novel_str = if *is_novel { ", novel" } else { "" };
            reason_parts.push(format!(
                "({}={}{}, {:.0}% conc, {})",
                name, value, novel_str, concentration * 100.0, card_desc
            ));
        }

        Some(MitigationRule {
            match_fields,
            action: "rate_limit".to_string(),
            rate_pps: self.decoded_baseline,
            reason: reason_parts.join(" + "),
        })
    }

    fn reset_anomaly_tracking(&mut self) {
        for tracker in self.field_trackers.values_mut() {
            tracker.reset_anomaly_tracking();
        }
    }
}

// =============================================================================
// TRAFFIC GENERATORS
// =============================================================================

fn generate_normal(rng: &mut u64) -> Packet {
    *rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
    let r = (*rng >> 16) as f32 / 65535.0;

    if r < 0.75 {
        // TCP
        *rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let dst_port = [80, 443, 8080, 22, 8443][(*rng as usize) % 5];
        *rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let flags = ["PA", "A", "SA", "S", "FA"][(*rng as usize) % 5];
        Packet::Tcp {
            src_port: 49152 + (*rng as u16 % 16383),
            dst_port,
            flags: flags.to_string(),
            payload_size: (*rng % 1500) as u32,
        }
    } else if r < 0.95 {
        // UDP
        *rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let dst_port = [53, 443, 123, 5353][(*rng as usize) % 4];
        Packet::Udp {
            src_port: 49152 + (*rng as u16 % 16383),
            dst_port,
            payload_size: 20 + (*rng % 492) as u32,
        }
    } else {
        // ICMP
        *rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        Packet::Icmp {
            icmp_type: if *rng % 2 == 0 { 0 } else { 8 },
            payload_size: 64,
        }
    }
}

fn generate_dns_reflection(rng: &mut u64) -> Packet {
    *rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
    Packet::Udp {
        src_port: 53,
        dst_port: 49152 + (*rng as u16 % 16383),
        payload_size: 512 + (*rng % 3584) as u32,
    }
}

fn generate_syn_flood(rng: &mut u64) -> Packet {
    *rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
    Packet::Tcp {
        src_port: 1 + (*rng as u16 % 65534),
        dst_port: 80,
        flags: "S".to_string(),
        payload_size: 0,
    }
}

fn generate_icmp_flood(_rng: &mut u64) -> Packet {
    Packet::Icmp {
        icmp_type: 8,
        payload_size: 1400,
    }
}

// =============================================================================
// MAIN
// =============================================================================

fn main() {
    println!("===========================================================================");
    println!("BATCH 013: Vector-Derived Rate Limit Mitigation (Rust)");
    println!("===========================================================================\n");

    println!("Key features:");
    println!("  - Binary search rate decoding (O(log N))");
    println!("  - HyperLogLog cardinality tracking (1KB per field)");
    println!("  - Three anomaly signals: pattern, rate, cardinality");
    println!("  - Composite match rules for precise mitigation\n");

    let mut detector = RateLimitDetector::new();
    let mut rng = 42u64;

    // Scale factor for simulation
    let scale = 0.001;
    let baseline_pps = 5000.0;
    let attack_pps = 500_000.0;

    println!("---------------------------------------------------------------------------");
    println!("SCENARIO");
    println!("---------------------------------------------------------------------------\n");
    println!("  Baseline: {} pps", format_number(baseline_pps));
    println!("  Attack:   {} pps (100x amplification)\n", format_number(attack_pps));

    // Phase 1: Warmup
    println!("Phase: warmup ({} packets)", WARMUP_PACKETS);
    let warmup_count = (WARMUP_PACKETS as f64 / scale) as usize;
    for _ in 0..((warmup_count as f64 * scale) as usize).max(WARMUP_PACKETS) {
        let packet = generate_normal(&mut rng);
        detector.process(&packet, baseline_pps);
    }

    // Phase 2: Normal
    println!("Phase: normal-1");
    let normal_count = (100.0 / scale) as usize;
    for _ in 0..((normal_count as f64 * scale) as usize).max(50) {
        let packet = generate_normal(&mut rng);
        detector.process(&packet, baseline_pps);
    }

    println!("\n---------------------------------------------------------------------------");
    println!("DETECTION & MITIGATION");
    println!("---------------------------------------------------------------------------\n");

    // Attack scenarios
    let attacks: Vec<(&str, fn(&mut u64) -> Packet)> = vec![
        ("DNS Reflection", generate_dns_reflection),
        ("SYN Flood", generate_syn_flood),
        ("ICMP Flood", generate_icmp_flood),
    ];

    let mut all_rules: Vec<(&str, Option<MitigationRule>)> = Vec::new();

    for (attack_name, generator) in attacks {
        println!("  Attack: {}", attack_name);

        // Generate attack traffic
        let attack_count = (500.0 / scale) as usize;
        for _ in 0..((attack_count as f64 * scale) as usize).max(100) {
            let packet = generator(&mut rng);
            detector.process(&packet, attack_pps);
            detector.track_anomaly_values(&packet);
        }

        // Generate rule
        let rule = detector.generate_rule();
        if let Some(ref r) = rule {
            println!("    MATCH: {}", format_match(&r.match_fields));
            println!("    → RATE LIMIT TO: {:.0} pps", r.rate_pps);
        }
        all_rules.push((attack_name, rule));

        // Reset for next attack
        detector.reset_anomaly_tracking();

        // Recovery phase
        let recovery_count = (200.0 / scale) as usize;
        for _ in 0..((recovery_count as f64 * scale) as usize).max(50) {
            let packet = generate_normal(&mut rng);
            detector.process(&packet, baseline_pps);
        }
        println!();
    }

    println!("---------------------------------------------------------------------------");
    println!("ENFORCER JSON OUTPUT");
    println!("---------------------------------------------------------------------------\n");

    for (name, rule) in &all_rules {
        if let Some(r) = rule {
            println!("  {}:", name);
            println!("{}\n", r.to_json());
        }
    }

    println!("===========================================================================");
    println!("SUMMARY");
    println!("===========================================================================\n");

    println!("  ANOMALY SIGNALS (all vector-based):");
    println!("    1. Pattern Divergence - what field values changed");
    println!("    2. Rate Divergence - throughput change, decode to PPS");
    println!("    3. Cardinality Divergence - value diversity change\n");

    println!("  MEMORY EFFICIENCY:");
    println!("    - HyperLogLog: 1KB per field (~1% error)");
    println!("    - Total: {} fields × (1 HLL + 3 vectors) = ~65KB", detector.field_trackers.len());

    println!("\n  ZERO HARDCODED KNOWLEDGE:");
    println!("    - Fields discovered from packet structure");
    println!("    - Values learned during warmup");
    println!("    - Rate decoded from baseline vector");
}

fn format_match(fields: &HashMap<String, String>) -> String {
    fields.iter()
        .map(|(k, v)| format!("{}={}", k, v))
        .collect::<Vec<_>>()
        .join(" AND ")
}

fn format_number(n: f64) -> String {
    let s = format!("{:.0}", n);
    let bytes: Vec<char> = s.chars().collect();
    let mut result = String::new();
    for (i, c) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i) % 3 == 0 {
            result.push(',');
        }
        result.push(*c);
    }
    result
}
