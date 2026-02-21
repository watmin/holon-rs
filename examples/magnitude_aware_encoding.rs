//! BATCH 15: Magnitude-Aware Numeric Encoding
//!
//! Demonstrates $log and $linear markers for VSA/HDC operations.
//!
//! HYPOTHESIS: Log encoding clusters similar magnitudes while string encoding
//! treats all values as unrelated. This enables "find similar intensity"
//! queries without hard-coded thresholds.
//!
//! Run: cargo run --example magnitude_aware_encoding --release

use holon::highlevel::Holon;
use holon::kernel::Similarity;

fn print_header(title: &str) {
    println!();
    println!("{}", "=".repeat(70));
    println!(" {}", title);
    println!("{}", "=".repeat(70));
}

fn print_similarity_matrix(labels: &[&str], vectors: &[holon::Vector], title: &str) {
    println!("\n{}", title);
    println!("{}", "-".repeat(60));

    // Header row
    print!("{:>10}", "");
    for label in labels {
        print!("{:>10}", label);
    }
    println!();

    // Matrix rows
    for (i, label) in labels.iter().enumerate() {
        print!("{:>10}", label);
        for j in 0..vectors.len() {
            let sim = Similarity::cosine(&vectors[i], &vectors[j]);
            print!("{:>10.3}", sim);
        }
        println!();
    }
}

/// PART 1: String vs Log Encoding
fn demo_string_vs_log_encoding(holon: &Holon) {
    print_header("PART 1: String vs Log Encoding");

    // Test values spanning multiple orders of magnitude
    let values = [100, 200, 1000, 10000, 100000];
    let labels: Vec<&str> = vec!["100", "200", "1000", "10000", "100000"];

    // String encoding (default behavior - numbers as strings)
    println!("\nString Encoding (default): Numbers → random vectors, no magnitude relationship");
    let string_vectors: Vec<_> = values
        .iter()
        .map(|v| {
            holon
                .encode_json(&format!(r#"{{"rate": {}}}"#, v))
                .unwrap()
        })
        .collect();
    print_similarity_matrix(&labels, &string_vectors, "String Encoding Similarity Matrix");

    // Log encoding
    println!("\n\nLog Encoding ($log): Similar magnitudes → similar vectors");
    let log_vectors: Vec<_> = values
        .iter()
        .map(|v| {
            holon
                .encode_json(&format!(r#"{{"$log": {}}}"#, v))
                .unwrap()
        })
        .collect();
    print_similarity_matrix(&labels, &log_vectors, "Log Encoding Similarity Matrix");

    // Analysis
    println!("\n{}", "-".repeat(60));
    println!("ANALYSIS:");
    println!("{}", "-".repeat(60));

    let sim_100_200_str = Similarity::cosine(&string_vectors[0], &string_vectors[1]);
    let sim_100_1000_str = Similarity::cosine(&string_vectors[0], &string_vectors[2]);

    let sim_100_200_log = Similarity::cosine(&log_vectors[0], &log_vectors[1]);
    let sim_100_1000_log = Similarity::cosine(&log_vectors[0], &log_vectors[2]);
    let sim_1000_10000_log = Similarity::cosine(&log_vectors[2], &log_vectors[3]);

    println!("\nString: 100 vs 200 = {:.3} (essentially random)", sim_100_200_str);
    println!("String: 100 vs 1000 = {:.3} (also random)", sim_100_1000_str);
    println!("\nLog: 100 vs 200 (2x) = {:.3} (high - same ballpark)", sim_100_200_log);
    println!("Log: 100 vs 1000 (10x) = {:.3} (moderate - one order apart)", sim_100_1000_log);
    println!("Log: 1000 vs 10000 (10x) = {:.3} (similar - also 10x ratio)", sim_1000_10000_log);

    println!("\nKEY INSIGHT: Log encoding makes 10x ratios produce consistent similarity drops,");
    println!("regardless of the absolute values involved.");
}

/// PART 2: Traffic Magnitude Clustering
fn demo_traffic_magnitude_clustering(holon: &Holon) {
    print_header("PART 2: Traffic Magnitude Clustering");

    // Traffic samples at different magnitude tiers
    let traffic_samples = [
        // Tier 1: Low-rate scanners (10-100 pps)
        (r#"{"type": "scanner", "src_ip": "10.0.0.1", "rate_pps": {"$log": 15}}"#, "scan-15"),
        (r#"{"type": "scanner", "src_ip": "10.0.0.2", "rate_pps": {"$log": 45}}"#, "scan-45"),
        (r#"{"type": "scanner", "src_ip": "10.0.0.3", "rate_pps": {"$log": 80}}"#, "scan-80"),
        // Tier 2: Medium traffic (1k-10k pps)
        (r#"{"type": "normal", "src_ip": "10.0.1.1", "rate_pps": {"$log": 2000}}"#, "norm-2k"),
        (r#"{"type": "normal", "src_ip": "10.0.1.2", "rate_pps": {"$log": 5000}}"#, "norm-5k"),
        (r#"{"type": "normal", "src_ip": "10.0.1.3", "rate_pps": {"$log": 8000}}"#, "norm-8k"),
        // Tier 3: High-volume attacks (100k+ pps)
        (r#"{"type": "attack", "src_ip": "10.0.2.1", "rate_pps": {"$log": 150000}}"#, "atk-150k"),
        (r#"{"type": "attack", "src_ip": "10.0.2.2", "rate_pps": {"$log": 500000}}"#, "atk-500k"),
        (r#"{"type": "attack", "src_ip": "10.0.2.3", "rate_pps": {"$log": 800000}}"#, "atk-800k"),
    ];

    let labels: Vec<&str> = traffic_samples.iter().map(|(_, l)| *l).collect();
    let vectors: Vec<_> = traffic_samples
        .iter()
        .map(|(json, _)| holon.encode_json(json).unwrap())
        .collect();

    println!("\nTraffic samples with $log-encoded rates:");
    for (json, label) in &traffic_samples {
        // Extract rate for display (simple parsing)
        if let Some(start) = json.find("\"$log\":") {
            let rest = &json[start + 7..];
            if let Some(end) = rest.find('}') {
                let rate_str = rest[..end].trim();
                println!("  {}: {} pps", label, rate_str);
            }
        }
    }

    print_similarity_matrix(&labels, &vectors, "\nTraffic Sample Similarity Matrix");

    // Analyze clustering
    println!("\n{}", "-".repeat(60));
    println!("CLUSTER ANALYSIS:");
    println!("{}", "-".repeat(60));

    // Within-tier similarities
    fn avg_similarity(vectors: &[holon::Vector], indices: &[usize]) -> f64 {
        let mut sims = Vec::new();
        for &i in indices {
            for &j in indices {
                if i < j {
                    sims.push(Similarity::cosine(&vectors[i], &vectors[j]));
                }
            }
        }
        if sims.is_empty() {
            0.0
        } else {
            sims.iter().sum::<f64>() / sims.len() as f64
        }
    }

    // Cross-tier similarities
    fn cross_similarity(vectors: &[holon::Vector], tier1: &[usize], tier2: &[usize]) -> f64 {
        let mut sims = Vec::new();
        for &i in tier1 {
            for &j in tier2 {
                sims.push(Similarity::cosine(&vectors[i], &vectors[j]));
            }
        }
        sims.iter().sum::<f64>() / sims.len() as f64
    }

    let tier1 = [0, 1, 2]; // scanners
    let tier2 = [3, 4, 5]; // normal
    let tier3 = [6, 7, 8]; // attacks

    println!("\nWithin-tier (should be HIGH):");
    println!("  Scanners (10-100 pps): {:.3}", avg_similarity(&vectors, &tier1));
    println!("  Normal (1k-10k pps):   {:.3}", avg_similarity(&vectors, &tier2));
    println!("  Attacks (100k+ pps):   {:.3}", avg_similarity(&vectors, &tier3));

    println!("\nCross-tier (should be LOWER):");
    println!("  Scanners ↔ Normal:  {:.3}", cross_similarity(&vectors, &tier1, &tier2));
    println!("  Normal ↔ Attacks:   {:.3}", cross_similarity(&vectors, &tier2, &tier3));
    println!("  Scanners ↔ Attacks: {:.3}", cross_similarity(&vectors, &tier1, &tier3));

    println!("\nKEY INSIGHT: Same-magnitude traffic clusters together naturally,");
    println!("without any hard-coded threshold configuration.");
}

/// PART 3: Find Similar Intensity Attacks
fn demo_find_similar_intensity(holon: &Holon) {
    print_header("PART 3: Find Similar Intensity Attacks");

    // Reference attack we're investigating
    let reference = r#"{"event_type": "ddos", "src_ip": "192.168.1.100", "rate_pps": {"$log": 50000}, "bytes_per_sec": {"$log": 75000000}}"#;

    // Historical attacks with varying intensities
    let historical = [
        (r#"{"event_type": "ddos", "rate_pps": {"$log": 45000}, "bytes_per_sec": {"$log": 68000000}}"#, "similar-1", 45000, 68000000),
        (r#"{"event_type": "ddos", "rate_pps": {"$log": 60000}, "bytes_per_sec": {"$log": 90000000}}"#, "similar-2", 60000, 90000000),
        (r#"{"event_type": "ddos", "rate_pps": {"$log": 1000}, "bytes_per_sec": {"$log": 1500000}}"#, "small-attack", 1000, 1500000),
        (r#"{"event_type": "ddos", "rate_pps": {"$log": 5000000}, "bytes_per_sec": {"$log": 7500000000}}"#, "massive-attack", 5000000, 7500000000u64),
        (r#"{"event_type": "exfiltration", "rate_pps": {"$log": 55000}, "bytes_per_sec": {"$log": 80000000}}"#, "exfil-similar", 55000, 80000000),
    ];

    let ref_vec = holon.encode_json(reference).unwrap();

    println!("\nReference Attack:");
    println!("  Type: ddos");
    println!("  Rate: 50,000 pps");
    println!("  Bandwidth: 75,000,000 bytes/sec");

    println!("\nHistorical Attacks (searching for similar intensity):");
    println!("{}", "-".repeat(60));

    // Calculate similarities and collect results
    let mut results: Vec<_> = historical
        .iter()
        .map(|(json, label, rate, bw)| {
            let vec = holon.encode_json(json).unwrap();
            let sim = Similarity::cosine(&ref_vec, &vec);
            (label, rate, bw, sim)
        })
        .collect();

    // Sort by similarity (descending)
    results.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());

    for (label, rate, bw, sim) in &results {
        println!("  {:.3}  {:20}  {:>10} pps  {:>15} B/s", sim, label, rate, bw);
    }

    println!("\n{}", "-".repeat(60));
    println!("ANALYSIS:");
    println!("{}", "-".repeat(60));
    println!("\n'Similar intensity' attacks rank highest (similar-1, similar-2)");
    println!("despite having different exact values. The exfiltration event also");
    println!("ranks high because its MAGNITUDE is similar, even though the type differs.");
    println!("\nSmall and massive attacks rank lower due to magnitude difference,");
    println!("not because of hard-coded thresholds.");
}

/// PART 4: Linear vs Log Encoding Use Cases
fn demo_linear_vs_log_encoding(holon: &Holon) {
    print_header("PART 4: Linear vs Log Encoding Use Cases");

    println!("\n--- Use Case: Response Time Monitoring ---\n");

    let latencies = [10, 20, 100, 110, 200];

    println!("Latency values: 10ms, 20ms, 100ms, 110ms, 200ms");
    println!("\nQuestion: Is 10→20ms more similar to 100→110ms or to 10→100ms?");

    // Linear encoding
    let lin_vecs: Vec<_> = latencies
        .iter()
        .map(|v| holon.encode_json(&format!(r#"{{"$linear": {}}}"#, v)).unwrap())
        .collect();

    // Log encoding
    let log_vecs: Vec<_> = latencies
        .iter()
        .map(|v| holon.encode_json(&format!(r#"{{"$log": {}}}"#, v)).unwrap())
        .collect();

    println!("\nLINEAR ENCODING (equal differences = equal similarity):");
    println!("  10ms ↔ 20ms (diff=10):   {:.3}", Similarity::cosine(&lin_vecs[0], &lin_vecs[1]));
    println!("  100ms ↔ 110ms (diff=10): {:.3}", Similarity::cosine(&lin_vecs[2], &lin_vecs[3]));
    println!("  10ms ↔ 100ms (diff=90):  {:.3}", Similarity::cosine(&lin_vecs[0], &lin_vecs[2]));

    println!("\nLOG ENCODING (equal ratios = equal similarity):");
    println!("  10ms ↔ 20ms (2x):        {:.3}", Similarity::cosine(&log_vecs[0], &log_vecs[1]));
    println!("  100ms ↔ 110ms (1.1x):    {:.3}", Similarity::cosine(&log_vecs[2], &log_vecs[3]));
    println!("  100ms ↔ 200ms (2x):      {:.3}", Similarity::cosine(&log_vecs[2], &log_vecs[4]));

    println!("\n{}", "-".repeat(60));
    println!("WHEN TO USE EACH:");
    println!("{}", "-".repeat(60));
    println!(r#"
LINEAR ($linear):
  - Latency/response times (added delay matters)
  - Temperature (5°C change is 5°C change)
  - Offsets and positions
  - When absolute difference matters

LOG ($log):
  - Packet rates (10x is 10x regardless of baseline)
  - Byte counts (orders of magnitude)
  - Prices spanning wide ranges
  - Resource usage percentages
  - When proportional change matters
"#);
}

/// PART 5: Practical Traffic Analysis
fn demo_practical_traffic_analysis(holon: &Holon) {
    print_header("PART 5: Practical Traffic Analysis Demo");

    let traffic_log = [
        // Normal baseline traffic
        (r#"{"src": "10.0.0.1", "dst": "web", "pps": {"$log": 500}}"#, 1, "normal", 500),
        (r#"{"src": "10.0.0.2", "dst": "web", "pps": {"$log": 800}}"#, 2, "normal", 800),
        (r#"{"src": "10.0.0.3", "dst": "web", "pps": {"$log": 1200}}"#, 3, "normal", 1200),
        // Reconnaissance
        (r#"{"src": "evil.1", "dst": "web", "pps": {"$log": 50}}"#, 4, "recon", 50),
        (r#"{"src": "evil.2", "dst": "web", "pps": {"$log": 30}}"#, 5, "recon", 30),
        // Attack ramp-up
        (r#"{"src": "botnet.1", "dst": "web", "pps": {"$log": 5000}}"#, 6, "ramp", 5000),
        (r#"{"src": "botnet.2", "dst": "web", "pps": {"$log": 15000}}"#, 7, "ramp", 15000),
        // Full attack
        (r#"{"src": "botnet.3", "dst": "web", "pps": {"$log": 100000}}"#, 8, "attack", 100000),
        (r#"{"src": "botnet.4", "dst": "web", "pps": {"$log": 250000}}"#, 9, "attack", 250000),
        (r#"{"src": "botnet.5", "dst": "web", "pps": {"$log": 180000}}"#, 10, "attack", 180000),
    ];

    println!("\nTraffic log with $log-encoded rates:");
    println!("{}", "-".repeat(60));
    for (_, ts, label, pps) in &traffic_log {
        println!("  ts={:2}  {:8}  {:>10} pps", ts, label, pps);
    }

    // Build baseline from normal traffic
    let normal_vecs: Vec<_> = traffic_log
        .iter()
        .filter(|(_, _, label, _)| *label == "normal")
        .map(|(json, _, _, _)| holon.encode_json(json).unwrap())
        .collect();

    let normal_refs: Vec<&holon::Vector> = normal_vecs.iter().collect();
    let baseline = holon.prototype(&normal_refs, 0.5);

    println!("\n\nBaseline built from normal traffic (ts 1-3)");
    println!("Anomaly score = 1 - similarity to baseline");
    println!("{}", "-".repeat(60));

    for (json, ts, label, pps) in &traffic_log {
        let vec = holon.encode_json(json).unwrap();
        let sim = holon.similarity(&baseline, &vec);
        let anomaly_score = 1.0 - sim;

        let bar_len = (anomaly_score * 40.0).max(0.0).min(40.0) as usize;
        let bar = format!("{}{}", "█".repeat(bar_len), "░".repeat(40 - bar_len));

        println!("  ts={:2}  {:8}  {:>10} pps  anomaly={:.3}  [{}]",
                 ts, label, pps, anomaly_score, bar);
    }

    println!("\n{}", "-".repeat(60));
    println!("OBSERVATIONS:");
    println!("{}", "-".repeat(60));
    println!(r#"
1. Normal traffic (500-1200 pps) has LOW anomaly scores
2. Recon (30-50 pps) has MODERATE scores - different magnitude from normal
3. Attack traffic (100k+ pps) has HIGH scores - orders of magnitude different

The magnitude-aware encoding naturally separates traffic tiers without
explicit threshold configuration. The vector similarity captures the
"unusual magnitude" signal directly.
"#);
}

fn main() {
    println!("\n{}", "=".repeat(70));
    println!(" BATCH 15: MAGNITUDE-AWARE NUMERIC ENCODING (Rust)");
    println!(" Demonstrating $log and $linear markers for VSA/HDC");
    println!("{}", "=".repeat(70));

    let holon = Holon::new(4096);

    demo_string_vs_log_encoding(&holon);
    demo_traffic_magnitude_clustering(&holon);
    demo_find_similar_intensity(&holon);
    demo_linear_vs_log_encoding(&holon);
    demo_practical_traffic_analysis(&holon);

    println!("\n{}", "=".repeat(70));
    println!(" SUMMARY");
    println!("{}", "=".repeat(70));
    println!(r#"
NEW MARKERS:
  {{"$log": value}}           - Log10 encoding (equal ratios = equal similarity)
  {{"$linear": value}}        - Positional encoding (equal differences = equal similarity)
  {{"$log": value, "$scale": n}}  - Custom decay rate

KEY BENEFITS:
  1. Cluster by magnitude without thresholds
  2. "Find similar intensity" queries work naturally
  3. Proportional changes captured in similarity
  4. Choose encoding mode based on domain semantics

DEFAULT BEHAVIOR UNCHANGED:
  Bare numbers still encode as strings (exact matching).
  Use $log/$linear markers to opt-in to magnitude awareness.
"#);
}
