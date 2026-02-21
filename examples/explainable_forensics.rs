//! Explainable Anomaly Forensics with Extended VSA Primitives
//!
//! Demonstrates the NEW extended primitives for investigating and
//! explaining anomalies - not just detecting them, but understanding WHY.
//!
//! PRIMITIVES DEMONSTRATED:
//! - segment()           - Find WHEN behavior changed
//! - complexity()        - Measure HOW MIXED the signal is
//! - invert()            - Decompose WHAT patterns are present
//! - similarity_profile() - See WHERE dimensions differ
//! - attend()            - Focus on RELEVANT dimensions
//! - project()           - Check IF in known attack subspace
//! - analogy()           - Transfer patterns between contexts
//! - conditional_bind()  - Gated feature binding
//!
//! Run: cargo run --example explainable_forensics

use holon::highlevel::Holon;
use holon::kernel::{AttendMode, GateMode, Primitives, SegmentMethod, Vector};

// =============================================================================
// TRAFFIC GENERATION
// =============================================================================

#[derive(Clone, Debug)]
enum TrafficType {
    Normal,
    ScanProbe,
    CredentialStuff,
    Exfil,
}

#[derive(Clone, Debug)]
struct TrafficEvent {
    source_ip: String,
    dest_port: u16,
    method: String,
    path_prefix: String,
    status_class: String,
    bytes_class: String,
    agent_type: String,
    traffic_type: TrafficType, // Ground truth
}

fn generate_normal_traffic(count: usize, seed: u64) -> Vec<TrafficEvent> {
    let mut events = Vec::with_capacity(count);
    let mut rng_state = seed;

    let paths = ["api", "health", "metrics", "users"];
    let ips = ["10.0.1.50", "10.0.1.51", "10.0.2.100"];

    for _ in 0..count {
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let idx = (rng_state >> 16) as usize;

        events.push(TrafficEvent {
            source_ip: ips[idx % ips.len()].to_string(),
            dest_port: 443,
            method: if idx % 3 == 0 { "POST" } else { "GET" }.to_string(),
            path_prefix: paths[idx % paths.len()].to_string(),
            status_class: "2xx".to_string(),
            bytes_class: "medium".to_string(),
            agent_type: "browser".to_string(),
            traffic_type: TrafficType::Normal,
        });
    }

    events
}

fn generate_scan_traffic(count: usize, seed: u64) -> Vec<TrafficEvent> {
    let mut events = Vec::with_capacity(count);
    let mut rng_state = seed;

    let paths = ["git", "env", "admin", "wp-admin", "phpMyAdmin"];

    for _ in 0..count {
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let idx = (rng_state >> 16) as usize;

        events.push(TrafficEvent {
            source_ip: "45.33.32.156".to_string(), // Single scanner
            dest_port: [80, 443, 8080][idx % 3],
            method: "GET".to_string(),
            path_prefix: paths[idx % paths.len()].to_string(),
            status_class: "4xx".to_string(),
            bytes_class: "small".to_string(),
            agent_type: "script".to_string(),
            traffic_type: TrafficType::ScanProbe,
        });
    }

    events
}

fn generate_credential_stuffing(count: usize, seed: u64) -> Vec<TrafficEvent> {
    let mut events = Vec::with_capacity(count);
    let mut rng_state = seed;

    for _ in 0..count {
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let idx = (rng_state >> 16) as usize;

        events.push(TrafficEvent {
            source_ip: format!("192.168.{}.{}", idx % 255, (idx / 255) % 255),
            dest_port: 443,
            method: "POST".to_string(),
            path_prefix: "auth".to_string(),
            status_class: "4xx".to_string(),
            bytes_class: "small".to_string(),
            agent_type: "script".to_string(),
            traffic_type: TrafficType::CredentialStuff,
        });
    }

    events
}

fn generate_exfil_traffic(count: usize, seed: u64) -> Vec<TrafficEvent> {
    let mut events = Vec::with_capacity(count);
    let mut rng_state = seed;

    for _ in 0..count {
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let idx = (rng_state >> 16) as usize;

        events.push(TrafficEvent {
            source_ip: "10.0.1.50".to_string(), // Compromised host
            dest_port: 443,
            method: "POST".to_string(),
            path_prefix: if idx % 2 == 0 { "export" } else { "backup" }.to_string(),
            status_class: "2xx".to_string(),
            bytes_class: "large".to_string(),
            agent_type: "script".to_string(),
            traffic_type: TrafficType::Exfil,
        });
    }

    events
}

fn encode_event(holon: &Holon, event: &TrafficEvent) -> Vector {
    let json = format!(
        r#"{{"source_ip":"{}","dest_port":"{}","method":"{}","path_prefix":"{}","status_class":"{}","bytes_class":"{}","agent_type":"{}"}}"#,
        event.source_ip,
        event.dest_port,
        event.method,
        event.path_prefix,
        event.status_class,
        event.bytes_class,
        event.agent_type
    );
    holon.encode_json(&json).unwrap()
}

// =============================================================================
// MAIN
// =============================================================================

fn main() {
    println!(
        r#"
╔══════════════════════════════════════════════════════════════════════╗
║         EXPLAINABLE ANOMALY FORENSICS WITH VSA PRIMITIVES            ║
║                                                                      ║
║  Primitives: segment, complexity, invert, similarity_profile,        ║
║              attend, project, analogy, conditional_bind              ║
╚══════════════════════════════════════════════════════════════════════╝
"#
    );

    let holon = Holon::new(4096);

    // =========================================================================
    // PHASE 1: Generate Traffic Stream
    // =========================================================================

    println!("======================================================================");
    println!("PHASE 1: TRAFFIC STREAM GENERATION");
    println!("======================================================================\n");

    let mut events = Vec::new();

    // Normal baseline (0-100)
    events.extend(generate_normal_traffic(100, 42));

    // Scan probe (100-150)
    events.extend(generate_scan_traffic(50, 123));

    // Normal recovery (150-200)
    events.extend(generate_normal_traffic(50, 456));

    // Credential stuffing (200-300)
    events.extend(generate_credential_stuffing(100, 789));

    // Brief normal (300-320)
    events.extend(generate_normal_traffic(20, 1011));

    // Exfiltration (320-370)
    events.extend(generate_exfil_traffic(50, 1213));

    // Final recovery (370-450)
    events.extend(generate_normal_traffic(80, 1415));

    println!("Generated {} events across attack phases:", events.len());
    println!("  - Normal baseline:      events 0-100");
    println!("  - Scan probe attack:    events 100-150");
    println!("  - Normal recovery:      events 150-200");
    println!("  - Credential stuffing:  events 200-300");
    println!("  - Brief normal:         events 300-320");
    println!("  - Data exfiltration:    events 320-370");
    println!("  - Final recovery:       events 370-450");

    // Encode all events
    let vectors: Vec<Vector> = events.iter().map(|e| encode_event(&holon, e)).collect();

    // =========================================================================
    // PHASE 2: segment() - Find Phase Changes
    // =========================================================================

    println!("\n======================================================================");
    println!("PHASE 2: segment() - DETECTING BEHAVIORAL PHASE CHANGES");
    println!("======================================================================");

    println!("\n    segment(stream, window, threshold, method) finds breakpoints.");
    println!("    This answers: \"WHEN did behavior change?\"\n");

    let breakpoints = Primitives::segment(&vectors, 20, 0.4, SegmentMethod::Diff);

    println!("Detected {} segment breakpoints: {:?}", breakpoints.len(), breakpoints);

    // =========================================================================
    // PHASE 3: complexity() - Measure Entropy
    // =========================================================================

    println!("\n======================================================================");
    println!("PHASE 3: complexity() - MEASURING SIGNAL ENTROPY");
    println!("======================================================================");

    println!("\n    complexity(vec) measures how \"mixed\" a vector is.");
    println!("    Low = clean signal, High = superposition.\n");

    let phases = [
        (0..100, "Normal baseline"),
        (100..150, "Scan probe"),
        (200..300, "Credential stuffing"),
        (320..370, "Exfiltration"),
    ];

    println!("{:<25} {:>15}", "Phase", "Avg Complexity");
    println!("{}", "-".repeat(45));

    for (range, name) in phases.iter() {
        let complexities: Vec<f64> = vectors[range.clone()]
            .iter()
            .map(|v| Primitives::complexity(v))
            .collect();
        let avg: f64 = complexities.iter().sum::<f64>() / complexities.len() as f64;
        println!("{:<25} {:>15.4}", name, avg);
    }

    // =========================================================================
    // PHASE 4: Build Prototypes for invert()
    // =========================================================================

    println!("\n======================================================================");
    println!("PHASE 4: invert() - ANOMALY ATTRIBUTION");
    println!("======================================================================");

    println!("\n    invert(vec, codebook) reconstructs what patterns are present.");
    println!("    This answers: \"WHAT known patterns explain this anomaly?\"\n");

    // Build prototypes
    let normal_vecs: Vec<&Vector> = vectors[0..50].iter().collect();
    let scan_vecs: Vec<&Vector> = vectors[100..130].iter().collect();
    let cred_vecs: Vec<&Vector> = vectors[200..250].iter().collect();
    let exfil_vecs: Vec<&Vector> = vectors[320..350].iter().collect();

    let normal_proto = Primitives::prototype(&normal_vecs, 0.5);
    let scan_proto = Primitives::prototype(&scan_vecs, 0.5);
    let cred_proto = Primitives::prototype(&cred_vecs, 0.5);
    let exfil_proto = Primitives::prototype(&exfil_vecs, 0.5);

    let codebook = vec![
        normal_proto.clone(),
        scan_proto.clone(),
        cred_proto.clone(),
        exfil_proto.clone(),
    ];
    let codebook_names = ["normal", "scan_probe", "credential_stuffing", "exfiltration"];

    let test_samples = [(115, "From scan phase"), (250, "From cred stuff phase"), (340, "From exfil phase")];

    for (idx, desc) in test_samples.iter() {
        let results = Primitives::invert(&vectors[*idx], &codebook, 3, 0.1);
        println!("  Sample {} ({}):", idx, desc);
        println!("    Ground truth: {:?}", events[*idx].traffic_type);
        println!("    Detected patterns:");
        for (cb_idx, sim) in results.iter() {
            println!("      - {}: {:.3}", codebook_names[*cb_idx], sim);
        }
        println!();
    }

    // =========================================================================
    // PHASE 5: similarity_profile()
    // =========================================================================

    println!("======================================================================");
    println!("PHASE 5: similarity_profile() - DIMENSION-WISE ANALYSIS");
    println!("======================================================================");

    println!("\n    similarity_profile(A, B) returns similarity as a VECTOR.");
    println!("    Shows WHERE two patterns agree/disagree.\n");

    let profiles = [
        (Primitives::similarity_profile(&scan_proto, &normal_proto), "Scan vs Normal"),
        (Primitives::similarity_profile(&cred_proto, &normal_proto), "Cred vs Normal"),
        (Primitives::similarity_profile(&exfil_proto, &normal_proto), "Exfil vs Normal"),
    ];

    println!("{:<20} {:>8} {:>10} {:>10}", "Comparison", "Agree", "Disagree", "Ratio");
    println!("{}", "-".repeat(50));

    for (profile, name) in profiles.iter() {
        let agree = profile.data().iter().filter(|&&x| x > 0).count();
        let disagree = profile.data().iter().filter(|&&x| x < 0).count();
        let ratio = if agree + disagree > 0 {
            agree as f64 / (agree + disagree) as f64
        } else {
            0.0
        };
        println!("{:<20} {:>8} {:>10} {:>9.1}%", name, agree, disagree, ratio * 100.0);
    }

    // =========================================================================
    // PHASE 6: attend()
    // =========================================================================

    println!("\n======================================================================");
    println!("PHASE 6: attend() - SOFT ATTENTION ANALYSIS");
    println!("======================================================================");

    println!("\n    attend(query, memory, strength, mode) applies soft attention.\n");

    let suspicious_idx = 340;
    let suspicious_vec = &vectors[suspicious_idx];

    let attended_hard = Primitives::attend(&scan_proto, suspicious_vec, 1.0, AttendMode::Hard);
    let attended_soft = Primitives::attend(&scan_proto, suspicious_vec, 1.0, AttendMode::Soft);
    let attended_amp = Primitives::attend(&scan_proto, suspicious_vec, 1.0, AttendMode::Amplify);

    println!("Attending to sample {} with scan_proto as query:", suspicious_idx);
    println!("  Original complexity:  {:.4}", Primitives::complexity(suspicious_vec));
    println!("  After hard attention: {:.4}", Primitives::complexity(&attended_hard));
    println!("  After soft attention: {:.4}", Primitives::complexity(&attended_soft));
    println!("  After amplify:        {:.4}", Primitives::complexity(&attended_amp));

    // =========================================================================
    // PHASE 7: project()
    // =========================================================================

    println!("\n======================================================================");
    println!("PHASE 7: project() - SUBSPACE CLASSIFICATION");
    println!("======================================================================");

    println!("\n    project(vec, subspace) projects onto exemplar subspace.\n");

    let attack_subspace: Vec<&Vector> = vec![&scan_proto, &cred_proto, &exfil_proto];

    let proj_samples = [
        (25, "Normal baseline"),
        (115, "Scan probe"),
        (250, "Credential stuffing"),
        (340, "Exfiltration"),
    ];

    println!("{:<30} {:>15} {:>15}", "Sample", "Original Norm", "Projected Norm");
    println!("{}", "-".repeat(65));

    for (idx, desc) in proj_samples.iter() {
        let vec = &vectors[*idx];
        let projected = Primitives::project(vec, &attack_subspace, true);
        let orig_norm = vec.norm();
        let proj_norm = projected.norm();
        println!("{:<30} {:>15.2} {:>15.2}", format!("{} ({})", desc, idx), orig_norm, proj_norm);
    }

    // =========================================================================
    // PHASE 8: analogy()
    // =========================================================================

    println!("\n======================================================================");
    println!("PHASE 8: analogy() - PATTERN TRANSFER");
    println!("======================================================================");

    println!("\n    analogy(A, B, C) computes: A is to B as C is to ?\n");

    let port_80_scan = holon
        .encode_json(r#"{"dest_port":"80","method":"GET","path_prefix":"admin","status_class":"4xx"}"#)
        .unwrap();
    let port_443_scan = holon
        .encode_json(r#"{"dest_port":"443","method":"GET","path_prefix":"admin","status_class":"4xx"}"#)
        .unwrap();
    let port_80_exfil = holon
        .encode_json(r#"{"dest_port":"80","method":"POST","path_prefix":"export","status_class":"2xx","bytes_class":"large"}"#)
        .unwrap();
    let actual_443_exfil = holon
        .encode_json(r#"{"dest_port":"443","method":"POST","path_prefix":"export","status_class":"2xx","bytes_class":"large"}"#)
        .unwrap();

    let predicted_443_exfil = Primitives::analogy(&port_80_scan, &port_443_scan, &port_80_exfil);
    let sim = holon.similarity(&predicted_443_exfil, &actual_443_exfil);

    println!("Analogy: port_80_scan : port_443_scan :: port_80_exfil : ?");
    println!("Predicted vs actual port_443_exfil similarity: {:.3}", sim);

    // =========================================================================
    // PHASE 9: conditional_bind()
    // =========================================================================

    println!("\n======================================================================");
    println!("PHASE 9: conditional_bind() - GATED FEATURE BINDING");
    println!("======================================================================");

    println!("\n    conditional_bind(A, B, gate, mode) binds only where gate passes.\n");

    let source_ip_vec = holon.get_vector("45.33.32.156");
    let behavior_vec = holon.get_vector("GET_admin");
    let gate_vec = holon.get_vector("4xx");

    let gated_binding = Primitives::conditional_bind(&source_ip_vec, &behavior_vec, &gate_vec, GateMode::Positive);
    let full_binding = Primitives::bind(&source_ip_vec, &behavior_vec);

    println!("Conditional binding (gated on 4xx status):");
    println!("  Full binding complexity:   {:.4}", Primitives::complexity(&full_binding));
    println!("  Gated binding complexity:  {:.4}", Primitives::complexity(&gated_binding));
    println!("  Active dims (gated):       {} / {}", gated_binding.nnz(), gated_binding.dimensions());
    println!("  Active dims (full):        {} / {}", full_binding.nnz(), full_binding.dimensions());

    // =========================================================================
    // SUMMARY
    // =========================================================================

    println!("\n======================================================================");
    println!("SUMMARY: EXTENDED PRIMITIVES FOR ANOMALY FORENSICS");
    println!("======================================================================");

    println!(
        r#"
    The extended primitives enable a complete forensics workflow:

    ┌─────────────────────────────────────────────────────────────────────┐
    │  DETECTION          │  INVESTIGATION       │  EXPLANATION          │
    ├─────────────────────┼──────────────────────┼───────────────────────┤
    │  segment()          │  invert()            │  similarity_profile() │
    │  Find WHEN          │  Find WHAT           │  Show WHERE           │
    │                     │                      │                       │
    │  complexity()       │  project()           │  analogy()            │
    │  Measure entropy    │  Classify subspace   │  Generalize patterns  │
    │                     │                      │                       │
    │                     │  attend()            │  conditional_bind()   │
    │                     │  Focus analysis      │  Context-aware encode │
    └─────────────────────┴──────────────────────┴───────────────────────┘

    Together, these primitives transform anomaly detection from
    "something is wrong" to "HERE is what changed, WHEN, and WHY."
"#
    );
}
