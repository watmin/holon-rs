//! Compositional Recall — Algebraic Queries Over Incident Memory
//!
//! 16 incidents encoded across 4 service layers. One probe. Five algebraic
//! queries on the same library — no re-indexing, no pipeline, no query rewrite.
//!
//! Run: cargo run --example compositional_recall --release

use holon::memory::OnlineSubspace;
use holon::primitives::{AttendMode, NegateMethod, Primitives};
use holon::{Holon, ScalarValue, WalkType, Walkable, WalkableValue};
use std::collections::HashMap;

// =============================================================================
// Incident struct + Walkable
// =============================================================================

#[derive(Clone)]
struct Incident {
    id: &'static str,
    service: &'static str,
    layer: &'static str,
    severity: &'static str,
    incident_type: &'static str,
    env: &'static str,
    team: &'static str,
    resolved: bool,
}

impl Walkable for Incident {
    fn walk_type(&self) -> WalkType {
        WalkType::Map
    }

    fn walk_map_items(&self) -> Vec<(&str, WalkableValue)> {
        vec![
            ("id",       WalkableValue::Scalar(ScalarValue::String(self.id.to_string()))),
            ("service",  WalkableValue::Scalar(ScalarValue::String(self.service.to_string()))),
            ("layer",    WalkableValue::Scalar(ScalarValue::String(self.layer.to_string()))),
            ("severity", WalkableValue::Scalar(ScalarValue::String(self.severity.to_string()))),
            ("type",     WalkableValue::Scalar(ScalarValue::String(self.incident_type.to_string()))),
            ("env",      WalkableValue::Scalar(ScalarValue::String(self.env.to_string()))),
            ("team",     WalkableValue::Scalar(ScalarValue::String(self.team.to_string()))),
            ("resolved", WalkableValue::Scalar(ScalarValue::Bool(self.resolved))),
        ]
    }
}

// =============================================================================
// Incident library
// =============================================================================

const INCIDENTS: &[Incident] = &[
    // Frontend / web
    Incident { id: "INC-001", service: "frontend", layer: "web",      severity: "high",     incident_type: "latency",   env: "prod",    team: "ui",       resolved: true  },
    Incident { id: "INC-002", service: "frontend", layer: "web",      severity: "medium",   incident_type: "crash",     env: "prod",    team: "ui",       resolved: true  },
    Incident { id: "INC-003", service: "frontend", layer: "web",      severity: "low",      incident_type: "latency",   env: "staging", team: "ui",       resolved: true  },
    Incident { id: "INC-004", service: "frontend", layer: "web",      severity: "critical", incident_type: "outage",    env: "prod",    team: "ui",       resolved: false },
    // Backend API
    Incident { id: "INC-005", service: "api",      layer: "backend",  severity: "high",     incident_type: "latency",   env: "prod",    team: "platform", resolved: true  },
    Incident { id: "INC-006", service: "api",      layer: "backend",  severity: "medium",   incident_type: "crash",     env: "prod",    team: "platform", resolved: true  },
    Incident { id: "INC-007", service: "api",      layer: "backend",  severity: "critical", incident_type: "outage",    env: "prod",    team: "platform", resolved: false },
    Incident { id: "INC-008", service: "api",      layer: "backend",  severity: "low",      incident_type: "latency",   env: "staging", team: "platform", resolved: true  },
    // Database
    Incident { id: "INC-009", service: "postgres", layer: "database", severity: "critical", incident_type: "outage",    env: "prod",    team: "infra",    resolved: false },
    Incident { id: "INC-010", service: "postgres", layer: "database", severity: "high",     incident_type: "latency",   env: "prod",    team: "infra",    resolved: true  },
    Incident { id: "INC-011", service: "redis",    layer: "database", severity: "medium",   incident_type: "crash",     env: "prod",    team: "infra",    resolved: true  },
    Incident { id: "INC-012", service: "postgres", layer: "database", severity: "low",      incident_type: "latency",   env: "staging", team: "infra",    resolved: true  },
    // Security
    Incident { id: "INC-013", service: "auth",     layer: "security", severity: "critical", incident_type: "intrusion", env: "prod",    team: "security", resolved: false },
    Incident { id: "INC-014", service: "auth",     layer: "security", severity: "high",     incident_type: "intrusion", env: "prod",    team: "security", resolved: true  },
    Incident { id: "INC-015", service: "api",      layer: "security", severity: "critical", incident_type: "intrusion", env: "prod",    team: "security", resolved: false },
    Incident { id: "INC-016", service: "auth",     layer: "security", severity: "medium",   incident_type: "crash",     env: "staging", team: "security", resolved: true  },
];

// =============================================================================
// Display helpers
// =============================================================================

fn print_header(title: &str) {
    println!();
    println!("{}", "=".repeat(65));
    println!("  {}", title);
    println!("{}", "=".repeat(65));
}

fn show_results(label: &str, query: &holon::Vector, incident_vecs: &[holon::Vector], holon: &Holon, top_k: usize) {
    println!("\n  Query   : {}", label);
    let mut sims: Vec<(usize, f64)> = incident_vecs
        .iter()
        .enumerate()
        .map(|(i, v)| (i, holon.similarity(query, v)))
        .collect();
    sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (rank, (idx, sim)) in sims.iter().take(top_k).enumerate() {
        let inc = &INCIDENTS[*idx];
        println!(
            "    {}. {}  {:<12}  {:<10}  {:<8}  {:<10}  env={:<7}  sim={:.3}",
            rank + 1,
            inc.id,
            inc.service,
            inc.layer,
            inc.severity,
            inc.incident_type,
            inc.env,
            sim
        );
    }
}

// =============================================================================
// Main
// =============================================================================

fn main() -> holon::Result<()> {
    print_header("COMPOSITIONAL RECALL\n  Algebraic Queries Over Incident Memory");

    let holon = Holon::with_seed(4096, 42);

    // Encode all incidents
    println!("\nEncoding {} incidents across 4 service layers...", INCIDENTS.len());
    let incident_vecs: Vec<holon::Vector> = INCIDENTS
        .iter()
        .map(|inc| holon.encode_walkable(inc))
        .collect();

    // Build layer + env prototypes
    let layer_vecs = |layer: &str| -> Vec<&holon::Vector> {
        incident_vecs
            .iter()
            .zip(INCIDENTS.iter())
            .filter(|(_, inc)| inc.layer == layer)
            .map(|(v, _)| v)
            .collect()
    };
    let env_vecs = |env: &str| -> Vec<&holon::Vector> {
        incident_vecs
            .iter()
            .zip(INCIDENTS.iter())
            .filter(|(_, inc)| inc.env == env)
            .map(|(v, _)| v)
            .collect()
    };

    let db_refs       = layer_vecs("database");
    let frontend_refs = layer_vecs("web");
    let backend_refs  = layer_vecs("backend");
    let security_refs = layer_vecs("security");
    let prod_refs     = env_vecs("prod");

    let db_proto       = Primitives::prototype(&db_refs,       0.5);
    let frontend_proto = Primitives::prototype(&frontend_refs, 0.5);
    let backend_proto  = Primitives::prototype(&backend_refs,  0.5);
    let security_proto = Primitives::prototype(&security_refs, 0.5);
    let prod_proto     = Primitives::prototype(&prod_refs,     0.5);

    // Build EngramLibrary — one subspace per layer
    let mut library = holon.create_engram_library();
    for (layer_name, refs) in &[
        ("database", &db_refs),
        ("frontend", &frontend_refs),
        ("backend",  &backend_refs),
        ("security", &security_refs),
    ] {
        let mut ss = OnlineSubspace::with_params(4096, 16, 2.0, 0.01, 3.5, 500);
        for _ in 0..3 {
            for v in refs.iter() {
                ss.update(&v.to_f64());
            }
        }
        library.add(layer_name, &ss, None, HashMap::new());
    }

    let mut layer_names = library.names();
    layer_names.sort();
    println!("  Layer subspaces: {}", layer_names.join(", "));

    // Probe: high-severity latency in backend/prod, unresolved
    let probe_incident = Incident {
        id: "PROBE",
        service: "api",
        layer: "backend",
        severity: "high",
        incident_type: "latency",
        env: "prod",
        team: "platform",
        resolved: false,
    };
    let probe = holon.encode_walkable(&probe_incident);

    println!();
    println!("{}", "-".repeat(65));
    println!("PROBE: high-severity latency in backend/prod (unresolved)");
    println!("{}", "-".repeat(65));

    // 1. Basic recall — cosine similarity, what any vector DB gives you
    show_results("Basic recall (cosine similarity)", &probe, &incident_vecs, &holon, 4);

    // 2. Negation — subtract the database structural signal
    let probe_no_db = Primitives::negate_with_method(&probe, &db_proto, NegateMethod::Subtract);
    show_results(
        "negate(probe, db_proto) — exclude database incidents",
        &probe_no_db, &incident_vecs, &holon, 4,
    );

    // 3. Amplification — strengthen the production-environment signal
    let probe_prod = Primitives::amplify(&probe, &prod_proto, 1.5);
    show_results(
        "amplify(probe, prod_proto) — prioritise production impact",
        &probe_prod, &incident_vecs, &holon, 4,
    );

    // 4. Analogy — transfer the backend→security relationship onto the probe
    //    analogy(a, b, c) = c + difference(a, b)
    let probe_as_security = Primitives::analogy(&backend_proto, &security_proto, &probe);
    show_results(
        "analogy(backend→security, probe) — transfer pattern to security layer",
        &probe_as_security, &incident_vecs, &holon, 4,
    );

    // 5. Attend — extract the security-resonant components of the probe
    let probe_security_view = Primitives::attend(&probe, &security_proto, 2.0, AttendMode::Soft);
    show_results(
        "attend(probe, security_proto) — surface security-resonant signal",
        &probe_security_view, &incident_vecs, &holon, 4,
    );

    // 6. EngramLibrary — which layer manifold does the probe fit best?
    println!("\n  Query   : EngramLibrary.match_vec — which layer manifold fits best?");
    let matches = library.match_vec(&probe.to_f64(), 4, 10);
    for (rank, (name, residual)) in matches.iter().enumerate() {
        println!(
            "    {}. layer='{}'  residual={:.2}  (lower = better fit)",
            rank + 1, name, residual
        );
    }

    println!();
    println!("{}", "=".repeat(65));
    println!("Each query is one algebraic operation on the same encoded library.");
    println!("No re-indexing. No query rewrite. No pipeline.");
    println!();
    println!("Try: negate(amplify(probe, prod_proto), db_proto)");
    println!("Try: build a resolved=true prototype, negate to find open incidents only");

    // Suppress unused variable warnings for protos we built but didn't use directly
    let _ = frontend_proto;

    Ok(())
}
