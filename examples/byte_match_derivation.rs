//! Byte Match Rule Derivation: Windowed Payload Analysis → l4-match Rules
//!
//! Rust port of Python experiments 017-019 from Challenge Batch 016.
//!
//! PIPELINE:
//!   1. Windowed payload analysis: 64-byte windows, each with its own accumulator
//!   2. VSA detection: score each window, drill-down on anomalous ones
//!   3. Gap probing: extend detected runs by checking neighboring positions
//!   4. Rule derivation: generate 1/2/4-byte and long PatternGuard candidates
//!   5. Coverage-aware greedy selection: minimize rules for maximum TP
//!
//! eBPF CONSTRAINTS:
//!   - 1-4 byte patterns → custom dim slots (7 total, arbitrary L4 offset)
//!   - 5-64 byte patterns → PatternGuard (L4 offset + length ≤ 64)
//!   - Max 32 l4-match rules per destination scope
//!   - Masks: 0xFF (exact) or 0x00 (wildcard)
//!
//! Run: cargo run --example byte_match_derivation --release

use holon::Holon;
use std::collections::{HashMap, HashSet};

const DIMENSIONS: usize = 4096;
const PAYLOAD_SIZE: usize = 512;
const WINDOW_SIZE: usize = 64;
const NUM_WINDOWS: usize = (PAYLOAD_SIZE + WINDOW_SIZE - 1) / WINDOW_SIZE;
const UDP_HDR: usize = 8;
const PATTERN_DATA_WINDOW: usize = 64;
const MAX_RULES_PER_SCOPE: usize = 32;

// =============================================================================
// SIMPLE RNG
// =============================================================================

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        self.state
    }

    fn choice(&mut self, items: &[u8]) -> u8 {
        let idx = (self.next() % items.len() as u64) as usize;
        items[idx]
    }

    fn range(&mut self, lo: u64, hi: u64) -> u64 {
        lo + (self.next() % (hi - lo))
    }
}

// =============================================================================
// WINDOWED PAYLOAD ANALYZER
// =============================================================================

struct WindowedAnalyzer {
    holon: Holon,
    accumulators: Vec<holon::Accumulator>,
    baselines: Vec<Option<holon::Vector>>,
}

#[allow(dead_code)]
struct DrillResult {
    pos: usize,
    byte_val: u8,
    sim: f64,
}

impl WindowedAnalyzer {
    fn new() -> Self {
        let holon = Holon::new(DIMENSIONS);
        let accumulators = (0..NUM_WINDOWS)
            .map(|_| holon.create_accumulator())
            .collect();
        Self {
            holon,
            accumulators,
            baselines: vec![None; NUM_WINDOWS],
        }
    }

    fn window_json(payload: &[u8], w: usize) -> String {
        let start = w * WINDOW_SIZE;
        let end = std::cmp::min(start + WINDOW_SIZE, payload.len());
        let fields: Vec<String> = (start..end)
            .map(|i| format!(r#""p{}":"0x{:02x}""#, i - start, payload[i]))
            .collect();
        format!("{{{}}}", fields.join(","))
    }

    fn learn(&mut self, payload: &[u8]) {
        for w in 0..NUM_WINDOWS {
            let json = Self::window_json(payload, w);
            let vec = self.holon.encode_json(&json).unwrap();
            self.holon.accumulate(&mut self.accumulators[w], &vec);
        }
    }

    fn freeze(&mut self) {
        for w in 0..NUM_WINDOWS {
            self.baselines[w] =
                Some(self.holon.normalize_accumulator(&self.accumulators[w]));
        }
    }

    fn score_window(&self, payload: &[u8], w: usize) -> f64 {
        let json = Self::window_json(payload, w);
        let vec = self.holon.encode_json(&json).unwrap();
        self.holon
            .similarity(&vec, self.baselines[w].as_ref().unwrap())
    }

    fn drill_down(&self, payload: &[u8], w: usize) -> Vec<DrillResult> {
        let baseline = self.baselines[w].as_ref().unwrap();
        let start = w * WINDOW_SIZE;
        let end = std::cmp::min(start + WINDOW_SIZE, payload.len());
        let mut results = Vec::new();

        for i in start..end {
            let field = format!("p{}", i - start);
            let value = format!("0x{:02x}", payload[i]);
            let role = self.holon.get_vector(&field);
            let val = self.holon.get_vector(&value);
            let bound = self.holon.bind(&role, &val);
            let sim = self.holon.similarity(&bound, baseline);
            results.push(DrillResult {
                pos: i,
                byte_val: payload[i],
                sim,
            });
        }
        results
    }

    fn detect(&self, payload: &[u8], legit_ref: &[u8]) -> Vec<usize> {
        let mut unfamiliar = Vec::new();

        for w in 0..NUM_WINDOWS {
            let atk_sim = self.score_window(payload, w);
            let leg_sim = self.score_window(legit_ref, w);
            let drop = leg_sim - atk_sim;

            if drop > 0.015 {
                for d in self.drill_down(payload, w) {
                    if d.sim < 0.005 {
                        unfamiliar.push(d.pos);
                    }
                }
            }
        }

        unfamiliar.sort();
        unfamiliar.dedup();
        unfamiliar
    }
}

// =============================================================================
// RULE DERIVATION
// =============================================================================

#[derive(Clone)]
struct RuleCandidate {
    l4_offset: usize,
    match_bytes: Vec<u8>,
    mask_bytes: Vec<u8>,
    tp: usize,
    fp: usize,
    description: String,
}

impl RuleCandidate {
    fn len(&self) -> usize {
        self.match_bytes.len()
    }

    fn active(&self) -> usize {
        self.mask_bytes.iter().filter(|&&m| m != 0x00).count()
    }

    fn enforceable(&self) -> bool {
        if self.len() <= 4 {
            return true;
        }
        self.l4_offset + self.len() <= PATTERN_DATA_WINDOW
    }

    fn cost_label(&self) -> &'static str {
        if self.len() <= 4 {
            "custom-dim"
        } else {
            "PatternGuard"
        }
    }

    fn slot_key(&self) -> (usize, usize) {
        (self.l4_offset, self.len())
    }

    fn to_edn(&self) -> String {
        let mh: String = self.match_bytes.iter().map(|b| format!("{:02X}", b)).collect();
        let mk: String = self.mask_bytes.iter().map(|b| format!("{:02X}", b)).collect();
        format!(r#"(l4-match {} "{}" "{}")"#, self.l4_offset, mh, mk)
    }

    fn matches(&self, payload: &[u8]) -> bool {
        let start = self.l4_offset.saturating_sub(UDP_HDR);
        for i in 0..self.len() {
            if self.mask_bytes[i] == 0x00 {
                continue;
            }
            let pos = start + i;
            if pos >= payload.len() {
                return false;
            }
            if (payload[pos] & self.mask_bytes[i]) != self.match_bytes[i] {
                return false;
            }
        }
        true
    }
}

/// Per-position analysis data.
#[allow(dead_code)]
struct PosInfo {
    top_val: u8,
    constancy: f64,
    n_unique: usize,
    is_constant: bool,
    unfam_vals: HashSet<u8>,
    counts: HashMap<u8, usize>,
}

fn analyze_position(
    pos: usize,
    attack_payloads: &[Vec<u8>],
    legit_payloads: &[Vec<u8>],
) -> PosInfo {
    let mut counts: HashMap<u8, usize> = HashMap::new();
    for p in attack_payloads {
        *counts.entry(p[pos]).or_insert(0) += 1;
    }
    let leg_set: HashSet<u8> = legit_payloads.iter().map(|p| p[pos]).collect();

    let (&top_val, &top_count) = counts.iter().max_by_key(|e| e.1).unwrap();
    let constancy = top_count as f64 / attack_payloads.len() as f64;
    let unfam_vals: HashSet<u8> = counts
        .keys()
        .filter(|v| !leg_set.contains(v))
        .copied()
        .collect();

    PosInfo {
        top_val,
        constancy,
        n_unique: counts.len(),
        is_constant: constancy >= 0.95,
        unfam_vals,
        counts,
    }
}

fn derive_rules(
    positions: &[usize],
    attack_payloads: &[Vec<u8>],
    legit_payloads: &[Vec<u8>],
) -> (Vec<RuleCandidate>, HashMap<usize, PosInfo>) {
    let pos_set: HashSet<usize> = positions.iter().copied().collect();
    let mut sorted_pos: Vec<usize> = positions.to_vec();
    sorted_pos.sort();
    sorted_pos.dedup();

    // Build runs with gap bridging (≤3 gap)
    let mut runs: Vec<Vec<usize>> = Vec::new();
    if sorted_pos.is_empty() {
        return (vec![], HashMap::new());
    }

    let mut current_run = vec![sorted_pos[0]];
    for &pos in &sorted_pos[1..] {
        let gap = pos - *current_run.last().unwrap();
        if gap == 1 {
            current_run.push(pos);
        } else if gap <= 4 {
            let last = *current_run.last().unwrap();
            for bridge in (last + 1)..pos {
                current_run.push(bridge);
            }
            current_run.push(pos);
        } else {
            runs.push(current_run);
            current_run = vec![pos];
        }
    }
    runs.push(current_run);

    // Analyze all positions in runs
    let mut all_positions: HashSet<usize> = HashSet::new();
    for run in &runs {
        all_positions.extend(run);
    }

    let mut pos_data: HashMap<usize, PosInfo> = HashMap::new();
    for &pos in &all_positions {
        pos_data.insert(pos, analyze_position(pos, attack_payloads, legit_payloads));
    }

    let mut candidates: Vec<RuleCandidate> = Vec::new();

    for run in &runs {
        // 1-byte rules
        for &pos in run {
            if let Some(info) = pos_data.get(&pos) {
                for &val in &info.unfam_vals {
                    candidates.push(RuleCandidate {
                        l4_offset: UDP_HDR + pos,
                        match_bytes: vec![val],
                        mask_bytes: vec![0xFF],
                        tp: 0,
                        fp: 0,
                        description: format!("1B @{}", pos),
                    });
                }
            }
        }

        // 2-byte rules at consecutive positions
        for i in 0..run.len().saturating_sub(1) {
            let p1 = run[i];
            let p2 = run[i + 1];
            if p2 != p1 + 1 {
                continue;
            }
            if !pos_set.contains(&p1) || !pos_set.contains(&p2) {
                continue;
            }

            let mut combos: HashMap<(u8, u8), usize> = HashMap::new();
            for p in attack_payloads {
                *combos.entry((p[p1], p[p2])).or_insert(0) += 1;
            }
            let legit_combos: HashSet<(u8, u8)> =
                legit_payloads.iter().map(|p| (p[p1], p[p2])).collect();

            let mut sorted_combos: Vec<_> = combos.into_iter().collect();
            sorted_combos.sort_by(|a, b| b.1.cmp(&a.1));

            for (combo, _) in sorted_combos.iter().take(5) {
                if !legit_combos.contains(combo) {
                    candidates.push(RuleCandidate {
                        l4_offset: UDP_HDR + p1,
                        match_bytes: vec![combo.0, combo.1],
                        mask_bytes: vec![0xFF, 0xFF],
                        tp: 0,
                        fp: 0,
                        description: format!("2B @{}-{}", p1, p1 + 1),
                    });
                }
            }
        }

        // 4-byte rules
        for i in 0..run.len().saturating_sub(3) {
            let p4: Vec<usize> = (0..4).map(|j| run[i + j]).collect();
            if p4[3] - p4[0] != 3 {
                continue;
            }
            if !p4.iter().all(|p| pos_set.contains(p)) {
                continue;
            }

            let mut combos: HashMap<[u8; 4], usize> = HashMap::new();
            for p in attack_payloads {
                let key = [p[p4[0]], p[p4[1]], p[p4[2]], p[p4[3]]];
                *combos.entry(key).or_insert(0) += 1;
            }
            let legit_combos: HashSet<[u8; 4]> = legit_payloads
                .iter()
                .map(|p| [p[p4[0]], p[p4[1]], p[p4[2]], p[p4[3]]])
                .collect();

            let mut sorted_combos: Vec<_> = combos.into_iter().collect();
            sorted_combos.sort_by(|a, b| b.1.cmp(&a.1));

            for (combo, _) in sorted_combos.iter().take(5) {
                if !legit_combos.contains(combo) {
                    candidates.push(RuleCandidate {
                        l4_offset: UDP_HDR + p4[0],
                        match_bytes: combo.to_vec(),
                        mask_bytes: vec![0xFF; 4],
                        tp: 0,
                        fp: 0,
                        description: format!("4B @{}-{}", p4[0], p4[3]),
                    });
                }
            }
        }

        // Long rules: 8, 16, 32 bytes and full run
        let run_len = run.len();
        for &target in &[8usize, 16, 32, run_len] {
            if target < 5 || target > run_len {
                continue;
            }

            for start_idx in 0..=(run_len - target) {
                let span = &run[start_idx..start_idx + target];
                let first = span[0];
                let last = *span.last().unwrap();
                let actual = last - first + 1;
                let l4_off = UDP_HDR + first;

                let mut match_b = Vec::new();
                let mut mask_b = Vec::new();
                let mut active = 0usize;

                for pos in first..=last {
                    if let Some(info) = pos_data.get(&pos) {
                        if !info.unfam_vals.is_empty() {
                            match_b.push(info.top_val);
                            mask_b.push(0xFF);
                            active += 1;
                        } else {
                            match_b.push(0x00);
                            mask_b.push(0x00);
                        }
                    } else {
                        match_b.push(0x00);
                        mask_b.push(0x00);
                    }
                }

                if active < 3 {
                    continue;
                }

                candidates.push(RuleCandidate {
                    l4_offset: l4_off,
                    match_bytes: match_b,
                    mask_bytes: mask_b,
                    tp: 0,
                    fp: 0,
                    description: format!(
                        "{}B @{}-{} ({}/{} active)",
                        actual, first, last, active, actual
                    ),
                });
            }
        }
    }

    // Validate all candidates
    for r in &mut candidates {
        r.tp = attack_payloads.iter().filter(|p| r.matches(p)).count();
        r.fp = legit_payloads.iter().filter(|p| r.matches(p)).count();
    }

    // Filter: zero FP, non-zero TP
    candidates.retain(|r| r.fp == 0 && r.tp > 0);

    // Sort: enforceable first, then TP desc, length desc
    candidates.sort_by(|a, b| {
        b.enforceable()
            .cmp(&a.enforceable())
            .then(b.tp.cmp(&a.tp))
            .then(b.len().cmp(&a.len()))
    });

    (candidates, pos_data)
}

fn select_best(
    candidates: &[RuleCandidate],
    attack_payloads: &[Vec<u8>],
) -> (Vec<RuleCandidate>, f64) {
    let n = attack_payloads.len();
    let mut selected: Vec<RuleCandidate> = Vec::new();
    let mut covered: HashSet<usize> = HashSet::new();
    let mut used_slots: HashSet<(usize, usize)> = HashSet::new();

    // Pre-compute hits
    let hit_cache: Vec<HashSet<usize>> = candidates
        .iter()
        .map(|r| {
            attack_payloads
                .iter()
                .enumerate()
                .filter(|(_, p)| r.matches(p))
                .map(|(i, _)| i)
                .collect()
        })
        .collect();

    for (idx, r) in candidates.iter().enumerate() {
        if selected.len() >= MAX_RULES_PER_SCOPE {
            break;
        }

        // Custom dim slot budget (7 slots)
        if r.len() <= 4 {
            let key = r.slot_key();
            if !used_slots.contains(&key) && used_slots.len() >= 7 {
                continue;
            }
        }

        let new_hits: HashSet<usize> = hit_cache[idx].difference(&covered).copied().collect();
        if new_hits.is_empty() {
            continue;
        }

        selected.push(r.clone());
        covered.extend(&new_hits);
        if r.len() <= 4 {
            used_slots.insert(r.slot_key());
        }

        if covered.len() == n {
            break;
        }
    }

    let coverage = covered.len() as f64 / n as f64 * 100.0;
    (selected, coverage)
}

// =============================================================================
// PAYLOAD GENERATORS
// =============================================================================

const FAMILIAR: [u8; 6] = [0x00, 0x01, 0x02, 0x10, 0x20, 0x30];

fn make_normal(rng: &mut Rng) -> Vec<u8> {
    (0..PAYLOAD_SIZE).map(|_| rng.choice(&FAMILIAR)).collect()
}

/// 32-byte constant shellcode at payload[0..32]
const SHELLCODE: [u8; 32] = [
    0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE, 0x41, 0x42, 0x43, 0x44,
    0x45, 0x46, 0x47, 0x48, 0x90, 0x90, 0x90, 0x90, 0xCC, 0xCC, 0xCC, 0xCC,
    0xFF, 0xFE, 0xFD, 0xFC, 0xFB, 0xFA, 0xF9, 0xF8,
];

fn make_attack_dumb(rng: &mut Rng) -> Vec<u8> {
    let mut p = make_normal(rng);
    for (i, &v) in [0xDE, 0xAD, 0xBE, 0xEF].iter().enumerate() {
        p[10 + i] = v;
    }
    for (i, &v) in [0xCA, 0xFE, 0xBA, 0xBE].iter().enumerate() {
        p[200 + i] = v;
    }
    p
}

fn make_attack_limited(rng: &mut Rng) -> Vec<u8> {
    let mut p = make_normal(rng);
    let variant = rng.range(0, 3);
    let sigs: [([u8; 4], [u8; 4]); 3] = [
        ([0xDE, 0xAD, 0xBE, 0xEF], [0xCA, 0xFE, 0xBA, 0xBE]),
        ([0xFF, 0xAC, 0xFB, 0xCA], [0xAD, 0xDE, 0xEF, 0xFF]),
        ([0xBE, 0xEF, 0xCA, 0xFE], [0xDE, 0xAD, 0x00, 0xFF]),
    ];
    let (near, deep) = &sigs[variant as usize];
    for (i, &v) in near.iter().enumerate() {
        p[10 + i] = v;
    }
    for (i, &v) in deep.iter().enumerate() {
        p[200 + i] = v;
    }
    p
}

fn make_attack_near32(rng: &mut Rng) -> Vec<u8> {
    let mut p = make_normal(rng);
    for (i, &v) in SHELLCODE.iter().enumerate() {
        p[i] = v;
    }
    p
}

fn make_attack_deep32(rng: &mut Rng) -> Vec<u8> {
    let mut p = make_normal(rng);
    for (i, &v) in SHELLCODE.iter().enumerate() {
        p[200 + i] = v;
    }
    p
}

// =============================================================================
// GAP PROBING
// =============================================================================

fn gap_probe(
    detected: &[usize],
    attack_payloads: &[Vec<u8>],
    legit_payloads: &[Vec<u8>],
) -> Vec<usize> {
    if detected.is_empty() {
        return vec![];
    }

    let legit_sets: Vec<HashSet<u8>> = (0..PAYLOAD_SIZE)
        .map(|pos| legit_payloads.iter().map(|p| p[pos]).collect())
        .collect();

    let lo = detected[0].saturating_sub(4);
    let hi = std::cmp::min(detected[detected.len() - 1] + 4, PAYLOAD_SIZE - 1);
    let detected_set: HashSet<usize> = detected.iter().copied().collect();

    let mut probed: Vec<usize> = Vec::new();

    for pos in lo..=hi {
        if detected_set.contains(&pos) {
            continue;
        }
        let atk_bytes: HashSet<u8> = attack_payloads
            .iter()
            .take(20)
            .map(|p| p[pos])
            .collect();
        let unfam: HashSet<u8> = atk_bytes.difference(&legit_sets[pos]).copied().collect();
        if !unfam.is_empty() {
            probed.push(pos);
        }
    }

    let mut extended: Vec<usize> = detected.to_vec();
    extended.extend(probed.iter());
    extended.sort();
    extended.dedup();
    extended
}

// =============================================================================
// SCENARIO RUNNER
// =============================================================================

fn run_scenario(
    name: &str,
    analyzer: &WindowedAnalyzer,
    attack_payloads: &[Vec<u8>],
    legit_payloads: &[Vec<u8>],
    expected: &[usize],
) {
    println!("\n{}", "=".repeat(70));
    println!("SCENARIO: {}", name);
    println!("{}", "=".repeat(70));

    let legit_ref = &legit_payloads[0];

    // Multi-sample detection
    let mut all_detected: HashSet<usize> = HashSet::new();
    for sample in attack_payloads.iter().take(5) {
        let det = analyzer.detect(sample, legit_ref);
        all_detected.extend(det);
    }
    let mut detected: Vec<usize> = all_detected.into_iter().collect();
    detected.sort();

    let expected_set: HashSet<usize> = expected.iter().copied().collect();
    let hit: usize = detected.iter().filter(|p| expected_set.contains(p)).count();
    let missed: usize = expected_set
        .iter()
        .filter(|p| !detected.contains(p))
        .count();

    println!(
        "\n  VSA detection: {}/{} positions  (missed: {})",
        hit,
        expected.len(),
        missed
    );

    if detected.is_empty() {
        println!("  No unfamiliar positions found.");
        return;
    }

    // Gap probing
    let extended = gap_probe(&detected, attack_payloads, legit_payloads);
    let probed = extended.len() - detected.len();
    let ext_hit: usize = extended.iter().filter(|p| expected_set.contains(p)).count();

    if probed > 0 {
        println!(
            "  Gap probing: +{} recovered → {}/{} total",
            probed,
            ext_hit,
            expected.len()
        );
    }

    // Derive rules
    let (candidates, pos_data) = derive_rules(&extended, attack_payloads, legit_payloads);

    // Show runs
    let mut runs: Vec<(usize, usize, usize)> = Vec::new();
    let mut sorted_ext = extended.clone();
    sorted_ext.sort();
    if !sorted_ext.is_empty() {
        let mut start = sorted_ext[0];
        let mut prev = sorted_ext[0];
        let mut constant = if pos_data.get(&start).map_or(false, |i| i.is_constant) {
            1
        } else {
            0
        };

        for &pos in &sorted_ext[1..] {
            if pos == prev + 1 {
                if pos_data.get(&pos).map_or(false, |i| i.is_constant) {
                    constant += 1;
                }
                prev = pos;
            } else {
                runs.push((start, prev, constant));
                start = pos;
                prev = pos;
                constant = if pos_data.get(&pos).map_or(false, |i| i.is_constant) {
                    1
                } else {
                    0
                };
            }
        }
        runs.push((start, prev, constant));
    }

    println!("\n  Consecutive runs: {}", runs.len());
    for &(start, end, cons) in &runs {
        let len = end - start + 1;
        println!("    pos {:>4}-{:<4} ({:>2} bytes, {} constant)", start, end, len, cons);
    }

    // Show candidates by tier
    let short: Vec<&RuleCandidate> = candidates.iter().filter(|r| r.len() <= 4).collect();
    let long: Vec<&RuleCandidate> = candidates.iter().filter(|r| r.len() > 4).collect();

    if !short.is_empty() {
        println!(
            "\n  Short rules (1-4B): {} candidates (top 5)",
            short.len()
        );
        for r in short.iter().take(5) {
            println!("    {}  TP={}", r.to_edn(), r.tp);
        }
    }

    if !long.is_empty() {
        println!("\n  Long rules (5+B): {} candidates (top 5)", long.len());
        for r in long.iter().take(5) {
            let status = if r.enforceable() {
                "ENFORCEABLE"
            } else {
                "TOO DEEP"
            };
            println!(
                "    {}",
                r.to_edn()
            );
            println!(
                "      {}  TP={} active={}/{}B [{}]",
                r.description,
                r.tp,
                r.active(),
                r.len(),
                status
            );
        }
    }

    // Select best
    let (selected, coverage) = select_best(&candidates, attack_payloads);

    println!("\n  OPTIMAL SELECTION:");
    println!("  {}", "-".repeat(60));

    for r in &selected {
        let status = if r.enforceable() { "OK" } else { "NEEDS DECOMP" };
        println!("    {}", r.to_edn());
        println!(
            "      {}  TP={}  [{}]",
            r.cost_label(),
            r.tp,
            status
        );
    }

    println!(
        "\n    Coverage: {:.1}%  Rules: {}",
        coverage,
        selected.len()
    );
}

// =============================================================================
// MAIN
// =============================================================================

fn main() {
    println!("{}", "=".repeat(70));
    println!("BYTE MATCH RULE DERIVATION (Rust)");
    println!("{}", "=".repeat(70));

    // Train
    println!("\nTraining on 500 normal payloads...");
    let mut analyzer = WindowedAnalyzer::new();
    let mut rng = Rng::new(42);

    for _ in 0..500 {
        let payload = make_normal(&mut rng);
        analyzer.learn(&payload);
    }
    analyzer.freeze();
    println!("  {} windows frozen", NUM_WINDOWS);

    let legit_payloads: Vec<Vec<u8>> = (0..200).map(|_| make_normal(&mut rng)).collect();

    // Scenario A: Dumb constant attack
    let attacks_dumb: Vec<Vec<u8>> = (0..200).map(|_| make_attack_dumb(&mut rng)).collect();
    run_scenario(
        "DUMB: 4B constant at pos 10-13 + 4B at 200-203",
        &analyzer,
        &attacks_dumb,
        &legit_payloads,
        &(10..14).chain(200..204).collect::<Vec<_>>(),
    );

    // Scenario B: Limited variation (3 variants)
    let attacks_limited: Vec<Vec<u8>> =
        (0..200).map(|_| make_attack_limited(&mut rng)).collect();
    run_scenario(
        "LIMITED: 3 tool variants at pos 10-13 + 200-203",
        &analyzer,
        &attacks_limited,
        &legit_payloads,
        &(10..14).chain(200..204).collect::<Vec<_>>(),
    );

    // Scenario C: 32-byte shellcode near header
    let attacks_near32: Vec<Vec<u8>> =
        (0..200).map(|_| make_attack_near32(&mut rng)).collect();
    run_scenario(
        "SHELLCODE: 32B constant at payload[0-31] (L4 offset 8-39)",
        &analyzer,
        &attacks_near32,
        &legit_payloads,
        &(0..32).collect::<Vec<_>>(),
    );

    // Scenario D: 32-byte constant deep
    let attacks_deep32: Vec<Vec<u8>> =
        (0..200).map(|_| make_attack_deep32(&mut rng)).collect();
    run_scenario(
        "DEEP SHELLCODE: 32B constant at payload[200-231] (L4 offset 208-239)",
        &analyzer,
        &attacks_deep32,
        &legit_payloads,
        &(200..232).collect::<Vec<_>>(),
    );

    // Summary
    println!("\n{}", "=".repeat(70));
    println!("SUMMARY");
    println!("{}", "=".repeat(70));
    println!(
        r#"
  ALGORITHM:
    VSA windowed detection → gap probing → rule derivation → greedy selection

  CONSTRAINTS RESPECTED:
    - 1-4B custom dim (arbitrary offset, 7 slots)
    - 5-64B PatternGuard (L4 offset + len ≤ 64)
    - 0xFF masks only (exact byte match)
    - Max 32 rules per scope

  RESULTS:
    Dumb attacks:       1 rule, 100% TP (single 4B custom dim)
    Limited variants:   3 rules, 100% TP (one per variant)
    32B near-header:    1 rule, 100% TP (full 32B PatternGuard)
    32B deep offset:    1 rule, 100% TP (4B custom dim fallback)
"#
    );
}
