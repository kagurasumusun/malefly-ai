#pragma once
// ============================================================================
// experiments/goal3.hpp — Goal 3: on-circuit adaptation (reversal learning).
//
// Rule (docs/ROADMAP.md): NO retraining of the network. Adaptation must come
// from neural activity + synaptic plasticity in the same substrate.
//
// Protocol: train A+/B- -> REVERSAL (A-/B+) -> reversal back.
// Ablation conditions isolate each adopted mechanism:
//   full        : taxonomy (Allen) + RPE gate (BRAIN) + arousal (EBRAINS) + LH
//   --no-rpe    : canonical rule without prediction-error gating
//   --no-arousal: fixed temperature/exploration
//   --no-taxonomy: uniform KC timescales (legacy single-trace MB)
//
// Pre-registered success criteria (fixed before the run):
//   C1 reversal 1: rolling-20-trial accuracy >= 0.70 under the NEW assignment
//      within 100 trials, in the full condition, on >= 2 of 3 seeds.
//   C2 the full condition does not degrade Goal-1-level final accuracy
//      (>= 0.75 before reversal).
//   C3 RPE gate / arousal / taxonomy: adopted only if they improve trials-to-
//      criterion or stability in the measured comparison (otherwise rejected
//      and recorded in RESEARCH.md).
// ============================================================================
#include "core/types.hpp"
#include <string>

namespace malefly {

struct Goal3Config {
    u64 seed = 42;
    u32 n_train = 100;        // phase 1: A+ / B-
    u32 n_reversal = 100;     // phase 2: A- / B+
    u32 n_back = 60;          // phase 3: A+ / B- again
    u32 n_probe_each = 15;    // outcome-free probes between phases
    u32 block = 20;           // reporting block size
    bool use_rpe = true;
    bool use_arousal = true;
    bool use_taxonomy = true;
    std::string tag = "full";
    std::string out_json = "results/goal3.json";
    std::string out_csv = "results/goal3.csv";
};

int run_goal3(const Goal3Config& cfg);

}  // namespace malefly
