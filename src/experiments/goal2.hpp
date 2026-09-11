#pragma once
// ============================================================================
// experiments/goal2.hpp — Goal 2: memory.
//
// Three fixed-in-advance measurements over the trained MaleCNS-MB circuit
// (PRINCIPLES: criteria are stated before the run; results go to RESULTS.md):
//
//  retention    learned valence survives a 120 s reinforcement-free gap
//               (ongoing spontaneous activity; tests consolidation, absence
//               of catastrophic forgetting).
//  interference a novel odor E sharing ~half its glomerular support with A is
//               trained as E+; A/B memories must persist (and structural
//               transfer through pattern overlap is quantified).
//  delayed      sample odor (A or B) for 1 s -> odor OFF for D s -> choice.
//  choice (STM) Decision is computed from the *circuit-state* eligibility
//               trace (decaying), with the software-counter readout measured
//               alongside as a labeled contrast (no fake memory claims).
// ============================================================================
#include "core/types.hpp"
#include <string>
#include <vector>

namespace malefly {

struct Goal2Config {
    u64 seed = 100;
    u32 n_train_trials = 150;       // phase 1: A+ / B- bandit (as Goal 1)
    u32 n_probe_each = 20;          // probe presentations per odor per round
    u32 n_interfere_trials = 40;    // phase 3: E+ training trials
    f32 gap_s = 120.0f;             // retention gap (clean air, spontaneous only)
    std::vector<u32> stm_delays_ms = {0, 500, 1000, 2000, 4000};
    u32 n_stm_reps = 24;            // per (delay, odor)
    u32 n_glom = 50;
    u32 t_on_ms = 500;
    u32 t_outcome_ms = 200;
    u32 t_iti_ms = 500;
    u32 sample_ms = 1000;           // STM sample presentation length
    f32 concentration = 1.0f;
    std::string out_json = "results/goal2.json";
    std::string out_csv = "results/goal2_stm.csv";
};

int run_goal2(const Goal2Config& cfg);

}  // namespace malefly
