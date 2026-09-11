#pragma once
// ============================================================================
// experiments/goal1.hpp — Goal 1: autonomous value learning.
//
// Question: can a MaleCNS-derived circuit (AL -> MB -> MBON valence balance,
// DAN-modulated plasticity) learn, from raw experience alone, to approach the
// rewarded odor and avoid the punished one — and does learned valence
// generalize to novel odors in proportion to circuit overlap (i.e., is the
// memory a structured neural representation rather than a lookup table)?
//
// Success criteria are fixed in docs/ROADMAP.md *before* the experiment and
// all results are measured from the actual run (PRINCIPLES.md).
// ============================================================================
#include "core/types.hpp"
#include <string>

namespace malefly {

struct Goal1Config {
    u64 seed = 42;
    u32 n_train_trials = 200;
    u32 n_probe_each = 25;      // probe presentations per odor per round
    u32 n_glom = 50;
    u32 t_on_ms = 500;          // odor presentation window
    u32 t_outcome_ms = 200;     // odor offset -> DAN reinforcement delay
    u32 t_iti_ms = 500;         // inter-trial interval
    f32 concentration = 1.0f;
    bool use_lh = true;         // lateral-horn innate valence (ablatable)
    bool neural_arbiter = true; // neural action integrator (false = legacy
                                // software arbiter, for before/after only)
    bool use_modes = true;      // slow rest/active mode network (ablatable)
    bool develop = false;       // activity-dependent pruning before training
    std::string out_json = "results/goal1.json";
    std::string out_csv = "results/goal1.csv";
};

int run_goal1(const Goal1Config& cfg);

}  // namespace malefly
