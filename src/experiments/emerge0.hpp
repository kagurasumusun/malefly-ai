#pragma once
// ============================================================================
// experiments/emerge0.hpp — "wire it and it already behaves".
//
// The zero-training experiment demanded by the design principle: connect the
// Brain, put it in an EMPTY world (no odors, no rewards, no protocol except
// time passing), and MEASURE what the wiring does on its own:
//   - spontaneous motor-nerve activity (action without any program)
//   - rest/active bouts (DMN/ZBrain state switching)
//   - side-switch events (the circuit "changes its mind" — future directedness)
//   - arousal transients
// Also a second phase: odors appear (still NO reward, NO training) — the
// naive wiring should show approach-biased asymmetric behavior (innate LH
// pathway) that a lookup table could not have.
// ============================================================================
#include "core/types.hpp"
#include <string>

namespace malefly {

struct Emerge0Config {
    u64 seed = 5;
    u32 empty_ms = 60000;        // phase 1: empty world
    u32 odor_ms = 60000;         // phase 2: alternating odors, still no outcomes
    u32 bin_ms = 1000;           // behavioral bin
    f32 motor_bout_thresh = 40;  // population spikes/bin that count as a bout
    std::string out_json = "results/emerge0.json";
    std::string out_csv = "results/emerge0.csv";
};

int run_emerge0(const Emerge0Config& cfg);

}  // namespace malefly
