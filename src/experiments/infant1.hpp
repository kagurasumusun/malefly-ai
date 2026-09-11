#pragma once
// ============================================================================
// experiments/infant1.hpp — AI-monkey / human-infant level language benchmarks
// on the wiring-only substrate (AI body interface = syllable token stream).
//   L1 word segmentation   (Saffran, Aslin & Newport 1996)
//   L2 babble/imitation + self-other  (Eliades & Wang 2003; Kuhl 2004)
//   L3 word-reward grounding          (same MB pathway as odors)
//   L4 contingent reply / word completion (TRP pattern completion)
// ============================================================================
#include "core/types.hpp"
#include <string>

namespace malefly {

struct Infant1Config {
    u64 seed = 300;
    std::string out = "results/infant1.json";
    u32 l1_words = 200;       // 200 tri-syllabic words (~90 s stream)
    u32 l3_trials = 12;
    u32 l4_pairs = 60;
    u32 l4_probes = 10;
};

int run_infant1(const Infant1Config& cfg);

}  // namespace malefly
