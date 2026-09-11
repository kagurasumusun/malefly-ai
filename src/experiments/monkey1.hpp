#pragma once
// ============================================================================
// experiments/monkey1.hpp — primate-cognition benchmark suite.
//
// Human/monkey-derived tasks run on the SAME MaleCNS substrate plus the two
// new primate modules (Hippocampus, PFC). Success criteria fixed in advance.
//
//  A. delayed match-to-sample (DMS; Funahashi/Goldman-Rakic paradigm):
//     sample odor -> odorless delay D -> probe (same odor or a lure).
//     Behavior: the agent approaches on MATCH (familiarity signal), avoids on
//     NON-MATCH. Condition set: D in {0, 1, 2, 4} s x {no distractor,
//     distractor in active mode, distractor in rest mode}.
//     Success: accuracy > 0.7 at D=1s without distractor; distractor damage
//     significantly smaller in active mode than rest mode (top-down gating).
//  B. episodic one-shot + cued recall (CLS/DG-CA3):
//     single presentations A+ / B- (one US each), 60 s gap, then PARTIAL cues
//     (65% of A's glomeruli). Success: completion quality (CA3 overlap with
//     the encoded episode's fingerprint) > uncued baseline, valence recall
//     accuracy > chance.
//  C. retrieval-practice consolidation (test-enhanced learning):
//     re-exposure of A/B 3 times spaced -> MB valence retention at +120 s
//     compared against no-practice control.
//  D. spacing effect (Ebbinghaus): massed 30 vs spaced 30 training on a new
//     odor pair X+/Y- -> probe accuracy at +60 s. Spaced must win.
// ============================================================================
#include "core/types.hpp"
#include <string>

namespace malefly {

struct Monkey1Config {
    u64 seed = 300;
    u32 n_dms_reps = 20;         // per (delay, condition)
    u32 t_sample_ms = 500;
    u32 t_probe_ms = 500;
    u32 t_iti_ms = 700;
    u32 t_gap_s = 60;            // episodic gap (B)
    f32 cue_frac = 0.65f;        // partial-cue fraction (B)
    u32 n_practice = 3;          // retrieval practices (C)
    u32 n_spacing = 30;          // trials per condition (D)
    std::string out_json = "results/monkey1.json";
    std::string out_csv = "results/monkey1_dms.csv";
};

int run_monkey1(const Monkey1Config& cfg);

}  // namespace malefly
