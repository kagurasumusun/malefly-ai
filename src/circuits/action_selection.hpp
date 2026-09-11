#pragma once
// ============================================================================
// circuits/action_selection.hpp — sensorimotor mapping: MBON valence -> action.
//
// In the fly, MBONs converge onto downstream command/selection structures
// (CX, LH, descending pathways). Here: stochastic choice over
// {Approach, Avoid} from the MBON valence balance, with
//   - innate approach bias (stand-in for unmodeled lateral-horn innate path),
//   - temperature-based Bernoulli sampling (exploration),
//   - epsilon floor (guaranteed exploration).
// This is intentionally minimal for Goal 1; the central complex ring
// attractor will take over selection in Goal 4 (docs/ROADMAP.md).
// ============================================================================
#include "circuits/mushroom_body.hpp"
#include "core/rng.hpp"
#include <cmath>

namespace malefly {

inline f32 sigmoidf(f32 x) { return 1.0f / (1.0f + std::exp(-x)); }

struct Decision {
    bool approach = false;
    f32 p_approach = 0.5f;
    f32 valence = 0.0f;
    bool weak_signal = false;  // no KCs active -> innate default applies
};

inline Decision decide(const MushroomBody::Readout& r,
                       const MushroomBodyConfig& cfg, Rng& rng,
                       bool sample_action) {
    Decision d;
    d.weak_signal = r.kc_active_frac < 0.005f;
    // weak-signal default: no MB activation -> innate approach tendency
    // (documented stand-in for the lateral-horn innate pathway)
    const f32 v = d.weak_signal ? cfg.naive_valence : r.valence;
    d.valence = v;
    const f32 x = (v + cfg.innate_bias - cfg.naive_valence) / cfg.decision_temp;
    d.p_approach = sigmoidf(x);
    if (!sample_action) {
        d.approach = d.p_approach > 0.5f;
        return d;
    }
    if (rng.uniform01() < cfg.epsilon)
        d.approach = rng.uniform01() < 0.5f;
    else
        d.approach = rng.uniform01() < d.p_approach;
    return d;
}

}  // namespace malefly
