#pragma once
// ============================================================================
// circuits/action_selection.hpp — sensorimotor mapping: MBON valence -> action.
//
// In the fly, MBONs converge onto downstream command/selection structures
// (CX, LH, descending pathways). Here: stochastic choice over
// {Approach, Avoid} from
//   - MBON valence balance (learned, plastic),
//   - innate valence from the lateral horn (stimulus-dependent, hardwired —
//     Brain/MINDS-style region plug-in),
//   - arousal-modulated temperature/exploration (EBRAINS multiscale adoption,
//     E2 internal state).
// Legacy decide() (no context) preserved for backward-compatible experiments.
// ============================================================================
#include "circuits/mushroom_body.hpp"
#include "core/rng.hpp"
#include <algorithm>
#include <cmath>

namespace malefly {

inline f32 sigmoidf(f32 x) { return 1.0f / (1.0f + std::exp(-x)); }

struct Decision {
    bool approach = false;
    f32 p_approach = 0.5f;
    f32 valence = 0.0f;
    bool weak_signal = false;  // no KCs active -> innate/LH default applies
};

// decision context: internal-state / innate inputs (all optional)
struct DecisionContext {
    f32 innate_valence = 0.0f;  // lateral-horn stimulus-dependent attraction
    f32 arousal = 0.0f;         // modulator state in [0,1]
    bool use_arousal = false;   // EBRAINS adoption ablation flag
};

inline Decision decide_ctx(const MushroomBody::Readout& r,
                           const MushroomBodyConfig& cfg, Rng& rng,
                           bool sample_action, const DecisionContext& ctx) {
    Decision d;
    d.weak_signal = r.kc_active_frac < 0.005f;
    // learned valence; weak MB signal -> decision rests on the innate path
    const f32 v_mb = d.weak_signal ? 0.0f : r.valence;
    // trained MB output suppresses the innate LH path (learned behavior wins
    // over innate attraction — necessary for avoidance AND reversal)
    const f32 innate_gate =
        1.0f - std::min(1.0f, std::fabs(v_mb) / cfg.innate_gate_vmax);
    const f32 v = v_mb + ctx.innate_valence * innate_gate;
    d.valence = v;

    // arousal modulates decision dynamics (EBRAINS multiscale / E2):
    // prediction errors raise arousal; arousal raises exploration (temperature
    // and epsilon). Rationale: after violations the agent must re-sample the
    // changed world; the measured reversal experiment (docs/RESULTS.md §8)
    // decides whether this direction of modulation is the useful one.
    f32 temp = cfg.decision_temp;
    f32 eps = cfg.epsilon;
    if (ctx.use_arousal) {
        temp *= (1.0f + 0.75f * ctx.arousal);
        eps *= (1.0f + 1.0f * ctx.arousal);
    }

    const f32 x = (v + cfg.innate_bias - cfg.naive_valence) / temp;
    d.p_approach = sigmoidf(x);
    if (!sample_action) {
        d.approach = d.p_approach > 0.5f;
        return d;
    }
    if (rng.uniform01() < eps)
        d.approach = rng.uniform01() < 0.5f;
    else
        d.approach = rng.uniform01() < d.p_approach;
    return d;
}

// legacy path (pre-adoption behavior): constant naive valence, no arousal
inline Decision decide(const MushroomBody::Readout& r,
                       const MushroomBodyConfig& cfg, Rng& rng,
                       bool sample_action) {
    Decision d;
    d.weak_signal = r.kc_active_frac < 0.005f;
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
