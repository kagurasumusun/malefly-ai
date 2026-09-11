// ============================================================================
// circuits/modulator.hpp — mesoscopic modulatory state network ("arousal" +
// behavioral mode).
//
// Adoption from HBP/EBRAINS (multiscale co-simulation, NEST<->TVB): macroscopic
// (population-level) state variables coupled bidirectionally with the
// microscopic spiking circuit. Extended into a small SLOW STATE NETWORK:
//   node 1 arousal — raised by outcome prediction errors, decays (tau 20 s);
//   node 2 mode    — slow spontaneous switching between an active,
//                    exploratory state and a rest-like passive state
//                    (task-positive vs default/rest antagonism — DMN
//                    principle; ZBrain whole-brain state switching /
//                    futility-induced passivity; hypothalamic state gating of
//                    approach/avoidance). Mode gates the action integrator's
//                    base drive (rest -> passivity / opt-outs).
// The nodes are coupled (arousal shortens rest episodes) — the seed of the
// TVB-style region graph. Arousal is also the E2 "emotion dimension 2".
// All ablatable (enabled / modes_enabled = false) for before/after measurement.
// ============================================================================
#pragma once
#include "core/rng.hpp"
#include "core/types.hpp"
#include <algorithm>
#include <cmath>

namespace malefly {

struct ModulatorConfig {
    f32 tau_arousal = 20.0f;  // s, persistence of the internal state
    f32 gain_rew = 0.25f;     // arousal rise per unit reward
    f32 gain_pun = 0.30f;     // arousal rise per unit punishment
    bool enabled = true;

    // slow behavioral-mode network (DMN/ZBrain adoption; ablatable)
    bool modes_enabled = true;
    f32 mode_rest_hz = 0.02f;          // active -> rest (mean episode 50 s)
    f32 mode_active_hz = 0.05f;        // rest -> active (shortened by arousal)
    f32 mode_arousal_coupling = 3.0f;  // arousal multiplies wake-up rate
};

class Modulator {
public:
    explicit Modulator(const ModulatorConfig& cfg) : cfg_(cfg) {}

    void step(f32 dt, Rng* rng = nullptr) {
        if (cfg_.enabled) arousal_ *= std::exp(-dt / cfg_.tau_arousal);
        if (cfg_.modes_enabled && rng) {
            if (rest_) {
                const f32 rate =
                    cfg_.mode_active_hz *
                    (1.0f + cfg_.mode_arousal_coupling * arousal_);
                if (rng->bernoulli(rate * dt)) rest_ = false;
            } else if (rng->bernoulli(cfg_.mode_rest_hz * dt)) {
                rest_ = true;
            }
            ++mode_ms_;
            if (rest_) ++rest_ms_;
        }
    }

    void on_outcome(f32 r) {
        if (!cfg_.enabled || r == 0.0f) return;
        arousal_ += (r > 0.0f) ? cfg_.gain_rew * r : cfg_.gain_pun * (-r);
        arousal_ = std::clamp(arousal_, 0.0f, 1.0f);
    }

    f32 arousal() const { return arousal_; }
    bool rest() const { return cfg_.modes_enabled && rest_; }
    f64 rest_frac() const {
        return mode_ms_ ? static_cast<f64>(rest_ms_) / static_cast<f64>(mode_ms_) : 0.0;
    }
    void set_for_test(f32 a) { arousal_ = std::clamp(a, 0.0f, 1.0f); }
    void set_for_test_rest(bool r) { rest_ = r; }

private:
    ModulatorConfig cfg_;
    f32 arousal_ = 0.0f;
    bool rest_ = false;
    u64 mode_ms_ = 0, rest_ms_ = 0;
};

}  // namespace malefly
