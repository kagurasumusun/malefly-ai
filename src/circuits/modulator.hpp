#pragma once
// ============================================================================
// circuits/modulator.hpp — mesoscopic modulatory state ("arousal").
//
// Adoption from HBP/EBRAINS (multiscale co-simulation, NEST<->TVB): a
// macroscopic (population-level) state variable coupled bidirectionally with
// the microscopic spiking circuit — outcomes perturb the macro state, the
// macro state modulates microscopic decision dynamics (temperature,
// exploration) and thereby future spiking. This is also the E2 "emotion
// dimension 2" of docs/ROADMAP.md: a persistent internal state that modulates
// behavior structurally, not a bolt-on label.
//
// Ablatable (enabled=false) for before/after measurement.
// ============================================================================
#include "core/types.hpp"
#include <algorithm>
#include <cmath>

namespace malefly {

struct ModulatorConfig {
    f32 tau_arousal = 20.0f;  // s, persistence of the internal state
    f32 gain_rew = 0.25f;     // arousal rise per unit reward
    f32 gain_pun = 0.30f;     // arousal rise per unit punishment
    bool enabled = true;
};

class Modulator {
public:
    explicit Modulator(const ModulatorConfig& cfg) : cfg_(cfg) {}

    void step(f32 dt) {
        if (cfg_.enabled) arousal_ *= std::exp(-dt / cfg_.tau_arousal);
    }

    void on_outcome(f32 r) {
        if (!cfg_.enabled || r == 0.0f) return;
        arousal_ += (r > 0.0f) ? cfg_.gain_rew * r : cfg_.gain_pun * (-r);
        arousal_ = std::clamp(arousal_, 0.0f, 1.0f);
    }

    f32 arousal() const { return arousal_; }
    void set_for_test(f32 a) { arousal_ = std::clamp(a, 0.0f, 1.0f); }

private:
    ModulatorConfig cfg_;
    f32 arousal_ = 0.0f;
};

}  // namespace malefly
