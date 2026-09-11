#pragma once
// ============================================================================
// env/environment.hpp — the world for Goal 1: a two-odor bandit.
//
// Each trial the world presents odor A ("good": reward when approached) or
// odor B ("bad": punishment when approached) with 50/50 probability.
// Avoiding an odor ends the trial with no outcome (like a real opt-out).
// Outcomes are stochastic by default (85%) so the agent must integrate
// evidence over trials — a genuine bandit, not a lookup task.
// ============================================================================
#include "core/rng.hpp"
#include "core/types.hpp"
#include "util/odors.hpp"

namespace malefly {

struct BanditConfig {
    f32 p_reward_good = 0.85f;   // P(reward    | approach good odor)
    f32 p_punish_bad = 0.85f;    // P(punishment| approach bad odor)
    bool deterministic = false;  // if true, outcomes are certain (for tests)
};

class BanditEnv {
public:
    BanditEnv(const Odor& good, const Odor& bad, const BanditConfig& cfg, Rng& rng);

    void begin_trial();                       // pick odor 50/50
    const Odor& odor() const { return *current_; }
    bool current_is_good() const { return current_is_good_; }

    // called AFTER the agent acts; returns +1 / -1 / 0
    f32 apply(Action a);

private:
    const Odor& good_;
    const Odor& bad_;
    BanditConfig cfg_;
    Rng& rng_;
    const Odor* current_ = nullptr;
    bool current_is_good_ = false;
};

}  // namespace malefly
