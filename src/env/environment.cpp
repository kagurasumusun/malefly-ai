#include "env/environment.hpp"

namespace malefly {

BanditEnv::BanditEnv(const Odor& good, const Odor& bad,
                     const BanditConfig& cfg, Rng& rng)
    : good_(good), bad_(bad), cfg_(cfg), rng_(rng) {}

void BanditEnv::begin_trial() {
    current_is_good_ = rng_.bernoulli(0.5f);
    current_ = current_is_good_ ? &good_ : &bad_;
}

f32 BanditEnv::apply(Action a) {
    if (a != Action::Approach) return 0.0f;  // avoiding: no outcome
    if (current_is_good_)
        return (cfg_.deterministic || rng_.bernoulli(cfg_.p_reward_good)) ? 1.0f : 0.0f;
    return (cfg_.deterministic || rng_.bernoulli(cfg_.p_punish_bad)) ? -1.0f : 0.0f;
}

}  // namespace malefly
