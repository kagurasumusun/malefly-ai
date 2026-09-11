#include "circuits/antennal_lobe.hpp"
#include <cmath>

namespace malefly {

AntennalLobe::AntennalLobe(const AntennalLobeConfig& cfg, Rng& rng)
    : cfg_(cfg), rng_(rng),
      pn_(cfg.n_glom, LifConfig{}),
      ln_(cfg.n_ln, LifConfig{}),
      orn_rate_(cfg.n_glom, cfg.orn_base_hz) {}

void AntennalLobe::set_odor(const Odor* odor, f32 concentration) {
    for (u32 g = 0; g < cfg_.n_glom; ++g) {
        const f32 s = odor ? odor->profile[g] : 0.0f;
        orn_rate_[g] = cfg_.orn_base_hz + cfg_.orn_max_hz * concentration * s;
    }
}

void AntennalLobe::step(f32 dt) {
    t_ += dt;
    // AL oscillation: shared rate modulation paces PN spike synchrony
    const f32 osc = 1.0f + cfg_.osc_depth * std::sin(6.2831853f * cfg_.osc_hz * t_);

    // 1) ORN transduction: inhomogeneous Poisson spikes -> PNs (1:1)
    for (u32 g = 0; g < cfg_.n_glom; ++g)
        if (rng_.bernoulli(orn_rate_[g] * osc * dt)) pn_.add_exc(g, cfg_.orn_pn_quanta);

    // 2) PN integration
    pn_.step(dt);

    // 3) PN -> LN pool (all PNs to all LNs, pooled)
    const u8* ps = pn_.spikes();
    for (u32 g = 0; g < cfg_.n_glom; ++g)
        if (ps[g]) { ln_.add_exc_all(cfg_.pn_ln_quanta); pool_ops_ += cfg_.n_ln; }

    // 4) LN integration
    ln_.step(dt);

    // 5) LN -> PN global inhibition (gain control / contrast enhancement)
    const u8* ls = ln_.spikes();
    for (u32 j = 0; j < cfg_.n_ln; ++j)
        if (ls[j]) { pn_.add_inh_all(cfg_.ln_pn_quanta); pool_ops_ += cfg_.n_glom; }
}

usize AntennalLobe::memory_bytes() const {
    return pn_.memory_bytes() + ln_.memory_bytes()
         + orn_rate_.capacity() * sizeof(f32);
}

}  // namespace malefly
