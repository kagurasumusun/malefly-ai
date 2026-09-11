#include "circuits/antennal_lobe.hpp"
#include <cmath>

namespace malefly {
namespace {

LifConfig ln_config(f32 tau_inh) {
    LifConfig c;
    c.tau_inh = tau_inh;
    return c;
}

}  // namespace

AntennalLobe::AntennalLobe(const AntennalLobeConfig& cfg, Rng& rng)
    : cfg_(cfg), rng_(rng),
      pn_(cfg.n_glom, LifConfig{}),
      ln_fast_(cfg.ln_subclasses ? cfg.n_ln_fast : cfg.n_ln_fast + cfg.n_ln_slow,
               ln_config(cfg.ln_subclasses ? cfg.tau_inh_fast : 0.010f)),
      ln_slow_(cfg.ln_subclasses ? cfg.n_ln_slow : 0,
               ln_config(cfg.ln_subclasses ? cfg.tau_inh_slow : 0.010f)),
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

    // 3) PN -> LN subclass pools (all PNs to all LNs, pooled per subclass)
    const u8* ps = pn_.spikes();
    if (ln_fast_.size() > 0) {
        const f32 q = cfg_.ln_subclasses ? cfg_.ln_fast_pn_quanta
                                         : cfg_.legacy_pn_ln_quanta;
        for (u32 g = 0; g < cfg_.n_glom; ++g)
            if (ps[g]) { ln_fast_.add_exc_all(q); pool_ops_ += ln_fast_.size(); }
    }
    if (ln_slow_.size() > 0) {
        for (u32 g = 0; g < cfg_.n_glom; ++g)
            if (ps[g]) { ln_slow_.add_exc_all(cfg_.ln_slow_pn_quanta); pool_ops_ += ln_slow_.size(); }
    }

    // 4) LN integration
    ln_fast_.step(dt);
    if (ln_slow_.size() > 0) ln_slow_.step(dt);

    // 5) LN -> PN pooled inhibition (gain control / contrast enhancement)
    {
        const u8* s = ln_fast_.spikes();
        const f32 q = cfg_.ln_subclasses ? cfg_.ln_fast_back_quanta
                                         : cfg_.legacy_ln_quanta;
        for (u32 j = 0; j < ln_fast_.size(); ++j)
            if (s[j]) { pn_.add_inh_all(q); pool_ops_ += cfg_.n_glom; }
    }
    if (ln_slow_.size() > 0) {
        const u8* s = ln_slow_.spikes();
        for (u32 j = 0; j < ln_slow_.size(); ++j)
            if (s[j]) { pn_.add_inh_all(cfg_.ln_slow_back_quanta); pool_ops_ += cfg_.n_glom; }
    }
}

usize AntennalLobe::memory_bytes() const {
    return pn_.memory_bytes() + ln_fast_.memory_bytes() + ln_slow_.memory_bytes()
         + orn_rate_.capacity() * sizeof(f32);
}

}  // namespace malefly
