#include "circuits/mushroom_body.hpp"
#include <algorithm>
#include <cmath>

namespace malefly {

MushroomBody::MushroomBody(const MushroomBodyConfig& cfg, Rng& rng, u32 n_pn)
    : cfg_(cfg), n_pn_(n_pn),
      kc_(cfg.n_kc, [&] {
          LifConfig c;
          c.tau_exc = cfg.kc_tau_exc;
          c.v_thresh = cfg.kc_v_thresh;
          return c;
      }()),
      rng_(rng) {
    // sparse random PN->KC expansion (fixed; NOT plastic — this is connectome
    // structure, and in vivo it is developmentally hardwired)
    pn2kc_ = make_random_fanin(n_pn, cfg.n_kc, cfg.kc_fanin, rng, cfg.kc_quanta, 0.0f);
    // valence branches: naive flies mildly approach odors (LH-driven default):
    // approach branch initialized stronger than avoid branch
    w_appr_.resize(cfg.n_kc);
    w_avoid_.resize(cfg.n_kc);
    for (u32 k = 0; k < cfg.n_kc; ++k) {
        w_appr_[k] = rng.uniform(cfg.w_appr_init_lo, cfg.w_appr_init_hi);
        w_avoid_[k] = rng.uniform(cfg.w_avoid_init_lo, cfg.w_avoid_init_hi);
    }
    elig_.assign(cfg.n_kc, 0.0f);
    trial_counts_.assign(cfg.n_kc, 0);
    trial_pattern_.assign(cfg.n_kc, 0);
}

void MushroomBody::step(f32 dt, const u8* pn_spikes) {
    // 1) PN -> KC feedforward
    pn2kc_.propagate(pn_spikes, kc_.ge());

    // 2) APL-like pooled feedback: last step's KC spikes inhibit all KCs
    //    (one-step synaptic delay; enforces sparsity robustly)
    {
        const u8* last = kc_.spikes();
        u32 c = 0;
        for (u32 k = 0; k < cfg_.n_kc; ++k) c += last[k];
        if (c) kc_.add_inh_all(cfg_.apl_gain * static_cast<f32>(c));
    }

    // 3) KC integration (coincidence detection)
    kc_.step(dt);

    // 4) eligibility trace + per-trial counters from new spikes
    const f32 decay = std::exp(-dt / cfg_.tau_elig);
    const u8* sp = kc_.spikes();
    for (u32 k = 0; k < cfg_.n_kc; ++k) {
        if (sp[k]) {
            elig_[k] = std::min(1.0f, elig_[k] * decay + 1.0f);
            ++trial_counts_[k];
            trial_pattern_[k] = 1;
            mbon_ops_ += 2;  // KC->MBON readout touches both branches
        } else {
            elig_[k] *= decay;
        }
    }
}

void MushroomBody::apply_reinforcement(f32 reward) {
    if (reward == 0.0f) return;
    // canonical MB sign rule (Aso et al. 2014; Felsenberg et al. 2018):
    //   reward    DANs depress KC->avoid-branch synapses  (odor becomes +)
    //   punishment DANs depress KC->approach-branch synapses (odor becomes -)
    const f32 eta = reward > 0.0f ? cfg_.eta_reward : cfg_.eta_punish;
    std::vector<f32>& w = reward > 0.0f ? w_avoid_ : w_appr_;
    for (u32 k = 0; k < cfg_.n_kc; ++k) {
        const f32 e = elig_[k];
        if (e > 0.0f) w[k] = std::max(0.0f, w[k] - eta * e);
    }
}

void MushroomBody::begin_trial() {
    std::fill(trial_counts_.begin(), trial_counts_.end(), 0u);
    std::fill(trial_pattern_.begin(), trial_pattern_.end(), u8{0});
    if (cfg_.reset_elig_per_trial)
        std::fill(elig_.begin(), elig_.end(), 0.0f);
}

MushroomBody::Readout MushroomBody::readout() const {
    // MBON readout: activity-weighted mean plastic weight over KCs that
    // spiked this trial, per branch. Valence = approach - avoid in [-1, 1].
    f64 sa = 0.0, sv = 0.0;
    u64 csum = 0;
    u32 active = 0;
    for (u32 k = 0; k < cfg_.n_kc; ++k) {
        const u32 c = trial_counts_[k];
        if (c) {
            sa += static_cast<f64>(w_appr_[k]) * c;
            sv += static_cast<f64>(w_avoid_[k]) * c;
            csum += c;
        }
        active += trial_pattern_[k];
    }
    Readout r;
    r.a_appr = csum ? static_cast<f32>(sa / csum) : 0.0f;
    r.a_avoid = csum ? static_cast<f32>(sv / csum) : 0.0f;
    r.valence = r.a_appr - r.a_avoid;
    r.kc_active_frac = static_cast<f32>(active) / static_cast<f32>(cfg_.n_kc);
    return r;
}

usize MushroomBody::memory_bytes() const {
    return kc_.memory_bytes() + pn2kc_.memory_bytes()
         + (w_appr_.capacity() + w_avoid_.capacity() + elig_.capacity()) * sizeof(f32)
         + trial_counts_.capacity() * sizeof(u32)
         + trial_pattern_.capacity() * sizeof(u8);
}

}  // namespace malefly
