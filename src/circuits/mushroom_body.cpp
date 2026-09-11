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
    // sparse random PN->KC expansion (fixed; NOT plastic — connectome structure).
    // Fixed values are an approximation: by default wiring is HETEROGENEOUS
    // (fan-in distribution + lognormal weights, docs/RESEARCH.md §MICrONS);
    // hetero_wiring=false restores the legacy homogeneous construction.
    pn2kc_ = cfg.hetero_wiring
                 ? make_random_fanin_dist(n_pn, cfg.n_kc,
                                          static_cast<f32>(cfg.kc_fanin),
                                          cfg.fanin_sd, rng, cfg.kc_quanta,
                                          cfg.w_lognorm_sigma)
                 : make_random_fanin(n_pn, cfg.n_kc, cfg.kc_fanin, rng,
                                     cfg.kc_quanta, 0.0f);

    // naive flies mildly approach odors (LH-driven default)
    w_appr_.resize(cfg.n_kc);
    w_avoid_.resize(cfg.n_kc);
    f64 sa = 0.0, sv = 0.0;
    for (u32 k = 0; k < cfg.n_kc; ++k) {
        w_appr_[k] = rng.uniform(cfg.w_appr_init_lo, cfg.w_appr_init_hi);
        w_avoid_[k] = rng.uniform(cfg.w_avoid_init_lo, cfg.w_avoid_init_hi);
        sa += w_appr_[k];
        sv += w_avoid_[k];
    }
    // baseline-referenced readout reference (opponent MBON downstream circuit:
    // valence is read as the CHANGE from the naive balance, which gives the
    // readout a symmetric bidirectional range — required for reversal, and
    // consistent with opponent MBON pairing, Aso et al. 2014)
    appr_naive_ = static_cast<f32>(sa / cfg.n_kc);
    avoid_naive_ = static_cast<f32>(sv / cfg.n_kc);

    // [Allen BICCN adoption] subtype assignment: deterministic, interleaved by
    // fractional position; fractions from MaleCNS v1.0 (generated header)
    subtype_.resize(cfg.n_kc);
    eta_scale_.resize(cfg.n_kc);
    tau_subtype_.assign(KC_N_SUBTYPES, 0.0f);
    for (int s = 0; s < KC_N_SUBTYPES; ++s)
        tau_subtype_[s] = cfg.use_taxonomy ? KC_SUBTYPES[s].tau_elig : cfg.tau_elig;
    f32 acc = 0.0f;
    int sid = 0;
    for (u32 k = 0; k < cfg.n_kc; ++k) {
        const f32 pos = (static_cast<f32>(k) + 0.5f) / static_cast<f32>(cfg.n_kc);
        while (sid < KC_N_SUBTYPES - 1 && pos > acc + KC_SUBTYPES[sid].frac) {
            acc += KC_SUBTYPES[sid].frac;
            ++sid;
        }
        subtype_[k] = static_cast<u8>(sid);
        eta_scale_[k] = cfg.use_taxonomy ? KC_SUBTYPES[sid].eta_scale : 1.0f;
    }

    // threshold heterogeneity (fixed threshold = approximation)
    if (cfg.kc_thresh_sigma > 0.0f)
        for (u32 k = 0; k < cfg.n_kc; ++k)
            kc_.set_threshold(k, cfg.kc_v_thresh +
                                     cfg.kc_thresh_sigma * rng.normal(0.0f, 1.0f));

    elig_.assign(cfg.n_kc, 0.0f);
    trial_counts_.assign(cfg.n_kc, 0);
    trial_pattern_.assign(cfg.n_kc, 0);
}

void MushroomBody::develop(Rng& rng, u32 ms) {
    // spontaneous clean-air-like input -> activity-dependent pruning.
    // Wiring (which synapses survive) EMERGES from dynamics instead of a
    // fixed construction; only the development RULE is designed.
    const u32 n_pre = pn2kc_.n_pre;
    std::vector<u32> recent(cfg_.n_kc, 0);      // spikes in trailing window
    constexpr u32 kWindow = 100;                // ms
    // start from overconnectivity
    pn2kc_ = make_random_fanin_dist(n_pre, cfg_.n_kc,
                                    static_cast<f32>(cfg_.dev_fanin),
                                    cfg_.fanin_sd, rng, cfg_.kc_quanta,
                                    cfg_.w_lognorm_sigma);
    dev_pruned_ = 0;
    u32 window_spikes_total = 0;
    std::vector<u32> window_counts(cfg_.n_kc, 0);
    for (u32 t = 0; t < ms; ++t) {
        // spontaneous ORN base-rate Poisson -> PN channels
        std::vector<u8> pn(n_pre, 0);
        for (u32 g = 0; g < n_pre; ++g)
            pn[g] = rng.bernoulli(4.0f * DT) ? 1 : 0;  // 4 Hz base rate
        step(DT, pn.data());
        // trailing-window spike bookkeeping
        const u8* sp = kc_.spikes();
        for (u32 k = 0; k < cfg_.n_kc; ++k) {
            if (window_counts[k] > 0 && t >= kWindow &&
                t % kWindow == static_cast<u32>(k) % kWindow) { /* noop */ }
        }
        for (u32 k = 0; k < cfg_.n_kc; ++k) {
            recent[k] += sp[k];
            window_spikes_total += sp[k];
        }
        if (t >= kWindow && t % 5 == 0) {
            // prune silent KCs: one input per silent KC per sweep
            for (u32 k = 0; k < cfg_.n_kc; ++k) {
                if (recent[k] == 0 &&
                    active_fanin(pn2kc_, k) > 3) {
                    if (prune_one_synapse(pn2kc_, k, rng)) ++dev_pruned_;
                }
            }
            for (u32 k = 0; k < cfg_.n_kc; ++k) recent[k] = 0;
            // stop when target sparsity reached
            if (active_frac_now() <= cfg_.dev_target_active) break;
        }
        (void)window_counts;
        (void)window_spikes_total;
    }
}

f64 MushroomBody::active_frac_now() const {
    u32 c = 0;
    for (u8 x : trial_pattern_) c += x;
    return static_cast<f64>(c) / static_cast<f64>(cfg_.n_kc);
}

f64 MushroomBody::mean_active_fanin() const {
    f64 s = 0;
    for (u32 k = 0; k < cfg_.n_kc; ++k) s += active_fanin(pn2kc_, k);
    return s / static_cast<f64>(cfg_.n_kc);
}

f64 MushroomBody::sd_active_fanin() const {
    const f64 m = mean_active_fanin();
    f64 s = 0;
    for (u32 k = 0; k < cfg_.n_kc; ++k) {
        const f64 d = static_cast<f64>(active_fanin(pn2kc_, k)) - m;
        s += d * d;
    }
    return std::sqrt(s / static_cast<f64>(std::max<u32>(1, cfg_.n_kc - 1)));
}

void MushroomBody::step(f32 dt, const u8* pn_spikes) {
    // 1) PN -> KC feedforward
    pn2kc_.propagate(pn_spikes, kc_.ge());

    // 2) APL-like pooled feedback (one-step delay)
    {
        const u8* last = kc_.spikes();
        u32 c = 0;
        for (u32 k = 0; k < cfg_.n_kc; ++k) c += last[k];
        if (c) kc_.add_inh_all(cfg_.apl_gain * static_cast<f32>(c));
    }

    // 3) KC integration
    kc_.step(dt);
    {
        const u8* s = kc_.spikes();
        u32 c = 0;
        for (u32 k = 0; k < cfg_.n_kc; ++k) c += s[k];
        last_step_spikes_ = c;
    }

    // 4) per-subtype eligibility decay + per-trial counters
    f32 decay[KC_N_SUBTYPES];
    for (int s = 0; s < KC_N_SUBTYPES; ++s)
        decay[s] = std::exp(-dt / tau_subtype_[s]);
    const u8* sp = kc_.spikes();
    for (u32 k = 0; k < cfg_.n_kc; ++k) {
        const u8 st = subtype_[k];
        if (sp[k]) {
            elig_[k] = std::min(1.0f, elig_[k] * decay[st] + 1.0f);
            ++trial_counts_[k];
            trial_pattern_[k] = 1;
            mbon_ops_ += 2;
        } else {
            elig_[k] *= decay[st];
        }
    }
}

void MushroomBody::apply_reinforcement(f32 reward) {
    if (reward == 0.0f) return;

    // [BRAIN Initiative adoption] prediction-error-gated plasticity:
    // expected valence of the just-presented odor is read from the residual
    // eligibility trace at outcome time; updates scale with |RPE|.
    f32 gate = 1.0f;
    if (cfg_.rpe_gating) {
        const f32 expected = std::clamp(trace_readout().valence, -1.0f, 1.0f);
        const f32 rpe = std::fabs(reward - expected);
        gate = std::clamp(0.30f + 0.35f * rpe, 0.30f, 1.0f);
    }
    last_gate_ = gate;

    // canonical MB sign rule (Aso et al. 2014) + opponent restoration:
    //   reward    -> depress KC->avoid   (odor becomes +), approach recovers
    //   punishment -> depress KC->approach (odor becomes -), avoid recovers
    // Without restoration the readout cannot re-value a previously punished
    // odor (approach branch already floored) — measured in RESULTS §8.
    const f32 eta = reward > 0.0f ? cfg_.eta_reward : cfg_.eta_punish;
    std::vector<f32>& w_dec = reward > 0.0f ? w_avoid_ : w_appr_;
    const f32 naive_dec = reward > 0.0f ? avoid_naive_ : appr_naive_;
    std::vector<f32>& w_opp = reward > 0.0f ? w_appr_ : w_avoid_;
    const f32 naive_opp = reward > 0.0f ? appr_naive_ : avoid_naive_;
    for (u32 k = 0; k < cfg_.n_kc; ++k) {
        const f32 e = elig_[k];
        if (e > 0.0f) {
            const f32 d = eta * gate * eta_scale_[k] * e;
            w_dec[k] = std::max(0.0f, w_dec[k] - d);
            if (cfg_.oppo_restore > 0.0f)
                w_opp[k] = std::min(naive_opp, w_opp[k] + cfg_.oppo_restore * d);
        }
    }
}

void MushroomBody::begin_trial() {
    std::fill(trial_counts_.begin(), trial_counts_.end(), 0u);
    std::fill(trial_pattern_.begin(), trial_pattern_.end(), u8{0});
    if (cfg_.reset_elig_per_trial)
        std::fill(elig_.begin(), elig_.end(), 0.0f);
}

MushroomBody::Readout MushroomBody::readout() const {
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
    r.a_appr = csum ? static_cast<f32>(sa / csum) : appr_naive_;
    r.a_avoid = csum ? static_cast<f32>(sv / csum) : avoid_naive_;
    // delta readout: change from the naive opponent balance (bidirectional)
    r.valence = (r.a_appr - appr_naive_) - (r.a_avoid - avoid_naive_);
    r.kc_active_frac = static_cast<f32>(active) / static_cast<f32>(cfg_.n_kc);
    return r;
}

MushroomBody::TraceReadout MushroomBody::trace_readout() const {
    constexpr f32 kTraceThresh = 0.05f;
    f64 s = 0.0, c = 0.0;
    u32 n = 0;
    for (u32 k = 0; k < cfg_.n_kc; ++k) {
        const f32 e = elig_[k];
        if (e > kTraceThresh) {
            s += e * (static_cast<f64>(w_appr_[k]) - w_avoid_[k]);
            c += e;
            ++n;
        }
    }
    TraceReadout r;
    // delta from naive balance (consistent with readout())
    r.valence = c > 0.0 ? static_cast<f32>(s / c - (static_cast<f64>(appr_naive_) -
                                                     static_cast<f64>(avoid_naive_)))
                        : 0.0f;
    r.active_frac = static_cast<f32>(n) / static_cast<f32>(cfg_.n_kc);
    return r;
}

std::vector<u32> MushroomBody::subtype_counts() const {
    std::vector<u32> c(KC_N_SUBTYPES, 0);
    for (u8 s : subtype_) ++c[s];
    return c;
}

usize MushroomBody::memory_bytes() const {
    return kc_.memory_bytes() + pn2kc_.memory_bytes()
         + (w_appr_.capacity() + w_avoid_.capacity() + elig_.capacity()
            + eta_scale_.capacity() + tau_subtype_.capacity()) * sizeof(f32)
         + (subtype_.capacity() * sizeof(u8))
         + trial_counts_.capacity() * sizeof(u32)
         + trial_pattern_.capacity() * sizeof(u8);
}

}  // namespace malefly
