// ============================================================================
// experiments/action_integrator.hpp — neural action selection (no software
// arbiter inside the agent).
//
// WHY THIS EXISTS (design principle, PRINCIPLES.md #11): everything the agent
// "decides" must be circuit dynamics. The previous decision step (sigmoid +
//Bernoulli sampling outside the circuit) was OS-like and is replaced by two
// competing LIF populations (approach / avoid) with mutual inhibition:
//   - MBON valence (+ gated lateral-horn innate valence) biases the drive;
//   - slow membrane integration (tau = 120 ms) accumulates evidence ->
//     ramping activity, as observed brain-wide before movement in larval
//     zebrafish (whole-brain light-sheet; docs/RESEARCH.md §ZBrain);
//   - spike noise provides the diffusion -> exploration EMERGES from neural
//     variability instead of an epsilon parameter;
//   - the first population whose spike count crosses the bound wins; if none
//     crosses within the deliberation window, the animal opts out (Avoid),
//     which is how rest-state passivity enters behavior (ZBrain state
//     switching; DMN rest/task antagonism).
// Arousal (EBRAINS/E2) modulates noise & gain; rest mode lowers the base
// drive (passivity). All parameters are distributions elsewhere; here the
// heterogeneity is the spiking itself.
// ============================================================================
#pragma once
#include "core/lif.hpp"
#include "core/rng.hpp"
#include "core/types.hpp"
#include <algorithm>
#include <cmath>

namespace malefly {

struct ActionIntegratorConfig {
    u32 n_cells = 16;          // per population
    f32 tau_pop = 0.12f;       // s — slow evidence integration (ramping)
    f32 v_thresh = -0.062f;    // integrator cells fire on small drives
    f32 t_refrac = 0.010f;
    f32 base_drive = 0.0125f;  // tonic drive (approach side gets innate bias)
    f32 appr_bias = 1.02f;    // innate approach asymmetry (LH baseline)
    f32 drive_gain = 0.020f;   // |valence| -> extra drive
    f32 mutual_inh = 0.0030f;  // per opponent spike (WTA)
    f32 noise = 0.008f;       // diffusion from spiking variability
    u32 bound = 3;             // first population spikes to win (early ramp race)
    u32 window_ms = 250;       // deliberation window; none crossed -> opt-out
    f32 rest_drive_scale = 0.45f;  // passivity in the rest state
    // drive-side adaptation (synaptic depression / receptor adaptation — the
    // universal mechanism keeping behavior re-samplable; without it a fully
    // devalued odor traps the agent in an avoidance deadlock, measured in
    // docs/RESULTS.md §Goal3)
    f32 adapt_beta = 0.060f;    // depression per population spike
    f32 adapt_tau_rec = 15.0f;  // s, recovery (cross-trial accumulation)
    f32 adapt_floor = 0.25f;    // never fully silence a side
};

class ActionIntegrator {
public:
    explicit ActionIntegrator(const ActionIntegratorConfig& cfg)
        : cfg_(cfg),
          appr_(cfg.n_cells, cfg_for(cfg)),
          avoid_(cfg.n_cells, cfg_for(cfg)) {}

    static LifConfig cfg_for(const ActionIntegratorConfig& c) {
        LifConfig lc;
        lc.tau_m = c.tau_pop;
        lc.t_refrac = c.t_refrac;
        lc.v_thresh = c.v_thresh;
        return lc;
    }

    // ---- per-decision interface -------------------------------------------
    // map valence (+arousal/rest internal state) to population drives
    void begin_decision(f32 valence, f32 innate_valence_gated, f32 arousal,
                        bool rest) {
        const f32 gate = rest ? cfg_.rest_drive_scale : 1.0f;
        base_ = cfg_.base_drive * gate;
        // E2 hypothesis (measured in Goal 3): prediction-error-raised arousal
        // means "the world violated expectations" -> trust learned valence
        // LESS (re-exploration) and diffuse MORE. Without this, a strongly
        // devalued odor traps the agent in avoidance deadlock (measured:
        // seed 42 reversal, docs/RESULTS.md).
        gain_ = cfg_.drive_gain * (1.0f + 0.35f * arousal) * (1.0f - 0.55f * arousal);
        noise_ = cfg_.noise * (1.0f + 2.5f * arousal);
        count_appr_ = 0;
        count_avoid_ = 0;
        decided_at_ = -1;
        for (u32 i = 0; i < cfg_.n_cells; ++i) {
            // (re)polarize: quick reset of integrator state between decisions
            appr_.reset_state();
            avoid_.reset_state();
        }
    }

    // one integration step; drives derived from the CURRENT valence reading
    // (callers pass the live readout each step — decision tracks the circuit,
    // not a snapshot)
    void step(f32 dt, Rng& rng, f32 valence, f32 innate_valence_gated) {
        const f32 v = valence + innate_valence_gated;
        const f32 d_appr = (base_ * cfg_.appr_bias + gain_ * std::max(0.0f, v)) * adapt_appr_;
        const f32 d_avoid = (base_ + gain_ * std::max(0.0f, -v)) * adapt_avoid_;
        const f32 inh_appr = cfg_.mutual_inh * static_cast<f32>(last_avoid_spikes_);
        const f32 inh_avoid = cfg_.mutual_inh * static_cast<f32>(last_appr_spikes_);
        u32 na = 0, nv = 0;
        for (u32 i = 0; i < cfg_.n_cells; ++i) {
            appr_.add_exc(i, std::max(0.0f, d_appr - inh_appr) +
                                 noise_ * rng.normal(0.0f, 1.0f));
            avoid_.add_exc(i, std::max(0.0f, d_avoid - inh_avoid) +
                                  noise_ * rng.normal(0.0f, 1.0f));
        }
        appr_.step(dt);
        avoid_.step(dt);
        {
            const u8* s = appr_.spikes();
            for (u32 i = 0; i < cfg_.n_cells; ++i) na += s[i];
            count_appr_ += na;
        }
        {
            const u8* s = avoid_.spikes();
            for (u32 i = 0; i < cfg_.n_cells; ++i) nv += s[i];
            count_avoid_ += nv;
        }
        last_appr_spikes_ = na;
        last_avoid_spikes_ = nv;
        // drive-side adaptation: depress with own spikes, recover slowly
        adapt_appr_ = std::max(cfg_.adapt_floor,
                               adapt_appr_ - cfg_.adapt_beta * na +
                               (1.0f - adapt_appr_) * dt / cfg_.adapt_tau_rec);
        adapt_avoid_ = std::max(cfg_.adapt_floor,
                                adapt_avoid_ - cfg_.adapt_beta * nv +
                                (1.0f - adapt_avoid_) * dt / cfg_.adapt_tau_rec);
        ++steps_;
        if (decided_at_ < 0 &&
            (count_appr_ >= static_cast<i32>(cfg_.bound) ||
             count_avoid_ >= static_cast<i32>(cfg_.bound))) {
            decided_at_ = static_cast<i32>(steps_);
        }
    }

    // ---- outcome -----------------------------------------------------------
    bool decided() const { return decided_at_ >= 0; }
    i32 motor_approach() const { return count_appr_; }
    i32 motor_avoid() const { return count_avoid_; }
    bool approach_wins() const { return count_appr_ >= count_avoid_; }
    i32 rt_steps() const { return decided_at_; }
    f32 margin() const {
        return static_cast<f32>(count_appr_ - count_avoid_) /
               static_cast<f32>(2 * cfg_.bound);
    }

private:
    ActionIntegratorConfig cfg_;
    LifLayer appr_, avoid_;
    f32 base_ = 0, gain_ = 0, noise_ = 0;
    f32 adapt_appr_ = 1.0f, adapt_avoid_ = 1.0f;  // persists across decisions
    i32 count_appr_ = 0, count_avoid_ = 0;
    u32 last_appr_spikes_ = 0, last_avoid_spikes_ = 0;
    u32 steps_ = 0;
    i32 decided_at_ = -1;
};

}  // namespace malefly
