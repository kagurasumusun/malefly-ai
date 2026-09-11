#pragma once
// ============================================================================
// circuits/mushroom_body.hpp — mushroom body (MB): the fly's learning center.
//
// Biology (docs/MALECNS.md, Aso et al. 2014, Takemura et al. 2023+, MaleCNS):
//   ~2000 Kenyon cells (KCs) receive a sparse random expansion from ~150
//   uniglomerular PNs (median ~7 PNs/KC). KC responses are sparse and rely on
//   coincidence detection of synchronized PN input; APL feedback inhibition
//   sharpens sparsity. KCs converge onto 34 MBONs; the KC->MBON synapses are
//   the memory substrate. Dopaminergic neurons (PPL1/PAM, ~155 DANs) carry
//   reinforcement and modulate those synapses. Canonical sign rule:
//     reward DAN activation  depresses KC->avoid-pathway synapses,
//     punishment DAN activation depresses KC->approach-pathway synapses,
//   so approach/avoidance is read from the balance of the two branches.
//
// Implementation (documented simplifications):
//   - 2000 LIF KCs (real scale), random fan-in 7 from 50 PN channels,
//     high threshold -> few-spike coincidence requirement.
//   - APL approximated as pooled KC->(APL)->KC feedback inhibition with a
//     one-step delay (activity-dependent global KC suppression).
//   - MBONs: 2 graded readout channels (approach / avoid) standing in for the
//     34-MBON valence axes. Readout = activity-weighted mean of the plastic
//     KC->MBON weights over KCs active in the current trial (windowed leaky
//     integration). Spiking MBON population is a planned upgrade (DESIGN §7).
//   - DAN: one scalar reinforcement channel (r = +1 reward, -1 punishment)
//     implementing the canonical depression rule over an eligibility trace.
//     The eligibility trace (tau ~ 250 ms) bridges the odor->outcome delay
//     and is the seed of short-term memory (Goal 2).
// ============================================================================
#include "core/lif.hpp"
#include "core/rng.hpp"
#include "core/synapses.hpp"
#include "core/types.hpp"
#include <vector>

namespace malefly {

struct MushroomBodyConfig {
    u32 n_kc = 2000;         // real-scale KC count
    u32 kc_fanin = 7;        // median ~7 PNs per KC (connectome data)
    f32 kc_quanta = 0.036f;  // drive per PN spike onto a KC (calibrated)
    f32 kc_v_thresh = -0.050f;   // high threshold -> ~3-PN coincidence required
    f32 apl_gain = 0.002f;   // pooled KC->APL->KC feedback inhibition
    f32 tau_elig = 0.6f;    // eligibility trace (s) — KC->MBON traces last seconds
                            // (Berry et al. 2018). Long ITI proxy: see
                            // reset_elig_per_trial below.
    bool reset_elig_per_trial = true;  // real fly experiments space trials by
                                       // seconds-minutes, which prevents cross-trial
                                       // credit leakage; we compress time and instead
                                       // clear the trace between trials
    f32 kc_tau_exc = 0.005f; // KC synaptic current decay

    // valence branches (KC->MBON plastic weights, per KC)
    f32 w_appr_init_lo = 0.45f, w_appr_init_hi = 0.75f;   // approach branch
    f32 w_avoid_init_lo = 0.30f, w_avoid_init_hi = 0.60f; // avoid branch

    // DAN-modulated plasticity (depression of eligible synapses)
    f32 eta_reward = 0.12f;   // reward: depress KC->avoid
    f32 eta_punish = 0.14f;   // punishment: depress KC->approach

    // action selection over MBON valence
    f32 decision_temp = 0.22f;   // exploration temperature
    f32 epsilon = 0.05f;         // random-action floor
    f32 innate_bias = 0.15f;     // innate approach tendency (LH proxy)
    f32 naive_valence = 0.15f;   // expected valence of an untrained odor
};

class MushroomBody {
public:
    MushroomBody(const MushroomBodyConfig& cfg, Rng& rng, u32 n_pn);

    // KCs from PN spikes; eligibility + per-trial counters update internally
    void step(f32 dt, const u8* pn_spikes);

    // DAN reinforcement applied at outcome time (may lag odor by ~100s of ms)
    void apply_reinforcement(f32 reward);

    // reset per-trial spike counters / pattern (eligibility persists!)
    void begin_trial();

    struct Readout {
        f32 a_appr = 0, a_avoid = 0, valence = 0;
        f32 kc_active_frac = 0;
    };
    Readout readout() const;

    // ---- introspection (measurement is not optional; PRINCIPLES.md) ----
    const std::vector<f32>& w_appr() const { return w_appr_; }
    const std::vector<f32>& w_avoid() const { return w_avoid_; }
    const std::vector<u8>& trial_pattern() const { return trial_pattern_; }
    const Synapses& pn2kc() const { return pn2kc_; }
    u64 mbon_ops() const { return mbon_ops_; }
    u64 kc_total_spikes() const { return kc_.total_spikes(); }

    // test hook (tests/test_core.cpp)
    void debug_set_elig(u32 kc, f32 e) { elig_[kc] = e; }

    usize memory_bytes() const;

private:
    MushroomBodyConfig cfg_;
    u32 n_pn_;
    LifLayer kc_;
    Synapses pn2kc_;
    std::vector<f32> w_appr_, w_avoid_, elig_;
    std::vector<u32> trial_counts_;
    std::vector<u8> trial_pattern_;
    u64 mbon_ops_ = 0;
    Rng& rng_;
};

}  // namespace malefly
