#pragma once
// ============================================================================
// circuits/mushroom_body.hpp — mushroom body (MB): the fly's learning center.
//
// Biology (docs/MALECNS.md, Aso et al. 2014, MaleCNS):
//   ~2000 Kenyon cells, sparse PN fan-in ~7, high-threshold coincidence
//   detection, APL feedback inhibition, KC->MBON plastic valence branches,
//   DAN reinforcement with the canonical sign rule.
//
// ADOPTED mechanisms (each ablatable; each measured in docs/RESULTS.md):
//   [Allen BICCN] cell-type taxonomy: KC subtypes (alpha-beta / alpha'-beta' /
//     gamma) with FRACTIONS MEASURED from MaleCNS v1.0 real data
//     (src/circuits/kc_taxonomy.gen.h). Subtypes differ in eligibility-trace
//     timescale and plasticity rate -> multi-timescale memory on one substrate.
//   [BRAIN Initiative] prediction-error-gated plasticity (predictive
//     processing): DAN updates are scaled by |outcome - expected valence|, so
//     satisfied expectations stop updating (memory protection) and violated
//     expectations update strongly (fast reversal).
// ============================================================================
#include "circuits/kc_taxonomy.gen.h"
#include "core/lif.hpp"
#include "core/rng.hpp"
#include "core/synapses.hpp"
#include "core/types.hpp"
#include <algorithm>
#include <vector>

namespace malefly {

struct MushroomBodyConfig {
    u32 n_kc = 2000;         // real-scale is 4064 (docs/DATA.md); growth is a
                             // measured future step
    u32 kc_fanin = 7;        // median ~7 PNs per KC (connectome data)
    f32 kc_quanta = 0.034f;  // drive per PN spike onto a KC (calibrated)
    f32 kc_v_thresh = -0.050f;
    f32 apl_gain = 0.002f;   // pooled KC->APL->KC feedback inhibition
    f32 tau_elig = 0.6f;     // uniform trace used when use_taxonomy == false
    bool reset_elig_per_trial = true;  // long-ITI proxy (docs/RESULTS.md §2)
    f32 kc_tau_exc = 0.005f;

    f32 w_appr_init_lo = 0.45f, w_appr_init_hi = 0.75f;
    f32 w_avoid_init_lo = 0.30f, w_avoid_init_hi = 0.60f;

    f32 eta_reward = 0.12f;  // reward: depress KC->avoid
    f32 eta_punish = 0.14f;  // punishment: depress KC->approach
    // opponent restoration (reversal substrate): the opposite branch recovers
    // toward its naive value at this fraction of eta (0 disables = legacy
    // depression-only rule; measured in docs/RESULTS.md §8)
    f32 oppo_restore = 0.5f;

    f32 decision_temp = 0.22f;
    f32 epsilon = 0.05f;
    f32 innate_bias = 0.15f;
    f32 naive_valence = 0.15f;
    // trained-MB suppression of the innate LH path (documented fly circuit
    // principle: learned MBON output suppresses LH-driven innate behavior).
    // |MB valence| reaching this value gates the innate contribution to ~0.
    f32 innate_gate_vmax = 0.30f;

    // ---- adoptable mechanisms (ablation flags for before/after measurement)
    bool use_taxonomy = true;  // [Allen BICCN] KC subtype differentiation
    bool rpe_gating = true;    // [BRAIN Initiative] |RPE|-gated plasticity
};

class MushroomBody {
public:
    MushroomBody(const MushroomBodyConfig& cfg, Rng& rng, u32 n_pn);

    void step(f32 dt, const u8* pn_spikes);
    void apply_reinforcement(f32 reward);
    void begin_trial();

    struct Readout {
        f32 a_appr = 0, a_avoid = 0, valence = 0;
        f32 kc_active_frac = 0;
    };
    Readout readout() const;

    // circuit-state memory readout (Goal 2 STM): decaying eligibility trace
    struct TraceReadout {
        f32 valence = 0;
        f32 active_frac = 0;
    };
    TraceReadout trace_readout() const;

    // ---- introspection (measurement is mandatory; PRINCIPLES.md) ----
    const std::vector<f32>& w_appr() const { return w_appr_; }
    const std::vector<f32>& w_avoid() const { return w_avoid_; }
    const std::vector<u8>& trial_pattern() const { return trial_pattern_; }
    const Synapses& pn2kc() const { return pn2kc_; }
    u64 mbon_ops() const { return mbon_ops_; }
    u64 kc_total_spikes() const { return kc_.total_spikes(); }
    // measured subtype composition (Allen adoption audit)
    std::vector<u32> subtype_counts() const;
    f32 last_rpe_gate() const { return last_gate_; }

    void debug_set_elig(u32 kc, f32 e) { elig_[kc] = e; }

    usize memory_bytes() const;

private:
    MushroomBodyConfig cfg_;
    u32 n_pn_;
    LifLayer kc_;
    Synapses pn2kc_;
    std::vector<f32> w_appr_, w_avoid_, elig_;
    std::vector<f32> eta_scale_;       // per-KC plasticity scale (taxonomy)
    std::vector<u8> subtype_;          // per-KC subtype index
    std::vector<f32> tau_subtype_;     // tau_elig per subtype
    f32 appr_naive_ = 0.0f;            // naive branch means (delta readout ref)
    f32 avoid_naive_ = 0.0f;
    std::vector<u32> trial_counts_;
    std::vector<u8> trial_pattern_;
    f32 last_gate_ = 1.0f;
    u64 mbon_ops_ = 0;
    Rng& rng_;
};

}  // namespace malefly
