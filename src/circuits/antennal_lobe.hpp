#pragma once
// ============================================================================
// circuits/antennal_lobe.hpp — antennal lobe (AL) abstraction.
//
// Biology (MaleCNS / hemibrain; see docs/MALECNS.md):
//   ORNs (~50 glomeruli) -> uniglomerular PNs; local neurons (LNs) mediate
//   inhibition -> gain control / contrast enhancement. Odor-evoked LFP
//   oscillations (~20-30 Hz) pace PN synchrony for KC coincidence detection.
//
// ADOPTED (Blue Brain Project, measured — docs/RESULTS.md §6): inhibitory
// subclass differentiation (cf. 207 m/e-types & inhibitory subclasses,
// Markram et al. 2015). The single LN pool is split into
//   - FAST subclass (short tau_inh): phasic, cycle-level contrast control
//   - SLOW subclass (long tau_inh):  tonic, trial-level gain control
// Ablation flag ln_subclasses=false restores the legacy single pool exactly
// (before/after comparison).
//
// Documented simplifications: 1 PN/glomerulus; ORN->PN 1:1 Poisson
// transduction; pooled (all-to-all) subclass inhibition.
// ============================================================================
#include "core/lif.hpp"
#include "core/rng.hpp"
#include "core/types.hpp"
#include "util/odors.hpp"
#include <vector>

namespace malefly {

struct AntennalLobeConfig {
    u32 n_glom = 50;  // Drosophila AL: ~50 glomeruli (53 in MaleCNS v1.0)

    // inhibitory subclasses (Blue Brain adoption; ablatable)
    bool ln_subclasses = true;
    u32 n_ln_fast = 8;   // fast/phasic pool (tau_inh_fast)
    u32 n_ln_slow = 12;  // slow/tonic pool (tau_inh_slow)
    f32 tau_inh_fast = 0.003f;   // s — fast feedback window
    f32 tau_inh_slow = 0.045f;   // s — tonic gain window
    f32 ln_fast_pn_quanta = 0.008f;  // PN spike -> each fast LN
    f32 ln_slow_pn_quanta = 0.008f;  // PN spike -> each slow LN
    f32 ln_fast_back_quanta = 0.0014f;  // fast LN spike -> each PN
    f32 ln_slow_back_quanta = 0.0016f;  // slow LN spike -> each PN

    // legacy single-pool parameters (ln_subclasses == false)
    f32 legacy_ln_quanta = 0.002f;  // legacy ln_pn_quanta
    f32 legacy_pn_ln_quanta = 0.012f;

    f32 orn_base_hz = 4.0f;
    f32 orn_max_hz = 60.0f;
    f32 osc_hz = 22.0f;
    f32 osc_depth = 0.85f;
    f32 orn_pn_quanta = 0.100f;
};

class AntennalLobe {
public:
    AntennalLobe(const AntennalLobeConfig& cfg, Rng& rng);

    void set_odor(const Odor* odor, f32 concentration);
    void step(f32 dt);

    const u8* pn_spikes() const { return pn_.spikes(); }
    const LifLayer& pn() const { return pn_; }
    const LifLayer& ln_fast() const { return ln_fast_; }
    const LifLayer& ln_slow() const { return ln_slow_; }
    u32 n_glom() const { return cfg_.n_glom; }
    u32 n_ln() const { return cfg_.n_ln_fast + cfg_.n_ln_slow; }
    u64 pool_ops() const { return pool_ops_; }

    usize memory_bytes() const;

private:
    AntennalLobeConfig cfg_;
    Rng& rng_;
    LifLayer pn_;
    LifLayer ln_fast_;
    LifLayer ln_slow_;
    std::vector<f32> orn_rate_;
    f32 t_ = 0.0f;
    u64 pool_ops_ = 0;
};

}  // namespace malefly
