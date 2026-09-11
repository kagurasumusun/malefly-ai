#pragma once
// ============================================================================
// circuits/antennal_lobe.hpp — antennal lobe (AL) abstraction.
//
// Biology (MaleCNS / hemibrain; see docs/MALECNS.md):
//   ORNs (~50 glomeruli, one ORN class each) -> uniglomerular projection
//   neurons (PNs); local neurons (LNs) mediate broad inhibition -> gain
//   control / contrast enhancement (Olson et al. 1994; Stopfer et al. 2003).
//   Odor-evoked LFP oscillations (~20-30 Hz) pace PN spike synchrony, which
//   downstream Kenyon cells use for coincidence detection (Cassenaer & Laurent).
//
// Implementation (documented simplifications):
//   - 1 PN per glomerulus (50 PNs); ORN->PN is a 1:1 Poisson transduction
//     (transduction is not the study object; PN/LN/KC spikes are simulated).
//   - LN pool: n_ln LIF neurons receiving all PN spikes, inhibiting all PNs
//     (pooled global inhibition = normalized gain control).
//   - 20-25 Hz oscillation: ORN rates are modulated sinusoidally, producing
//     PN synchrony cycles that KC coincidence detection can gate on.
// ============================================================================
#include "core/lif.hpp"
#include "core/rng.hpp"
#include "core/types.hpp"
#include "util/odors.hpp"
#include <vector>

namespace malefly {

struct AntennalLobeConfig {
    u32 n_glom = 50;             // Drosophila AL: ~50 glomeruli
    u32 n_ln = 20;               // pooled LN population
    f32 orn_base_hz = 4.0f;      // spontaneous ORN rate
    f32 orn_max_hz = 60.0f;      // ORN rate at full profile strength, unit concentration
    f32 osc_hz = 22.0f;          // AL oscillation pacing PN synchrony (Hz)
    f32 osc_depth = 0.85f;       // modulation depth of ORN rate
    f32 orn_pn_quanta = 0.100f;  // drive per ORN spike onto its PN
    f32 pn_ln_quanta = 0.012f;   // drive per PN spike onto each LN
    f32 ln_pn_quanta = 0.0020f;   // inhibition per LN spike onto each PN
};

class AntennalLobe {
public:
    AntennalLobe(const AntennalLobeConfig& cfg, Rng& rng);

    // odor == nullptr -> clean air (base rate only)
    void set_odor(const Odor* odor, f32 concentration);
    void step(f32 dt);

    const u8* pn_spikes() const { return pn_.spikes(); }
    const LifLayer& pn() const { return pn_; }
    const LifLayer& ln() const { return ln_; }
    u32 n_glom() const { return cfg_.n_glom; }
    u32 n_ln() const { return cfg_.n_ln; }
    u64 pool_ops() const { return pool_ops_; }

    usize memory_bytes() const;

private:
    AntennalLobeConfig cfg_;
    Rng& rng_;
    LifLayer pn_, ln_;
    std::vector<f32> orn_rate_;  // per-glomerulus ORN rate (Hz), incl. odor
    f32 t_ = 0.0f;
    u64 pool_ops_ = 0;
};

}  // namespace malefly
