#pragma once
// ============================================================================
// core/lif.hpp — current-based (CUBA) leaky integrate-and-fire layer, SoA.
//
//   dv/dt = ((v_rest - v) + g_e - g_i) / tau_m
//   dg_e/dt = -g_e / tau_exc ;  dg_i/dt = -g_i / tau_inh
//   spike when v >= v_thresh (then v = v_reset, refractory t_refrac)
//
// Model-hierarchy note (docs/DESIGN.md §3):
//  - 1-compartment, current-based LIF was chosen as the *minimum* detail level
//    that supports the required computation (coincidence detection in Kenyon
//    cells, gain control in the antennal lobe). AdEx / multi-compartment /
//    morphology+ion-channel models are part of the ladder and will only be
//    adopted where measured AI capability demands it.
//
// Layout: structure-of-arrays, contiguous, no per-neuron heap objects, no
// pointers between neurons. State is flat and trivially SIMD/parallel friendly.
// ============================================================================
#include "core/types.hpp"
#include <vector>

namespace malefly {

struct LifConfig {
    f32 tau_m     = 0.020f;   // s
    f32 v_rest    = -0.070f;  // V
    f32 v_reset   = -0.075f;  // V
    f32 v_thresh  = -0.050f;  // V
    f32 t_refrac  = 0.002f;   // s
    f32 tau_exc   = 0.005f;   // s
    f32 tau_inh   = 0.010f;   // s
};

class LifLayer {
public:
    LifLayer(u32 n, const LifConfig& cfg);

    // per-neuron threshold heterogeneity (fixed values are an approximation;
    // real populations are heterogeneous — docs/RESEARCH.md §MICrONS/BBP)
    void set_threshold(u32 i, f32 v) { v_thresh_[i] = v; }
    f32 threshold(u32 i) const { return v_thresh_[i]; }

    // integrate one timestep; spikes_ reflects *this* step afterwards
    void step(f32 dt);

    // input drive (called between steps)
    void add_exc(u32 i, f32 q) { ge_[i] += q; }
    void add_exc_all(f32 q);
    void add_inh_all(f32 q);

    // raw access for Synapses::propagate scatter
    f32* ge() { return ge_.data(); }
    f32* gi() { return gi_.data(); }

    u32 size() const { return n_; }
    const u8* spikes() const { return spikes_.data(); }
    u64 total_spikes() const { return total_spikes_; }
    void reset_counters() { total_spikes_ = 0; }
    void reset_state();

    usize memory_bytes() const;

private:
    u32 n_;
    LifConfig cfg_;
    f32 t_now_ = 0.0f;
    u64 total_spikes_ = 0;
    std::vector<f32> v_, ge_, gi_, refrac_until_, v_thresh_;
    std::vector<u8> spikes_;
};

}  // namespace malefly
