#pragma once
// ============================================================================
// circuits/lateral_horn.hpp — lateral horn (LH): innate odor valence.
//
// Biology: in the fly, innate approach/avoidance to odors is mediated by a
// PN->LH pathway that is hardwired and experience-independent, parallel to
// the plastic MB pathway. Plug-in region #1 behind the Region interface
// (Brain/MINDS adoption).
//
// Implementation: 20 LIF cells with FIXED random convergent PN input
// (fan-in 10, no plasticity). Response = mean active fraction of LH cells
// over the presentation window. Strong patterned odors drive LH; weak
// uniform background (clean air) does not. innate_valence() maps response
// to a stimulus-dependent innate attraction bias for the decision circuit.
// ============================================================================
#include "circuits/region.hpp"
#include "core/lif.hpp"
#include "core/rng.hpp"
#include "core/synapses.hpp"
#include "core/types.hpp"
#include <cmath>

namespace malefly {

struct LateralHornConfig {
    u32 n_cells = 20;
    u32 fanin = 10;       // PN channels per LH cell
    f32 quanta = 0.060f;  // drive per PN spike (calibrated; see docs/RESULTS.md)
    f32 beta = 0.15f;     // innate valence gain in decision units
    f32 resp_scale = 0.03f;  // response saturation scale (calibrated)
};

class LateralHorn final : public Region {
public:
    LateralHorn(const LateralHornConfig& cfg, Rng& rng, u32 n_pn)
        : cfg_(cfg), lh_(cfg.n_cells, LifConfig{}),
          w_(make_random_fanin(n_pn, cfg.n_cells, cfg.fanin, rng, cfg.quanta, 0.0f)) {}

    void set_input(const u8* pn_spikes) { input_ = pn_spikes; }

    // Region interface
    const char* name() const override { return "lateral_horn"; }
    void step(f32 dt) override {
        if (input_) w_.propagate(input_, lh_.ge());
        lh_.step(dt);
        const u8* s = lh_.spikes();
        u32 c = 0;
        for (u32 k = 0; k < cfg_.n_cells; ++k) c += s[k];
        resp_acc_ += static_cast<f32>(c) / static_cast<f32>(cfg_.n_cells);
        ++resp_n_;
    }
    usize memory_bytes() const override {
        return lh_.memory_bytes() + w_.memory_bytes();
    }

    void begin_window() { resp_acc_ = 0.0f; resp_n_ = 0; }

    // mean active fraction of LH cells over the presentation window
    f32 response() const {
        return resp_n_ ? resp_acc_ / static_cast<f32>(resp_n_) : 0.0f;
    }

    // stimulus-dependent innate attraction (>= 0), saturating with response
    f32 innate_valence() const {
        return cfg_.beta * std::tanh(response() / cfg_.resp_scale);
    }

    u64 ops() const { return w_.ops; }

private:
    LateralHornConfig cfg_;
    LifLayer lh_;
    Synapses w_;
    const u8* input_ = nullptr;
    f64 resp_acc_ = 0.0;
    u64 resp_n_ = 0;
};

}  // namespace malefly
