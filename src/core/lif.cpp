#include "core/lif.hpp"
#include <cmath>
#include <algorithm>

namespace malefly {

LifLayer::LifLayer(u32 n, const LifConfig& cfg) : n_(n), cfg_(cfg) {
    v_.assign(n, cfg_.v_rest);
    ge_.assign(n, 0.0f);
    gi_.assign(n, 0.0f);
    refrac_until_.assign(n, -1.0f);
    spikes_.assign(n, 0);
}

void LifLayer::step(f32 dt) {
    const f32 decay_e = std::exp(-dt / cfg_.tau_exc);
    const f32 decay_i = std::exp(-dt / cfg_.tau_inh);
    t_now_ += dt;
    for (u32 i = 0; i < n_; ++i) {
        ge_[i] *= decay_e;
        gi_[i] *= decay_i;
        f32 v = v_[i];
        u8 s = 0;
        if (t_now_ >= refrac_until_[i]) {
            v += dt * ((cfg_.v_rest - v) + ge_[i] - gi_[i]) / cfg_.tau_m;
            if (v >= cfg_.v_thresh) {
                v = cfg_.v_reset;
                refrac_until_[i] = t_now_ + cfg_.t_refrac;
                s = 1;
                ++total_spikes_;
            }
        }
        v_[i] = v;
        spikes_[i] = s;
    }
}

void LifLayer::add_exc_all(f32 q) {
    for (u32 i = 0; i < n_; ++i) ge_[i] += q;
}

void LifLayer::add_inh_all(f32 q) {
    for (u32 i = 0; i < n_; ++i) gi_[i] += q;
}

void LifLayer::reset_state() {
    v_.assign(n_, cfg_.v_rest);
    ge_.assign(n_, 0.0f);
    gi_.assign(n_, 0.0f);
    refrac_until_.assign(n_, -1.0f);
    std::fill(spikes_.begin(), spikes_.end(), u8{0});
    t_now_ = 0.0f;
    total_spikes_ = 0;
}

usize LifLayer::memory_bytes() const {
    return (v_.capacity() + ge_.capacity() + gi_.capacity() + refrac_until_.capacity()) * sizeof(f32)
         + spikes_.capacity() * sizeof(u8);
}

}  // namespace malefly
