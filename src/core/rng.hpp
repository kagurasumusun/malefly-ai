#pragma once
// ============================================================================
// core/rng.hpp — deterministic RNG (splitmix64 seeding + xoshiro256++ stream).
//
// Every stochastic part of the simulation (ORN spike sampling, connectivity
// construction, weight init, action sampling, environment outcomes) draws
// from explicitly owned Rng instances so that a run is bit-for-bit
// reproducible for a given seed. This is a hard requirement of the project
// (reproducibility principle, docs/PRINCIPLES.md).
// ============================================================================
#include "core/types.hpp"
#include <cmath>

namespace malefly {

class Rng {
public:
    explicit Rng(u64 seed) {
        u64 x = seed;
        for (auto& s : s_) s = splitmix64(x);
        if ((s_[0] | s_[1] | s_[2] | s_[3]) == 0) s_[0] = 1;
    }

    u64 next_u64() {
        const u64 result = s_[0] + s_[3];
        const u64 t = s_[1] << 17;
        s_[2] ^= s_[0]; s_[3] ^= s_[1]; s_[1] ^= s_[2]; s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = (s_[3] << 45) | (s_[3] >> 19);
        return result;
    }

    // uniform in [0,1), 24-bit mantissa
    f32 uniform01() { return static_cast<f32>((next_u64() >> 40) * 0x1.0p-24f); }
    f32 uniform(f32 lo, f32 hi) { return lo + (hi - lo) * uniform01(); }
    bool bernoulli(f32 p) { return uniform01() < p; }

    f32 normal(f32 mean, f32 stddev) {
        // Box-Muller with cached second sample
        if (has_spare_) { has_spare_ = false; return mean + stddev * spare_; }
        f32 u1 = uniform01(), u2 = uniform01();
        if (u1 < 1e-12f) u1 = 1e-12f;
        const f32 r = std::sqrt(-2.0f * std::log(u1));
        const f32 th = 6.2831853f * u2;
        spare_ = r * std::sin(th);
        has_spare_ = true;
        return mean + stddev * r * std::cos(th);
    }

private:
    static u64 splitmix64(u64& x) {
        x += 0x9E3779B97F4A7C15ull;
        u64 z = x;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }

    u64 s_[4];
    f32 spare_ = 0.0f;
    bool has_spare_ = false;
};

}  // namespace malefly
