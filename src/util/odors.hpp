#pragma once
// ============================================================================
// util/odors.hpp — odor stimulus construction and overlap metrics.
//
// An odor is a profile over glomeruli: p[g] in [0,1]. In the real fly, ~50
// glomeruli each receive one ORN class; odors activate overlapping subsets.
// ============================================================================
#include "core/types.hpp"
#include <string>
#include <vector>

namespace malefly {

struct Odor {
    std::string name;
    std::vector<f32> profile;  // per-glomerulus strength in [0,1]
};

// random subset of `support` glomeruli, strengths ~ U(s_min, s_max)
Odor make_odor(u32 n_glom, u32 support, u64 seed, f32 s_min, f32 s_max);

// elementwise min(1, ca*a + cb*b) — binary mixture
Odor mix_odors(const Odor& a, const Odor& b, f32 ca, f32 cb);

// copy of base scaled by `scale`, plus `n_new` fresh random glomeruli
Odor make_variant(const Odor& base, f32 scale, u32 n_new, u32 n_glom, u64 seed);

// weak uniform background ("neutral", nearly-odorless air)
Odor uniform_odor(u32 n_glom, f32 level);

// weighted Jaccard overlap: sum(min) / sum(max), in [0,1]
f32 profile_overlap(const Odor& a, const Odor& b);

}  // namespace malefly
