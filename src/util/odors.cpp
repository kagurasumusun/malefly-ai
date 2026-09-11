#include "util/odors.hpp"
#include "core/rng.hpp"
#include <algorithm>

namespace malefly {

Odor make_odor(u32 n_glom, u32 support, u64 seed, f32 s_min, f32 s_max) {
    Rng rng(seed);
    Odor o;
    o.profile.assign(n_glom, 0.0f);
    std::vector<u32> idx(n_glom);
    for (u32 i = 0; i < n_glom; ++i) idx[i] = i;
    const u32 sup = std::min(support, n_glom);
    for (u32 m = 0; m < sup; ++m) {
        u32 avail = n_glom - m;
        u32 k = m + static_cast<u32>(rng.uniform01() * static_cast<f32>(avail));
        if (k >= avail) k = avail - 1;
        std::swap(idx[m], idx[m + k]);
        o.profile[idx[m]] = rng.uniform(s_min, s_max);
    }
    return o;
}

Odor mix_odors(const Odor& a, const Odor& b, f32 ca, f32 cb) {
    Odor o;
    const u32 n = static_cast<u32>(a.profile.size());
    o.profile.assign(n, 0.0f);
    for (u32 g = 0; g < n; ++g)
        o.profile[g] = std::min(1.0f, ca * a.profile[g] + cb * b.profile[g]);
    return o;
}

Odor make_variant(const Odor& base, f32 scale, u32 n_new, u32 n_glom, u64 seed) {
    Rng rng(seed);
    Odor o = base;
    o.profile.assign(n_glom, 0.0f);
    for (u32 g = 0; g < n_glom; ++g) o.profile[g] = scale * base.profile[g];
    u32 added = 0, guard = 0;
    while (added < n_new && guard++ < 100 * n_new) {
        u32 g = static_cast<u32>(rng.uniform01() * static_cast<f32>(n_glom));
        if (g >= n_glom) g = n_glom - 1;
        if (o.profile[g] > 0.0f) continue;
        o.profile[g] = rng.uniform(0.6f, 1.0f);
        ++added;
    }
    return o;
}

Odor uniform_odor(u32 n_glom, f32 level) {
    Odor o;
    o.profile.assign(n_glom, level);
    return o;
}

f32 profile_overlap(const Odor& a, const Odor& b) {
    f32 smin = 0.0f, smax = 0.0f;
    const u32 n = static_cast<u32>(a.profile.size());
    for (u32 g = 0; g < n; ++g) {
        smin += std::min(a.profile[g], b.profile[g]);
        smax += std::max(a.profile[g], b.profile[g]);
    }
    return smax > 0.0f ? smin / smax : 0.0f;
}

}  // namespace malefly
