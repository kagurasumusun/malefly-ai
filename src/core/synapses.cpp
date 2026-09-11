#include "core/synapses.hpp"
#include <algorithm>
#include <cmath>
#include <utility>

namespace malefly {

Synapses make_random_fanin(u32 n_pre, u32 n_post, u32 fanin, Rng& rng,
                           f32 w_mean, f32 w_jitter) {
    Synapses s;
    s.n_pre = n_pre;
    s.n_post = n_post;
    s.indptr.assign(n_pre + 1, 0);

    // generate all edges (pre, post): each post samples `fanin` distinct pres
    std::vector<std::pair<u32, u32>> edges;
    edges.reserve(usize(n_post) * fanin);
    std::vector<u32> scratch(n_pre);
    for (u32 i = 0; i < n_pre; ++i) scratch[i] = i;
    for (u32 j = 0; j < n_post; ++j) {
        for (u32 m = 0; m < fanin && m < n_pre; ++m) {
            u32 avail = n_pre - m;
            u32 idx = static_cast<u32>(rng.uniform01() * static_cast<f32>(avail));
            if (idx >= avail) idx = avail - 1;
            std::swap(scratch[m], scratch[m + idx]);
            edges.emplace_back(scratch[m], j);
        }
    }
    std::sort(edges.begin(), edges.end());  // group by pre; deterministic

    s.indices.resize(edges.size());
    s.weights.resize(edges.size());
    usize e = 0;
    for (u32 pre = 0; pre < n_pre; ++pre) {
        while (e < edges.size() && edges[e].first == pre) {
            s.indices[e] = edges[e].second;
            s.weights[e] = w_mean * (1.0f + w_jitter * (2.0f * rng.uniform01() - 1.0f));
            ++e;
        }
        s.indptr[pre + 1] = static_cast<u32>(e);
    }
    return s;
}

Synapses make_random_fanin_dist(u32 n_pre, u32 n_post, f32 fanin_mean,
                                f32 fanin_sd, Rng& rng, f32 w_mean, f32 w_sigma) {
    Synapses s;
    s.n_pre = n_pre;
    s.n_post = n_post;
    s.indptr.assign(n_pre + 1, 0);

    // lognormal with mean w_mean: mu = ln(w_mean) - sigma^2/2
    const f32 mu = std::log(std::max(1e-6f, w_mean)) - 0.5f * w_sigma * w_sigma;
    std::vector<std::pair<u32, u32>> edges;
    std::vector<u32> scratch(n_pre);
    for (u32 i = 0; i < n_pre; ++i) scratch[i] = i;
    for (u32 j = 0; j < n_post; ++j) {
        // per-post fan-in ~ round(N(mean, sd)) clamped — heterogeneous wiring
        const f32 fi = fanin_mean + fanin_sd * rng.normal(0.0f, 1.0f);
        i32 k = static_cast<i32>(fi + 0.5f);
        k = std::clamp(k, 3, 15);
        k = std::min<i32>(k, static_cast<i32>(n_pre));
        for (i32 m = 0; m < k; ++m) {
            u32 avail = n_pre - static_cast<u32>(m);
            u32 idx = m + static_cast<u32>(rng.uniform01() * static_cast<f32>(avail));
            if (idx >= avail) idx = avail - 1;
            std::swap(scratch[m], scratch[idx]);
            edges.emplace_back(scratch[m], j);
        }
    }
    std::sort(edges.begin(), edges.end());  // group by pre; deterministic

    s.indices.resize(edges.size());
    s.weights.resize(edges.size());
    usize e = 0;
    for (u32 pre = 0; pre < n_pre; ++pre) {
        while (e < edges.size() && edges[e].first == pre) {
            s.indices[e] = edges[e].second;
            const f32 w = std::exp(mu + w_sigma * rng.normal(0.0f, 1.0f));
            s.weights[e] = std::max(1e-4f, w);
            ++e;
        }
        s.indptr[pre + 1] = static_cast<u32>(e);
    }
    return s;
}

bool prune_one_synapse(Synapses& s, u32 post, Rng& rng) {
    // zero one uniformly-chosen nonzero-weight input of `post`
    std::vector<u32> candidates;
    for (u32 k = 0; k < s.num(); ++k)
        if (s.indices[k] == post && s.weights[k] > 0.0f)
            candidates.push_back(k);
    if (candidates.empty()) return false;
    const u32 k = candidates[static_cast<usize>(
        rng.uniform01() * static_cast<f32>(candidates.size())) % candidates.size()];
    s.weights[k] = 0.0f;
    return true;
}

u32 active_fanin(const Synapses& s, u32 post) {
    u32 c = 0;
    for (u32 k = 0; k < s.num(); ++k)
        if (s.indices[k] == post && s.weights[k] > 0.0f) ++c;
    return c;
}

}  // namespace malefly
