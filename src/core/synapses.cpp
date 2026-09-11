#include "core/synapses.hpp"
#include <algorithm>
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

}  // namespace malefly
