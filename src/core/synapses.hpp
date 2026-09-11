#pragma once
// ============================================================================
// core/synapses.hpp — sparse connectivity in CSR (compressed sparse row) form.
//
// Design (docs/DESIGN.md §5):
//  - flat contiguous arrays: indptr / indices / weights. No per-synapse
//    objects, no pointers. 8 bytes per synapse + 4 bytes per presynaptic row.
//  - propagate() scatter-adds weight * pre_spike into the postsynaptic current
//    buffer. This is the engine hot loop; it is branch-light and contiguous
//    per active row, ready for SIMD/parallel decomposition later.
//  - `ops` counts synapses touched (real perf measurement, not estimates).
//
// Fixed 1-step (1 ms) transmission delay is implicit in the per-timestep
// update order (documented simplification; graded delays are future work).
// ============================================================================
#include "core/types.hpp"
#include "core/rng.hpp"
#include <vector>

namespace malefly {

struct Synapses {
    u32 n_pre = 0;
    u32 n_post = 0;
    std::vector<u32> indptr;   // size n_pre + 1
    std::vector<u32> indices;  // post id per synapse, rows sorted by pre
    std::vector<f32> weights;  // per-synapse weight
    mutable u64 ops = 0;       // perf counter: synapses touched by propagate

    usize num() const { return indices.size(); }

    void propagate(const u8* pre_spikes, f32* post_ge) const {
        for (u32 p = 0; p < n_pre; ++p) {
            if (!pre_spikes[p]) continue;
            const u32 end = indptr[p + 1];
            for (u32 k = indptr[p]; k < end; ++k) {
                post_ge[indices[k]] += weights[k];
                ++ops;
            }
        }
    }

    usize memory_bytes() const {
        return indptr.capacity() * sizeof(u32)
             + indices.capacity() * sizeof(u32)
             + weights.capacity() * sizeof(f32);
    }
};

// Each of the n_post neurons samples `fanin` distinct presynaptic neurons
// (partial Fisher-Yates), fixed weight w_mean with relative jitter.
// Deterministic given rng state. Rows are built sorted by pre.
Synapses make_random_fanin(u32 n_pre, u32 n_post, u32 fanin, Rng& rng,
                           f32 w_mean, f32 w_jitter);

}  // namespace malefly
