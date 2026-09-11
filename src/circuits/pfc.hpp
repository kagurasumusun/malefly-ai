// ============================================================================
// circuits/pfc.hpp — primate-inspired prefrontal working-memory bank.
//
// LITERATURE GROUNDING (docs/HUMAN.md):
//   - DLPFC neurons maintain PERSISTENT, stimulus-selective activity across
//     the delay of delayed-response tasks (Funahashi, Bruce & Goldman-Rakic
//     1989; Curtis & D'Esposito 2003); the canonical circuit model is
//     stimulus-tuned recurrent excitation balanced by inhibition with slow
//     kinetics (Compte 2000; Wang 2001).
//   - Top-down signals protect the maintained code against distractors
//     (top-down modulation engaged during stimulus-absent stages; Gazzaley &
//     D'Esposito). Here the protection gate is the brain's OWN arousal/mode
//     state (rest mode -> low gate -> interference gets in), i.e. attentional
//     protection is an internal state effect, not a software flag.
//   - Familiarity/mismatch signal perirhinal-style: overlap between current
//     percept and the maintained assembly yields a match/mismatch signal.
//
// Implementation: LIF bank + one-shot Hebbian binding of the active
// assembly during sample presentation; maintenance = recurrent excitation
// with bounded weights; novelty at test time degrades the code (interference).
// ============================================================================
#pragma once
#include "core/lif.hpp"
#include "core/rng.hpp"
#include "core/synapses.hpp"
#include "core/types.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace malefly {

struct PfcConfig {
    u32 n_cells = 400;
    u32 in_fanin = 10;          // KC -> PFC
    f32 in_quanta = 0.028f;
    f32 thresh = -0.058f;       // low-ish: persistent recruitment
    f32 tau_m = 0.040f;         // slow integrator (NMDA-like persistence)
    f32 tau_exc = 0.030f;       // NMDA-like slow excitation (Wang 2001)
    // E-I feedback (no fixed values: heterogeneous fan-in/weights)
    u32 n_inh = 100;
    u32 e2i_fanin = 24;
    u32 i2e_fanin = 24;
    f32 w_e2i = 0.008f;
    f32 w_i2e = 0.010f;

    u32 rec_fanin = 16;
    f32 rec_quanta = 0.0020f;    // naive recurrent weight
    f32 eta_bind = 0.40f;       // one-shot assembly binding at sample time
    f32 w_rec_max = 0.016f;

    f32 gate_active = 1.00f;    // top-down protection (arousal/active mode)
    f32 gate_rest = 0.25f;      // rest: distractors write through

    f32 familiarity_tau = 1.5f;
};

class Pfc {
public:
    Pfc(const PfcConfig& cfg, Rng& rng, u32 n_input)
        : cfg_(cfg), cells_(cfg.n_cells, pfc_cfg(cfg)),
          inh_(cfg.n_inh, inh_cfg()) {
        e2i_ = make_random_fanin_dist(cfg.n_cells, cfg.n_inh,
                                      static_cast<f32>(cfg.e2i_fanin), 3.0f,
                                      rng, cfg.w_e2i, 0.30f);
        i2e_ = make_random_fanin_dist(cfg.n_inh, cfg.n_cells,
                                      static_cast<f32>(cfg.i2e_fanin), 3.0f,
                                      rng, cfg.w_i2e, 0.30f);
        kc2pfc_ = make_random_fanin_dist(n_input, cfg.n_cells,
                                         static_cast<f32>(cfg.in_fanin), 1.5f,
                                         rng, cfg.in_quanta, 0.30f);
        rec_ = make_random_fanin_dist(cfg.n_cells, cfg.n_cells,
                                      static_cast<f32>(cfg.rec_fanin), 2.5f,
                                      rng, cfg.rec_quanta, 0.35f);
        for (f32& w : rec_.weights) w = std::max(0.0f, w - 0.006f);
        assembly_.assign(cfg.n_cells, 0);
        assembly_size_ = 0;
        familiarity_ = 0.0f;
        fam_fast_ = 0.0f;
        sample_ms_ = 0.0f;
    }

    static LifConfig inh_cfg() {
        LifConfig c;
        c.tau_m = 0.010f;
        c.v_thresh = -0.054f;
        c.t_refrac = 0.002f;
        return c;
    }
    static LifConfig pfc_cfg(const PfcConfig& pc) {
        LifConfig c;
        c.tau_m = pc.tau_m;
        c.tau_exc = pc.tau_exc;
        c.v_thresh = -0.058f;
        c.t_refrac = 0.004f;
        return c;
    }

    void set_gate(f32 g) { gate_ = std::clamp(g, 0.0f, 1.0f); }

    void step(f32 dt, const u8* kc_pattern, bool sample_phase) {
        // sensory drive, gated by top-down protection (distractor filtering)
        if (kc_pattern && gate_ > 0.0f) {
            std::vector<f32> ge(cfg_.n_cells, 0.0f);
            kc2pfc_.propagate(kc_pattern, ge.data());
            for (u32 k = 0; k < cfg_.n_cells; ++k)
                cells_.add_exc(k, ge[k] * gate_);
        }
        // recurrent maintenance from last step (E->E, NMDA-like)
        rec_.propagate(cells_.spikes(), cells_.ge());
        // E-I feedback: E spikes drive interneurons; interneurons inhibit E
        e2i_.propagate(cells_.spikes(), inh_.ge());
        inh_.step(dt);
        i2e_.propagate(inh_.spikes(), cells_.gi());
        cells_.step(dt);

        const u8* s = cells_.spikes();
        u32 act = 0;
        for (u32 k = 0; k < cfg_.n_cells; ++k) act += s[k];

        // one-shot assembly binding during sample presentations. The bound
        // assembly is the stimulus-ONSET response (first ~150 ms: the KC-
        // identity-specific population, before recurrent recruitment spreads
        // the activity) — Funahashi/Bruce/Goldman-Rakic 1989 cue-period
        // selectivity; binding beyond the window would encode the whole
        // recruited net and destroy stimulus selectivity.
        if (sample_phase) {
            sample_ms_ += dt;
            if (sample_ms_ <= 0.150f) {
                for (u32 k = 0; k < cfg_.n_cells; ++k)
                    assembly_[k] = static_cast<u8>(assembly_[k] | s[k]);
                for (u32 pre = 0; pre < cfg_.n_cells; ++pre) {
                    if (!s[pre]) continue;
                    const u32 e = rec_.indptr[pre + 1];
                    for (u32 k = rec_.indptr[pre]; k < e; ++k) {
                        f32& w = rec_.weights[k];
                        w = std::min(cfg_.w_rec_max, w + cfg_.eta_bind);
                    }
                }
            }
            assembly_size_ = 0;
            for (u32 k = 0; k < cfg_.n_cells; ++k) assembly_size_ += assembly_[k];
        } else {
            sample_ms_ = 0.0f;
        }

        // familiarity = maintained assembly reactivated by the current input.
        // Two time constants: slow (experience-level novelty context) and
        // fast (tau 0.15s — probe-evoked MATCH signal, Miller & Desimone
        // 1994 match enhancement; Compte 2000 spatial WM rate profile).
        u32 match = 0;
        for (u32 k = 0; k < cfg_.n_cells; ++k) match += (s[k] && assembly_[k]);
        const f32 f = (assembly_size_ > 0)
                          ? static_cast<f32>(match) /
                                static_cast<f32>(assembly_size_)
                          : 0.0f;
        familiarity_ += (f - familiarity_) *
                        (1.0f - std::exp(-dt / cfg_.familiarity_tau));
        fam_fast_ += (f - fam_fast_) * (1.0f - std::exp(-dt / 0.150f));
    }

    void clear_assembly() {
        std::fill(assembly_.begin(), assembly_.end(), u8{0});
        assembly_size_ = 0;
    }

    f32 familiarity() const { return fam_fast_; }   // fast match signal
    f32 familiarity_slow() const { return familiarity_; }
    // measurement accessors (diagnostics only)
    u32 debug_match_now() const {
        const u8* s = cells_.spikes();
        u32 m = 0;
        for (u32 k = 0; k < cfg_.n_cells; ++k) m += (s[k] && assembly_[k]);
        return m;
    }
    u32 debug_assembly_size() const { return assembly_size_; }
    f32 debug_inh_frac() const {
        const u8* sp = inh_.spikes();
        u32 a = 0;
        for (u32 k = 0; k < cfg_.n_inh; ++k) a += sp[k];
        return static_cast<f32>(a) / static_cast<f32>(cfg_.n_inh);
    }
    f32 debug_gi_mean() const {
        const f32* gi = const_cast<LifLayer&>(cells_).gi();
        f32 m = 0;
        for (u32 k = 0; k < cfg_.n_cells; ++k) m += gi[k];
        return m / static_cast<f32>(cfg_.n_cells);
    }
    f32 debug_ge_mean() const {
        const f32* ge = const_cast<LifLayer&>(cells_).ge();
        f32 m = 0;
        for (u32 k = 0; k < cfg_.n_cells; ++k) m += ge[k];
        return m / static_cast<f32>(cfg_.n_cells);
    }
    f32 active_frac() const {
        u32 a = 0;
        const u8* sp = cells_.spikes();
        for (u32 k = 0; k < cfg_.n_cells; ++k) a += sp[k];
        return static_cast<f32>(a) / static_cast<f32>(cfg_.n_cells);
    }
    u64 bound_synapses() const {
        u64 c = 0;
        for (f32 w : rec_.weights)
            if (w > cfg_.rec_quanta + 0.05f) ++c;
        return c;
    }

    usize memory_bytes() const {
        return cells_.memory_bytes() + kc2pfc_.memory_bytes() +
               rec_.memory_bytes() + inh_.memory_bytes() +
               e2i_.memory_bytes() + i2e_.memory_bytes();
    }

private:
    PfcConfig cfg_;
    LifLayer cells_;
    LifLayer inh_;
    Synapses e2i_, i2e_;
    Synapses kc2pfc_, rec_;
    f32 familiarity_ = 0.0f;
    f32 fam_fast_ = 0.0f;
    f32 sample_ms_ = 0.0f;
    f32 gate_ = 1.0f;
    std::vector<u8> assembly_;
    u32 assembly_size_ = 0;
};

}  // namespace malefly
