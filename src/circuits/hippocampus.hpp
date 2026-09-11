// ============================================================================
// circuits/hippocampus.hpp — primate-inspired episodic memory system.
//
// LITERATURE GROUNDING (docs/HUMAN.md):
//   - Complementary Learning Systems (McClelland, McNaughton & O'Reilly 1995):
//     a FAST, sparse, pattern-separated system (hippocampus) for one-shot
//     episodic encoding, complementing the SLOW overlapping system (here: the
//     mushroom-body 'neocortical' substrate) that integrates across events.
//   - Dentate gyrus performs PATTERN SEPARATION via extremely sparse coding;
//     CA3 stores auto-associative memories in DILUTED recurrent collaterals
//     and performs PATTERN COMPLETION from a fragment (Rolls 2013; 
//     mossy-fiber expansion + sparse CA3 + LTP/heterosynaptic LTD).
//   - Novelty/TD-error gates WHAT is stored (CLS; Kumaran 2016): here the
//     encoder fires when the agent's own novelty signal (1 - familiarity)
//     is high — no external 'store this' calls.
//   - Retrieval practice transfers episodic content into slow systems
//     (test-enhanced learning, Roediger & Karpicke 2006) — implemented as
//     re-exposure-driven MB plasticity reactivation (systems consolidation).
//
// Implementation: all projections are sparse random wiring (lognormal
// weights); encode is a one-shot Hebbian event on co-active CA3 pairs and on
// valence-binding cells. No tables, no keys — memory IS the changed synapses.
// ============================================================================
#pragma once
#include "core/lif.hpp"
#include "core/rng.hpp"
#include "core/synapses.hpp"
#include "core/types.hpp"
#include <algorithm>
#include <cmath>
#include <vector>
#include <algorithm>

namespace malefly {

struct HippocampusConfig {
    u32 n_dg = 1500;         // dentate granule cells
    u32 dg_fanin = 8;        // KC -> DG
    f32 dg_quanta = 0.130f;
    f32 dg_thresh = -0.054f; // tuned -> DG active ~2-5% (pattern separation)

    u32 n_ca3 = 1200;
    u32 mossy_fanin = 40;    // DG -> CA3 (mossy fibers dominate CA3 drive)
    f32 mossy_quanta = 0.016f;
    f32 ca3_thresh = -0.056f;

    u32 ca3_rec_fanin = 60;   // ~5% of n_ca3 (diluted, Rolls 2013)  // diluted recurrent collaterals (Rolls 2013)
    f32 ca3_rec_quanta = 0.012f;
    f32 eta_enc = 0.55f;     // one-shot LTP on encode
    f32 w_rec_max = 0.90f;
    f32 eta_homeo = 0.020f;  // slow heterosynaptic LTD toward baseline

    u32 n_ca1 = 200;
    u32 ca1_fanin = 15;      // CA3 -> CA1 readout (fixed)

    f32 familiarity_tau = 3.0f;   // s, familiarity trace (perirhinal-like)
    f32 novelty_gate = 0.30f;     // encode when novelty (1-fam) above this
    f32 enc_refrac_ms = 1500;     // min spacing between encode events

    // valence binding (CA3 -> valence cells, one-shot)
    f32 eta_val = 0.45f;

    // perirhinal familiarity readout (Brown & Aggleton 2001): silent-init
    // CA3->PR synapses stamped at encode; PR firing = cumulative match
    u32 n_pr = 64;
    u32 pr_fanin = 30;

    f32 decay_strength = 0.0f;    // episodic trace decay per step (0 = none;
                                  // forgetting emerges via interference)
};

class Hippocampus {
public:
    Hippocampus(const HippocampusConfig& cfg, Rng& rng, u32 n_input)
        : cfg_(cfg), n_in_(n_input),
          dg_(cfg.n_dg, dg_cfg()), ca3_(cfg.n_ca3, ca3_cfg()),
          vpos_(1, val_cfg()), vneg_(1, val_cfg()),
          pr_(cfg.n_pr, pr_cfg()) {
        kc2dg_ = make_random_fanin_dist(n_input, cfg.n_dg,
                                        static_cast<f32>(cfg.dg_fanin), 1.2f,
                                        rng, cfg.dg_quanta, 0.30f);
        dg2ca3_ = make_random_fanin_dist(cfg.n_dg, cfg.n_ca3,
                                         static_cast<f32>(cfg.mossy_fanin), 1.5f,
                                         rng, cfg.mossy_quanta, 0.35f);
        ca3rec_ = make_random_fanin_dist(cfg.n_ca3, cfg.n_ca3,
                                         static_cast<f32>(cfg.ca3_rec_fanin), 3.0f,
                                         rng, cfg.ca3_rec_quanta, 0.40f);
        // recurrent weights start at ~0: storage ADDS to them
        for (f32& w : ca3rec_.weights) w *= 0.02f;
        ca32val_ = make_random_fanin_dist(cfg.n_ca3, 2, 150.0f, 8.0f,
                                          rng, 0.020f, 0.35f);
        // start near-silent: valence cells fire only for US-stamped assemblies
        for (f32& w : ca32val_.weights) w *= 0.02f;
        ca32pr_ = make_random_fanin_dist(cfg.n_ca3, cfg.n_pr,
                                         static_cast<f32>(cfg.pr_fanin), 3.0f,
                                         rng, 0.030f, 0.35f);
        for (f32& w : ca32pr_.weights) w *= 0.02f;  // silent until stamped
        familiarity_ = 0.0f;
        refrac_ms_ = 0;
        episodes_ = 0;
        pr_ge_.assign(cfg.n_pr, 0.0f);
    }

    static LifConfig dg_cfg() {
        LifConfig c;
        c.tau_m = 0.012f;
        c.v_thresh = -0.046f;
        return c;
    }
    static LifConfig ca3_cfg() {
        LifConfig c;
        c.tau_m = 0.018f;
        c.tau_exc = 0.008f;
        c.v_thresh = -0.036f;
        return c;
    }
    static LifConfig pr_cfg() {
        LifConfig c;
        c.tau_m = 0.020f;
        c.v_thresh = -0.055f;
        return c;
    }
    static LifConfig val_cfg() {
        LifConfig c;
        c.tau_m = 0.020f;
        c.v_thresh = -0.055f;
        return c;
    }

    // world sensor port (sugar / pain) — the ONLY instructive input
    void set_us(f32 reward, f32 punish) {
        us_pos_ = reward;
        us_neg_ = punish;
    }

    // ---- the system, running on the perceptual stream ----------------------
    void step(f32 dt, const u8* kc_pattern) {
        // familiarity trace decays
        familiarity_ *= std::exp(-dt / cfg_.familiarity_tau);
        if (refrac_ms_ > 0) --refrac_ms_;
        if (us_pos_ > 0.0f || us_neg_ > 0.0f) {
            if (++us_hold_ms_ > 800) { us_pos_ = 0.0f; us_neg_ = 0.0f; us_hold_ms_ = 0; }
        }

        // DG pattern separation
        kc2dg_.propagate(kc_pattern, dg_.ge());
        dg_.step(dt);

        // CA3: mossy drive + recurrent drive from LAST step (one delay)
        dg2ca3_.propagate(dg_.spikes(), ca3_.ge());
        ca3rec_.propagate(ca3_.spikes(), ca3_.ge());
        ca3_.step(dt);

        // valence binding cells (driven by current CA3 assembly)
        ca32val_.propagate(ca3_.spikes(), val_ge_);
        vpos_.add_exc(0, val_ge_[0]);
        vneg_.add_exc(0, val_ge_[1]);
        vpos_.step(dt);
        vneg_.step(dt);
        val_pos_total_ += vpos_.spikes()[0];
        val_neg_total_ += vneg_.spikes()[0];
        val_ge_[0] *= 0.7f;
        val_ge_[1] *= 0.7f;

        // perirhinal familiarity: PR cells fire to the extent the current
        // CA3 assembly overlaps ENCODED assemblies (synaptic match signal)
        ca32pr_.propagate(ca3_.spikes(), pr_ge_.data());
        { for (u32 k = 0; k < cfg_.n_pr; ++k) pr_.add_exc(k, pr_ge_[k]); }
        pr_.step(dt);

        // familiarity = how much current CA3 matches the strongest stored
        // assembly (measured as recurrent-supported excess activation)
        u32 act = 0;
        {
            const u8* sp = ca3_.spikes();
            for (u32 k = 0; k < cfg_.n_ca3; ++k) act += sp[k];
        }
        const f32 frac = static_cast<f32>(act) / static_cast<f32>(cfg_.n_ca3);
        // recall support = fraction above the naive CA3 rate (~2%): the part
        // of activity that recurrent weights are holding up
        // one-shot ENCODE: a strong CA3 assembly marks an episode; the
        // familiarity EMA provides novelty context for measurement
        {
            u32 pr_act = 0;
            const u8* ps = pr_.spikes();
            for (u32 k = 0; k < cfg_.n_pr; ++k) pr_act += ps[k];
            const f32 pr_rate = static_cast<f32>(pr_act) / static_cast<f32>(cfg_.n_pr);
            familiarity_ += (std::min(1.0f, pr_rate * 25.0f) - familiarity_) *
                            (1.0f - std::exp(-dt / cfg_.familiarity_tau));
        }
        // Encode gate: a sparse CA3 assembly marks an episode. A live US
        // event OVERRIDES the refractory (synaptic tagging & capture:
        // neuromodulatory strong events stamp in the ongoing trace —
        // Frey & Morris 1997; Li et al. 2003).
        const bool us_live = (us_hold_ms_ > 0);
        if (us_refrac_ms_ > 0) --us_refrac_ms_;
        if (frac > 0.030f && (refrac_ms_ == 0 || (us_live && us_refrac_ms_ == 0))) {
            encode_event();
            refrac_ms_ = cfg_.enc_refrac_ms;
            us_refrac_ms_ = 500;  // ~one stamp per US event
            ++episodes_;
        }

        // slow heterosynaptic homeostasis ( Roll's LTD term)
        if (cfg_.eta_homeo > 0.0f)
            for (f32& w : ca3rec_.weights)
                if (w > cfg_.ca3_rec_quanta * 0.02f)
                    w -= cfg_.eta_homeo * w * dt;
    }

    // valence(+/-) spike counts since last call (readout + auto-clear)
    void read_valence(u32& pos, u32& neg) {
        pos = val_pos_total_;
        neg = val_neg_total_;
        val_pos_total_ = 0;
        val_neg_total_ = 0;
    }

    f32 familiarity() const { return familiarity_; }
    f32 novelty() const { return 1.0f - familiarity_; }
    u32 episodes() const { return episodes_; }
    f32 dg_frac() const {
        u32 a = 0;
        const u8* sp = dg_.spikes();
        for (u32 k = 0; k < cfg_.n_dg; ++k) a += sp[k];
        return static_cast<f32>(a) / static_cast<f32>(cfg_.n_dg);
    }
    f32 ca3_frac() const {
        u32 a = 0;
        const u8* sp = ca3_.spikes();
        for (u32 k = 0; k < cfg_.n_ca3; ++k) a += sp[k];
        return static_cast<f32>(a) / static_cast<f32>(cfg_.n_ca3);
    }
    u64 rec_nonzero() const {
        u64 c = 0;
        for (f32 w : ca3rec_.weights)
            if (w > 0.01f) ++c;
        return c;
    }

    usize memory_bytes() const {
        return kc2dg_.memory_bytes() + dg2ca3_.memory_bytes() +
               ca3rec_.memory_bytes() + ca32val_.memory_bytes() + ca32pr_.memory_bytes() +
               dg_.memory_bytes() + ca3_.memory_bytes() +
               vpos_.memory_bytes() + vneg_.memory_bytes();
    }

private:
    // one-shot Hebbian binding among co-active CA3 cells + valence cells.
    // Synapses are CSR (rows = presynaptic id): iterate ACTIVE PRESYNAPTIC
    // cells and strengthen their outgoing synapses (co-activity = pre fires
    // while the assembly is up; postsynaptic cells are the active set).
    void encode_event() {
        const u8* s = ca3_.spikes();
        for (u32 pre = 0; pre < cfg_.n_ca3; ++pre) {
            if (!s[pre]) continue;
            // recurrent collaterals: strengthen existing diluted synapses
            const u32 e = ca3rec_.indptr[pre + 1];
            for (u32 k = ca3rec_.indptr[pre]; k < e; ++k) {
                f32& w = ca3rec_.weights[k];
                if (w > 0.0005f)  // only EXISTING diluted collaterals
                    w = std::min(cfg_.w_rec_max, w + cfg_.eta_enc);
            }
            // perirhinal stamp: THIS assembly was experienced (one-shot)
            const u32 ep = ca32pr_.indptr[pre + 1];
            for (u32 k = ca32pr_.indptr[pre]; k < ep; ++k)
                ca32pr_.weights[k] = std::min(0.50f, ca32pr_.weights[k] + cfg_.eta_enc);
            // valence binding: only when a US neuromodulatory event accompanied
            // the episode (VUM/DAN state — set via set_us from world sensors)
            if (us_pos_ > 0.0f || us_neg_ > 0.0f) {
                const u32 ev = ca32val_.indptr[pre + 1];
                for (u32 k = ca32val_.indptr[pre]; k < ev; ++k) {
                    const u32 target = ca32val_.indices[k];  // 0 = v+, 1 = v-
                    if (((target == 0) ? us_pos_ : us_neg_) > 0.0f)
                        ca32val_.weights[k] =
                            std::min(0.20f, ca32val_.weights[k] + cfg_.eta_val);
                }
            }
        }
    }

    HippocampusConfig cfg_;
    u32 n_in_;
    LifLayer dg_, ca3_, vpos_, vneg_, pr_;
    std::vector<f32> pr_ge_;
    Synapses kc2dg_, dg2ca3_, ca3rec_, ca32val_, ca32pr_;
    f32 val_ge_[2] = {0.0f, 0.0f};
    f32 familiarity_ = 0.0f;
    f32 us_pos_ = 0.0f, us_neg_ = 0.0f;
    u32 us_hold_ms_ = 0;
    i32 refrac_ms_ = 0;
    i32 us_refrac_ms_ = 0;
    u32 episodes_ = 0;
    u32 val_pos_total_ = 0, val_neg_total_ = 0;
};

}  // namespace malefly
