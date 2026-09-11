#pragma once
// ============================================================================
// circuits/language.hpp — AI body interface + infant speech circuits.
//
// POSITIONING (docs/RESULTS §13): the agent keeps the brain-structured
// substrate (sparse sensory recoding -> statistical association cortex ->
// motor production -> corollary-discharge self monitoring) but its BODY is
// an AI interface: syllable tokens in/out instead of larynx/cochlea. No
// claim of real-world embodiment; all mechanisms are wiring+state.
//
// Circuits (infant/monkey-baby level, literature per HUMAN.md §5):
//  - SyllableSensorium (superior-temporal-like): sparse token assemblies,
//    TRP sequence memory via trace-timed Hebbian learning (Saffran, Aslin &
//    Newport 1996: transitional probabilities segment words; tamarins too,
//    Hauser et al. 2001). Word completion = pattern completion over the TRP
//    synapses (dialogue primitive).
//    SELF/OTHER: a corollary-discharge gate from the vocal motor circuit
//    (Eliades & Wang 2003/2008: marmoset auditory cortex suppresses
//    responses to self-vocalization; premotor origin) BOTH suppresses the
//    auditory layer during self-vocalization AND blocks self-produced
//    syllables from entering the caregiver statistics — the agent learns
//    language from the outside, not from its own babble (measurable).
//  - VocalMotor (premotor/Broca-like): spontaneous babbling (internal
//    noise-generated bouts, like the emerge0 motor system), arcuate-style
//    Hebbian audio->motor mapping — a heard syllable paired with own motor
//    activity becomes producible (imitation/echo). Kuhl 2004: social
//    gating — external speech raises babbling rate.
// All learning is trace-timed Hebbian on synaptic weights; no dictionaries,
// no supervision, no software state beyond the interface counters.
// ============================================================================
#include "core/lif.hpp"
#include "core/rng.hpp"
#include "core/synapses.hpp"
#include "core/types.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace malefly {

// AI body interface: syllabary (u8 token ids). The driver/world converts
// text streams <-> token ids; circuits only ever see spike patterns.
struct LanguageConfig {
    u32 n_tokens = 12;        // syllabary size (Saffran: 4 words x 3 syl)
    u32 cells_per_token = 6;  // auditory assembly per syllable

    // auditory layer
    f32 aud_quanta = 0.030f;  // drive per ms while a token sounds
    f32 aud_thresh = -0.055f;
    f32 aud_tau_m = 0.010f;
    f32 aud_refrac = 0.006f;  // ~40-80 Hz rate code

    // TRP sequence memory (aud -> aud, trace-timed Hebbian). Row sums are
    // homeostatically normalized (synaptic scaling), so each synapse's share
    // tracks the co-occurrence RATE = the transitional probability
    // (Saffran 1996; Turrigiano & Nelson 2004 scaling).
    f32 trp_tau = 0.8f;       // s, eligibility trace (Saffran window)
    f32 eta_trp = 0.15f;      // multiplicative LTP per co-occurrence
    f32 w_trp_max = 0.10f;
    u32 trp_fanin = 24;
    f32 trp_init = 0.001f;    // naive collateral; row target = fanin*init

    // completion layer (pattern completion over TRP synapses)
    f32 comp_tau_m = 0.014f;
    f32 comp_thresh = -0.030f;   // gated: only learned transitions fire
    f32 comp_refrac = 0.010f;
    f32 comp_gain = 30.0f;       // completion propagation gain

    // motor layer / babbling
    u32 n_motor_cells = 48;   // (n_tokens syllables x 6 at n_tokens=8)
    f32 motor_tau_m = 0.012f;
    f32 motor_thresh = -0.054f;
    f32 babble_burst_mean_ms = 2200.0f;  // exp ISI between babble bursts
    f32 babble_burst_ms = 180.0f;
    f32 babble_drive = 0.030f;
    f32 social_babble_gain = 3.0f;       // caregiver present -> more babble

    // arcuate audio->motor mapping (Hebbian, STDP window)
    u32 arc_fanin = 30;   // E[inputs from matching token block] ~2 (learnable)
    f32 eta_arc = 0.040f;
    f32 w_arc_max = 0.50f;
    f32 arc_stdp_tau = 0.08f;   // s: must be << inter-burst interval, else
                              // echoes bind to the WRONG rehearsal

    // corollary discharge (self-vocalization gate)
    f32 cd_suppress = 0.120f;   // tonic inhibition while self-vocalizing
};

// ---------------------------------------------------------------------------
class SyllableSensorium {
public:
    SyllableSensorium(const LanguageConfig& cfg, Rng& rng)
        : cfg_(cfg), aud_(n_aud_cells(cfg), aud_cfg()),
          comp_(n_aud_cells(cfg), comp_cfg()),
          n_aud_(n_aud_cells(cfg)) {
        // TRP sequence memory: diluted collaterals between auditory cells,
        // near-silent init — the STATISTICS of external speech write them
        trp_ = make_random_fanin_dist(n_aud_, n_aud_,
                                      static_cast<f32>(cfg.trp_fanin), 3.0f,
                                      rng, cfg.trp_init, 0.30f);
        trace_.assign(n_aud_, 0.0f);
        ext_count_.assign(cfg.n_tokens, 0);
        self_count_.assign(cfg.n_tokens, 0);
    }

    static u32 n_aud_cells(const LanguageConfig& cfg) {
        return cfg.n_tokens * cfg.cells_per_token;
    }
    static LifConfig aud_cfg() {
        LifConfig c;
        c.tau_m = 0.010f;
        c.v_thresh = -0.055f;
        c.t_refrac = 0.006f;
        return c;
    }
    static LifConfig comp_cfg() {
        LifConfig c;
        c.tau_m = 0.014f;
        c.v_thresh = -0.048f;
        c.t_refrac = 0.010f;
        return c;
    }

    u32 n_aud() const { return n_aud_; }
    const u8* aud_spikes() const { return aud_.spikes(); }

    // ---- AI body interface port -------------------------------------------
    // caregiver (or self feedback) sounds token t — driver holds it per ms;
    // external=false marks self-produced feedback (gated by corollary
    // discharge, so own voice is heard only weakly and never as statistics)
    void drive(u32 token, bool external) {
        if (token >= cfg_.n_tokens) return;
        drive_on_[token] = true;
        drive_ext_[token] = external;
    }
    // corollary discharge from VocalMotor (wired by the driver each step)
    void set_self_tag(bool self_speaking) { self_tag_ = self_speaking; }

    void step(f32 dt) {
        for (u32 t = 0; t < cfg_.n_tokens; ++t) {
            if (!drive_on_[t]) continue;
            if (self_tag_ && !drive_ext_[t]) {  // own voice: suppressed input
                ++self_blocked_;
                continue;
            }
            const u32 base = t * cfg_.cells_per_token;
            for (u32 k = 0; k < cfg_.cells_per_token; ++k)
                aud_.add_exc(base + k, cfg_.aud_quanta);
            if (drive_ext_[t]) ++ext_count_[t];
        }
        if (self_tag_ && self_blocked_ == 0) ++self_blocked_;  // tag w/o token
        // corollary-discharge tonic suppression of the auditory layer
        if (self_tag_) aud_.add_inh_all(cfg_.cd_suppress);

        aud_.step(dt);

        // presynaptic traces (the Saffran window)
        const u8* s = aud_.spikes();
        for (u32 k = 0; k < n_aud_; ++k)
            trace_[k] = s[k] ? 1.0f
                             : trace_[k] * std::exp(-dt / cfg_.trp_tau);
        // TRP learning from EXTERNAL speech only (self is gated out above —
        // the measurable self/other function for statistics). Potentiation is
        // followed by row normalization, so weights converge to P(post|pre).
        if (!self_tag_) {
            for (u32 pre = 0; pre < n_aud_; ++pre) {
                if (trace_[pre] < 0.10f) continue;
                const u32 e = trp_.indptr[pre + 1];
                bool touched = false;
                const u32 blk = cfg_.cells_per_token;
                for (u32 k = trp_.indptr[pre]; k < e; ++k) {
                    const u32 post = trp_.indices[k];
                    if (post / blk == pre / blk) continue;  // no token-autapses
                    if (!s[post]) continue;
                    f32& w = trp_.weights[k];
                    if (w < cfg_.w_trp_max) {
                        // multiplicative LTP: after row normalization the
                        // synapse's SHARE converges to the transition
                        // probability P(post|pre) — the Saffran statistic
                        w = std::min(cfg_.w_trp_max, w * (1.0f + cfg_.eta_trp));
                        touched = true;
                    }
                }
                if (touched) normalize_row(pre);
            }
        }

        // pattern completion: TRP synapses drive the completion layer; the
        // threshold sits between boundary (TP~1/3) and within-word (TP~1)
        // drives, so only word-internal transitions complete (word recall
        // from a cue — the primitive behind contingent replying)
        {
            const u32 e_all = trp_.indptr[n_aud_];
            for (u32 pre = 0; pre < n_aud_; ++pre) {
                if (!s[pre]) continue;
                const u32 e = trp_.indptr[pre + 1];
                for (u32 k = trp_.indptr[pre]; k < e; ++k)
                    comp_.ge()[trp_.indices[k]] +=
                        trp_.weights[k] * cfg_.comp_gain;
            }
            (void)e_all;
        }
        comp_.step(dt);

        // drive flags are per-ms (driver re-asserts)
        for (u32 t = 0; t < cfg_.n_tokens; ++t) drive_on_[t] = false;
        self_tag_ = false;
    }

    // ---- measurements -------------------------------------------------------
    // mean TRP weight token(tok_pre) -> token(tok_post) (cell-block mean)
    f64 trp_pair(u32 tok_pre, u32 tok_post) const {
        const u32 b0 = tok_pre * cfg_.cells_per_token;
        const u32 e0 = b0 + cfg_.cells_per_token;
        const u32 b1 = tok_post * cfg_.cells_per_token;
        const u32 e1 = b1 + cfg_.cells_per_token;
        f64 sum = 0; u32 n = 0;
        for (u32 pre = b0; pre < e0; ++pre) {
            const u32 e = trp_.indptr[pre + 1];
            for (u32 k = trp_.indptr[pre]; k < e; ++k) {
                const u32 post = trp_.indices[k];
                if (post < b1 || post >= e1) continue;
                sum += trp_.weights[k]; ++n;
            }
        }
        return n ? sum / n : 0.0;
    }
    // completion-layer activity for a token's assembly right now
    u32 comp_active_for(u32 token) const {
        return act_in(comp_.spikes(), token);
    }
    u32 aud_active_for(u32 token) const {
        return act_in(aud_.spikes(), token);
    }
    f32 aud_ge_for(u32 token) const {
        const f32* ge = const_cast<LifLayer&>(aud_).ge();
        const u32 base = token * cfg_.cells_per_token;
        f32 g = 0;
        for (u32 k = 0; k < cfg_.cells_per_token; ++k) g += ge[base + k];
        return g;
    }
    u64 self_blocked() const { return self_blocked_; }
    const std::vector<u32>& ext_counts() const { return ext_count_; }

    usize memory_bytes() const {
        return trp_.memory_bytes() + aud_.memory_bytes() + comp_.memory_bytes();
    }

private:
    // synaptic scaling: hold the presynaptic row sum at the naive target so
    // each synapse's share tracks the co-occurrence rate (Turrigiano)
    void normalize_row(u32 pre) {
        const u32 e = trp_.indptr[pre + 1];
        f32 sum = 0;
        for (u32 k = trp_.indptr[pre]; k < e; ++k) sum += trp_.weights[k];
        if (sum <= 1e-9f) return;
        const f32 target = static_cast<f32>(cfg_.trp_fanin) * cfg_.trp_init;
        const f32 scale = target / sum;
        for (u32 k = trp_.indptr[pre]; k < e; ++k) trp_.weights[k] *= scale;
    }
    u32 act_in(const u8* s, u32 token) const {
        const u32 base = token * cfg_.cells_per_token;
        u32 a = 0;
        for (u32 k = 0; k < cfg_.cells_per_token; ++k) a += s[base + k];
        return a;
    }

    LanguageConfig cfg_;
    LifLayer aud_, comp_;
    Synapses trp_;
    u32 n_aud_;
    std::vector<f32> trace_;
    bool drive_on_[16] = {false};
    bool drive_ext_[16] = {false};
    bool self_tag_ = false;
    std::vector<u32> ext_count_, self_count_;
    u64 self_blocked_ = 0;
public:
};

// ---------------------------------------------------------------------------
class VocalMotor {
public:
    VocalMotor(const LanguageConfig& cfg, Rng& rng)
        : cfg_(cfg), mot_(cfg.n_motor_cells, motor_cfg()), rng_(rng) {
        m_trace_.assign(cfg.n_motor_cells, 0.0f);
        // first babble comes soon (infants babble from the start); later
        // ISIs are exponential with the config mean
        babble_timer_ms_ =
            std::min(-std::log(std::max(1e-6f, rng_.uniform01())) *
                         cfg_.babble_burst_mean_ms,
                     1500.0f);
    }

    // wire arcuate from the sensorium (driver calls once after building both)
    void bind_sensorium(SyllableSensorium& s) {
        arc_ = make_random_fanin_dist(s.n_aud(), cfg_.n_motor_cells,
                                      static_cast<f32>(cfg_.arc_fanin), 2.0f,
                                      rng_, 0.0f, 0.30f);  // silent init
        sens_ = &s;
    }

    static LifConfig motor_cfg() {
        LifConfig c;
        c.tau_m = 0.012f;
        c.v_thresh = -0.054f;
        c.t_refrac = 0.008f;
        return c;
    }

    u32 cells_per_syl() const { return cfg_.n_motor_cells / cfg_.n_tokens; }
    void set_social_presence(bool present) { social_ = present; }
    // quiet attentive state (probe phase): spontaneous babbling suspended —
    // the driver-side "testing window", wired as a modulatory port
    void set_active(bool a) { active_ = a; }

    // current vocalization (token id) or 0xff if silent; the DRIVER plays
    // this back into the sensorium as self feedback (the AI body loop)
    u8 current_vocalization() const { return vocal_now_; }
    bool speaking() const { return vocal_ms_left_ > 0; }
    u64 vocalizations() const { return n_vocal_; }

    void step(f32 dt, const u8* aud_spikes) {
        // spontaneous babbling (internal noise; social presence raises rate
        // — Kuhl 2004 social gating)
        if (vocal_ms_left_ > 0) {
            vocal_ms_left_ -= static_cast<u32>(dt * 1000.0f + 0.5f);
            // NOTE: vocal_now_ intentionally keeps the LAST syllable after
            // the burst (speaking()==false marks silence) — a caregiver
            // responding to the babble must still know WHAT was said
        } else {
            // timers are in ms; dt is seconds (the decrement is the bug that
            // made babbling 1000x too slow)
            if (babble_timer_ms_ > 0.0f)
                babble_timer_ms_ -= dt * 1000.0f *
                                    (social_ ? cfg_.social_babble_gain : 1.0f);
            if (babble_timer_ms_ <= 0.0f && active_) {
                vocal_now_ = static_cast<u8>(
                    static_cast<u32>(rng_.uniform01() * cfg_.n_tokens) %
                    cfg_.n_tokens);
                vocal_ms_left_ = static_cast<u32>(cfg_.babble_burst_ms);
                babble_timer_ms_ =
                    -std::log(std::max(1e-6f, rng_.uniform01())) *
                    cfg_.babble_burst_mean_ms;
                ++n_vocal_;
            }
        }
        if (vocal_ms_left_ > 0) {
            const u32 base = vocal_now_ * cells_per_syl();
            for (u32 k = 0; k < cells_per_syl(); ++k)
                mot_.add_exc(base + k, cfg_.babble_drive);
        }
        // arcuate: heard audio drives the learned motor mapping (echo).
        // Self feedback never reaches the auditory layer as spikes (gated),
        // so own vocalizations are not re-learned as echoes.
        arc_.propagate(aud_spikes, mot_.ge());
        mot_.step(dt);

        // arcuate Hebbian: audio spike x recently-active motor (STDP window)
        const u8* ms = mot_.spikes();
        for (u32 k = 0; k < cfg_.n_motor_cells; ++k)
            m_trace_[k] = ms[k] ? 1.0f
                                : m_trace_[k] * std::exp(-dt / cfg_.arc_stdp_tau);
        for (u32 pre = 0; pre < sens_->n_aud(); ++pre) {
            if (!aud_spikes[pre]) continue;
            const u32 e = arc_.indptr[pre + 1];
            for (u32 k = arc_.indptr[pre]; k < e; ++k) {
                if (m_trace_[arc_.indices[k]] < 0.05f) continue;
                f32& w = arc_.weights[k];
                if (w < cfg_.w_arc_max) {
                    w = std::min(cfg_.w_arc_max, w + cfg_.eta_arc);
                    ++dbg_arc_events;
                }
            }
        }
    }

    // measurement: motor response per syllable (echo strength)
    u32 mot_active_for(u32 token) const {
        const u8* s = mot_.spikes();
        const u32 base = token * cells_per_syl();
        u32 a = 0;
        for (u32 k = 0; k < cells_per_syl(); ++k) a += s[base + k];
        return a;
    }
    u32 cells_per_syl_count() const { return cells_per_syl(); }
    // mean arcuate weight aud-token j -> motor-token j (echo learnability)
    f64 dbg_arc_pair(u32 tok, u32 cells_per_tok_aud, u32 cells_per_syl) const {
        const u32 b0 = tok * cells_per_tok_aud, e0 = b0 + cells_per_tok_aud;
        const u32 b1 = tok * cells_per_syl, e1 = b1 + cells_per_syl;
        f64 sum = 0; u32 n = 0;
        for (u32 pre = b0; pre < e0; ++pre) {
            const u32 e = arc_.indptr[pre + 1];
            for (u32 k = arc_.indptr[pre]; k < e; ++k) {
                const u32 post = arc_.indices[k];
                if (post < b1 || post >= e1) continue;
                sum += arc_.weights[k]; ++n;
            }
        }
        return n ? sum / n : -1.0;   // -1: no synapse sampling this pair
    }
    u64 arc_bound() const {
        u64 c = 0;
        for (f32 w : arc_.weights) if (w > 0.05f) ++c;
        return c;
    }

private:
    LanguageConfig cfg_;
    LifLayer mot_;
    Synapses arc_;
    Rng rng_;
    SyllableSensorium* sens_ = nullptr;
    std::vector<f32> m_trace_;
    f32 babble_timer_ms_ = 0.0f;
    u8 vocal_now_ = 0xff;
    u32 vocal_ms_left_ = 0;
    bool social_ = false;
    bool active_ = true;
    u64 n_vocal_ = 0;
public:
    u64 dbg_arc_events = 0;
};

}  // namespace malefly
