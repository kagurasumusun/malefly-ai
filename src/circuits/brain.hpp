// ============================================================================
// circuits/brain.hpp — the SELF-CONTAINED agent.
//
// DESIGN PRINCIPLE (user directive, PRINCIPLES.md #11): the agent contains NO
// software arbiter, no resets from outside, no reinforcement function calls.
// The agent is a WIRING: sensory nerves in, motor nerves out, everything
// between them is circuits. Behavior — including spontaneous behavior — is
// what this wiring DOES when it runs.
//
// What the world may do (and all an experimenter may do):
//   - apply odor stimuli to the antenna (set_odor)
//   - touch the taste/reward sensor or the pain sensor (reward/punish input)
//   - READ the motor nerves (motor counts) and run stats
// Everything else happens inside:
//   - odor-onset detection is a sensory transient gate (rising edge) that
//     lets the MB clear its per-episode counters — internal to the brain
//   - reward is delivered by a VUMmx1-like octopaminergic neuron
//     (Honeybee Connectome: Hammer & Menzel 1995 — one identified neuron
//     whose activation IS the US; it also codes RPE, which converges with
//     our BRAIN-Initiative RPE gate) wired to the MB plasticity machinery
//   - punishment is delivered by a PPL1-like dopaminergic neuron
//   - action is two racing motor populations with mutual inhibition
//     (ramping-to-bound; ZBrain whole-brain ramping before movement);
//     their spikes ARE the behavior — the world just reads which nerve won
//   - rest/active mode + arousal (DMN antagonism, ZBrain state switching,
//     EBRAINS macro-micro coupling) continuously modulate the integrator
//   - drive-side adaptation keeps behavior re-samplable (no deadlock)
// Fixed values are treated as population parameters; heterogeneity lives in
// the neurons (thresholds) and synapses (lognormal weights, fan-in spread).
// ============================================================================
#pragma once
#include "circuits/action_integrator.hpp"
#include "circuits/antennal_lobe.hpp"
#include "circuits/lateral_horn.hpp"
#include "circuits/modulator.hpp"
#include "circuits/mushroom_body.hpp"
#include "core/lif.hpp"
#include "core/rng.hpp"
#include "core/types.hpp"

namespace malefly {

struct BrainConfig {
    AntennalLobeConfig al;
    MushroomBodyConfig mb;
    LateralHornConfig lh;
    ModulatorConfig mod;
    ActionIntegratorConfig integ;

    // reward / punishment pathways (Honeybee VUMmx1 / fly PPL1 analogues)
    f32 us_quanta = 0.090f;      // sensor drive per ms onto VUM / DAN cell
    f32 us_thresh = -0.055f;
    f32 us_min_isi_ms = 100;     // one US pulse ~ 1-2 plasticity events (VUM burst)

    // spontaneous behavior bookkeeping
    u32 motor_bins = 0;          // stats only
};

struct BrainIn {
    f32 reward_sensor = 0.0f;    // 0/1: sugar sensor (world sets per step)
    f32 punish_sensor = 0.0f;    // 0/1: pain/quinine sensor
};

struct BrainOut {
    i64 motor_approach = 0;      // TOTAL approach-nerve spikes ever emitted
    i64 motor_avoid = 0;         // TOTAL avoid-nerve spikes (world diffs them)
    u64 decisions = 0;           // completed decision events (races)
    u64 choice_approach = 0;     // races won by the approach population
    u64 choice_avoid = 0;        // races won by the avoid population
    f32 arousal = 0.0f;          // introspection (measurement only)
    bool rest = false;           // introspection (measurement only)
};

class Brain {
public:
    Brain(const BrainConfig& cfg, Rng& rng)
        : cfg_(cfg), rng_(rng), al_(cfg.al, rng),
          mb_(cfg.mb, rng, cfg.al.n_glom), lh_(cfg.lh, rng, cfg.al.n_glom),
          mod_(cfg.mod),
          vum_(1, us_config(cfg)), dan_(1, us_config(cfg)),
          integ_(cfg.integ) {
        // the integrator runs from birth: spontaneous decisions exist with
        // no sensory history at all (they come from neural variability)
        integ_.begin_decision(0.0f, 0.0f, 0.0f, false);
    }

    // ---- world interface (the ONLY inputs) ---------------------------------
    void set_odor(const Odor* o, f32 conc) { pending_odor_ = o; conc_ = conc; }
    void set_sensors(f32 reward, f32 punish) {
        in_.reward_sensor = reward;
        in_.punish_sensor = punish;
    }

    // ---- the brain, running -------------------------------------------------
    void step(f32 dt) {
        // sensory onset transient (internal gate): rising edge of odor input
        const bool odor_now = (pending_odor_ != nullptr);
        if (odor_now && !odor_prev_) {
            // MB clears its per-episode counters — this is a sensory-gated
            // internal event, not an OS reset of memory (weights & traces
            // persist; only the episode accumulator clears)
            mb_.begin_trial();
            lh_.begin_window();  // innate-path response window = this episode
            const auto r = mb_.trace_readout();
            integ_.begin_decision(r.valence, gated_innate(r.valence),
                                  mod_.arousal(), mod_.rest());
        }
        odor_prev_ = odor_now;
        al_.set_odor(pending_odor_, conc_);

        // olfactory pathway
        al_.step(dt);
        const u8* pn = al_.pn_spikes();
        mb_.step(dt, pn);
        lh_.set_input(pn);
        lh_.step(dt);

        // internal state network (arousal + rest/active mode)
        mod_.step(dt, &rng_);

        // US pathways: sensor -> identified neuron -> plasticity signal
        step_us(vum_, in_.reward_sensor, dt, +1.0f);
        step_us(dan_, in_.punish_sensor, dt, -1.0f);

        // action integrator runs CONTINUOUSLY: a completed race is a
        // behavior event; a new race starts immediately (a stream of
        // spontaneous decisions — no one schedules them)
        const auto tr = mb_.trace_readout();
        integ_.step(dt, rng_, tr.valence, gated_innate(tr.valence));
        if (integ_.decided()) {
            tot_appr_ += integ_.motor_approach();
            tot_avoid_ += integ_.motor_avoid();
            ++events_;
            if (integ_.approach_wins()) ++wins_appr_; else ++wins_avoid_;
            integ_.begin_decision(tr.valence, gated_innate(tr.valence),
                                  mod_.arousal(), mod_.rest());
        }
    }

    BrainOut out() const {
        BrainOut o;
        o.motor_approach = tot_appr_;
        o.motor_avoid = tot_avoid_;
        o.decisions = events_;
        o.choice_approach = wins_appr_;
        o.choice_avoid = wins_avoid_;
        o.arousal = mod_.arousal();
        o.rest = mod_.rest();
        return o;
    }

    // ---- introspection (measurement only — never used for behavior) --------
    const AntennalLobe& al() const { return al_; }
    const MushroomBody& mb() const { return mb_; }
    const LateralHorn& lh() const { return lh_; }
    const Modulator& modulator() const { return mod_; }
    const ActionIntegrator& integrator() const { return integ_; }
    u64 vum_spikes() const { return vum_.total_spikes(); }
    u64 dan_spikes() const { return dan_.total_spikes(); }

private:
    static LifConfig us_config(const BrainConfig& c) {
        LifConfig lc;
        lc.tau_m = 0.020f;
        lc.v_thresh = c.us_thresh;
        lc.t_refrac = static_cast<f32>(c.us_min_isi_ms) * 0.001f;
        return lc;
    }

    // sensor -> identified neuromodulatory cell -> MB plasticity event
    void step_us(LifLayer& cell, f32 sensor, f32 dt, f32 sign) {
        if (sensor > 0.0f) cell.add_exc(0, cfg_.us_quanta);
        cell.step(dt);
        if (cell.spikes()[0]) mb_.apply_reinforcement(sign);
    }

    f32 gated_innate(f32 v_mb) const {
        const f32 gate =
            1.0f - std::min(1.0f, std::fabs(v_mb) / cfg_.mb.innate_gate_vmax);
        return lh_.innate_valence() * gate;
    }

    BrainConfig cfg_;
    Rng& rng_;
    AntennalLobe al_;
    MushroomBody mb_;
    LateralHorn lh_;
    Modulator mod_;
    LifLayer vum_;   // octopaminergic reward neuron (VUMmx1 analogue)
    LifLayer dan_;   // dopaminergic punishment neuron (PPL1 analogue)
    ActionIntegrator integ_;
    BrainIn in_{};
    const Odor* pending_odor_ = nullptr;
    f32 conc_ = 1.0f;
    bool odor_prev_ = false;
    i64 tot_appr_ = 0, tot_avoid_ = 0;
    u64 events_ = 0, wins_appr_ = 0, wins_avoid_ = 0;
};

}  // namespace malefly
