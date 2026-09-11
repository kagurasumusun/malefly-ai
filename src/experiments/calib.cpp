// ============================================================================
// experiments/calib.cpp — measures AL/MB response statistics for given params.
// Prints: PN rate during odor, KC active fraction, KC spikes per presentation
// for the trained pair (A, B) and a neutral odor. Used to freeze defaults.
// ============================================================================
#include "experiments/calib.hpp"

#include "circuits/antennal_lobe.hpp"
#include "circuits/mushroom_body.hpp"
#include "core/rng.hpp"
#include "util/odors.hpp"

#include <cstdio>
#include <vector>

namespace malefly {

int run_calib(const CalibConfig& cfg) {
    Odor odor_a = make_odor(50, 15, cfg.seed + 101, 0.60f, 1.00f);
    odor_a.name = "A";
    Odor odor_b = make_odor(50, 15, cfg.seed + 202, 0.60f, 1.00f);
    odor_b.name = "B";
    Odor odor_n = uniform_odor(50, 0.12f);
    odor_n.name = "N";

    Rng rng(cfg.seed);
    AntennalLobe al(AntennalLobeConfig{}, rng);
    MushroomBodyConfig mbc;
    mbc.kc_quanta = cfg.kc_quanta;
    mbc.kc_v_thresh = cfg.kc_thresh;
    mbc.apl_gain = cfg.apl_gain;
    MushroomBody mb(mbc, rng, 50);

    const std::vector<const Odor*> odors = {&odor_a, &odor_b, &odor_n};

    std::printf("calib: kc_quanta=%.4f kc_thresh=%.4f apl_gain=%.5f\n",
                cfg.kc_quanta, cfg.kc_thresh, cfg.apl_gain);
    for (const Odor* o : odors) {
        u64 pn_spikes = 0, kc_spikes = 0, ln_spikes = 0;
        u32 active_total = 0;
        for (u32 p = 0; p < cfg.n_pres; ++p) {
            mb.begin_trial();
            al.set_odor(o, 1.0f);
            const u64 pn0 = al.pn().total_spikes();
            const u64 ln0 = al.ln().total_spikes();
            const u64 kc0 = mb.kc_total_spikes();
            for (u32 t = 0; t < 500; ++t) {
                al.step(DT);
                mb.step(DT, al.pn_spikes());
            }
            pn_spikes += al.pn().total_spikes() - pn0;
            ln_spikes += al.ln().total_spikes() - ln0;
            kc_spikes += mb.kc_total_spikes() - kc0;
            u32 act = 0;
            for (u8 x : mb.trial_pattern()) act += x;
            active_total += act;
            al.set_odor(nullptr, 0.0f);
            for (u32 t = 0; t < 500; ++t) {
                al.step(DT);
                mb.step(DT, al.pn_spikes());
            }
        }
        const f64 pres = static_cast<f64>(cfg.n_pres);
        std::printf(
            "odor %s: PN=%.1f Hz  LN=%.1f Hz  KC_active=%.1f%%  KC_spikes/pres=%.2f\n",
            o->name.c_str(),
            static_cast<f64>(pn_spikes) / (pres * 0.5 * al.n_glom()),
            static_cast<f64>(ln_spikes) / (pres * 0.5 * al.n_ln()),
            100.0 * static_cast<f64>(active_total) / (pres * mbc.n_kc),
            static_cast<f64>(kc_spikes) / pres);
    }
    return 0;
}

}  // namespace malefly
