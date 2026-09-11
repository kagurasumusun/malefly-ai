// ============================================================================
// experiments/calib.cpp — measures AL/MB/LH response statistics.
// Prints per-subclass LN rates, PN rate, KC sparsity, LH innate response for
// the odor pair (A, B) and neutral air. Used to freeze defaults with evidence.
// ============================================================================
#include "experiments/calib.hpp"

#include "circuits/antennal_lobe.hpp"
#include "circuits/lateral_horn.hpp"
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
    LateralHorn lh(LateralHornConfig{}, rng, 50);

    const std::vector<const Odor*> odors = {&odor_a, &odor_b, &odor_n};

    std::printf("calib: kc_quanta=%.4f kc_thresh=%.4f apl_gain=%.5f ln_subclasses=%d\n",
                cfg.kc_quanta, cfg.kc_thresh, cfg.apl_gain,
                AntennalLobeConfig{}.ln_subclasses ? 1 : 0);
    for (const Odor* o : odors) {
        u64 pn_spikes = 0, kc_spikes = 0, lnf_spikes = 0, lns_spikes = 0;
        u32 active_total = 0;
        f64 lh_resp = 0;
        for (u32 p = 0; p < cfg.n_pres; ++p) {
            mb.begin_trial();
            lh.begin_window();
            al.set_odor(o, 1.0f);
            const u64 pn0 = al.pn().total_spikes();
            const u64 lnf0 = al.ln_fast().total_spikes();
            const u64 lns0 = al.ln_slow().total_spikes();
            const u64 kc0 = mb.kc_total_spikes();
            for (u32 t = 0; t < 500; ++t) {
                al.step(DT);
                const u8* pn = al.pn_spikes();
                mb.step(DT, pn);
                lh.set_input(pn);
                lh.step(DT);
            }
            pn_spikes += al.pn().total_spikes() - pn0;
            lnf_spikes += al.ln_fast().total_spikes() - lnf0;
            lns_spikes += al.ln_slow().total_spikes() - lns0;
            kc_spikes += mb.kc_total_spikes() - kc0;
            lh_resp += lh.response();
            u32 act = 0;
            for (u8 x : mb.trial_pattern()) act += x;
            active_total += act;
            al.set_odor(nullptr, 0.0f);
            for (u32 t = 0; t < 500; ++t) {
                al.step(DT);
                const u8* pn = al.pn_spikes();
                mb.step(DT, pn);
                lh.set_input(pn);
                lh.step(DT);
            }
        }
        const f64 pres = static_cast<f64>(cfg.n_pres);
        std::printf(
            "odor %s: PN=%.1f Hz  LNf=%s%.1f Hz  LNs=%s%.1f Hz  KC_active=%.1f%%  "
            "KC_sp/pres=%.2f  LH_resp=%.3f LH_v=%.3f\n",
            o->name.c_str(),
            static_cast<f64>(pn_spikes) / (pres * 0.5 * al.n_glom()),
            al.ln_fast().size() ? "" : "-",
            al.ln_fast().size()
                ? static_cast<f64>(lnf_spikes) / (pres * 0.5 * al.ln_fast().size())
                : 0.0,
            al.ln_slow().size() ? "" : "-",
            al.ln_slow().size()
                ? static_cast<f64>(lns_spikes) / (pres * 0.5 * al.ln_slow().size())
                : 0.0,
            100.0 * static_cast<f64>(active_total) / (pres * mbc.n_kc),
            static_cast<f64>(kc_spikes) / pres,
            lh_resp / pres,
            LateralHornConfig{}.beta * std::tanh((lh_resp / pres) / LateralHornConfig{}.resp_scale));
    }
    return 0;
}

}  // namespace malefly
