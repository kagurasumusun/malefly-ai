// ============================================================================
// experiments/emerge0.cpp — see emerge0.hpp. Every number is measured from
// the running wiring; nothing here trains or programs the agent.
// ============================================================================
#include "experiments/emerge0.hpp"

#include "circuits/brain.hpp"
#include "util/jsonw.hpp"
#include "util/odors.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

namespace malefly {

int run_emerge0(const Emerge0Config& cfg) {
    Rng rng(cfg.seed);
    Brain brain(BrainConfig{}, rng);

    Odor odor_a = make_odor(50, 15, cfg.seed + 101, 0.60f, 1.00f);
    odor_a.name = "A";
    Odor odor_b = make_odor(50, 15, cfg.seed + 202, 0.60f, 1.00f);
    odor_b.name = "B";

    struct Bin {
        i32 appr = 0, avoid = 0;
        u8 rest = 0;
        f32 arousal = 0;
        u8 odor = 0;  // 0 none, 1 A, 2 B
    };
    std::vector<Bin> bins;
    Bin cur;
    u32 bin_t = 0;

    u32 rest_switches = 0, side_switches = 0;
    bool prev_rest = false;
    i32 prev_sign = 0;

    i64 prev_appr = 0, prev_avoid = 0;
    auto step_phase = [&](u32 ms, u8 odor_code) {
        for (u32 t = 0; t < ms; ++t) {
            // phase 2 alternates odors every 5 s (still outcome-free)
            if (odor_code == 1) {
                const u32 slot = (t / 5000) % 2;
                brain.set_odor(slot ? &odor_b : &odor_a, 1.0f);
            } else {
                brain.set_odor(nullptr, 0.0f);
            }
            brain.set_sensors(0.0f, 0.0f);  // NO rewards, EVER
            brain.step(DT);
            const auto o = brain.out();
            cur.appr += static_cast<i32>(o.motor_approach - prev_appr);
            cur.avoid += static_cast<i32>(o.motor_avoid - prev_avoid);
            prev_appr = o.motor_approach;
            prev_avoid = o.motor_avoid;
            cur.rest = o.rest;
            cur.arousal = o.arousal;
            cur.odor = odor_code;
            const bool r = o.rest;
            if (r != prev_rest) ++rest_switches;
            prev_rest = r;
            if (++bin_t == cfg.bin_ms) {
                const i32 net = cur.appr - cur.avoid;
                const i32 sign = (std::abs(net) > 10) ? (net > 0 ? 1 : -1) : 0;
                if (sign != 0 && prev_sign != 0 && sign != prev_sign)
                    ++side_switches;
                if (sign != 0) prev_sign = sign;
                bins.push_back(cur);
                cur = Bin{};
                bin_t = 0;
            }
        }
    };

    // ---- phase 1: empty world ----
    step_phase(cfg.empty_ms, 0);
    const u32 empty_bins = static_cast<u32>(bins.size());

    // ---- phase 2: odors appear (still outcome-free) ----
    step_phase(cfg.odor_ms, 1);

    // ---- measurements ----
    u32 bouts_empty = 0, bouts_odor = 0;
    i32 net_empty = 0, net_odor = 0;
    u32 rest_bins = 0, rest_bins_empty = 0;
    for (u32 i = 0; i < bins.size(); ++i) {
        const bool in_empty = i < empty_bins;
        const i32 tot = bins[i].appr + bins[i].avoid;
        if (tot >= static_cast<i32>(cfg.motor_bout_thresh))
            (in_empty ? bouts_empty : bouts_odor)++;
        (in_empty ? net_empty : net_odor) += bins[i].appr - bins[i].avoid;
        rest_bins += bins[i].rest;
        if (in_empty) rest_bins_empty += bins[i].rest;
    }
    const f64 s_empty = static_cast<f64>(net_empty);
    const f64 s_odor = static_cast<f64>(net_odor);
    const f64 asym_empty = (s_empty + s_odor) > 0 ? s_empty / (s_empty + s_odor) : 0.5;

    std::printf("==== emerge0: wiring-only behavior (no training, no program) ====\n");
    std::printf("empty world: motor bouts %u/%u bins (%.0f%%), net side balance %.3f, "
                "rest %u/%u bins, switches %u\n",
                bouts_empty, empty_bins,
                100.0 * bouts_empty / std::max<u32>(empty_bins, 1),
                asym_empty, rest_bins_empty, empty_bins, rest_switches);
    u32 odor_bins = static_cast<u32>(bins.size()) - empty_bins;
    u32 rest_bins_odor = rest_bins - rest_bins_empty;
    std::printf("odor world (no outcomes): bouts %u/%u (%.0f%%), approach-side net %+d, "
                "rest %u/%u\n",
                bouts_odor, odor_bins,
                100.0 * bouts_odor / std::max<u32>(odor_bins, 1), net_odor,
                rest_bins_odor, odor_bins);
    std::printf("spontaneous side-switches: %u | VUM spikes %llu, DAN spikes %llu (never driven)\n",
                side_switches,
                static_cast<unsigned long long>(brain.vum_spikes()),
                static_cast<unsigned long long>(brain.dan_spikes()));

    std::ofstream f(cfg.out_json);
    JsonW j(f);
    j.kv("experiment", std::string("emerge0_spontaneous_behavior"));
    j.kv("seed", cfg.seed);
    j.k("empty_world");
    j.obj();
    j.kv("bins", empty_bins);
    j.kv("motor_bouts", bouts_empty);
    j.kv("rest_bins", rest_bins_empty);
    j.end_obj();
    j.k("odor_world_no_outcome");
    j.obj();
    j.kv("bins", odor_bins);
    j.kv("motor_bouts", bouts_odor);
    j.kv("approach_net", net_odor);
    j.kv("rest_bins", rest_bins_odor);
    j.end_obj();
    j.kv("rest_switches", rest_switches);
    j.kv("side_switches", side_switches);
    j.end_obj();

    std::ofstream c(cfg.out_csv);
    c << "bin,odor,rest,arousal,motor_approach,motor_avoid\n";
    for (u32 i = 0; i < bins.size(); ++i) {
        char b[128];
        std::snprintf(b, sizeof b, "%u,%u,%u,%.3f,%d,%d\n", i, bins[i].odor,
                      bins[i].rest, bins[i].arousal, bins[i].appr, bins[i].avoid);
        c << b;
    }
    std::printf("results: %s | %s\n", cfg.out_json.c_str(), cfg.out_csv.c_str());
    return 0;
}

}  // namespace malefly
