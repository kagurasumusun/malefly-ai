// ============================================================================
// experiments/goal3.cpp — Goal 3 driver (see goal3.hpp for design & criteria).
// ============================================================================
#include "experiments/goal3.hpp"

#include "circuits/action_selection.hpp"
#include "circuits/antennal_lobe.hpp"
#include "circuits/lateral_horn.hpp"
#include "circuits/modulator.hpp"
#include "circuits/mushroom_body.hpp"
#include "core/rng.hpp"
#include "util/jsonw.hpp"
#include "util/odors.hpp"
#include "util/timer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace malefly {
namespace {

u64 fnv1a(const void* data, usize n) {
    const u8* p = static_cast<const u8*>(data);
    u64 h = 1469598103934665603ull;
    for (usize i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}
void hash_mix_f32(u64& h, f32 v) { h = h * 1099511628211ull ^ fnv1a(&v, sizeof(v)); }

void ensure_parent_dir(const std::string& path) {
    const usize pos = path.find_last_of('/');
    if (pos == std::string::npos) return;
    ::mkdir(path.substr(0, pos).c_str(), 0755);
}

}  // namespace

int run_goal3(const Goal3Config& cfg) {
    // ================= world =================
    Odor odor_a = make_odor(50, 15, cfg.seed + 101, 0.60f, 1.00f);
    odor_a.name = "A";
    Odor odor_b = make_odor(50, 15, cfg.seed + 202, 0.60f, 1.00f);
    odor_b.name = "B";

    Rng rng(cfg.seed);
    AntennalLobe al(AntennalLobeConfig{}, rng);
    MushroomBodyConfig mbc;
    mbc.rpe_gating = cfg.use_rpe;
    mbc.use_taxonomy = cfg.use_taxonomy;
    MushroomBody mb(mbc, rng, 50);
    LateralHorn lh(LateralHornConfig{}, rng, 50);
    ModulatorConfig mod_cfg;
    mod_cfg.enabled = cfg.use_arousal;
    Modulator mod(mod_cfg);

    u64 sim_ms = 0;
    auto step_both = [&]() {
        al.step(DT);
        const u8* pn = al.pn_spikes();
        mb.step(DT, pn);
        lh.set_input(pn);
        lh.step(DT);
        mod.step(DT);
        ++sim_ms;
    };
    auto present = [&](const Odor& o, bool sense_only = false) {
        mb.begin_trial();
        lh.begin_window();
        al.set_odor(&o, 1.0f);
        for (u32 t = 0; t < 500; ++t) step_both();
        (void)sense_only;
    };
    auto gap = [&](u32 ms) {
        al.set_odor(nullptr, 0.0f);
        for (u32 t = 0; t < ms; ++t) step_both();
    };

    // outcome-free probe: returns per-odor p_approach via decision samples
    auto probe = [&](const Odor& o, u32 reps) {
        u32 appr = 0;
        f64 v = 0;
        for (u32 i = 0; i < reps; ++i) {
            present(o);
            const auto r = mb.readout();
            DecisionContext ctx;
            ctx.innate_valence = lh.innate_valence();
            ctx.arousal = mod.arousal();
            ctx.use_arousal = cfg.use_arousal;
            const Decision d = decide_ctx(r, mbc, rng, false, ctx);
            appr += d.approach ? 1u : 0u;
            v += d.valence;
            gap(500);
        }
        return std::pair<f64, f64>(static_cast<f64>(appr) / reps, v / reps);
    };

    // training phase with a mutable valence assignment
    struct PhaseRow {
        u32 trial = 0;
        u8 phase = 0, is_a = 0, approach = 0, correct = 0;
        f32 reward = 0, p_appr = 0, arousal = 0;
    };
    std::vector<PhaseRow> rows;
    std::vector<std::pair<f64, f64>> probes;  // (pA, pB) per probe round
    Timer timer;
    timer.start();

    auto run_phase = [&](u8 phase, u32 n_trials, bool a_good) {
        for (u32 t = 0; t < n_trials; ++t) {
            const bool is_a = rng.bernoulli(0.5f);
            present(is_a ? odor_a : odor_b);
            const auto r = mb.readout();
            DecisionContext ctx;
            ctx.innate_valence = lh.innate_valence();
            ctx.arousal = mod.arousal();
            ctx.use_arousal = cfg.use_arousal;
            const Decision d = decide_ctx(r, mbc, rng, true, ctx);
            const bool good = (is_a == a_good);
            f32 rew = 0.0f;
            if (d.approach)
                rew = good ? (rng.bernoulli(0.85f) ? 1.0f : 0.0f)
                           : (rng.bernoulli(0.85f) ? -1.0f : 0.0f);
            mod.on_outcome(rew);
            rows.push_back({t + 1, phase, static_cast<u8>(is_a),
                            static_cast<u8>(d.approach),
                            static_cast<u8>(d.approach == good), rew,
                            d.p_approach, mod.arousal()});
            gap(200);
            if (rew != 0.0f) mb.apply_reinforcement(rew);
            gap(500);
        }
    };

    auto probe_round = [&]() {
        probes.push_back({probe(odor_a, cfg.n_probe_each).first,
                          probe(odor_b, cfg.n_probe_each).first});
    };

    // ---- phases ----
    run_phase(1, cfg.n_train, /*a_good=*/true);
    probe_round();
    run_phase(2, cfg.n_reversal, /*a_good=*/false);
    probe_round();
    run_phase(3, cfg.n_back, /*a_good=*/true);
    probe_round();
    const f64 wall_s = timer.seconds();

    // ---- metrics: per-phase blocks & trials-to-0.70 ----
    struct Block {
        u8 phase = 0;
        f64 acc = 0;
        u32 n = 0;
    };
    std::vector<Block> blocks;
    auto blocks_of = [&](u8 phase) {
        std::vector<Block> out;
        std::vector<PhaseRow> sel;
        for (auto& r : rows)
            if (r.phase == phase) sel.push_back(r);
        for (usize i = 0; i < sel.size(); i += cfg.block) {
            const usize j = std::min(sel.size(), i + cfg.block);
            u32 c = 0;
            for (usize k = i; k < j; ++k) c += sel[k].correct ? 1u : 0u;
            out.push_back({phase, static_cast<f64>(c) / static_cast<f64>(j - i),
                           static_cast<u32>(j - i)});
        }
        return out;
    };
    for (u8 p : {1, 2, 3}) {
        auto b = blocks_of(p);
        blocks.insert(blocks.end(), b.begin(), b.end());
    }
    auto t70 = [&](u8 phase) {
        i64 hit = -1;
        const u32 bl = cfg.block;
        std::vector<PhaseRow> sel;
        for (auto& r : rows)
            if (r.phase == phase) sel.push_back(r);
        for (usize i = bl; i <= sel.size(); ++i) {
            u32 c = 0;
            for (usize k = i - bl; k < i; ++k) c += sel[k].correct ? 1u : 0u;
            if (static_cast<f64>(c) / bl >= 0.70) {
                hit = static_cast<i64>(i);
                break;
            }
        }
        return hit;
    };
    const i64 t70_p1 = t70(1), t70_p2 = t70(2), t70_p3 = t70(3);

    // ---- perf / hash ----
    const f64 sim_s = static_cast<f64>(sim_ms) * 0.001;
    u64 run_hash = 1469598103934665603ull;
    for (const PhaseRow& r : rows) hash_mix_f32(run_hash, r.reward);
    for (const auto& pr : probes) {
        hash_mix_f32(run_hash, static_cast<f32>(pr.first));
        hash_mix_f32(run_hash, static_cast<f32>(pr.second));
    }
    for (f32 w : mb.w_appr()) hash_mix_f32(run_hash, w);
    for (f32 w : mb.w_avoid()) hash_mix_f32(run_hash, w);
    run_hash = run_hash * 1099511628211ull ^ 0x9e3779b97f4a7c15ull;

    // ---- outputs ----
    ensure_parent_dir(cfg.out_csv);
    {
        std::ofstream f(cfg.out_csv);
        f << "trial,phase,is_a,approach,correct,reward,p_approach,arousal\n";
        char buf[192];
        for (const PhaseRow& r : rows) {
            std::snprintf(buf, sizeof buf, "%u,%d,%d,%d,%d,%.1f,%.6f,%.4f\n",
                          r.trial, r.phase, r.is_a, r.approach, r.correct,
                          r.reward, r.p_appr, r.arousal);
            f << buf;
        }
    }
    ensure_parent_dir(cfg.out_json);
    {
        std::ofstream f(cfg.out_json);
        JsonW j(f);
        j.kv("experiment", std::string("goal3_reversal_adaptation"));
        j.kv("tag", cfg.tag);
        j.kv("seed", cfg.seed);
        j.k("config");
        j.obj();
        j.kv("n_train", cfg.n_train);
        j.kv("n_reversal", cfg.n_reversal);
        j.kv("n_back", cfg.n_back);
        j.kv("use_rpe", cfg.use_rpe);
        j.kv("use_arousal", cfg.use_arousal);
        j.kv("use_taxonomy", cfg.use_taxonomy);
        j.end_obj();
        j.k("probes_p_approach");
        j.arr();
        for (const auto& p : probes) {
            j.obj();
            j.kv("pA", p.first);
            j.kv("pB", p.second);
            j.end_obj();
        }
        j.end_arr();
        j.k("blocks");
        j.arr();
        for (const Block& b : blocks) {
            j.obj();
            j.kv("phase", b.phase);
            j.kv("acc", b.acc);
            j.end_obj();
        }
        j.end_arr();
        j.k("t70");
        j.obj();
        j.kv("phase1_train", t70_p1);
        j.kv("phase2_reversal", t70_p2);
        j.kv("phase3_back", t70_p3);
        j.end_obj();
        j.k("perf");
        j.obj();
        j.kv("wall_s", wall_s);
        j.kv("sim_s", sim_s);
        j.kv("rt_factor", sim_s / wall_s);
        j.end_obj();
        j.k("run_hash");
        char hb[32];
        std::snprintf(hb, sizeof hb, "%016llx",
                      static_cast<unsigned long long>(run_hash));
        j.val(std::string(hb));
        j.end_obj();
    }

    // ---- stdout ----
    std::printf("==== Goal 3: reversal adaptation [%s] seed=%llu ====\n",
                cfg.tag.c_str(), static_cast<unsigned long long>(cfg.seed));
    for (usize i = 0; i < probes.size(); ++i)
        std::printf("probe %zu: pA=%.2f pB=%.2f\n", i, probes[i].first,
                    probes[i].second);
    std::printf("blocks: ");
    for (const Block& b : blocks)
        std::printf("P%d:%.2f ", b.phase, b.acc);
    std::printf("\nt70: train=%lld reversal=%lld back=%lld (trial index, -1 = not reached)\n",
                static_cast<long long>(t70_p1), static_cast<long long>(t70_p2),
                static_cast<long long>(t70_p3));
    std::printf("perf: sim=%.0fs wall=%.1fs rt=%.0fx hash=%016llx\n", sim_s,
                wall_s, sim_s / wall_s,
                static_cast<unsigned long long>(run_hash));
    return 0;
}

}  // namespace malefly
