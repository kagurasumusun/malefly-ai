// ============================================================================
// experiments/goal2.cpp — Goal 2 driver (see goal2.hpp for the design).
//
// All reported numbers are measured from this run (no analytic shortcuts).
// ============================================================================
#include "experiments/goal2.hpp"

#include "circuits/action_selection.hpp"
#include "circuits/antennal_lobe.hpp"
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
    for (usize i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}
void hash_mix_f32(u64& h, f32 v) { h = h * 1099511628211ull ^ fnv1a(&v, sizeof(v)); }

f64 vec_mean(const std::vector<f64>& v) {
    if (v.empty()) return 0.0;
    f64 s = 0;
    for (f64 x : v) s += x;
    return s / static_cast<f64>(v.size());
}
f64 vec_sd(const std::vector<f64>& v, f64 m) {
    if (v.size() < 2) return 0.0;
    f64 s = 0;
    for (f64 x : v) s += (x - m) * (x - m);
    return std::sqrt(s / static_cast<f64>(v.size() - 1));
}

void ensure_parent_dir(const std::string& path) {
    const usize pos = path.find_last_of('/');
    if (pos == std::string::npos) return;
    ::mkdir(path.substr(0, pos).c_str(), 0755);
}

struct ProbeRow {
    std::string odor;
    f32 p_approach = 0, valence = 0, kc_active = 0;
};

struct OdorAgg {
    f64 p = 0, p_sd = 0, v = 0, kc = 0;
    u32 n = 0;
};

OdorAgg aggregate(const std::vector<ProbeRow>& rows, const std::string& name) {
    std::vector<f64> ps, vs, ks;
    for (const ProbeRow& r : rows)
        if (r.odor == name) {
            ps.push_back(r.p_approach);
            vs.push_back(r.valence);
            ks.push_back(r.kc_active);
        }
    OdorAgg a;
    a.n = static_cast<u32>(ps.size());
    if (!ps.empty()) {
        a.p = vec_mean(ps);
        a.v = vec_mean(vs);
        a.kc = vec_mean(ks);
        a.p_sd = vec_sd(ps, a.p);
    }
    return a;
}

f64 jaccard_u8(const std::vector<u8>& a, const std::vector<u8>& b) {
    u32 inter = 0, uni = 0;
    for (usize i = 0; i < a.size(); ++i) {
        inter += (a[i] && b[i]) ? 1 : 0;
        uni += (a[i] || b[i]) ? 1 : 0;
    }
    return uni ? static_cast<f64>(inter) / static_cast<f64>(uni) : 0.0;
}

}  // namespace

int run_goal2(const Goal2Config& cfg) {
    // ================= world =================
    Odor odor_a = make_odor(cfg.n_glom, 15, cfg.seed + 101, 0.60f, 1.00f);
    odor_a.name = "A_rewarded";
    Odor odor_b = make_odor(cfg.n_glom, 15, cfg.seed + 202, 0.60f, 1.00f);
    odor_b.name = "B_punished";
    Odor fresh = make_odor(cfg.n_glom, 15, cfg.seed + 404, 0.60f, 1.00f);
    Odor odor_e = mix_odors(odor_a, fresh, 0.5f, 0.5f);  // ~half support shared with A
    odor_e.name = "E_new_positive";
    Odor odor_n = uniform_odor(cfg.n_glom, 0.12f);
    odor_n.name = "N_neutral";
    const std::vector<const Odor*> probe_list = {&odor_a, &odor_b, &odor_e, &odor_n};
    const usize np = probe_list.size();

    Rng rng(cfg.seed);
    AntennalLobe al(AntennalLobeConfig{}, rng);
    MushroomBodyConfig mbc;
    MushroomBody mb(mbc, rng, cfg.n_glom);

    // ================= helpers =================
    u64 sim_ms = 0;
    auto step_both = [&]() {
        al.step(DT);
        mb.step(DT, al.pn_spikes());
        ++sim_ms;
    };
    auto present = [&](const Odor& odor) {
        mb.begin_trial();
        al.set_odor(&odor, cfg.concentration);
        for (u32 t = 0; t < cfg.t_on_ms; ++t) step_both();
    };
    auto run_gap = [&](u32 ms) {
        al.set_odor(nullptr, 0.0f);
        for (u32 t = 0; t < ms; ++t) step_both();
    };
    auto run_gap_s = [&](f32 s) { run_gap(static_cast<u32>(s * 1000.0f)); };

    auto probe_all = [&](std::vector<ProbeRow>& out) {
        std::vector<u32> order;
        order.reserve(usize(cfg.n_probe_each) * np);
        for (u32 rep = 0; rep < cfg.n_probe_each; ++rep)
            for (u32 i = 0; i < np; ++i) order.push_back(i);
        for (usize i = order.size(); i > 1; --i) {
            const usize j = static_cast<usize>(rng.next_u64() % i);
            std::swap(order[i - 1], order[j]);
        }
        for (u32 idx : order) {
            present(*probe_list[idx]);
            const auto r = mb.readout();
            const Decision d = decide(r, mbc, rng, false);
            out.push_back({probe_list[idx]->name, d.p_approach, d.valence, r.kc_active_frac});
            run_gap(cfg.t_iti_ms);
        }
    };

    auto kc_pattern_of = [&](const Odor& o) {
        present(o);
        std::vector<u8> p = mb.trial_pattern();
        run_gap(cfg.t_iti_ms);
        return p;
    };

    Timer timer;
    timer.start();

    // ================= phase 1: A+ / B- bandit training =================
    std::vector<ProbeRow> probe_train;
    probe_all(probe_train);

    u32 n_eff_reward = 0, n_eff_punish = 0;
    for (u32 trial = 0; trial < cfg.n_train_trials; ++trial) {
        const bool good = rng.bernoulli(0.5f);
        present(good ? odor_a : odor_b);
        const auto r = mb.readout();
        const Decision d = decide(r, mbc, rng, true);
        const f32 rew = d.approach ? (good ? (rng.bernoulli(0.85f) ? 1.0f : 0.0f)
                                           : (rng.bernoulli(0.85f) ? -1.0f : 0.0f))
                                   : 0.0f;
        n_eff_reward += (rew > 0.0f) ? 1u : 0u;
        n_eff_punish += (rew < 0.0f) ? 1u : 0u;
        run_gap(cfg.t_outcome_ms);
        if (rew != 0.0f) mb.apply_reinforcement(rew);
        run_gap(cfg.t_iti_ms);
    }

    std::vector<ProbeRow> probe_post_train;
    probe_all(probe_post_train);
    const OdorAgg tA = aggregate(probe_post_train, "A_rewarded");
    const OdorAgg tB = aggregate(probe_post_train, "B_punished");

    // ================= phase 2: retention across a 120 s gap =================
    run_gap_s(cfg.gap_s);
    std::vector<ProbeRow> probe_post_gap;
    probe_all(probe_post_gap);
    const OdorAgg gA = aggregate(probe_post_gap, "A_rewarded");
    const OdorAgg gB = aggregate(probe_post_gap, "B_punished");

    // ================= phase 3: interference — train E+ =================
    for (u32 trial = 0; trial < cfg.n_interfere_trials; ++trial) {
        present(odor_e);
        const auto r = mb.readout();
        const Decision d = decide(r, mbc, rng, true);
        run_gap(cfg.t_outcome_ms);
        if (d.approach) mb.apply_reinforcement(+1.0f);  // E always rewarded
        run_gap(cfg.t_iti_ms);
    }
    std::vector<ProbeRow> probe_post_interfere;
    probe_all(probe_post_interfere);
    const OdorAgg iA = aggregate(probe_post_interfere, "A_rewarded");
    const OdorAgg iB = aggregate(probe_post_interfere, "B_punished");
    const OdorAgg iE = aggregate(probe_post_interfere, "E_new_positive");
    const OdorAgg iN = aggregate(probe_post_interfere, "N_neutral");

    // ================= phase 4: delayed choice (STM) =================
    struct StmRow {
        u32 delay_ms = 0;
        f64 p_appr_trace_A = 0, p_appr_trace_B = 0;
        f64 p_appr_count_A = 0, p_appr_count_B = 0;
        f64 acc_trace = 0, acc_count = 0;
        f64 trace_valence_A = 0, trace_valence_B = 0, trace_active_A = 0;
        u32 n = 0;
    };
    std::vector<StmRow> stm;

    for (const u32 delay_ms : cfg.stm_delays_ms) {
        StmRow row;
        row.delay_ms = delay_ms;
        u32 nA = 0, apprA_trace = 0, apprA_count = 0;
        u32 nB = 0, apprB_trace = 0, apprB_count = 0;
        f64 tvA = 0, tvB = 0, taA = 0;

        for (u32 rep = 0; rep < cfg.n_stm_reps; ++rep) {
            for (u8 which = 0; which < 2; ++which) {
                const Odor& s = which ? odor_a : odor_b;
                present(s);                       // 1 s sample
                run_gap(delay_ms);                // odor OFF (clean air), D seconds
                // circuit-state decision (decaying trace)
                const auto tr = mb.trace_readout();
                MushroomBody::Readout rr;
                rr.valence = tr.valence;
                rr.kc_active_frac = tr.active_frac;
                const Decision d_trace = decide(rr, mbc, rng, false);
                // software-counter decision (contrast; NOT circuit memory)
                const auto rc = mb.readout();
                const Decision d_count = decide(rc, mbc, rng, false);

                const bool correct_trace = (d_trace.approach == (which == 1));
                const bool correct_count = (d_count.approach == (which == 1));
                if (which) {
                    ++nA;
                    apprA_trace += d_trace.approach ? 1u : 0u;
                    apprA_count += d_count.approach ? 1u : 0u;
                    tvA += tr.valence;
                    taA += tr.active_frac;
                } else {
                    ++nB;
                    apprB_trace += d_trace.approach ? 1u : 0u;
                    apprB_count += d_count.approach ? 1u : 0u;
                    tvB += tr.valence;
                }
                row.acc_trace += correct_trace ? 1.0 : 0.0;
                row.acc_count += correct_count ? 1.0 : 0.0;
                run_gap(cfg.t_iti_ms);
            }
        }
        row.n = nA;
        row.p_appr_trace_A = static_cast<f64>(apprA_trace) / nA;
        row.p_appr_count_A = static_cast<f64>(apprA_count) / nA;
        row.p_appr_trace_B = static_cast<f64>(apprB_trace) / nB;
        row.p_appr_count_B = static_cast<f64>(apprB_count) / nB;
        row.trace_valence_A = tvA / nA;
        row.trace_valence_B = tvB / nB;
        row.trace_active_A = taA / nA;
        row.acc_trace /= static_cast<f64>(nA + nB);
        row.acc_count /= static_cast<f64>(nA + nB);
        stm.push_back(row);
    }

    const f64 wall_s = timer.seconds();

    // ================= KC pattern analysis =================
    const auto pat_a = kc_pattern_of(odor_a);
    const auto pat_e = kc_pattern_of(odor_e);
    const auto pat_b = kc_pattern_of(odor_b);
    const f64 jac_ae = jaccard_u8(pat_a, pat_e);
    const f64 jac_ab = jaccard_u8(pat_a, pat_b);
    const f64 ov_ae = profile_overlap(odor_a, odor_e);
    const f64 ov_ab = profile_overlap(odor_a, odor_b);

    // ================= perf / hash =================
    const u32 n_neurons = al.n_glom() + al.n_ln() + mbc.n_kc;
    const f64 sim_s = static_cast<f64>(sim_ms) * 0.001;
    u64 run_hash = 1469598103934665603ull;
    for (const ProbeRow& p : probe_post_interfere) hash_mix_f32(run_hash, p.p_approach);
    for (const StmRow& s : stm) hash_mix_f32(run_hash, static_cast<f32>(s.acc_trace));
    for (f32 w : mb.w_appr()) hash_mix_f32(run_hash, w);
    for (f32 w : mb.w_avoid()) hash_mix_f32(run_hash, w);
    run_hash = run_hash * 1099511628211ull ^ 0xff51afd7ed558ccdull;

    // ================= outputs: CSV =================
    ensure_parent_dir(cfg.out_csv);
    {
        std::ofstream f(cfg.out_csv);
        f << "delay_ms,n,p_appr_trace_A,p_appr_trace_B,acc_trace,"
             "p_appr_count_A,p_appr_count_B,acc_count,trace_valence_A,trace_valence_B,trace_active_A\n";
        char buf[256];
        for (const StmRow& s : stm) {
            std::snprintf(buf, sizeof buf,
                          "%u,%u,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                          s.delay_ms, s.n, s.p_appr_trace_A, s.p_appr_trace_B,
                          s.acc_trace, s.p_appr_count_A, s.p_appr_count_B,
                          s.acc_count, s.trace_valence_A, s.trace_valence_B,
                          s.trace_active_A);
            f << buf;
        }
    }

    // ================= outputs: JSON =================
    ensure_parent_dir(cfg.out_json);
    {
        std::ofstream f(cfg.out_json);
        JsonW j(f);
        j.kv("experiment", std::string("goal2_memory"));
        j.kv("version", std::string("0.1.0"));
        j.kv("seed", cfg.seed);

        j.k("config");
        j.obj();
        j.kv("seed", cfg.seed);
        j.kv("n_train_trials", cfg.n_train_trials);
        j.kv("n_probe_each", cfg.n_probe_each);
        j.kv("n_interfere_trials", cfg.n_interfere_trials);
        j.kv("gap_s", cfg.gap_s);
        j.kv("n_stm_reps", cfg.n_stm_reps);
        j.kv("sample_ms", cfg.sample_ms);
        j.kv("tau_elig_s", mbc.tau_elig);
        j.end_obj();

        j.k("structure");
        j.obj();
        j.kv("profile_overlap_A_E", ov_ae);
        j.kv("profile_overlap_A_B", ov_ab);
        j.kv("kc_jaccard_A_E", jac_ae);
        j.kv("kc_jaccard_A_B", jac_ab);
        j.end_obj();

        j.k("retention");
        j.obj();
        j.kv("gap_s", cfg.gap_s);
        j.kv("pA_before", tA.p);
        j.kv("pA_after", gA.p);
        j.kv("delta_pA", gA.p - tA.p);
        j.kv("pB_before", tB.p);
        j.kv("pB_after", gB.p);
        j.kv("delta_pB", gB.p - tB.p);
        j.end_obj();

        j.k("interference");
        j.obj();
        j.kv("pA_after", iA.p);
        j.kv("pB_after", iB.p);
        j.kv("pE_after", iE.p);
        j.kv("pN_after", iN.p);
        j.end_obj();

        j.k("stm");
        j.arr();
        for (const StmRow& s : stm) {
            j.obj();
            j.kv("delay_ms", s.delay_ms);
            j.kv("n", s.n);
            j.kv("p_appr_trace_A", s.p_appr_trace_A);
            j.kv("p_appr_trace_B", s.p_appr_trace_B);
            j.kv("acc_trace", s.acc_trace);
            j.kv("p_appr_count_A", s.p_appr_count_A);
            j.kv("p_appr_count_B", s.p_appr_count_B);
            j.kv("acc_count", s.acc_count);
            j.kv("trace_valence_A", s.trace_valence_A);
            j.kv("trace_valence_B", s.trace_valence_B);
            j.kv("trace_active_frac_A", s.trace_active_A);
            j.end_obj();
        }
        j.end_arr();

        j.k("perf");
        j.obj();
        j.kv("wall_s", wall_s);
        j.kv("sim_s", sim_s);
        j.kv("sim_ms", sim_ms);
        j.kv("rt_factor", sim_s / wall_s);
        j.kv("neurons", n_neurons);
        j.end_obj();

        j.k("run_hash");
        char hb[32];
        std::snprintf(hb, sizeof hb, "%016llx", static_cast<unsigned long long>(run_hash));
        j.val(std::string(hb));
        j.end_obj();
    }

    // ================= stdout summary =================
    std::printf("==== Goal 2: memory (retention / interference / delayed choice) ====\n");
    std::printf("training: pA=%.3f pB=%.3f (outcomes +/-%d/%d)\n",
                tA.p, tB.p, n_eff_reward, n_eff_punish);
    std::printf("retention after %.0fs gap: A %.3f -> %.3f (d=%+.3f) | B %.3f -> %.3f (d=%+.3f)\n",
                cfg.gap_s, tA.p, gA.p, gA.p - tA.p, tB.p, gB.p, gB.p - tB.p);
    std::printf("interference (E+ trained, overlap A-E: profile %.2f / KC jac %.3f): "
                "A=%.3f B=%.3f E=%.3f N=%.3f\n",
                ov_ae, jac_ae, iA.p, iB.p, iE.p, iN.p);
    std::printf("STM (trace-based choice after odor off):\n");
    for (const StmRow& s : stm)
        std::printf("  D=%4dms | pA=%.2f pB=%.2f acc=%.2f | counter-acc=%.2f | traceV(A)=%+.3f(B)=%+.3f act=%.1f%%\n",
                    s.delay_ms, s.p_appr_trace_A, s.p_appr_trace_B, s.acc_trace,
                    s.acc_count, s.trace_valence_A, s.trace_valence_B,
                    100.0 * s.trace_active_A);
    std::printf("perf: sim=%.0fs wall=%.1fs rt=%.0fx | run_hash=%016llx\n",
                sim_s, wall_s, sim_s / wall_s,
                static_cast<unsigned long long>(run_hash));
    std::printf("results: %s | %s\n", cfg.out_json.c_str(), cfg.out_csv.c_str());
    return 0;
}

}  // namespace malefly
