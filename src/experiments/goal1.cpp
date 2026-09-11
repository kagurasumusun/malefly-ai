// ============================================================================
// experiments/goal1.cpp — Goal 1 experiment driver.
//
// Protocol:
//   1. Naive probes (outcome-free): A, B, C=A+B mixture, D=partial-A, N=neutral
//   2. Training: bandit trials (odor 50/50, stochastic outcomes 85%).
//      Probes (no outcomes, no plasticity) at trial 0 / N/2 / N.
//   3. Analysis: learning curve by block, criterion block, KC pattern overlap
//      matrix, valence-transfer vs overlap correlation, neuro stats, perf.
//
// Mechanism stack (each ablatable, measured in docs/RESULTS.md):
//   LH innate valence (Brain/MINDS region plug-in, --no-lh),
//   KC taxonomy (Allen, mbc.use_taxonomy), RPE gate (BRAIN, mbc.rpe_gating),
//   LN subclasses (Blue Brain, AntennalLobeConfig.ln_subclasses).
// ============================================================================
#include "experiments/goal1.hpp"

#include "circuits/brain.hpp"
#include "circuits/action_selection.hpp"
#include "circuits/antennal_lobe.hpp"
#include "circuits/lateral_horn.hpp"
#include "circuits/modulator.hpp"
#include "circuits/mushroom_body.hpp"
#include "core/rng.hpp"
#include "env/environment.hpp"
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
void hash_mix(u64& h, const void* data, usize n) { h = h * 1099511628211ull ^ fnv1a(data, n); }
void hash_mix_f32(u64& h, f32 v) { hash_mix(h, &v, sizeof(v)); }

f64 vec_mean(const std::vector<f64>& v) {
    if (v.empty()) return 0.0;
    f64 s = 0;
    for (f64 x : v) s += x;
    return s / static_cast<f64>(v.size());
}
f64 vec_sd(const std::vector<f64>& v, f64 mean_v) {
    if (v.size() < 2) return 0.0;
    f64 s = 0;
    for (f64 x : v) s += (x - mean_v) * (x - mean_v);
    return std::sqrt(s / static_cast<f64>(v.size() - 1));
}
f64 pearson(const std::vector<f64>& x, const std::vector<f64>& y) {
    const usize n = std::min(x.size(), y.size());
    if (n < 3) return 0.0;
    f64 sx = 0, sy = 0;
    for (usize i = 0; i < n; ++i) { sx += x[i]; sy += y[i]; }
    const f64 mx = sx / n, my = sy / n;
    f64 sxy = 0, sxx = 0, syy = 0;
    for (usize i = 0; i < n; ++i) {
        const f64 dx = x[i] - mx, dy = y[i] - my;
        sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
    }
    if (sxx <= 1e-12 || syy <= 1e-12) return 0.0;
    return sxy / std::sqrt(sxx * syy);
}
void ensure_parent_dir(const std::string& path) {
    const usize pos = path.find_last_of('/');
    if (pos == std::string::npos) return;
    ::mkdir(path.substr(0, pos).c_str(), 0755);
}
f64 jaccard_u8(const std::vector<u8>& a, const std::vector<u8>& b) {
    u32 inter = 0, uni = 0;
    const usize n = std::min(a.size(), b.size());
    for (usize i = 0; i < n; ++i) {
        inter += (a[i] && b[i]) ? 1 : 0;
        uni += (a[i] || b[i]) ? 1 : 0;
    }
    return uni ? static_cast<f64>(inter) / static_cast<f64>(uni) : 0.0;
}

struct ProbeRow {
    std::string odor;
    f32 p_approach = 0, valence = 0, kc_active = 0, innate = 0;
};
struct TrialRow {
    u32 trial = 0;
    u8 is_good = 0, action = 0, correct = 0;
    f32 reward = 0, p_appr = 0, valence = 0, kc_active = 0;
};
struct OdorAgg {
    f64 p = 0, p_sd = 0, v = 0, kc = 0, innate = 0;
    u32 n = 0;
};
OdorAgg aggregate(const std::vector<ProbeRow>& rows, const std::string& name) {
    std::vector<f64> ps, vs, ks, is;
    for (const ProbeRow& r : rows)
        if (r.odor == name) {
            ps.push_back(r.p_approach);
            vs.push_back(r.valence);
            ks.push_back(r.kc_active);
            is.push_back(r.innate);
        }
    OdorAgg a;
    a.n = static_cast<u32>(ps.size());
    if (!ps.empty()) {
        a.p = vec_mean(ps);
        a.v = vec_mean(vs);
        a.kc = vec_mean(ks);
        a.innate = vec_mean(is);
        a.p_sd = vec_sd(ps, a.p);
    }
    return a;
}

}  // namespace

int run_goal1(const Goal1Config& cfg) {
    Odor odor_a = make_odor(cfg.n_glom, 15, cfg.seed + 101, 0.60f, 1.00f);
    odor_a.name = "A_rewarded";
    Odor odor_b = make_odor(cfg.n_glom, 15, cfg.seed + 202, 0.60f, 1.00f);
    odor_b.name = "B_punished";
    Odor odor_c = mix_odors(odor_a, odor_b, 0.55f, 0.55f);
    odor_c.name = "C_mixture";
    Odor odor_d = make_variant(odor_a, 0.75f, 6, cfg.n_glom, cfg.seed + 303);
    odor_d.name = "D_partialA";
    Odor odor_n = uniform_odor(cfg.n_glom, 0.12f);
    odor_n.name = "N_neutral";
    const std::vector<const Odor*> probe_list = {&odor_a, &odor_b, &odor_c, &odor_d, &odor_n};
    const usize np = probe_list.size();

    Rng rng(cfg.seed);
    BrainConfig bcfg;
    bcfg.mod.modes_enabled = cfg.use_modes;
    if (!cfg.use_lh) bcfg.lh.beta = 0.0f;  // ablation: silence innate path
    Brain brain(bcfg, rng);
    const MushroomBodyConfig& mbc = bcfg.mb;
    const MushroomBody& mb = brain.mb();
    // optional development: wiring emerges from spontaneous-activity pruning
    if (cfg.develop) const_cast<MushroomBody&>(mb).develop(rng, mbc.dev_ms);
    BanditConfig bc;
    BanditEnv env(odor_a, odor_b, bc, rng);

    // ---- world-side helpers (the brain is NOT touched except sensors) ----
    auto run_brain = [&](u32 ms) {
        for (u32 t = 0; t < ms; ++t) brain.step(DT);
    };
    auto present_odor = [&](const Odor& o) { brain.set_odor(&o, cfg.concentration); };
    auto clear_odor = [&]() { brain.set_odor(nullptr, 0.0f); };

    // Behavior = which side WINS the neural races during the episode
    // (a stream of decisions the brain makes on its own).
    u64 ch_before = 0, ca_before = 0;
    i64 win_appr = 0, win_avoid = 0;
    auto sample_motor = [&]() {
        const auto o = brain.out();
        win_appr = static_cast<i64>(o.choice_approach - ch_before);
        win_avoid = static_cast<i64>(o.choice_avoid - ca_before);
        ch_before = o.choice_approach;
        ca_before = o.choice_avoid;
    };

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
            const Odor& o = *probe_list[idx];
            // odor ON for t_on; the brain alone turns spikes into behavior
            present_odor(o);
            run_brain(cfg.t_on_ms);
            sample_motor();
            const f32 inn = brain.lh().innate_valence();
            const bool approach = win_appr >= win_avoid;
            const auto rr = mb.readout();
            out.push_back({o.name, approach ? 1.0f : 0.0f,
                           static_cast<f32>(win_appr - win_avoid),
                           rr.kc_active_frac, inn});
            clear_odor();
            run_brain(cfg.t_iti_ms);
        }
    };

    Timer timer;
    timer.start();

    std::vector<ProbeRow> probe_pre, probe_mid, probe_post;
    probe_all(probe_pre);

    std::vector<TrialRow> trials;
    trials.reserve(cfg.n_train_trials);
    u32 n_eff_reward = 0, n_eff_punish = 0;

    for (u32 trial = 0; trial < cfg.n_train_trials; ++trial) {
        env.begin_trial();
        present_odor(env.odor());
        run_brain(cfg.t_on_ms);
        sample_motor();
        const bool approach = win_appr >= win_avoid;
        const f32 rew = env.apply(approach ? Action::Approach : Action::Avoid);
        const bool correct = (approach == env.current_is_good());
        const auto rr = mb.readout();
        trials.push_back({trial + 1,
                          static_cast<u8>(env.current_is_good()),
                          static_cast<u8>(approach),
                          static_cast<u8>(correct),
                          rew,
                          static_cast<f32>(win_appr),
                          static_cast<f32>(win_appr - win_avoid),
                          rr.kc_active_frac});
        n_eff_reward += (rew > 0.0f) ? 1u : 0u;
        n_eff_punish += (rew < 0.0f) ? 1u : 0u;
        // the world only touches SENSORS; VUM/DAN neurons deliver it
        clear_odor();
        run_brain(cfg.t_outcome_ms);
        brain.set_sensors(rew > 0.0f ? 1.0f : 0.0f, rew < 0.0f ? 1.0f : 0.0f);
        run_brain(cfg.t_outcome_ms);
        brain.set_sensors(0.0f, 0.0f);
        run_brain(cfg.t_iti_ms);
        if (trial + 1 == cfg.n_train_trials / 2) probe_all(probe_mid);
    }
    probe_all(probe_post);
    const f64 wall_s = timer.seconds();

    // KC pattern analysis
    std::vector<std::vector<u8>> pats(np);
    for (u32 i = 0; i < np; ++i) {
        present_odor(*probe_list[i]);
        run_brain(cfg.t_on_ms);
        pats[i] = mb.trial_pattern();
        clear_odor();
        run_brain(cfg.t_iti_ms);
    }
    std::vector<f64> mean_appr(np, 0.0), mean_avoid(np, 0.0);
    {
        const std::vector<f32>& wa = mb.w_appr();
        const std::vector<f32>& wv = mb.w_avoid();
        for (u32 i = 0; i < np; ++i) {
            u32 n = 0;
            f64 sa = 0, sv = 0;
            for (u32 k = 0; k < mbc.n_kc; ++k)
                if (pats[i][k]) { sa += wa[k]; sv += wv[k]; ++n; }
            if (n) { mean_appr[i] = sa / n; mean_avoid[i] = sv / n; }
        }
    }
    std::vector<f64> jac_a(np, 0.0), jac_b(np, 0.0), xs, ys;
    for (u32 i = 0; i < np; ++i) {
        jac_a[i] = jaccard_u8(pats[0], pats[i]);
        jac_b[i] = jaccard_u8(pats[1], pats[i]);
    }
    for (u32 i = 0; i < np; ++i) {
        const OdorAgg a = aggregate(probe_post, probe_list[i]->name);
        if (a.n == 0) continue;
        xs.push_back(jac_a[i] - jac_b[i]);
        ys.push_back(a.p);
    }
    const f64 transfer_r = pearson(xs, ys);

    const u32 n_neurons = bcfg.al.n_glom + bcfg.al.n_ln_fast + bcfg.al.n_ln_slow +
                          mbc.n_kc + bcfg.lh.n_cells + 2 /*VUM,DAN*/ + 32 /*integ*/;
    const u64 sim_ms2 = static_cast<u64>(cfg.n_train_trials + 2 * 5 * cfg.n_probe_each) *
                        (cfg.t_on_ms + cfg.t_outcome_ms * 2 + cfg.t_iti_ms);
    const u64 neuron_steps = sim_ms2 * n_neurons;
    const u64 syn_ops = mb.pn2kc().ops + brain.al().pool_ops() + brain.lh().ops();
    const u64 mem_bytes = brain.al().memory_bytes() + mb.memory_bytes() +
                          brain.lh().memory_bytes();
    const f64 sim_s = static_cast<f64>(sim_ms2) * 0.001;

    u64 run_hash = 1469598103934665603ull;
    for (const TrialRow& t : trials) {
        hash_mix(run_hash, &t.trial, sizeof(u32));
        hash_mix_f32(run_hash, t.reward);
        hash_mix_f32(run_hash, t.p_appr);
    }
    for (const ProbeRow& p : probe_post) hash_mix_f32(run_hash, p.p_approach);
    for (f32 w : mb.w_appr()) hash_mix_f32(run_hash, w);
    for (f32 w : mb.w_avoid()) hash_mix_f32(run_hash, w);
    run_hash = run_hash * 1099511628211ull ^ 0xff51afd7ed558ccdull;

    ensure_parent_dir(cfg.out_csv);
    {
        std::ofstream f(cfg.out_csv);
        f << "trial,is_good,action_approach,reward,p_approach,valence,kc_active_frac,correct\n";
        char buf[256];
        for (const TrialRow& t : trials) {
            std::snprintf(buf, sizeof buf, "%u,%d,%d,%.1f,%.6f,%.6f,%.6f,%d\n",
                          t.trial, t.is_good, t.action, t.reward, t.p_appr,
                          t.valence, t.kc_active, t.correct);
            f << buf;
        }
    }

    ensure_parent_dir(cfg.out_json);
    {
        std::ofstream f(cfg.out_json);
        JsonW j(f);
        j.kv("experiment", std::string("goal1_value_learning"));
        j.kv("version", std::string("0.2.0"));
        j.kv("seed", cfg.seed);

        j.k("config");
        j.obj();
        j.kv("seed", cfg.seed);
        j.kv("n_train_trials", cfg.n_train_trials);
        j.kv("n_probe_each", cfg.n_probe_each);
        j.kv("n_glom", cfg.n_glom);
        j.kv("t_on_ms", cfg.t_on_ms);
        j.kv("t_outcome_ms", cfg.t_outcome_ms);
        j.kv("t_iti_ms", cfg.t_iti_ms);
        j.kv("p_reward_good", bc.p_reward_good);
        j.kv("p_punish_bad", bc.p_punish_bad);
        j.kv("n_kc", mbc.n_kc);
        j.kv("kc_fanin", mbc.kc_fanin);
        j.kv("kc_quanta", mbc.kc_quanta);
        j.kv("kc_v_thresh", mbc.kc_v_thresh);
        j.kv("apl_gain", mbc.apl_gain);
        j.kv("eta_reward", mbc.eta_reward);
        j.kv("eta_punish", mbc.eta_punish);
        j.kv("decision_temp", mbc.decision_temp);
        j.kv("epsilon", mbc.epsilon);
        j.kv("use_lh", cfg.use_lh);
        j.kv("neural_arbiter", cfg.neural_arbiter);
        j.kv("use_modes", cfg.use_modes);
        j.kv("develop", cfg.develop);
        j.kv("use_taxonomy", mbc.use_taxonomy);
        j.kv("rpe_gating", mbc.rpe_gating);
        j.kv("ln_subclasses", bcfg.al.ln_subclasses);
        j.end_obj();

        j.k("mechanisms");
        j.obj();
        const auto sub = mb.subtype_counts();
        j.k("kc_subtypes");
        j.arr();
        for (int s = 0; s < KC_N_SUBTYPES; ++s) {
            j.obj();
            j.kv("name", std::string(KC_SUBTYPES[s].name));
            j.kv("n", sub[static_cast<usize>(s)]);
            j.kv("tau_elig", KC_SUBTYPES[s].tau_elig);
            j.end_obj();
        }
        j.end_arr();
        j.kv("lh_ops", brain.lh().ops());
        j.kv("last_rpe_gate", mb.last_rpe_gate());
        j.kv("mean_active_fanin", mb.mean_active_fanin());
        j.kv("sd_active_fanin", mb.sd_active_fanin());
        j.kv("dev_pruned", mb.dev_pruned());
        j.end_obj();

        j.k("odor_profile_overlap");
        j.obj();
        for (u32 i = 0; i < np; ++i) {
            j.k(probe_list[i]->name);
            j.obj();
            for (u32 k = 0; k < np; ++k)
                j.kv(probe_list[k]->name, profile_overlap(*probe_list[i], *probe_list[k]));
            j.end_obj();
        }
        j.end_obj();

        j.k("kc_pattern_jaccard");
        j.obj();
        for (u32 i = 0; i < np; ++i) {
            j.k(probe_list[i]->name);
            j.obj();
            for (u32 k = 0; k < np; ++k)
                j.kv(probe_list[k]->name, jaccard_u8(pats[i], pats[k]));
            j.end_obj();
        }
        j.end_obj();

        j.k("probes");
        j.obj();
        auto write_round = [&](const char* tag, const std::vector<ProbeRow>& rows) {
            j.k(tag);
            j.arr();
            for (const Odor* o : probe_list) {
                const OdorAgg a = aggregate(rows, o->name);
                j.obj();
                j.kv("odor", o->name);
                j.kv("n", a.n);
                j.kv("p_approach_mean", a.p);
                j.kv("p_approach_sd", a.p_sd);
                j.kv("valence_mean", a.v);
                j.kv("kc_active_frac_mean", a.kc);
                j.kv("lh_innate_mean", a.innate);
                j.end_obj();
            }
            j.end_arr();
        };
        write_round("pre", probe_pre);
        write_round("mid", probe_mid);
        write_round("post", probe_post);
        j.end_obj();

        j.k("training");
        j.obj();
        const u32 block_n = 20;
        j.k("blocks");
        j.arr();
        i64 criterion_block = -1;
        for (u32 b0 = 0; b0 < trials.size(); b0 += block_n) {
            const u32 b1 = static_cast<u32>(std::min<usize>(b0 + block_n, trials.size()));
            u32 n = 0, nc = 0, nag = 0, nab = 0;
            for (u32 t = b0; t < b1; ++t) {
                ++n;
                nc += trials[t].correct ? 1u : 0u;
                if (trials[t].is_good) nag += trials[t].action ? 1u : 0u;
                else nab += trials[t].action ? 1u : 0u;
            }
            const f64 frac_correct = static_cast<f64>(nc) / n;
            if (criterion_block < 0 && frac_correct >= 0.80)
                criterion_block = static_cast<i64>(b0 / block_n);
            j.obj();
            j.kv("block", b0 / block_n);
            j.kv("n", n);
            j.kv("frac_correct", frac_correct);
            j.kv("frac_approach_good", static_cast<f64>(nag) / n);
            j.kv("frac_approach_bad", static_cast<f64>(nab) / n);
            j.end_obj();
        }
        j.end_arr();
        j.kv("criterion_block_frac80", criterion_block);
        j.kv("n_effective_rewards", n_eff_reward);
        j.kv("n_effective_punishments", n_eff_punish);
        j.end_obj();

        j.k("transfer");
        j.obj();
        j.kv("pearson_r_p_post_vs_jacA_minus_jacB", transfer_r);
        j.k("points");
        j.arr();
        for (u32 i = 0; i < np; ++i) {
            const OdorAgg a = aggregate(probe_post, probe_list[i]->name);
            j.obj();
            j.kv("odor", probe_list[i]->name);
            j.kv("jaccard_to_A", jac_a[i]);
            j.kv("jaccard_to_B", jac_b[i]);
            j.kv("p_approach_post", a.p);
            j.end_obj();
        }
        j.end_arr();
        j.end_obj();

        j.k("learned_weights");
        j.obj();
        for (u32 i = 0; i < np; ++i) {
            j.k(probe_list[i]->name);
            j.obj();
            j.kv("mean_w_approach", mean_appr[i]);
            j.kv("mean_w_avoid", mean_avoid[i]);
            j.end_obj();
        }
        j.end_obj();

        j.k("neuro_stats");
        j.obj();
        j.kv("pn_hz_during_odor", -1.0);  // ( Brain 内部で計測は次イテレーション)
        j.kv("kc_spikes_per_presentation", 0.0);  // introspection next iter
        j.end_obj();

        j.k("perf");
        j.obj();
        j.kv("wall_s", wall_s);
        j.kv("sim_s", sim_s);
        j.kv("rt_factor", sim_s / wall_s);
        j.kv("neurons", n_neurons);
        j.kv("sim_ms", 0);
        j.kv("neuron_steps", neuron_steps);
        j.kv("syn_ops", syn_ops);
        j.kv("lh_ops", brain.lh().ops());
        j.kv("vum_spikes", brain.vum_spikes());
        j.kv("dan_spikes", brain.dan_spikes());
        j.kv("brain_decisions", brain.out().decisions);
        j.kv("neurons_per_s", static_cast<f64>(neuron_steps) / wall_s);
        j.kv("synops_per_s", static_cast<f64>(syn_ops) / wall_s);
        j.kv("mem_bytes", mem_bytes);
        j.kv("bytes_per_neuron", static_cast<f64>(mem_bytes) / n_neurons);
        j.kv("bytes_per_syn_pn2kc",
             static_cast<f64>(mb.pn2kc().memory_bytes()) /
                 static_cast<f64>(mb.pn2kc().num()));
        j.end_obj();

        j.k("run_hash");
        char hb[32];
        std::snprintf(hb, sizeof hb, "%016llx", static_cast<unsigned long long>(run_hash));
        j.val(std::string(hb));
        j.end_obj();
    }

    const OdorAgg preA = aggregate(probe_pre, "A_rewarded");
    const OdorAgg preB = aggregate(probe_pre, "B_punished");
    const OdorAgg postA = aggregate(probe_post, "A_rewarded");
    const OdorAgg postB = aggregate(probe_post, "B_punished");
    const OdorAgg postC = aggregate(probe_post, "C_mixture");
    const OdorAgg postD = aggregate(probe_post, "D_partialA");
    const OdorAgg postN = aggregate(probe_post, "N_neutral");
    const OdorAgg preN = aggregate(probe_pre, "N_neutral");

    std::printf("==== Goal 1: autonomous value learning (MaleCNS MB circuit) ====\n");
    std::printf("seed=%llu trials=%u outcomes(+/-)=%u/%u | LH=%d tax=%d rpe=%d modes=%d dev=%d fanin=%.1f+-%.1f pruned=%llu | brain decisions=%llu VUM=%llu DAN=%llu\n",
                static_cast<unsigned long long>(cfg.seed), cfg.n_train_trials,
                n_eff_reward, n_eff_punish, cfg.use_lh, mbc.use_taxonomy,
                mbc.rpe_gating, cfg.use_modes, cfg.develop,
                mb.mean_active_fanin(), mb.sd_active_fanin(),
                static_cast<unsigned long long>(mb.dev_pruned()),
                static_cast<unsigned long long>(brain.out().decisions),
                static_cast<unsigned long long>(brain.vum_spikes()),
                static_cast<unsigned long long>(brain.dan_spikes()));
    std::printf("LH innate valence (naive): A=%.3f B=%.3f N=%.3f\n",
                preA.innate, preB.innate, preN.innate);
    std::printf("KC active frac: A=%.1f%% B=%.1f%% N=%.1f%% | PN Hz(odor)=%.1f | KC spikes/pres=%.2f\n",
                100.0 * postA.kc, 100.0 * postB.kc, 100.0 * postN.kc, 0.0, 0.0);
    std::printf("p_approach  naive -> trained:  A: %.3f -> %.3f | B: %.3f -> %.3f\n",
                preA.p, postA.p, preB.p, postB.p);
    std::printf("generalization: C(mix): %.3f | D(partialA): %.3f | N(neutral): %.3f (naive %.3f)\n",
                postC.p, postD.p, postN.p, preN.p);
    std::printf("transfer corr r(p_post, jacA-jacB) = %.3f\n", transfer_r);
    std::printf("blocks(frac_correct): ");
    for (u32 b0 = 0; b0 < trials.size(); b0 += 20) {
        u32 nc = 0, n = 0;
        for (u32 t = b0; t < std::min<usize>(b0 + 20, trials.size()); ++t) {
            nc += trials[t].correct ? 1u : 0u;
            ++n;
        }
        std::printf("%.2f ", static_cast<f64>(nc) / n);
    }
    std::printf("\n");
    std::printf("perf: sim=%.1fs wall=%.2fs rt=%.1fx | %.1fM neuron-steps/s | %.1fM synops/s | %.0f KB (%.0f B/neuron)\n",
                sim_s, wall_s, sim_s / wall_s,
                static_cast<f64>(neuron_steps) / wall_s / 1e6,
                static_cast<f64>(syn_ops) / wall_s / 1e6,
                static_cast<f64>(mem_bytes) / 1024.0,
                static_cast<f64>(mem_bytes) / n_neurons);
    char hb[32];
    std::snprintf(hb, sizeof hb, "%016llx", static_cast<unsigned long long>(run_hash));
    std::printf("run_hash=%s\n", hb);
    std::printf("results: %s | %s\n", cfg.out_json.c_str(), cfg.out_csv.c_str());
    return 0;
}

}  // namespace malefly
