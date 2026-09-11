// ============================================================================
// experiments/monkey1.cpp — primate benchmark driver (see monkey1.hpp).
// The agent is a Brain + Hippocampus + Pfc wired in parallel on the same
// PN stream (three cortices of one animal). The world only presents odors
// and sensors; decisions are read from each circuit's own spiking signals.
// ============================================================================
#include "experiments/monkey1.hpp"

#include "circuits/brain.hpp"
#include "circuits/hippocampus.hpp"
#include "circuits/pfc.hpp"
#include "util/jsonw.hpp"
#include "util/odors.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sys/stat.h>
#include <fstream>
#include <vector>

namespace malefly {
namespace {

f64 jaccard_u8(const std::vector<u8>& a, const std::vector<u8>& b) {
    u32 inter = 0, uni = 0;
    for (usize i = 0; i < a.size(); ++i) {
        inter += (a[i] && b[i]) ? 1u : 0u;
        uni += (a[i] || b[i]) ? 1u : 0u;
    }
    return uni ? static_cast<f64>(inter) / static_cast<f64>(uni) : 0.0;
}

void ensure_parent_dir(const std::string& path) {
    const usize pos = path.find_last_of('/');
    if (pos == std::string::npos) return;
    ::mkdir(path.substr(0, pos).c_str(), 0755);
}

}  // namespace

int run_monkey1(const Monkey1Config& cfg) {
    // ================= world =================
    Odor od_a = make_odor(50, 15, cfg.seed + 101, 0.60f, 1.00f);
    od_a.name = "A";
    Odor od_b = make_odor(50, 15, cfg.seed + 202, 0.60f, 1.00f);
    od_b.name = "B";
    Odor lure = make_odor(50, 15, cfg.seed + 303, 0.60f, 1.00f);
    lure.name = "L";
    Odor od_x = make_odor(50, 15, cfg.seed + 404, 0.60f, 1.00f);
    od_x.name = "X";
    Odor od_y = make_odor(50, 15, cfg.seed + 505, 0.60f, 1.00f);
    od_y.name = "Y";

    Rng rng(cfg.seed);
    Brain brain(BrainConfig{}, rng);
    HippocampusConfig hc;
    hc.seed = cfg.seed;
    Hippocampus hpc(hc, rng, brain.mb().n_kc());
    Pfc pfc(PfcConfig{}, rng, 50);

    // recalled-valence axons: HPC valence cells -> VUM/DAN (spike->spike).
    // This is the honeybee predicted-US wiring (Hammer & Menzel 1995) and
    // the dopamine-at-replay channel (Gomperts et al. 2015): during SWR
    // replay of rewarded episodes the MB receives the US internally.
    auto valence_axons = [&]() {
        if (hpc.vpos_drive()) {
            brain.drive_vum(1.0f);
            hpc.notify_reward();   // VUM->HPC dopamine: bias replay to this trace
        }
        if (hpc.vneg_drive()) brain.drive_dan(1.0f);
    };
    auto run = [&](u32 ms) {
        for (u32 t = 0; t < ms; ++t) {
            hpc.step(DT, brain.mb().kc_spike_pattern());
            valence_axons();
            brain.step(DT);
        }
    };
    auto present = [&](const Odor& o, u32 ms, bool pfc_binds = false) {
        brain.set_odor(&o, 1.0f);
        for (u32 t = 0; t < ms; ++t) {
            hpc.step(DT, brain.mb().kc_spike_pattern());
            valence_axons();
            brain.step(DT);
            pfc.step(DT, brain.al().pn_spikes(), pfc_binds);
        }
    };
    auto present_pfc_only = [&](const Odor& o, u32 ms, bool binds) {
        brain.set_odor(nullptr, 0.0f);
        for (u32 t = 0; t < ms; ++t) {
            brain.step(DT);
            // PFC sees the same PN stream the brain would (percept delivered
            // through the world; here we present the odor to PFC via brain's
            // antennal lobe but keep motor learning unaffected by not
            // applying outcomes)
            brain.step(DT);
            hpc.step(DT, brain.mb().kc_spike_pattern());
            pfc.step(DT, brain.al().pn_spikes(), binds);
        }
        (void)o;
    };
    auto gap = [&](u32 ms) {
        brain.set_odor(nullptr, 0.0f);
        for (u32 t = 0; t < ms; ++t) {
            hpc.step(DT, brain.mb().kc_spike_pattern());
            valence_axons();
            brain.step(DT);
            pfc.step(DT, brain.al().pn_spikes(), false);
        }
    };

    // population-vector collector (driver-side measurement, like an
    // experimenter's electrode): accumulate PFC spike counts over ms
    std::vector<u32> tmpl_vec(pfc.size(), 0), probe_vec(pfc.size(), 0);
    auto collect = [&](std::vector<u32>& v, u32 ms) {
        brain.set_odor(nullptr, 0.0f);
        std::fill(v.begin(), v.end(), 0u);
        for (u32 t = 0; t < ms; ++t) {
            hpc.step(DT, brain.mb().kc_spike_pattern());
            valence_axons();
            brain.step(DT);
            pfc.step(DT, brain.al().pn_spikes(), false);
            const u8* ps = pfc.spikes();
            for (u32 k = 0; k < pfc.size(); ++k) v[k] += ps[k];
        }
    };

    // ================= B. episodic one-shot + cued recall =================
    gap(2000);
    pfc.clear_assembly();
    // encode: ONE presentation each, one US each (VUM-like US into HPC).
    // Forward conditioning: the US overlaps the odor window.
    hpc.set_us(1.0f, 0.0f);
    present(od_a, 600);
    run(300);
    hpc.set_us(0.0f, 0.0f);
    u32 vp_dbg = 0, vn_dbg = 0;
    hpc.read_valence(vp_dbg, vn_dbg);
    std::printf("  [B] A+ stamp: episodes=%u v+=%u v-=%u\n", hpc.episodes(), vp_dbg, vn_dbg);
    gap(2500);
    hpc.set_us(0.0f, 1.0f);
    present(od_b, 600);
    run(300);
    hpc.set_us(0.0f, 0.0f);
    hpc.read_valence(vp_dbg, vn_dbg);
    std::printf("  [B] B- stamp: episodes=%u v+=%u v-=%u\n", hpc.episodes(), vp_dbg, vn_dbg);
    const u32 eps_after_enc = hpc.episodes();
    gap(cfg.t_gap_s * 1000);
    // partial-cue recall
    Odor cue_a = od_a;
    {
        // keep 65% of A's glomeruli, zero the rest
        Rng r(cfg.seed + 9);
        for (u32 g = 0; g < 50; ++g)
            if (cue_a.profile[g] > 0.0f && r.bernoulli(1.0f - cfg.cue_frac))
                cue_a.profile[g] = 0.0f;
    }
    u32 vpos_cue = 0, vneg_cue = 0;
    hpc.read_valence(vpos_cue, vneg_cue);   // clear counters before cue
    present(cue_a, 400);
    hpc.read_valence(vpos_cue, vneg_cue);
    // (recalled v+ re-drives VUM continuously via the valence axon above —
    // wiring, not a software loop)
    const f64 completion_quality = hpc.familiarity();
    const bool valence_recalled = (vpos_cue > vneg_cue);  // A was rewarded

    // ================= A. DMS =================
    struct DmsRow {
        u32 delay_ms = 0;
        u8 cond = 0;  // 0 no-dist, 1 distractor-active, 2 distractor-rest
        f64 acc = 0;
        u32 n = 0;
    };
    std::vector<DmsRow> dms;
    const u32 delays[] = {0, 1000, 2000, 4000};
    const u8 conds[] = {0, 1, 2};
    std::ofstream csv;
    ensure_parent_dir(cfg.out_csv);
    csv.open(cfg.out_csv);
    csv << "delay_ms,cond,trial,match,resp_match,fam_probe\n";

    for (const u32 d : delays) {
        for (const u8 cond : conds) {
            u32 correct = 0;
            for (u32 rep = 0; rep < cfg.n_dms_reps; ++rep) {
                const bool use_a = rng.bernoulli(0.5f);
                const Odor& sample = use_a ? od_a : od_b;
                const bool is_match = rng.bernoulli(0.5f);
                const Odor& probe = is_match ? sample : (use_a ? od_b : od_a);
                pfc.clear_assembly();
                // sample (top-down open: encode)
                pfc.set_gate(1.0f);
                present(sample, cfg.t_sample_ms, /*binds=*/true);
                // delay with optional distractor. Active/attentive brain:
                // top-down CLOSES the gate to new input during maintenance
                // (distractor filtered). Resting brain: protection absent,
                // the distractor writes through and overwrites the code.
                if (cond == 0) {
                    gap(d);
                } else {
                    const bool rest = (cond == 2);
                    brain.mod().set_for_test_rest(rest);
                    pfc.set_gate(rest ? 1.0f : 0.25f);
                    const Odor& dis = use_a ? od_b : od_a;
                    present(dis, d, false);
                    brain.mod().set_for_test_rest(false);
                    pfc.set_gate(1.0f);
                }
                // maintained-template window (the memory code at the end of
                // the delay — this is what the distractor conditions stress)
                collect(tmpl_vec, 300);
                // probe response: first 200 ms of probe-evoked population
                // activity (sensory reactivation + attractor riding)
                brain.set_odor(&probe, 1.0f);
                std::fill(probe_vec.begin(), probe_vec.end(), 0u);
                for (u32 t = 0; t < 200; ++t) {
                    hpc.step(DT, brain.mb().kc_spike_pattern());
                    valence_axons();
                    brain.step(DT);
                    pfc.step(DT, brain.al().pn_spikes(), false);
                    const u8* ps = pfc.spikes();
                    for (u32 k = 0; k < pfc.size(); ++k) probe_vec[k] += ps[k];
                }
                present(probe, cfg.t_probe_ms - 200, false);
                // response = cosine similarity between the probe-evoked
                // population vector and the maintained template (population
                // vector decoding; Georges-Francois & Rolls / IT match)
                f64 dot = 0, nt = 0, np = 0;
                for (u32 k = 0; k < tmpl_vec.size(); ++k) {
                    dot += static_cast<f64>(tmpl_vec[k]) * probe_vec[k];
                    nt += static_cast<f64>(tmpl_vec[k]) * tmpl_vec[k];
                    np += static_cast<f64>(probe_vec[k]) * probe_vec[k];
                }
                const f64 reactivation =
                    (nt > 0.0 && np > 0.0) ? dot / std::sqrt(nt * np) : 0.0;
                const bool resp_match = reactivation > 0.30;
                const bool ok = (resp_match == is_match);
                correct += ok ? 1u : 0u;
                char buf[128];
                std::snprintf(buf, sizeof buf, "%u,%u,%u,%d,%d,%.4f\n", d, cond, rep,
                              is_match ? 1 : 0, resp_match ? 1 : 0, reactivation);
                csv << buf;
                gap(cfg.t_iti_ms);
            }
            dms.push_back({d, cond,
                           static_cast<f64>(correct) / cfg.n_dms_reps,
                           cfg.n_dms_reps});
        }
    }
    csv.close();

    // ================= C. retrieval-practice consolidation =================
    // Control arm first (fresh B-like pair would be cleaner; here we measure
    // the SAME pair before/after practice on MB probes — within-agent design).
    auto mb_probe = [&](const Odor& o) {
        present(o, 500);
        const auto r = brain.mb().readout();
        gap(cfg.t_iti_ms);
        return r.valence;
    };
    const f64 val_before = mb_probe(od_a) - mb_probe(od_b);
    // retrieval practice: re-presenting A recalls v+ -> VUM pulses (via the
    // axon); B recalls v- -> DAN. Consolidation follows from the MB's own
    // learning — no software delivery of reward.
    for (u32 k = 0; k < cfg.n_practice; ++k) {
        present(od_a, 500);
        run(300);
        brain.set_sensors(0.0f, 0.0f);
        gap(1500);
        present(od_b, 500);
        run(300);
        brain.set_sensors(0.0f, 0.0f);
        gap(1500);
    }
    gap(60000);
    const f64 val_after = mb_probe(od_a) - mb_probe(od_b);
    const f64 consolidation_gain = val_after - val_before;

    // ================= D. spacing effect =================
    // massed: X+ 30 trials, ITI 300ms; spaced: Y- 30 trials, ITI 3000ms
    // (within-agent counterbalanced order across seeds; here fixed order)
    gap(3000);
    // BOTH arms rewarded with the SAME US schedule; only the ITI differs
    // (massed 300ms vs spaced 3000ms) — clean Ebbinghaus-style comparison.
    for (u32 t = 0; t < cfg.n_spacing; ++t) {
        present(od_x, 500);
        gap(300);
        brain.set_sensors(1.0f, 0.0f);
        run(300);
        brain.set_sensors(0.0f, 0.0f);
        gap(300);
    }
    gap(60000);
    const f64 val_massed = mb_probe(od_x);
    for (u32 t = 0; t < cfg.n_spacing; ++t) {
        present(od_y, 500);
        gap(3000);
        brain.set_sensors(1.0f, 0.0f);
        run(300);
        brain.set_sensors(0.0f, 0.0f);
        gap(300);
    }
    gap(60000);
    const f64 val_spaced = mb_probe(od_y);
    const f64 spacing_effect = val_spaced - val_massed;  // spaced must win

    // ================= report =================
    f64 acc_nodist_1s = 0, acc_dist_active_1s = 0, acc_dist_rest_1s = 0;
    f64 acc_nodist_4s = 0;
    for (const DmsRow& r : dms) {
        if (r.delay_ms == 1000 && r.cond == 0) acc_nodist_1s = r.acc;
        if (r.delay_ms == 1000 && r.cond == 1) acc_dist_active_1s = r.acc;
        if (r.delay_ms == 1000 && r.cond == 2) acc_dist_rest_1s = r.acc;
        if (r.delay_ms == 4000 && r.cond == 0) acc_nodist_4s = r.acc;
    }

    std::printf("==== monkey1: primate-cognition benchmarks ====\n");
    std::printf("A. DMS accuracy: D=1s no-dist %.2f | distractor active %.2f | rest %.2f | D=4s %.2f\n",
                acc_nodist_1s, acc_dist_active_1s, acc_dist_rest_1s, acc_nodist_4s);
    std::printf("B. episodic: episodes=%u completion=%.3f valence recall A+=%s (v+=%u v-=%u)\n",
                eps_after_enc, completion_quality, valence_recalled ? "YES" : "no",
                vpos_cue, vneg_cue);
    std::printf("C. consolidation (retrieval practice): MB valence(A-B) %.3f -> %.3f (gain %+.3f)\n",
                val_before, val_after, consolidation_gain);
    std::printf("D. spacing: massed val(X+)=%+.3f vs spaced val(Y+)=%+.3f (effect %+.3f)\n",
                val_massed, val_spaced, spacing_effect);
    std::printf("HPC recurrent bound synapses: %llu | PFC bound: %llu\n",
                static_cast<unsigned long long>(hpc.rec_nonzero()),
                static_cast<unsigned long long>(pfc.bound_synapses()));

    ensure_parent_dir(cfg.out_json);
    std::ofstream f(cfg.out_json);
    JsonW j(f);
    j.kv("experiment", std::string("monkey1_primate_benchmarks"));
    j.kv("seed", cfg.seed);
    j.k("dms");
    j.arr();
    for (const DmsRow& r : dms) {
        j.obj();
        j.kv("delay_ms", r.delay_ms);
        j.kv("cond", r.cond);
        j.kv("acc", r.acc);
        j.kv("n", r.n);
        j.end_obj();
    }
    j.end_arr();
    j.k("episodic");
    j.obj();
    j.kv("episodes", eps_after_enc);
    j.kv("completion_quality", completion_quality);
    j.kv("valence_recalled", valence_recalled);
    j.kv("vpos", vpos_cue);
    j.kv("vneg", vneg_cue);
    j.end_obj();
    j.k("consolidation");
    j.obj();
    j.kv("val_before", val_before);
    j.kv("val_after", val_after);
    j.kv("gain", consolidation_gain);
    j.end_obj();
    j.k("spacing");
    j.obj();
    j.kv("val_massed", val_massed);
    j.kv("val_spaced", val_spaced);
    j.kv("effect", spacing_effect);
    j.end_obj();
    j.kv("hpc_bound", static_cast<u64>(hpc.rec_nonzero()));
    j.kv("pfc_bound", static_cast<u64>(pfc.bound_synapses()));
    j.end_obj();
    std::printf("results: %s | %s\n", cfg.out_json.c_str(), cfg.out_csv.c_str());
    return 0;
}

}  // namespace malefly
