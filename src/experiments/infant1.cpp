// ============================================================================
// experiments/infant1.cpp — AI-body language benchmarks (see infant1.hpp).
// The world is the AI interface: a caregiver "speaks" syllable tokens; the
// agent's body is a VocalMotor whose output is played back as self feedback.
// No strings, no dictionaries inside the agent — tokens are u8 ids at the
// interface and spike patterns inside.
// ============================================================================
#include "experiments/infant1.hpp"

#include "circuits/brain.hpp"
#include "circuits/language.hpp"
#include "util/jsonw.hpp"
#include "util/odors.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sys/stat.h>
#include <fstream>

namespace malefly {
namespace {

const char* SYL[12] = {"ba", "da", "ga", "ma", "na", "pa",
                       "ta", "ka", "chi", "te", "no", "lu"};

void ensure_parent_dir(const std::string& path) {
    const usize pos = path.find_last_of('/');
    if (pos == std::string::npos) return;
    ::mkdir(path.substr(0, pos).c_str(), 0755);
}

// one caregiver syllable: 150 ms sound
constexpr u32 T_Syl = 150;

}  // namespace

int run_infant1(const Infant1Config& cfg) {
    Rng rng(cfg.seed);

    // ======================================================================
    // L1. statistical word segmentation (Saffran 1996): 4 words x 3 syl,
    // 12 syllables, continuous stream, TP=1.0 within / ~0.33 across.
    // ======================================================================
    LanguageConfig lc1;  // n_tokens=12
    SyllableSensorium stg1(lc1, rng);
    const u32 words[4][3] = {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}};
    for (u32 w = 0; w < cfg.l1_words; ++w) {
        const u32* word = words[static_cast<u32>(rng.uniform01() * 4.0f) % 4u];
        for (u32 s = 0; s < 3; ++s) {
            for (u32 t = 0; t < T_Syl; ++t) {
                stg1.drive(word[s], true);
                stg1.step(DT);
            }
        }
    }
    f64 within[4], wmean = 0;
    for (u32 i = 0; i < 4; ++i) {
        within[i] = 0.5 * (stg1.trp_pair(words[i][0], words[i][1]) +
                           stg1.trp_pair(words[i][1], words[i][2]));
        wmean += within[i] / 4.0;
    }
    f64 across = 0;
    {   // boundary pairs: last syllable of word i -> first of word j (i!=j)
        f64 sum = 0; u32 n = 0;
        for (u32 i = 0; i < 4; ++i)
            for (u32 j = 0; j < 4; ++j) {
                if (i == j) continue;
                sum += stg1.trp_pair(words[i][2], words[j][0]);
                ++n;
            }
        across = sum / n;
    }
    const f64 seg_ratio = (across > 0.0) ? wmean / across : 0.0;
    const bool l1_pass = (wmean > across * 2.0);
    std::printf("L1 segmentation: within=%.3f across=%.3f ratio=%.1fx  %s\n",
                wmean, across, seg_ratio, l1_pass ? "PASS" : "FAIL");

    // ======================================================================
    // L2. babbling, imitation, and the self/other gate.
    // Body loop: VocalMotor produces a syllable -> driver plays it back as
    // SELF feedback (corollary-discharge tagged: suppressed + never treated
    // as caregiver statistics). Caregiver CONTINGENTLY echoes babbles
    // (proto-dialogue) — this is what teaches the arcuate echo mapping.
    // ======================================================================
    LanguageConfig lc2;
    lc2.n_tokens = 8;
    SyllableSensorium stg2(lc2, rng);
    VocalMotor mot2(lc2, rng);
    mot2.bind_sensorium(stg2);

    auto body_step = [&](bool caregiver_token, u32 tok) {
        mot2.step(DT, stg2.aud_spikes());
        if (mot2.speaking())
            stg2.drive(mot2.current_vocalization(), /*external=*/false);
        stg2.set_self_tag(mot2.speaking());
        if (caregiver_token) stg2.drive(tok, true);
        stg2.step(DT);
    };

    // phase A: solo babbling (15 s) — no caregiver, no echo learning
    const u64 voc_before = mot2.vocalizations();
    for (u32 t = 0; t < 15000; ++t) body_step(false, 0);
    const f64 babble_alone_rate =
        static_cast<f64>(mot2.vocalizations() - voc_before) / 15.0;

    // phase B: contingent-echo dialogue (40 s). The caregiver repeats each
    // babble back while the infant's motor trace is still active (STDP
    // window) — proto-conversation, the social gate for echo learning.
    mot2.set_social_presence(true);
    u32 echo_planned = 0;
    {
        bool prev_speaking = false;
        u32 echo_left = 0; u8 echo_tok = 0xff;
        for (u32 t = 0; t < 40000; ++t) {
            const bool sp = mot2.speaking();
            if (!sp && prev_speaking) {          // babble ended — caregiver
                echo_left = T_Syl;               // answers AFTER the infant
                echo_tok = mot2.current_vocalization();
                ++echo_planned;
            }
            prev_speaking = sp;
            const bool cg = (echo_left > 0);
            if (cg) --echo_left;
            body_step(cg, echo_tok);
        }
    }
    const u64 voc_total = mot2.vocalizations() - voc_before;
    const f64 babble_social_rate =
        static_cast<f64>(voc_total > static_cast<u64>(babble_alone_rate * 15.0)
                             ? voc_total - static_cast<u64>(babble_alone_rate * 15.0)
                             : 0u) / 40.0;

    // phase C: probe — caregiver says each syllable once (quiet state);
    // echo = the matching motor assembly responds.
    mot2.set_active(false);
    u32 echo_correct = 0;
    for (u32 j = 0; j < 8; ++j) {
        for (u32 t = 0; t < 800; ++t) body_step(false, 0);  // settle
        u32 best_tok = 0xff; u32 best_act = 0;
        for (u32 t = 0; t < 300; ++t) {
            body_step(true, j);
            if (t > 20)
                for (u32 k = 0; k < 8; ++k) {
                    const u32 a = mot2.mot_active_for(k);
                    if (a > best_act) { best_act = a; best_tok = k; }
                }
        }
        if (best_tok == j && best_act >= 2) ++echo_correct;
    }
    const bool l2_pass = (echo_correct >= 6) && stg2.self_blocked() > 100;
    std::printf(
        "L2 imitation: echo %u/8  self-blocked=%llu  babbles=%llu  %s\n",
        echo_correct, (unsigned long long)stg2.self_blocked(),
        (unsigned long long)mot2.vocalizations(), l2_pass ? "PASS" : "FAIL");

    // ======================================================================
    // L3. word-reward grounding: tokens are stimuli for the SAME Brain
    // pathway as odors (in this AI body, symbols enter like smells).
    // "pa"(0) -> reward, "ti"(1) -> neutral. Grounded valence readout.
    // ======================================================================
    Brain brain(BrainConfig{}, rng);
    Odor po = make_odor(50, 15, cfg.seed + 701, 0.60f, 1.00f);
    Odor ti = make_odor(50, 15, cfg.seed + 702, 0.60f, 1.00f);
    for (u32 tr = 0; tr < cfg.l3_trials; ++tr) {
        // po + reward (US overlaps the token)
        brain.set_odor(&po, 1.0f);
        for (u32 t = 0; t < 500; ++t) {
            brain.step(DT);
            if (t == 200) brain.set_sensors(1.0f, 0.0f);
        }
        brain.set_sensors(0.0f, 0.0f);
        brain.set_odor(nullptr, 0.0f);
        for (u32 t = 0; t < 2000; ++t) brain.step(DT);
        // ti alone
        brain.set_odor(&ti, 1.0f);
        for (u32 t = 0; t < 500; ++t) brain.step(DT);
        brain.set_odor(nullptr, 0.0f);
        for (u32 t = 0; t < 2000; ++t) brain.step(DT);
    }
    auto valence_probe = [&](const Odor& o) {
        brain.set_odor(&o, 1.0f);
        for (u32 t = 0; t < 500; ++t) brain.step(DT);
        const f64 v = brain.mb().trace_readout().valence;
        brain.set_odor(nullptr, 0.0f);
        for (u32 t = 0; t < 2000; ++t) brain.step(DT);
        return v;
    };
    const f64 v_po = valence_probe(po);
    const f64 v_ti = valence_probe(ti);
    const bool l3_pass = (v_po - v_ti) > 0.10;
    std::printf("L3 grounding: val(po)=%.3f val(ti)=%.3f  %s\n",
                v_po, v_ti, l3_pass ? "PASS" : "FAIL");

    // ======================================================================
    // L4. contingent reply: hear "ka" -> complete "ka pa" -> produce/complete
    // "pa" (TRP pattern completion; the primitive behind turn-taking).
    // ======================================================================
    LanguageConfig lc4;
    lc4.n_tokens = 4;   // "ka"(0) "pa"(1) "ma"(2) "ta"(3)
    SyllableSensorium stg4(lc4, rng);
    for (u32 p = 0; p < cfg.l4_pairs; ++p) {
        for (u32 t = 0; t < 200; ++t) { stg4.drive(0, true); stg4.step(DT); }
        for (u32 t = 0; t < 200; ++t) { stg4.drive(1, true); stg4.step(DT); }
        for (u32 t = 0; t < 1500; ++t) stg4.step(DT);
        for (u32 t = 0; t < 200; ++t) { stg4.drive(2, true); stg4.step(DT); }
        for (u32 t = 0; t < 200; ++t) { stg4.drive(3, true); stg4.step(DT); }
        for (u32 t = 0; t < 1500; ++t) stg4.step(DT);
    }
    // probe: cue -> completed syllable = completion-layer winner. Correct
    // reply = the TRAINED successor; a reverse completion (probe pa->ka)
    // would indicate TP-direction failure.
    u32 correct = 0, wrong = 0, reverse = 0;
    auto probe = [&](u32 cue, u32 expect, u32 self_id) {
        for (u32 t = 0; t < 800; ++t) stg4.step(DT);   // settle
        u32 best = 0xff; u32 best_act = 0;
        for (u32 t = 0; t < 250; ++t) {
            stg4.drive(cue, true);
            stg4.step(DT);
            if (t > 50)
                for (u32 k = 0; k < 4; ++k) {
                    const u32 a = stg4.comp_active_for(k);
                    if (a > best_act) { best_act = a; best = k; }
                }
        }
        if (best == expect && best_act >= 2) ++correct;
        else if (best == self_id) ++reverse;  // self-echo or backward: wrong
        else ++wrong;
    };
    for (u32 pr = 0; pr < cfg.l4_probes; ++pr) {
        probe(0, 1, 0);   // "ka" -> expect "pa"
        probe(2, 3, 2);   // "ma" -> expect "ta"
    }
    const u32 total_probes = cfg.l4_probes * 2;
    const bool l4_pass = (correct >= total_probes * 7 / 10 && reverse == 0);
    std::printf(
        "L4 reply: %u/%u correct  reverse %u  miss %u  %s\n",
        correct, total_probes, reverse, wrong, l4_pass ? "PASS" : "FAIL");

    // ======================================================================
    ensure_parent_dir(cfg.out);
    std::ofstream f(cfg.out);
    JsonW j(f);
    j.k("experiment"); j.val("infant1");
    j.k("seed"); j.val(static_cast<u64>(cfg.seed));
    j.k("L1_segmentation");
    j.kv("within_mean", wmean);
    j.kv("across_mean", across);
    j.kv("ratio", seg_ratio);
    j.kv("w0", within[0]); j.kv("w1", within[1]);
    j.kv("w2", within[2]); j.kv("w3", within[3]);
    j.kv("pass", l1_pass ? 1u : 0u);
    j.k("L2_imitation");
    j.kv("echo_correct_of8", echo_correct);
    j.kv("self_blocked", static_cast<u64>(stg2.self_blocked()));
    j.kv("babble_rate_alone_per_s", babble_alone_rate);
    j.kv("babble_rate_social_per_s", babble_social_rate);
    j.kv("pass", l2_pass ? 1u : 0u);
    j.k("L3_grounding");
    j.kv("valence_po", v_po);
    j.kv("valence_ti", v_ti);
    j.kv("pass", l3_pass ? 1u : 0u);
    j.k("L4_reply");
    j.kv("correct", correct);
    j.kv("probes", total_probes);
    j.kv("reverse_completions", reverse);
    j.kv("pass", l4_pass ? 1u : 0u);
    j.k("memory_bytes");
    j.kv("sensorium_l1", stg1.memory_bytes());
    j.kv("sensorium_l2", stg2.memory_bytes());
    j.k("pass_count");
    j.val(static_cast<u64>((l1_pass ? 1 : 0) + (l2_pass ? 1 : 0) +
                           (l3_pass ? 1 : 0) + (l4_pass ? 1 : 0)));
    // JsonW writes on destruction (root_.dump)
    std::printf("infant1: %u/4 pass | results: %s\n",
                (l1_pass ? 1 : 0) + (l2_pass ? 1 : 0) + (l3_pass ? 1 : 0) +
                    (l4_pass ? 1 : 0),
                cfg.out.c_str());
    return 0;
}

}  // namespace malefly
