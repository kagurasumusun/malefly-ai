// ============================================================================
// tests/test_core.cpp — unit + integration tests (no external framework).
//
// Covers: RNG reproducibility, LIF dynamics (fire/silent/refractory), CSR
// integrity & propagation, MB plasticity signs (canonical MB rule), bandit
// environment semantics, odor overlap math, and a bit-for-bit determinism
// smoke test of a full AL+MB simulation.
// ============================================================================
#include "circuits/action_selection.hpp"
#include "circuits/antennal_lobe.hpp"
#include "circuits/brain.hpp"
#include "circuits/lateral_horn.hpp"
#include "circuits/modulator.hpp"
#include "circuits/mushroom_body.hpp"
#include "core/lif.hpp"
#include "core/rng.hpp"
#include "core/synapses.hpp"
#include "env/environment.hpp"
#include "util/odors.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace malefly;

static int g_fail = 0;
static int g_run = 0;

#define CHECK(cond)                                                    \
    do {                                                               \
        ++g_run;                                                       \
        if (!(cond)) {                                                 \
            ++g_fail;                                                  \
            std::fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, \
                         __LINE__, #cond);                             \
        }                                                              \
    } while (0)

static bool near(f32 a, f32 b, f32 tol) { return std::fabs(a - b) <= tol; }

// ---------- RNG ----------
static void test_rng() {
    Rng a(7), b(7), c(8);
    bool same = true, diff = false;
    for (int i = 0; i < 4096; ++i) {
        const u64 x = a.next_u64(), y = b.next_u64(), z = c.next_u64();
        same &= (x == y);
        diff |= (x != z);
    }
    CHECK(same);
    CHECK(diff);

    Rng d(1);
    u32 below = 0, above = 0;
    for (int i = 0; i < 10000; ++i) {
        const f32 u = d.uniform01();
        CHECK(u >= 0.0f && u < 1.0f);
        below += (u < 0.5f);
        above += (u >= 0.5f);
    }
    CHECK(below > 4000 && above > 4000);  // roughly uniform
}

// ---------- LIF ----------
static void test_lif() {
    LifConfig cfg;
    LifLayer n(4, cfg);

    // silent without input
    for (int t = 0; t < 200; ++t) n.step(DT);
    CHECK(n.total_spikes() == 0);

    // strong tonic drive -> fires
    for (int t = 0; t < 100; ++t) {
        n.add_exc(0, 0.20f);
        n.add_exc(1, 0.20f);
        n.step(DT);
    }
    CHECK(n.total_spikes() >= 10);

    // refractory: with huge drive, rate is capped by t_refrac
    LifConfig c2;
    c2.t_refrac = 0.010f;
    LifLayer m(1, c2);
    for (int t = 0; t < 1000; ++t) {
        m.add_exc(0, 5.0f);
        m.step(DT);
    }
    // max rate = 1/(1ms+10ms) => ~90 spikes in 1s; allow margin
    CHECK(m.total_spikes() <= 100);
    CHECK(m.total_spikes() >= 50);
}

// ---------- CSR synapses ----------
static void test_synapses() {
    Rng rng(3);
    Synapses s = make_random_fanin(50, 500, 7, rng, 0.1f, 0.0f);
    CHECK(s.num() == 500 * 7);
    CHECK(s.indptr.size() == 51);
    CHECK(s.indptr[0] == 0);
    CHECK(s.indptr[50] == s.num());

    // degrees: each post has exactly 7 inputs
    std::vector<u32> deg(500, 0);
    for (u32 idx : s.indices) {
        CHECK(idx < 500);
        ++deg[idx];
    }
    for (u32 d : deg) CHECK(d == 7);

    // propagate a single pre spike; expected = 0.1 * out-degree(pre)
    std::vector<f32> post(500, 0.0f);
    std::vector<u8> pre(50, 0);
    u32 outdeg = s.indptr[5 + 1] - s.indptr[5];
    pre[5] = 1;
    s.propagate(pre.data(), post.data());
    u32 hit = 0;
    for (f32 x : post) hit += (x > 0.0f) ? 1u : 0u;
    CHECK(hit == outdeg);
    for (f32 x : post)
        if (x > 0.0f) CHECK(near(x, 0.1f, 1e-6f));
}

// ---------- MB plasticity: canonical sign rule ----------
static void test_mb_plasticity() {
    Rng rng(11);
    MushroomBodyConfig cfg;
    cfg.n_kc = 100;
    cfg.kc_fanin = 3;
    cfg.use_taxonomy = false;  // isolate the canonical rule from adoptions
    cfg.rpe_gating = false;
    cfg.oppo_restore = 0.0f;   // legacy depression-only rule (tested below)
    MushroomBody mb(cfg, rng, 10);

    const f32 wa0 = mb.w_appr()[0], wv0 = mb.w_avoid()[0];
    const f32 wa1 = mb.w_appr()[1], wv1 = mb.w_avoid()[1];
    const f32 wa2 = mb.w_appr()[2], wv2 = mb.w_avoid()[2];

    // reward: depress avoid branch on eligible KCs only
    mb.debug_set_elig(0, 1.0f);
    mb.debug_set_elig(1, 0.5f);
    mb.apply_reinforcement(+1.0f);
    CHECK(near(mb.w_avoid()[0], wv0 - cfg.eta_reward * 1.0f, 1e-5f));
    CHECK(near(mb.w_avoid()[1], wv1 - cfg.eta_reward * 0.5f, 1e-5f));
    CHECK(near(mb.w_avoid()[2], wv2, 1e-6f));                      // not eligible
    CHECK(near(mb.w_appr()[0], wa0, 1e-6f));                       // approach untouched
    CHECK(near(mb.w_appr()[1], wa1, 1e-6f));

    // punishment: depress approach branch on eligible KCs only
    mb.debug_set_elig(2, 1.0f);
    mb.apply_reinforcement(-1.0f);
    CHECK(near(mb.w_appr()[2], wa2 - cfg.eta_punish * 1.0f, 1e-5f));
    CHECK(near(mb.w_avoid()[2], wv2, 1e-6f));

    // zero outcome: no change
    mb.apply_reinforcement(0.0f);
    CHECK(near(mb.w_appr()[2], wa2 - cfg.eta_punish, 1e-5f));

    // floor at zero
    mb.debug_set_elig(3, 1.0f);
    for (int i = 0; i < 100; ++i) mb.apply_reinforcement(-1.0f);
    CHECK(mb.w_appr()[3] == 0.0f);
}

// ---------- environment semantics ----------
static void test_env() {
    Odor a = make_odor(50, 15, 5, 0.6f, 1.0f);
    Odor b = make_odor(50, 15, 6, 0.6f, 1.0f);
    BanditConfig bc;
    bc.deterministic = true;
    Rng rng(9);
    BanditEnv env(a, b, bc, rng);

    int found_good = 0;
    for (int i = 0; i < 200 && found_good < 5; ++i) {
        env.begin_trial();
        if (env.current_is_good()) {
            ++found_good;
            CHECK(env.apply(Action::Approach) == +1.0f);
        } else {
            CHECK(env.apply(Action::Approach) == -1.0f);
        }
        CHECK(env.apply(Action::Avoid) == 0.0f);  // avoid -> no outcome
    }
    CHECK(found_good == 5);
}

// ---------- odors ----------
static void test_odors() {
    Odor a = make_odor(50, 15, 123, 0.6f, 1.0f);
    CHECK(profile_overlap(a, a) > 0.999f);
    Odor b = make_odor(50, 15, 456, 0.6f, 1.0f);
    const f32 ov = profile_overlap(a, b);
    CHECK(ov > 0.0f && ov < 0.6f);  // random 15/50 subsets: partial overlap
    Odor c = mix_odors(a, b, 0.5f, 0.5f);
    for (u32 g = 0; g < 50; ++g)
        CHECK(c.profile[g] <= 1.0f + 1e-6f);
}

// ---------- [Allen BICCN] KC subtype taxonomy ----------
static void test_taxonomy() {
    MushroomBodyConfig cfg;
    cfg.use_taxonomy = true;
    Rng rng(31);
    MushroomBody mb(cfg, rng, 50);
    const auto counts = mb.subtype_counts();
    CHECK(counts.size() == 3);
    u32 total = 0;
    for (u32 c : counts) total += c;
    CHECK(total == cfg.n_kc);
    // measured fractions from MaleCNS v1.0 (allow 2% absolute tolerance)
    const f64 frac_ab = static_cast<f64>(counts[0]) / cfg.n_kc;
    const f64 frac_abp = static_cast<f64>(counts[1]) / cfg.n_kc;
    const f64 frac_g = static_cast<f64>(counts[2]) / cfg.n_kc;
    CHECK(std::fabs(frac_ab - KC_SUBTYPES[0].frac) < 0.02);
    CHECK(std::fabs(frac_abp - KC_SUBTYPES[1].frac) < 0.02);
    CHECK(std::fabs(frac_g - KC_SUBTYPES[2].frac) < 0.02);
    // fractions sum to ~1 and come from real data (alpha-beta largest)
    CHECK(frac_ab > frac_abp && frac_g > frac_abp);
}

// ---------- [BRAIN Initiative] RPE-gated plasticity ----------
static void test_rpe_gating() {
    Rng rng(41);
    MushroomBodyConfig cfg;
    cfg.n_kc = 100;
    cfg.kc_fanin = 3;
    cfg.use_taxonomy = false;
    cfg.rpe_gating = true;
    MushroomBody mb(cfg, rng, 10);

    // set all traces eligible; trace valence = naive (appr-avoid ~ +0.15)
    for (u32 k = 0; k < cfg.n_kc; ++k) mb.debug_set_elig(k, 1.0f);
    const f32 expected = mb.trace_readout().valence;
    const f32 rpe = std::fabs(1.0f - expected);
    const f32 gate = std::clamp(0.30f + 0.35f * rpe, 0.30f, 1.0f);
    CHECK(mb.last_rpe_gate() == 1.0f);  // not applied yet

    const f32 wv0 = mb.w_avoid()[0];
    mb.apply_reinforcement(+1.0f);
    CHECK(near(mb.w_avoid()[0], wv0 - cfg.eta_reward * gate, 1e-4f));
    CHECK(near(mb.last_rpe_gate(), gate, 1e-5f));

    // gate is bounded in [0.3, 1.0]; with baseline-referenced readout the
    // naive expectation is 0, so +1 and -1 gate symmetrically at 0.65
    CHECK(gate >= 0.30f && gate <= 1.0f);
    CHECK(near(gate, 0.65f, 1e-3f));
}

// ---------- opponent restoration (reversal substrate) ----------
static void test_opponent_restore() {
    Rng rng(45);
    MushroomBodyConfig cfg;
    cfg.n_kc = 100;
    cfg.kc_fanin = 3;
    cfg.use_taxonomy = false;
    cfg.rpe_gating = false;
    cfg.oppo_restore = 1.0f;
    MushroomBody mb(cfg, rng, 10);
    const f32 naive_appr = mb.readout().a_appr;   // empty counters -> naive ref
    const f32 wa0 = mb.w_appr()[0], wv0 = mb.w_avoid()[0];
    mb.debug_set_elig(0, 1.0f);
    mb.apply_reinforcement(+1.0f);
    const f32 exp_appr = std::min(wa0 + cfg.eta_reward, naive_appr);
    CHECK(near(mb.w_avoid()[0], wv0 - cfg.eta_reward, 1e-5f));       // canonical
    CHECK(near(mb.w_appr()[0], exp_appr, 1e-5f));                    // restore toward naive
    mb.debug_set_elig(0, 1.0f);
    mb.apply_reinforcement(-1.0f);
    const f32 exp_appr2 = std::max(0.0f, exp_appr - cfg.eta_punish);
    CHECK(near(mb.w_appr()[0], exp_appr2, 1e-4f));                   // punished back
    CHECK(mb.w_avoid()[0] <= naive_appr);
    // restore=0 reproduces the legacy depression-only rule
    MushroomBodyConfig cfg2 = cfg;
    cfg2.oppo_restore = 0.0f;
    MushroomBody mb2(cfg2, rng, 10);
    const f32 wa2 = mb2.w_appr()[0];
    mb2.debug_set_elig(0, 1.0f);
    mb2.apply_reinforcement(+1.0f);
    CHECK(near(mb2.w_appr()[0], wa2, 1e-6f));
}

// ---------- [Brain/MINDS] lateral horn region ----------
static void test_lateral_horn() {
    Rng rng(51);
    LateralHorn lh(LateralHornConfig{}, rng, 50);
    AntennalLobe al(AntennalLobeConfig{}, rng);
    CHECK(std::string(lh.name()) == "lateral_horn");  // Region interface

    auto response_to = [&](const Odor& o) {
        lh.begin_window();
        al.set_odor(&o, 1.0f);
        for (int t = 0; t < 500; ++t) {
            al.step(DT);
            lh.set_input(al.pn_spikes());
            lh.step(DT);
        }
        al.set_odor(nullptr, 0.0f);
        return lh.innate_valence();
    };
    Odor strong = make_odor(50, 18, 77, 0.7f, 1.0f);
    Odor air = uniform_odor(50, 0.12f);
    const f32 v_strong = response_to(strong);
    const f32 v_air = response_to(air);
    CHECK(v_strong > 0.03f);            // patterned odor -> innate attraction
    CHECK(v_air < 0.3f * v_strong);     // clean air -> (near) none
    CHECK(v_strong >= 0.0f && v_air >= 0.0f);
}

// ---------- [EBRAINS] modulator & arousal-modulated decision ----------
static void test_modulator_arousal() {
    ModulatorConfig mc;
    Modulator mod(mc);
    mod.on_outcome(+1.0f);
    CHECK(near(mod.arousal(), 0.25f, 1e-5f));
    CHECK(mod.arousal() > 0.0f);
    mod.on_outcome(-1.0f);
    CHECK(mod.arousal() > 0.0f);  // punishments also raise arousal
    mod.set_for_test(2.0f);
    CHECK(mod.arousal() == 1.0f);  // clamped
    for (int i = 0; i < 100000; ++i) mod.step(DT);  // 100 s >> tau_arousal(20 s)
    CHECK(mod.arousal() < 0.01f);  // decays

    // arousal raises exploration (temperature & epsilon): a strong valence
    // stays decisive but is softened, and near-zero valence moves toward 0.5
    Rng rng(61);
    MushroomBodyConfig cfg;
    MushroomBody::Readout r;
    r.valence = +0.60f;
    r.kc_active_frac = 0.10f;
    DecisionContext calm, hot;
    hot.arousal = 1.0f;
    hot.use_arousal = true;
    const Decision d_calm = decide_ctx(r, cfg, rng, false, calm);
    const Decision d_hot = decide_ctx(r, cfg, rng, false, hot);
    CHECK(d_hot.p_approach < d_calm.p_approach);   // softened, still > 0.5
    CHECK(d_hot.p_approach > 0.5f);
    r.valence = -0.60f;
    const Decision d_calm2 = decide_ctx(r, cfg, rng, false, calm);
    const Decision d_hot2 = decide_ctx(r, cfg, rng, false, hot);
    CHECK(d_hot2.p_approach > d_calm2.p_approach); // softened, still < 0.5
    CHECK(d_hot2.p_approach < 0.5f);
}

// ---------- MB trace-based readout (circuit-state memory) ----------
static void test_trace_readout() {
    Rng rng(23);
    MushroomBodyConfig cfg;
    cfg.n_kc = 200;
    cfg.kc_fanin = 3;
    MushroomBody mb(cfg, rng, 10);

    auto t0 = mb.trace_readout();
    CHECK(t0.active_frac == 0.0f);
    CHECK(t0.valence == 0.0f);  // empty trace -> zero valence, not innate bias

    for (u32 k = 0; k < cfg.n_kc; ++k) mb.debug_set_elig(k, 1.0f);
    auto t1 = mb.trace_readout();
    CHECK(t1.active_frac == 1.0f);
    // baseline-referenced readout: naive weights -> delta valence ~ 0
    CHECK(near(t1.valence, 0.0f, 0.02f));
}

// ---------- weak-signal default ----------
static void test_weak_signal_default() {
    Rng rng(17);
    AntennalLobe al(AntennalLobeConfig{}, rng);
    MushroomBodyConfig cfg;
    MushroomBody mb(cfg, rng, 50);
    al.set_odor(nullptr, 0.0f);
    mb.begin_trial();
    for (int t = 0; t < 500; ++t) {
        al.step(DT);
        mb.step(DT, al.pn_spikes());
    }
    const auto r = mb.readout();
    // heterogeneous wiring: clean-air activity is sparse but not exactly zero
    CHECK(r.kc_active_frac < 0.005f);
    const Decision d = decide(r, cfg, rng, false);
    CHECK(d.weak_signal);
    CHECK(near(d.valence, cfg.naive_valence, 1e-6f));  // innate default applies
    CHECK(d.p_approach > 0.55f);                       // naive: mild approach
}

// ---------- Brain: wiring-only spontaneous behavior + VUM plasticity ----------
static void test_brain_emergent() {
    Rng rng(71);
    Brain brain(BrainConfig{}, rng);

    // empty world for 10 s: behavior must happen with NO inputs at all
    brain.set_odor(nullptr, 0.0f);
    brain.set_sensors(0.0f, 0.0f);
    i64 a0 = 0, v0 = 0;
    u64 d0 = 0;
    for (int t = 0; t < 10000; ++t) {
        brain.step(DT);
        const auto o = brain.out();
        a0 = o.motor_approach;
        v0 = o.motor_avoid;
        d0 = o.decisions;
    }
    CHECK(d0 > 0);                       // spontaneous decisions exist
    CHECK(a0 + v0 > 0);                  // motor nerves carried spikes

    // reward sensor (sugar) without any odor: VUM fires -> MB weights change
    const f32 w_before = brain.mb().w_avoid()[0];
    BrainConfig cfg2;
    Rng rng2(72);
    Brain brain2(cfg2, rng2);
    brain2.set_odor(nullptr, 0.0f);
    for (int t = 0; t < 3000; ++t) {
        brain2.set_sensors(t >= 1000 && t < 1200 ? 1.0f : 0.0f, 0.0f);
        brain2.step(DT);
    }
    CHECK(brain2.vum_spikes() > 0);      // the reward neuron fired
    bool w_changed = false;
    for (u32 k = 0; k < cfg2.mb.n_kc; ++k)
        if (brain2.mb().w_avoid()[k] < 0.60f) w_changed = true;
    (void)w_before;
    CHECK(w_changed);                    // plasticity followed the VUM

    // punishment sensor: DAN fires
    Rng rng3(73);
    Brain brain3(BrainConfig{}, rng3);
    brain3.set_odor(nullptr, 0.0f);
    for (int t = 0; t < 3000; ++t) {
        brain3.set_sensors(0.0f, t >= 1000 && t < 1200 ? 1.0f : 0.0f);
        brain3.step(DT);
    }
    CHECK(brain3.dan_spikes() > 0);
}

// ---------- end-to-end bit-for-bit determinism ----------
static u64 simulate_once(u64 seed) {
    Rng rng(seed);
    Odor a = make_odor(50, 15, seed + 1, 0.6f, 1.0f);
    Odor b = make_odor(50, 15, seed + 2, 0.6f, 1.0f);
    AntennalLobe al(AntennalLobeConfig{}, rng);
    MushroomBodyConfig cfg;
    MushroomBody mb(cfg, rng, 50);

    u64 h = 1469598103934665603ull;
    auto hash_bytes = [&h](const void* p, usize n) {
        const u8* q = static_cast<const u8*>(p);
        for (usize i = 0; i < n; ++i) {
            h ^= q[i];
            h *= 1099511628211ull;
        }
    };

    for (int rep = 0; rep < 3; ++rep) {
        for (const Odor* o : {&a, &b}) {
            mb.begin_trial();
            al.set_odor(o, 1.0f);
            for (int t = 0; t < 200; ++t) {
                al.step(DT);
                mb.step(DT, al.pn_spikes());
                hash_bytes(al.pn_spikes(), 50);
                hash_bytes(mb.trial_pattern().data(), cfg.n_kc);
            }
            const auto r = mb.readout();
            hash_bytes(&r.valence, sizeof(f32));
            hash_bytes(&r.a_appr, sizeof(f32));
            al.set_odor(nullptr, 0.0f);
            for (int t = 0; t < 100; ++t) {
                al.step(DT);
                mb.step(DT, al.pn_spikes());
            }
            // DAN reinforcement on the odor just presented
            mb.apply_reinforcement(rep == 1 ? -1.0f : +1.0f);
        }
    }
    for (f32 w : mb.w_appr()) hash_bytes(&w, sizeof(f32));
    for (f32 w : mb.w_avoid()) hash_bytes(&w, sizeof(f32));
    return h;
}

static void test_determinism() {
    const u64 h1 = simulate_once(9);
    const u64 h2 = simulate_once(9);
    const u64 h3 = simulate_once(10);
    CHECK(h1 == h2);  // same seed -> bit-identical
    CHECK(h1 != h3);  // different seed -> different dynamics
}

int main() {
    test_rng();
    test_lif();
    test_synapses();
    test_mb_plasticity();
    test_opponent_restore();
    test_env();
    test_odors();
    test_taxonomy();
    test_rpe_gating();
    test_lateral_horn();
    test_modulator_arousal();
    test_trace_readout();
    test_weak_signal_default();
    test_brain_emergent();
    test_determinism();
    std::printf("%d checks, %d failures\n", g_run, g_fail);
    return g_fail == 0 ? 0 : 1;
}
