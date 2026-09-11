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
#include "circuits/mushroom_body.hpp"
#include "core/lif.hpp"
#include "core/rng.hpp"
#include "core/synapses.hpp"
#include "env/environment.hpp"
#include "util/odors.hpp"

#include <algorithm>
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
    CHECK(r.kc_active_frac == 0.0f);
    const Decision d = decide(r, cfg, rng, false);
    CHECK(d.weak_signal);
    CHECK(near(d.valence, cfg.naive_valence, 1e-6f));  // innate default applies
    CHECK(d.p_approach > 0.55f);                       // naive: mild approach
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
    test_env();
    test_odors();
    test_weak_signal_default();
    test_determinism();
    std::printf("%d checks, %d failures\n", g_run, g_fail);
    return g_fail == 0 ? 0 : 1;
}
