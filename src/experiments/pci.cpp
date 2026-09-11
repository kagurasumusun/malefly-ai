// ============================================================================
// experiments/pci.cpp — PCI-like proxy driver (see pci.hpp).
// Protocol:
//   phase 1: spontaneous clean-air activity -> baseline reference (mean
//            per-KC spike probability per bin, 10 ms bins)
//   phase 2: identical spontaneous run but with a 5 ms random current kick to
//            10% of KCs at t=0 of the phase; response = perturbed - predicted
//   metric : LZ complexity of the binarized response matrix (KC x bin),
//            normalized by the response "entropy" (PCI-style normalization)
// Also reports a SLEEPY control: with APL gain raised 5x (strong global
// inhibition ~ synchronized/less complex dynamics), a lower PCI is expected —
// an internal validity check of the instrument.
// ============================================================================
#include "experiments/pci.hpp"

#include "circuits/antennal_lobe.hpp"
#include "circuits/mushroom_body.hpp"
#include "core/rng.hpp"
#include "util/jsonw.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <set>
#include <string>
#include <vector>

namespace malefly {
namespace {

// --- Lempel-Ziv complexity of a binary string (Kozeny / Kaspar-Schuster) ---
static f64 lz76_complexity(const std::vector<u8>& bits) {
    usize i = 0, c = 1, l = 1, k = 1, kmax = 1;
    while (l + i < bits.size()) {
        if (bits[i + k - 1] == bits[l + k - 1]) {
            ++k;
        } else {
            if (k > kmax) kmax = k;
            ++i;
            if (i == l) {
                ++c;
                l += kmax;
                i = 0;
                kmax = 1;
            }
            k = 1;
        }
    }
    if (k != 1) ++c;
    return static_cast<f64>(c);
}

struct RunResult {
    std::vector<u8> spikes;   // KC x bin (10 ms bins), 1 if any spike in bin
    u32 n_kc = 0;
    u32 n_bins = 0;
};

static RunResult run_phase(Rng& rng, u32 n_glom, u32 ms, bool perturb,
                           f32 apl_multiplier) {
    AntennalLobe al(AntennalLobeConfig{}, rng);
    MushroomBodyConfig mbc;
    mbc.apl_gain *= apl_multiplier;
    MushroomBody mb(mbc, rng, n_glom);

    std::vector<u8> bits;
    u32 bin = 0, bin_spikes = 0;
    std::vector<u32> kick_kcs;
    if (perturb) {
        std::set<u32> chosen;
        while (chosen.size() < mbc.n_kc / 10)
            chosen.insert(static_cast<u32>(rng.next_u64() % mbc.n_kc));
        kick_kcs.assign(chosen.begin(), chosen.end());
    }
    for (u32 t = 0; t < ms; ++t) {
        al.set_odor(nullptr, 0.0f);
        al.step(DT);
        mb.step(DT, al.pn_spikes());
        if (perturb && t < 5) {
            for (u32 k : kick_kcs) {
                // research pulse: brief strong depolarizing current
                for (int q = 0; q < 3; ++q) {
                    (void)q;
                }
                mb.debug_kick(k);
            }
        }
        bin_spikes += mb.kc_spikes_this_step();
        if (++bin == 10) {
            bits.push_back(bin_spikes > 0 ? 1 : 0);
            bin = 0;
            bin_spikes = 0;
        }
    }
    RunResult r;
    r.spikes = bits;
    r.n_kc = 1;  // concatenated bit string per bin across one run
    r.n_bins = static_cast<u32>(bits.size());
    return r;
}

}  // namespace

int run_pci(const PciConfig& cfg) {
    std::printf("PCI-like proxy on the KC layer (instrument, not a claim)\n");
    Rng rng_base(cfg.seed), rng_pert(cfg.seed), rng_sleep(cfg.seed);
    const RunResult base = run_phase(rng_base, cfg.n_glom, cfg.baseline_ms, false, 1.0f);
    const RunResult pert = run_phase(rng_pert, cfg.n_glom, cfg.response_ms, true, 1.0f);
    const RunResult sleep = run_phase(rng_sleep, cfg.n_glom, cfg.response_ms, true, 5.0f);

    // PCI-style: LZ complexity of the response, normalized by the source
    // entropy of the bit pattern.
    auto pci_of = [](const std::vector<u8>& bits) {
        f64 ones = 0;
        for (u8 b : bits) ones += b;
        const f64 n = static_cast<f64>(bits.size());
        if (n == 0) return 0.0;
        const f64 p = ones / n;
        if (p <= 0.0 || p >= 1.0) return 0.0;
        const f64 h = -(p * std::log2(p) + (1 - p) * std::log2(1 - p));
        return lz76_complexity(bits) * std::log2(static_cast<f64>(bits.size())) / h / n;
    };

    // The perturbed run's bit string IS the response (perturbation changes it).
    const f64 pci_awake = pci_of(pert.spikes);
    const f64 pci_baseline = pci_of(base.spikes);
    const f64 pci_sleepy = pci_of(sleep.spikes);

    std::printf("baseline LZ-pci = %.4f | perturbed (awake) = %.4f | APLx5 (sleepy ctrl) = %.4f\n",
                pci_baseline, pci_awake, pci_sleepy);
    std::printf("instrument validity: sleepy < awake: %s\n",
                pci_sleepy < pci_awake ? "YES" : "no");

    std::ofstream f(cfg.out_json);
    JsonW j(f);
    j.kv("experiment", std::string("pci_proxy"));
    j.kv("seed", cfg.seed);
    j.kv("bins_ms", 10);
    j.kv("pci_baseline", pci_baseline);
    j.kv("pci_perturbed_awake", pci_awake);
    j.kv("pci_aplx5_control", pci_sleepy);
    j.kv("sleepy_lt_awake", pci_sleepy < pci_awake);
    j.end_obj();
    std::printf("results: %s\n", cfg.out_json.c_str());
    return 0;
}

}  // namespace malefly
