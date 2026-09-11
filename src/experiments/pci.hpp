#pragma once
// ============================================================================
// experiments/pci.hpp — PCI-like perturbational complexity instrument.
//
// Adoption from consciousness science / IIT-adjacent methodology (llT in the
// roadmap): the Perturbational Complexity Index (Casali et al. 2013) is a
// MEASUREMENT INSTRUMENT, not a claim of consciousness. We compute a PCI-like
// proxy on the KC layer: perturb spontaneous activity with a random current
// kick, and measure the spatiotemporal compressibility (Lempel-Ziv) of the
// binarized difference response. Numbers are reported as measured proxies;
// docs/PRINCIPLES.md #7 applies — no "consciousness achieved" claims.
// ============================================================================
#include "core/types.hpp"

namespace malefly {

struct PciConfig {
    u64 seed = 7;
    u32 baseline_ms = 2000;
    u32 response_ms = 2000;
    u32 n_glom = 50;
    std::string out_json = "results/pci.json";
};

int run_pci(const PciConfig& cfg);

}  // namespace malefly
