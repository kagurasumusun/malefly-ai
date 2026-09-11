#pragma once
// ============================================================================
// circuits/region.hpp — standard brain-region interface.
//
// Adoption from 革新脳 Brain/MINDS (multiscale atlas, Okano et al. 2016):
// regions are modular units behind a standard interface, so new areas (CX,
// LH, VNC motor layers…) plug in without touching existing circuits.
// Migration is incremental (PRINCIPLES #8: existing working circuits are not
// carelessly discarded) — LateralHorn is the first region behind this
// interface; AL/MB will migrate as they are next touched.
// ============================================================================
#include "core/types.hpp"

namespace malefly {

class Region {
public:
    virtual ~Region() = default;
    virtual const char* name() const = 0;
    virtual void step(f32 dt) = 0;
    virtual usize memory_bytes() const = 0;
};

}  // namespace malefly
