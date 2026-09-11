#pragma once
// ============================================================================
// experiments/calib.hpp — circuit calibration harness.
//
// Calibration is measurement, not tuning by eye: KC sparsity, PN rates and
// MBON readouts are driven into biologically-motivated target ranges by
// adjusting synaptic quanta/thresholds, and the resulting constants are frozen
// as defaults with the measured evidence recorded (docs/RESULTS.md).
// ============================================================================
#include "core/types.hpp"

namespace malefly {

struct CalibConfig {
    u64 seed = 1;
    u32 n_pres = 20;            // presentations per odor
    f32 kc_quanta = 0.042f;     // PN->KC drive per spike (override)
    f32 kc_thresh = -0.050f;    // KC spike threshold (override)
    f32 apl_gain = 0.001f;      // APL feedback gain (override)
};

int run_calib(const CalibConfig& cfg);

}  // namespace malefly
