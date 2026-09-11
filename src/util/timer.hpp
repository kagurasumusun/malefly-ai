#pragma once
// ============================================================================
// util/timer.hpp — wall-clock measurement (all perf claims are measured).
// ============================================================================
#include <chrono>

namespace malefly {

struct Timer {
    std::chrono::steady_clock::time_point t0{};
    void start() { t0 = std::chrono::steady_clock::now(); }
    f64 seconds() const {
        return std::chrono::duration<f64>(std::chrono::steady_clock::now() - t0).count();
    }
};

}  // namespace malefly
