#pragma once
// ============================================================================
// core/types.hpp — scalar types and simulation constants.
//
// Units:
//   time     [s]      (integration timestep DT = 1 ms)
//   voltage  [V]
//   "current"          conductance-normalized drive, expressed in volts
//                      (current-based LIF: g_e * (E_e - v) is folded into g_e)
//
// No external dependencies. Everything in the engine is deterministic and
// owned by explicitly-passed Rng instances (see core/rng.hpp).
// ============================================================================
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace malefly {

using f32 = float;
using f64 = double;
using i32 = std::int32_t;
using i64 = std::int64_t;
using u8  = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using usize = std::size_t;

constexpr f32 DT = 0.001f;  // integration timestep: 1 ms

// shared action vocabulary (sensorimotor interface between circuits and world)
enum class Action : u8 { Avoid = 0, Approach = 1 };

}  // namespace malefly
