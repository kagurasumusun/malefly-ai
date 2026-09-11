// compile/link check for the header-only Brain composition
#include "circuits/brain.hpp"
namespace malefly {
usize brain_memory_probe() {
    Rng rng(1);
    Brain b(BrainConfig{}, rng);
    return b.mb().memory_bytes() + b.al().memory_bytes();
}
}
