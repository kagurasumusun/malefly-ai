#ifndef MALEFLY_CODE_BRAIN_DEVOPS_HPP
#define MALEFLY_CODE_BRAIN_DEVOPS_HPP

#include "core/types.hpp"
#include <string>

namespace malefly {

struct CodeBrainDevOpsConfig {
    u64 seed = 42;
    u32 n_trials = 100;
    std::string out_json = "results/code_brain_devops.json";
    std::string out_csv = "results/code_brain_devops.csv";
};

int run_code_brain_devops(const CodeBrainDevOpsConfig& cfg);

} // namespace malefly

#endif // MALEFLY_CODE_BRAIN_DEVOPS_HPP
