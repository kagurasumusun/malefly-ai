#ifndef MALEFLY_CODE_AGENT_EXP_HPP
#define MALEFLY_CODE_AGENT_EXP_HPP

#include "core/types.hpp"
#include <string>

namespace malefly {

struct CodeAgentExpConfig {
    u64 seed = 42;
    u32 n_trials = 100;
    std::string out_json = "results/code_agent.json";
    std::string out_csv = "results/code_agent.csv";
};

int run_code_agent(const CodeAgentExpConfig& cfg);

} // namespace malefly

#endif // MALEFLY_CODE_AGENT_EXP_HPP
