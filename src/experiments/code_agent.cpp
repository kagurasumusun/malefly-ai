#include "experiments/code_agent.hpp"
#include "circuits/code_agent_engine.hpp"
#include "env/code_env.hpp"
#include "core/rng.hpp"
#include "util/jsonw.hpp"
#include <cstdio>
#include <vector>
#include <fstream>
#include <numeric>

namespace malefly {

int run_code_agent(const CodeAgentExpConfig& cfg) {
    Rng rng(cfg.seed);
    CodeAgentConfig agent_cfg;
    CodeAgentEngine agent(agent_cfg, rng);
    CodeEnv env(rng);

    std::printf("=== Running Code Agent Experiment (seed=%llu, trials=%u) ===\n",
                (unsigned long long)cfg.seed, cfg.n_trials);

    u32 correct_count = 0;
    std::vector<f32> accuracy_history;
    std::vector<f32> reward_history;

    for (u32 trial = 0; trial < cfg.n_trials; ++trial) {
        auto task = env.get_task(trial);
        auto input_vec = agent.perceive(task.prompt_code);

        u32 choice = agent.select_candidate_action(input_vec, task.candidate_completions);
        f32 reward = env.evaluate(task, choice);

        agent.learn(reward);

        if (reward > 0.0f) {
            correct_count++;
        }

        f32 acc = static_cast<f32>(correct_count) / (trial + 1);
        accuracy_history.push_back(acc);
        reward_history.push_back(reward);

        if ((trial + 1) % 10 == 0 || trial == cfg.n_trials - 1) {
            std::printf("Trial %3u/%3u: Task '%s' -> Chosen %u (%s) | Reward: %+.1f | Acc: %.2f\n",
                        trial + 1, cfg.n_trials, task.task_id.c_str(), choice,
                        (reward > 0.0f ? "PASS" : "FAIL"), reward, acc);
        }
    }

    f32 final_acc = accuracy_history.empty() ? 0.0f : accuracy_history.back();
    std::printf("=== Experiment Complete: Final Accuracy = %.2f (%u/%u) ===\n",
                final_acc, correct_count, cfg.n_trials);

    // Save JSON output
    if (!cfg.out_json.empty()) {
        std::ofstream json_out(cfg.out_json);
        if (json_out.is_open()) {
            json_out << "{\n";
            json_out << "  \"seed\": " << cfg.seed << ",\n";
            json_out << "  \"n_trials\": " << cfg.n_trials << ",\n";
            json_out << "  \"correct_count\": " << correct_count << ",\n";
            json_out << "  \"final_accuracy\": " << final_acc << ",\n";
            json_out << "  \"history\": [";
            for (size_t i = 0; i < accuracy_history.size(); ++i) {
                json_out << accuracy_history[i] << (i + 1 < accuracy_history.size() ? ", " : "");
            }
            json_out << "]\n";
            json_out << "}\n";
        }
    }

    // Save CSV output
    if (!cfg.out_csv.empty()) {
        std::ofstream csv_out(cfg.out_csv);
        if (csv_out.is_open()) {
            csv_out << "trial,reward,accuracy\n";
            for (size_t i = 0; i < accuracy_history.size(); ++i) {
                csv_out << (i + 1) << "," << reward_history[i] << "," << accuracy_history[i] << "\n";
            }
        }
    }

    return 0;
}

} // namespace malefly
