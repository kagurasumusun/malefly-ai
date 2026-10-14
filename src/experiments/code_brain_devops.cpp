#include "experiments/code_brain_devops.hpp"
#include "circuits/code_brain_variants.hpp"
#include "circuits/code_corpus.hpp"
#include "env/code_sandbox.hpp"
#include "core/rng.hpp"
#include "util/jsonw.hpp"
#include <cstdio>
#include <vector>
#include <fstream>
#include <numeric>
#include <map>

namespace malefly {

int run_code_brain_devops(const CodeBrainDevOpsConfig& cfg) {
    auto corpus = CodeCorpus::load_default_corpus();
    std::vector<BrainVariantType> variants = {
        BrainVariantType::StandardHebbian,
        BrainVariantType::RPEGatedDopamine,
        BrainVariantType::PFCHippocampalReplay,
        BrainVariantType::FullIntegratedBrain
    };

    std::printf("========================================================================\n");
    std::printf("  DEVOPS COMPARATIVE BENCHMARK: NEURAL CODE BRAIN VARIANTS (seed=%llu)\n",
                (unsigned long long)cfg.seed);
    std::printf("========================================================================\n");

    std::map<std::string, f32> final_accuracies;
    std::map<std::string, std::vector<f32>> accuracy_histories;

    for (size_t v = 0; v < variants.size(); ++v) {
        auto var_type = variants[v];
        // Give each variant a distinct seed branch derived from base seed
        u64 var_seed = cfg.seed + (v * 1000);
        CodeBrainVariantRunner runner(var_type, var_seed);
        std::string var_name = runner.variant_name();

        u32 correct_count = 0;
        std::vector<f32> acc_hist;

        for (u32 trial = 0; trial < cfg.n_trials; ++trial) {
            const auto& item = corpus[trial % corpus.size()];

            u32 choice = runner.select_candidate(item.prompt, item.choices);
            std::string full_candidate_code = item.prompt + item.choices[choice];

            // Real Python Sandbox Execution Check
            auto sb_res = CodeSandbox::verify_code_python(full_candidate_code, item.test_eval_code);

            runner.learn(sb_res.reward);

            if (sb_res.passed) {
                correct_count++;
            }

            f32 current_acc = static_cast<f32>(correct_count) / (trial + 1);
            acc_hist.push_back(current_acc);
        }

        f32 final_acc = acc_hist.empty() ? 0.0f : acc_hist.back();
        final_accuracies[var_name] = final_acc;
        accuracy_histories[var_name] = acc_hist;

        std::printf("Variant %-25s | Final Accuracy: %.2f (%u/%u)\n",
                    var_name.c_str(), final_acc, correct_count, cfg.n_trials);
    }

    std::printf("========================================================================\n");

    // Save output JSON
    if (!cfg.out_json.empty()) {
        std::ofstream json_out(cfg.out_json);
        if (json_out.is_open()) {
            json_out << "{\n";
            json_out << "  \"seed\": " << cfg.seed << ",\n";
            json_out << "  \"n_trials\": " << cfg.n_trials << ",\n";
            json_out << "  \"variants\": {\n";
            size_t v_idx = 0;
            for (auto const& [name, acc] : final_accuracies) {
                json_out << "    \"" << name << "\": " << acc;
                if (++v_idx < final_accuracies.size()) json_out << ",";
                json_out << "\n";
            }
            json_out << "  }\n";
            json_out << "}\n";
        }
    }

    // Save output CSV
    if (!cfg.out_csv.empty()) {
        std::ofstream csv_out(cfg.out_csv);
        if (csv_out.is_open()) {
            csv_out << "trial";
            for (auto const& [name, acc] : final_accuracies) {
                csv_out << "," << name;
            }
            csv_out << "\n";

            for (u32 t = 0; t < cfg.n_trials; ++t) {
                csv_out << (t + 1);
                for (auto const& [name, acc] : final_accuracies) {
                    csv_out << "," << accuracy_histories[name][t];
                }
                csv_out << "\n";
            }
        }
    }

    return 0;
}

} // namespace malefly
