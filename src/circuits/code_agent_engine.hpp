#ifndef MALEFLY_CODE_AGENT_ENGINE_HPP
#define MALEFLY_CODE_AGENT_ENGINE_HPP

#include "circuits/code_sensorium.hpp"
#include "circuits/mushroom_body.hpp"
#include "circuits/hippocampus.hpp"
#include "core/lif.hpp"
#include "core/rng.hpp"
#include "core/synapses.hpp"
#include "core/types.hpp"
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

namespace malefly {

struct CodeAgentConfig {
    u32 n_input_channels = CodeSensorium::N_CHANNELS;
    u32 n_kc = 500;
    u32 kc_fanin = 5;
    u32 n_output_actions = 16; // Candidate actions/token choices
    f32 eta_reward = 0.05f;
    f32 eta_punish = 0.05f;
};

class CodeAgentEngine {
public:
    CodeAgentEngine(const CodeAgentConfig& cfg, Rng& rng)
        : cfg_(cfg), rng_(rng), sensorium_(),
          mb_cfg_(make_mb_config(cfg)),
          mb_(mb_cfg_, rng, cfg.n_input_channels) {
        // Transition associative memory (N_CHANNELS x N_CHANNELS)
        transition_weights_.assign(CodeSensorium::N_CHANNELS * CodeSensorium::N_CHANNELS, 0.01f);
        trace_.assign(CodeSensorium::N_CHANNELS, 0.0f);
    }

    static MushroomBodyConfig make_mb_config(const CodeAgentConfig& cfg) {
        MushroomBodyConfig mbc;
        mbc.n_kc = cfg.n_kc;
        mbc.kc_fanin = cfg.kc_fanin;
        mbc.eta_reward = cfg.eta_reward;
        mbc.eta_punish = cfg.eta_punish;
        mbc.use_taxonomy = true;
        mbc.rpe_gating = true;
        return mbc;
    }

    const CodeSensorium& sensorium() const { return sensorium_; }

    // Step 1: Perceive current code snippet context
    std::vector<f32> perceive(const std::string& code_context) {
        auto tokens = sensorium_.tokenize(code_context);
        auto input_vec = sensorium_.encode_sequence(tokens, 4);

        // Set context trace for current snippet
        trace_ = input_vec;

        // Simulate AL -> MB dynamics for 100 timesteps (10ms)
        mb_.begin_trial();
        std::vector<u8> pn_spikes(CodeSensorium::N_CHANNELS, 0);
        for (u32 step = 0; step < 100; ++step) {
            for (u32 ch = 0; ch < CodeSensorium::N_CHANNELS; ++ch) {
                pn_spikes[ch] = (rng_.uniform01() < input_vec[ch] * 0.2f) ? 1 : 0;
            }
            mb_.step(DT, pn_spikes.data());
        }

        return input_vec;
    }

    // Step 2: Read decision / candidate action evaluation from circuit (valence and approach vs avoid)
    MushroomBody::Readout readout() const {
        return mb_.readout();
    }

    // Select among candidate actions based on MB valuation and candidate token transition prediction
    u32 select_candidate_action(const std::vector<f32>& input_vec, const std::vector<std::string>& candidates) {
        if (candidates.empty()) return 0;
        f32 base_valence = mb_.readout().valence;

        std::vector<std::vector<f32>> candidate_vecs;
        candidate_vecs.reserve(candidates.size());
        for (const auto& cand_str : candidates) {
            auto cand_tokens = sensorium_.tokenize(cand_str);
            candidate_vecs.push_back(sensorium_.encode_sequence(cand_tokens, 4));
        }

        std::vector<f32> scores(candidates.size(), 0.0f);
        for (size_t c = 0; c < candidates.size(); ++c) {
            f32 trans_score = 0.0f;
            f32 cand_sum = 0.0f;
            const auto& cand_vec = candidate_vecs[c];
            for (u32 j = 0; j < CodeSensorium::N_CHANNELS; ++j) {
                cand_sum += cand_vec[j];
            }
            if (cand_sum < 1e-5f) cand_sum = 1.0f;

            for (u32 i = 0; i < CodeSensorium::N_CHANNELS; ++i) {
                if (input_vec[i] < 0.05f) continue;
                for (u32 j = 0; j < CodeSensorium::N_CHANNELS; ++j) {
                    if (cand_vec[j] > 0.05f) {
                        trans_score += transition_weights_[i * CodeSensorium::N_CHANNELS + j] * input_vec[i] * cand_vec[j];
                    }
                }
            }
            scores[c] = base_valence + (trans_score / cand_sum) + 0.02f * rng_.uniform01();
        }

        // Epsilon-greedy exploration (15% random exploration during learning)
        u32 chosen_idx = 0;
        if (rng_.uniform01() < 0.15f) {
            chosen_idx = static_cast<u32>(rng_.next_u64()) % candidates.size();
        } else {
            auto max_it = std::max_element(scores.begin(), scores.end());
            chosen_idx = static_cast<u32>(std::distance(scores.begin(), max_it));
        }

        last_chosen_cand_vec_ = candidate_vecs[chosen_idx];
        return chosen_idx;
    }

    // Step 3: Reinforce based on environment execution score (+1 for compile/test pass, -1 for error)
    void learn(f32 reward) {
        mb_.apply_reinforcement(reward);

        // Update transition association weights between context trace and last chosen candidate pattern
        f32 eta = (reward > 0.0f) ? 0.30f : -0.15f;
        if (last_chosen_cand_vec_.size() == CodeSensorium::N_CHANNELS) {
            for (u32 i = 0; i < CodeSensorium::N_CHANNELS; ++i) {
                if (trace_[i] < 0.05f) continue;
                for (u32 j = 0; j < CodeSensorium::N_CHANNELS; ++j) {
                    if (last_chosen_cand_vec_[j] > 0.05f) {
                        f32& w = transition_weights_[i * CodeSensorium::N_CHANNELS + j];
                        w = std::clamp(w + eta * trace_[i] * last_chosen_cand_vec_[j], 0.0001f, 2.0f);
                    }
                }
            }
        }
    }

private:
    CodeAgentConfig cfg_;
    Rng rng_;
    CodeSensorium sensorium_;
    MushroomBodyConfig mb_cfg_;
    MushroomBody mb_;
    std::vector<f32> transition_weights_;
    std::vector<f32> trace_;
    std::vector<f32> last_chosen_cand_vec_;
};

} // namespace malefly

#endif // MALEFLY_CODE_AGENT_ENGINE_HPP
