#ifndef MALEFLY_CODE_BRAIN_HPP
#define MALEFLY_CODE_BRAIN_HPP

#include "circuits/code_ast.hpp"
#include "circuits/hippocampus.hpp"
#include "circuits/mushroom_body.hpp"
#include "circuits/pfc.hpp"
#include "core/lif.hpp"
#include "core/rng.hpp"
#include "core/synapses.hpp"
#include "core/types.hpp"
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <memory>

namespace malefly {

struct CodeBrainConfig {
    u32 n_sensory_channels = CodeASTEncoder::N_CHANNELS; // 256 channels
    u32 n_pfc_units = 256;
    u32 n_kc = 1000;
    u32 kc_fanin = 7;
    f32 eta_reward = 0.10f;
    f32 eta_punish = 0.10f;
};

class CodeBrain {
public:
    CodeBrain(const CodeBrainConfig& cfg, Rng& rng)
        : cfg_(cfg), rng_(rng), ast_encoder_(),
          pfc_(make_pfc_config(cfg), rng, cfg.n_sensory_channels),
          hpc_(make_hpc_config(), rng, cfg.n_sensory_channels),
          mb_(make_mb_config(cfg), rng, cfg.n_sensory_channels) {

        // Context-to-Candidate transition weight matrix (256 x 256)
        transition_weights_.assign(CodeASTEncoder::N_CHANNELS * CodeASTEncoder::N_CHANNELS, 0.01f);
        context_trace_.assign(CodeASTEncoder::N_CHANNELS, 0.0f);
    }

    static PfcConfig make_pfc_config(const CodeBrainConfig& cfg) {
        PfcConfig pc;
        pc.n_cells = cfg.n_pfc_units;
        pc.tau_m = 0.040f; // Persistent delay-period time constant
        pc.thresh = -0.050f;
        return pc;
    }

    static HippocampusConfig make_hpc_config() {
        HippocampusConfig hc;
        hc.n_dg = 1500;
        hc.n_ca3 = 1200;
        return hc;
    }

    static MushroomBodyConfig make_mb_config(const CodeBrainConfig& cfg) {
        MushroomBodyConfig mbc;
        mbc.n_kc = cfg.n_kc;
        mbc.kc_fanin = cfg.kc_fanin;
        mbc.eta_reward = cfg.eta_reward;
        mbc.eta_punish = cfg.eta_punish;
        mbc.use_taxonomy = true;
        mbc.rpe_gating = true;
        return mbc;
    }

    const CodeASTEncoder& ast_encoder() const { return ast_encoder_; }

    // Step 1: Perceive prompt code context through AST/CFG encoder & drive PFC/HPC/MB
    std::vector<f32> perceive(const std::string& prompt_code) {
        auto ast = CodeASTEncoder::parse_simple_python(prompt_code);
        auto input_vec = ast_encoder_.encode_ast(ast);

        context_trace_ = input_vec;

        std::vector<u8> kc_pattern(CodeASTEncoder::N_CHANNELS, 0);
        for (u32 ch = 0; ch < CodeASTEncoder::N_CHANNELS; ++ch) {
            kc_pattern[ch] = (input_vec[ch] > 0.1f) ? 1 : 0;
        }

        // Step PFC working memory persistent firing & Hippocampus
        pfc_.step(DT, kc_pattern.data(), true);
        hpc_.step(DT, kc_pattern.data());

        // Drive Mushroom Body valuation circuit
        mb_.begin_trial();
        std::vector<u8> pn_spikes(CodeASTEncoder::N_CHANNELS, 0);
        for (u32 step = 0; step < 50; ++step) {
            for (u32 ch = 0; ch < CodeASTEncoder::N_CHANNELS; ++ch) {
                pn_spikes[ch] = (rng_.uniform01() < input_vec[ch] * 0.3f) ? 1 : 0;
            }
            mb_.step(DT, pn_spikes.data());
        }

        return input_vec;
    }

    // Step 2: Evaluate candidates semantically and select action
    u32 select_candidate(const std::vector<f32>& context_vec, const std::vector<std::string>& candidate_codes) {
        if (candidate_codes.empty()) return 0;

        f32 mb_valence = mb_.readout().valence;

        std::vector<std::vector<f32>> candidate_vecs;
        candidate_vecs.reserve(candidate_codes.size());

        for (const auto& cand_code : candidate_codes) {
            auto cand_ast = CodeASTEncoder::parse_simple_python(cand_code);
            candidate_vecs.push_back(ast_encoder_.encode_ast(cand_ast));
        }

        std::vector<f32> scores(candidate_codes.size(), 0.0f);
        for (size_t c = 0; c < candidate_codes.size(); ++c) {
            f32 trans_score = 0.0f;
            f32 cand_sum = 0.0f;
            const auto& cand_vec = candidate_vecs[c];

            for (u32 j = 0; j < CodeASTEncoder::N_CHANNELS; ++j) {
                cand_sum += cand_vec[j];
            }
            if (cand_sum < 1e-5f) cand_sum = 1.0f;

            for (u32 i = 0; i < CodeASTEncoder::N_CHANNELS; ++i) {
                if (context_vec[i] < 0.05f) continue;
                for (u32 j = 0; j < CodeASTEncoder::N_CHANNELS; ++j) {
                    if (cand_vec[j] > 0.05f) {
                        trans_score += transition_weights_[i * CodeASTEncoder::N_CHANNELS + j] * context_vec[i] * cand_vec[j];
                    }
                }
            }

            scores[c] = mb_valence + (trans_score / cand_sum) + 0.01f * rng_.uniform01();
        }

        // Epsilon-greedy exploration during early learning
        u32 chosen_idx = 0;
        if (rng_.uniform01() < exploration_rate_) {
            chosen_idx = static_cast<u32>(rng_.next_u64()) % candidate_codes.size();
        } else {
            auto max_it = std::max_element(scores.begin(), scores.end());
            chosen_idx = static_cast<u32>(std::distance(scores.begin(), max_it));
        }

        last_chosen_cand_vec_ = candidate_vecs[chosen_idx];
        return chosen_idx;
    }

    // Step 3: Learn from sandbox execution outcome with configurable brain mechanism gating
    void learn_variant(f32 reward, bool use_rpe_gating, bool use_hippocampal_replay) {
        mb_.apply_reinforcement(reward);
        if (reward > 0.0f) {
            hpc_.set_us(1.0f, 0.0f);
        } else {
            hpc_.set_us(0.0f, 1.0f);
        }

        f32 rpe_gate = use_rpe_gating ? mb_.last_rpe_gate() : 1.0f;
        f32 base_eta = (reward > 0.0f) ? 0.35f : -0.15f;
        f32 eta = base_eta * rpe_gate;

        if (last_chosen_cand_vec_.size() == CodeASTEncoder::N_CHANNELS) {
            for (u32 i = 0; i < CodeASTEncoder::N_CHANNELS; ++i) {
                if (context_trace_[i] < 0.05f) continue;
                for (u32 j = 0; j < CodeASTEncoder::N_CHANNELS; ++j) {
                    if (last_chosen_cand_vec_[j] > 0.05f) {
                        f32& w = transition_weights_[i * CodeASTEncoder::N_CHANNELS + j];
                        w = std::clamp(w + eta * context_trace_[i] * last_chosen_cand_vec_[j], 0.0001f, 3.0f);
                    }
                }
            }
        }

        // Hippocampal consolidation replay during offline phase
        if (use_hippocampal_replay && reward > 0.0f) {
            replay_rest_phase();
            // Consolidate CA3 replay activity onto transition associations
            const u8* ca3_spikes = hpc_.ca3_spikes();
            for (u32 i = 0; i < CodeASTEncoder::N_CHANNELS && i < hpc_.n_dg(); ++i) {
                if (!ca3_spikes[i]) continue;
                for (u32 j = 0; j < CodeASTEncoder::N_CHANNELS; ++j) {
                    if (last_chosen_cand_vec_[j] > 0.05f) {
                        f32& w = transition_weights_[i * CodeASTEncoder::N_CHANNELS + j];
                        w = std::min(3.0f, w + 0.05f * last_chosen_cand_vec_[j]);
                    }
                }
            }
        }

        if (reward > 0.0f) {
            exploration_rate_ = std::max(0.02f, exploration_rate_ * 0.95f);
        }
    }

    void learn(f32 reward) {
        learn_variant(reward, true, true);
    }

    // Hippocampal SWR replay step during quiet/rest phases
    void replay_rest_phase() {
        std::vector<u8> empty_pattern(CodeASTEncoder::N_CHANNELS, 0);
        hpc_.step(DT, empty_pattern.data());
    }

private:
    CodeBrainConfig cfg_;
    Rng rng_;
    CodeASTEncoder ast_encoder_;
    Pfc pfc_;
    Hippocampus hpc_;
    MushroomBody mb_;

    std::vector<f32> transition_weights_;
    std::vector<f32> context_trace_;
    std::vector<f32> last_chosen_cand_vec_;
    f32 exploration_rate_ = 0.20f;
};

} // namespace malefly

#endif // MALEFLY_CODE_BRAIN_HPP
