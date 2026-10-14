#ifndef MALEFLY_CODE_BRAIN_VARIANTS_HPP
#define MALEFLY_CODE_BRAIN_VARIANTS_HPP

#include "circuits/code_brain.hpp"
#include <string>
#include <vector>

namespace malefly {

enum class BrainVariantType {
    StandardHebbian,
    RPEGatedDopamine,
    PFCHippocampalReplay,
    FullIntegratedBrain
};

class CodeBrainVariantRunner {
public:
    CodeBrainVariantRunner(BrainVariantType type, u64 seed)
        : type_(type), rng_(seed), brain_(CodeBrainConfig{}, rng_) {}

    u32 select_candidate(const std::string& prompt_code, const std::vector<std::string>& candidates) {
        auto input_vec = brain_.perceive(prompt_code);
        return brain_.select_candidate(input_vec, candidates);
    }

    void learn(f32 reward) {
        switch (type_) {
            case BrainVariantType::StandardHebbian:
                brain_.learn_variant(reward, false, false);
                break;
            case BrainVariantType::RPEGatedDopamine:
                brain_.learn_variant(reward, true, false);
                break;
            case BrainVariantType::PFCHippocampalReplay:
                brain_.learn_variant(reward, false, true);
                break;
            case BrainVariantType::FullIntegratedBrain:
                brain_.learn_variant(reward, true, true);
                break;
        }
    }

    std::string variant_name() const {
        switch (type_) {
            case BrainVariantType::StandardHebbian: return "A_StandardHebbian";
            case BrainVariantType::RPEGatedDopamine: return "B_RPEGatedDopamine";
            case BrainVariantType::PFCHippocampalReplay: return "C_PFCHippocampalReplay";
            case BrainVariantType::FullIntegratedBrain: return "D_FullIntegratedBrain";
        }
        return "Unknown";
    }

private:
    BrainVariantType type_;
    Rng rng_;
    CodeBrain brain_;
};

} // namespace malefly

#endif // MALEFLY_CODE_BRAIN_VARIANTS_HPP
