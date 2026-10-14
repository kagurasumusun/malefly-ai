#ifndef MALEFLY_CODE_SENSORIUM_HPP
#define MALEFLY_CODE_SENSORIUM_HPP

#include "core/types.hpp"
#include "core/rng.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <algorithm>

namespace malefly {

// Token categories for structured code representation
enum class CodeTokenType : u8 {
    Keyword = 0,
    Identifier,
    Operator,
    LiteralNumber,
    LiteralString,
    Delimiter,
    Indent,
    Dedent,
    Newline,
    Unknown
};

struct CodeToken {
    std::string text;
    CodeTokenType type{CodeTokenType::Unknown};
    u32 line{0};
    u32 col{0};
};

// Converts code text/tokens into multi-channel neural input spikes/activations
class CodeSensorium {
public:
    static constexpr u32 N_CHANNELS = 64; // e.g., 64 glomeruli / sensory input channels

    CodeSensorium() {
        init_vocab();
    }

    void init_vocab() {
        // Build base mapping for keywords, operators, delimiters
        const std::vector<std::string> keywords = {
            "def", "return", "if", "else", "elif", "for", "while", "in", "import",
            "from", "as", "class", "try", "except", "with", "lambda", "yield", "pass",
            "break", "continue", "and", "or", "not", "is", "None", "True", "False",
            "int", "str", "list", "dict", "set", "len", "range", "print"
        };
        for (u32 i = 0; i < keywords.size(); ++i) {
            vocab_map_[keywords[i]] = i + 1;
        }
    }

    // Simple lexical analyzer for code strings
    std::vector<CodeToken> tokenize(const std::string& code) const {
        std::vector<CodeToken> tokens;
        u32 i = 0;
        u32 line = 1, col = 1;

        while (i < code.size()) {
            char c = code[i];

            if (c == '\n') {
                tokens.push_back({"\n", CodeTokenType::Newline, line, col});
                line++;
                col = 1;
                i++;
                continue;
            }

            if (std::isspace(static_cast<unsigned char>(c))) {
                i++;
                col++;
                continue;
            }

            // Identifiers / Keywords
            if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
                u32 start_col = col;
                std::string s;
                while (i < code.size() && (std::isalnum(static_cast<unsigned char>(code[i])) || code[i] == '_')) {
                    s += code[i];
                    i++;
                    col++;
                }
                CodeTokenType type = CodeTokenType::Identifier;
                if (vocab_map_.count(s)) {
                    type = CodeTokenType::Keyword;
                }
                tokens.push_back({s, type, line, start_col});
                continue;
            }

            // Numbers
            if (std::isdigit(static_cast<unsigned char>(c))) {
                u32 start_col = col;
                std::string s;
                while (i < code.size() && (std::isdigit(static_cast<unsigned char>(code[i])) || code[i] == '.')) {
                    s += code[i];
                    i++;
                    col++;
                }
                tokens.push_back({s, CodeTokenType::LiteralNumber, line, start_col});
                continue;
            }

            // Operators & Delimiters
            std::string op(1, c);
            CodeTokenType type = CodeTokenType::Delimiter;
            if (std::string("+-*/%=<>!&|^~").find(c) != std::string::npos) {
                type = CodeTokenType::Operator;
                if (i + 1 < code.size()) {
                    char next_c = code[i+1];
                    if ((c == '=' || c == '!' || c == '<' || c == '>') && next_c == '=') {
                        op += next_c;
                        i++;
                        col++;
                    }
                }
            }
            tokens.push_back({op, type, line, col});
            i++;
            col++;
        }
        return tokens;
    }

    // Encodes a single token into a N_CHANNELS neural activation pattern [0, 1]
    std::vector<f32> encode_token(const CodeToken& token) const {
        std::vector<f32> pattern(N_CHANNELS, 0.0f);

        u64 hash = 14695981039346656037ULL; // FNV-1a
        for (char c : token.text) {
            hash ^= static_cast<u64>(c);
            hash *= 1099511628211ULL;
        }

        // Activate 4-8 specific sparse input channels deterministically based on hash
        Rng rng(hash);
        u32 n_active = 4 + (static_cast<u32>(rng.next_u64()) % 5);
        for (u32 k = 0; k < n_active; ++k) {
            u32 ch = static_cast<u32>(rng.next_u64()) % N_CHANNELS;
            f32 intensity = 0.5f + 0.5f * rng.uniform01();
            pattern[ch] = std::max(pattern[ch], intensity);
        }

        // Channel modulation based on token type
        u32 type_offset = static_cast<u32>(token.type) % 8;
        pattern[type_offset] = std::min(1.0f, pattern[type_offset] + 0.6f);

        return pattern;
    }

    // Encodes a sequence of tokens into a sensory activation window
    std::vector<f32> encode_sequence(const std::vector<CodeToken>& tokens, u32 context_len = 4) const {
        std::vector<f32> pattern(N_CHANNELS, 0.0f);
        if (tokens.empty()) return pattern;

        size_t start = tokens.size() > context_len ? tokens.size() - context_len : 0;
        f32 weight = 1.0f;
        for (size_t i = start; i < tokens.size(); ++i) {
            auto tok_pat = encode_token(tokens[i]);
            for (u32 ch = 0; ch < N_CHANNELS; ++ch) {
                pattern[ch] = std::max(pattern[ch], tok_pat[ch] * weight);
            }
            weight *= 0.8f; // Decay for older context tokens
        }
        return pattern;
    }

private:
    std::unordered_map<std::string, u32> vocab_map_;
};

} // namespace malefly

#endif // MALEFLY_CODE_SENSORIUM_HPP
