#ifndef MALEFLY_CODE_AST_HPP
#define MALEFLY_CODE_AST_HPP

#include "core/types.hpp"
#include "core/rng.hpp"
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>

namespace malefly {

enum class ASTNodeType : u16 {
    Module = 0,
    FunctionDef,
    ReturnStmt,
    IfStmt,
    ForLoop,
    WhileLoop,
    AssignStmt,
    BinaryOp,
    CompareOp,
    NameIdent,
    ConstantNum,
    ConstantStr,
    ListExpr,
    CallExpr,
    Unknown
};

struct ASTNode {
    ASTNodeType type{ASTNodeType::Unknown};
    std::string value;
    u32 depth{0};
    u32 scope_id{0};
    std::vector<std::shared_ptr<ASTNode>> children;
};

class CodeASTEncoder {
public:
    static constexpr u32 N_CHANNELS = 256; // High-capacity sensory channel vector

    // Parses code into a simple AST structure
    static std::shared_ptr<ASTNode> parse_simple_python(const std::string& code) {
        auto root = std::make_shared<ASTNode>();
        root->type = ASTNodeType::Module;
        root->value = "module";
        root->depth = 0;

        // Lines processing
        std::vector<std::string> lines;
        std::string current_line;
        for (char c : code) {
            if (c == '\n') {
                if (!current_line.empty()) lines.push_back(current_line);
                current_line.clear();
            } else {
                current_line += c;
            }
        }
        if (!current_line.empty()) lines.push_back(current_line);

        u32 current_scope = 0;
        for (const auto& line : lines) {
            auto child = std::make_shared<ASTNode>();
            child->depth = 1;

            if (line.find("def ") != std::string::npos) {
                child->type = ASTNodeType::FunctionDef;
                child->value = line;
                current_scope++;
            } else if (line.find("return ") != std::string::npos) {
                child->type = ASTNodeType::ReturnStmt;
                child->value = line;
            } else if (line.find("if ") != std::string::npos || line.find("elif ") != std::string::npos || line.find("else:") != std::string::npos) {
                child->type = ASTNodeType::IfStmt;
                child->value = line;
            } else if (line.find("for ") != std::string::npos || line.find("while ") != std::string::npos) {
                child->type = ASTNodeType::ForLoop;
                child->value = line;
            } else if (line.find("=") != std::string::npos) {
                child->type = ASTNodeType::AssignStmt;
                child->value = line;
            } else {
                child->type = ASTNodeType::CallExpr;
                child->value = line;
            }

            child->scope_id = current_scope;
            root->children.push_back(child);
        }

        return root;
    }

    // Encodes an AST node tree into a 256-channel neural activation pattern [0.0, 1.0]
    std::vector<f32> encode_ast(const std::shared_ptr<ASTNode>& node) const {
        std::vector<f32> pattern(N_CHANNELS, 0.0f);
        if (!node) return pattern;

        // Traverse AST and map nodes to channel activations
        std::vector<std::shared_ptr<ASTNode>> queue = {node};
        size_t head = 0;

        while (head < queue.size()) {
            auto curr = queue[head++];

            u64 hash = 14695981039346656037ULL;
            u16 type_val = static_cast<u16>(curr->type);
            hash ^= static_cast<u64>(type_val);
            hash *= 1099511628211ULL;

            for (char c : curr->value) {
                hash ^= static_cast<u64>(c);
                hash *= 1099511628211ULL;
            }

            Rng rng(hash);
            u32 n_active = 8 + static_cast<u32>(rng.next_u64() % 9);
            for (u32 k = 0; k < n_active; ++k) {
                u32 ch = static_cast<u32>(rng.next_u64()) % N_CHANNELS;
                f32 val = 0.5f + 0.5f * rng.uniform01();
                pattern[ch] = std::max(pattern[ch], val);
            }

            // Scope & Depth modulation
            u32 scope_ch = (curr->scope_id * 16 + static_cast<u32>(curr->type)) % N_CHANNELS;
            pattern[scope_ch] = std::min(1.0f, pattern[scope_ch] + 0.5f);

            for (const auto& child : curr->children) {
                queue.push_back(child);
            }
        }

        return pattern;
    }
};

} // namespace malefly

#endif // MALEFLY_CODE_AST_HPP
