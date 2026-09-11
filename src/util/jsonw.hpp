#pragma once
// ============================================================================
// util/jsonw.hpp — JSON writer facade over nlohmann/json (vendored).
//
// The streaming writer used here previously was hand-rolled and had a
// comma-bug; correctness is now delegated to the battle-tested nlohmann
// implementation (third_party/nlohmann/json.hpp, v3.11.3, MIT). The legacy
// streaming API (k/kv/obj/arr/val) is preserved so experiment code does not
// change; output is written on destruction as indented JSON.
// ============================================================================
#include "core/types.hpp"
#include <nlohmann/json.hpp>
#include <ostream>
#include <string>

namespace malefly {

class JsonW {
public:
    explicit JsonW(std::ostream& os) : os_(os) { stack_.push_back(&root_); }
    ~JsonW() {
        try {
            os_ << root_.dump(2);
        } catch (...) {
        }
    }

    void k(const std::string& key) { pending_ = key; }

    void obj() {
        if (stack_.back()->is_array()) {
            stack_.back()->push_back(nlohmann::json::object());
            stack_.push_back(&stack_.back()->back());
        } else {
            nlohmann::json& p = (*stack_.back())[pending_];
            p = nlohmann::json::object();
            stack_.push_back(&p);
        }
        pending_.clear();
    }
    void end_obj() { pop(); }

    void arr() {
        if (stack_.back()->is_array()) {
            stack_.back()->push_back(nlohmann::json::array());
            stack_.push_back(&stack_.back()->back());
        } else {
            nlohmann::json& p = (*stack_.back())[pending_];
            p = nlohmann::json::array();
            stack_.push_back(&p);
        }
        pending_.clear();
    }
    void end_arr() { pop(); }

    void val(f64 v)  { put(v); }
    void val(f32 v)  { put(static_cast<f64>(v)); }
    void val(u64 v)  { put(v); }
    void val(i64 v)  { put(v); }
    void val(int v)  { put(v); }
    void val(u32 v)  { put(v); }
    void val(bool v) { put(v); }
    void val(const std::string& s) { put(s); }

    template <class T>
    void kv(const std::string& key, const T& v) {
        k(key);
        val(v);
    }
    void kv(const std::string& key, f32 v) { k(key); val(static_cast<f64>(v)); }
    void kv(const std::string& key, u32 v) { k(key); val(static_cast<u64>(v)); }

private:
    template <class T>
    void put(const T& v) {
        if (stack_.back()->is_array()) stack_.back()->push_back(nlohmann::json(v));
        else (*stack_.back())[pending_] = v;
    }
    void pop() {
        if (stack_.size() > 1) stack_.pop_back();
        pending_.clear();
    }

    std::ostream& os_;
    nlohmann::json root_ = nlohmann::json::object();
    std::vector<nlohmann::json*> stack_;
    std::string pending_;
};

}  // namespace malefly
