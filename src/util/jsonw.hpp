#pragma once
// ============================================================================
// util/jsonw.hpp — minimal dependency-free JSON writer for experiment results.
// Not a general JSON library; just enough for flat/nested result documents.
// ============================================================================
#include "core/types.hpp"
#include <cstdio>
#include <ostream>
#include <string>

namespace malefly {

class JsonW {
public:
    explicit JsonW(std::ostream& os) : os_(os) {
        stack_.push_back(false);
        os_ << '{';
    }

    void k(const std::string& key) {
        sep();
        esc(key);
        os_ << ':';
    }

    void obj()  { sep(); stack_.push_back(false); os_ << '{'; }
    void end_obj() { os_ << '}'; pop(); }

    void arr()  { sep(); stack_.push_back(true); os_ << '['; }
    void end_arr() { os_ << ']'; pop(); }

    void val(f64 v) {
        sep();
        char b[40];
        std::snprintf(b, sizeof b, "%.8g", v);
        os_ << b;
        just_valued_ = true;
    }
    void val(f32 v)  { val(static_cast<f64>(v)); }
    void val(u64 v)  { sep(); os_ << v; just_valued_ = true; }
    void val(i64 v)  { sep(); os_ << v; just_valued_ = true; }
    void val(int v)  { sep(); os_ << v; just_valued_ = true; }
    void val(u32 v)  { val(static_cast<u64>(v)); }
    void val(bool v) { sep(); os_ << (v ? "true" : "false"); just_valued_ = true; }
    void val(const std::string& s) { sep(); esc(s); just_valued_ = true; }

    template <class T>
    void kv(const std::string& key, const T& v) { k(key); val(v); }

private:
    void sep() {
        if (just_valued_) os_ << ',';
        just_valued_ = false;
    }
    void pop() {
        stack_.pop_back();
        just_valued_ = true;
    }
    void esc(const std::string& s) {
        os_ << '"';
        for (char c : s) {
            if (c == '"' || c == '\\') os_ << '\\';
            os_ << c;
        }
        os_ << '"';
    }
    std::ostream& os_;
    std::vector<bool> stack_;
    bool just_valued_ = false;
};

}  // namespace malefly
