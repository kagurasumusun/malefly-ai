#ifndef MALEFLY_CODE_ENV_HPP
#define MALEFLY_CODE_ENV_HPP

#include "core/types.hpp"
#include "core/rng.hpp"
#include <string>
#include <vector>
#include <functional>

namespace malefly {

struct CodeTask {
    std::string task_id;
    std::string prompt_code;
    std::vector<std::string> candidate_completions;
    u32 correct_candidate_idx;
    std::string description;
};

class CodeEnv {
public:
    explicit CodeEnv(Rng& rng) : rng_(rng) {
        init_tasks();
    }

    void init_tasks() {
        tasks_ = {
            {
                "task_01_loop_syntax",
                "for i in range(10)",
                {":", " do", " {", "()"},
                0,
                "Syntax completion for Python loop statement"
            },
            {
                "task_02_return_statement",
                "def add(a, b):\n    ",
                {"pass", "return a + b", "return a - b", "print(a)"},
                1,
                "Function return logic for addition"
            },
            {
                "task_03_if_condition",
                "if x > 0:\n    return True\nel",
                {"if x == 0:", "se True:", "se:", "se pass"},
                2,
                "Syntax completion for else branch"
            },
            {
                "task_04_list_comprehension",
                "squares = [x*x for x in ",
                {"10]", "list(10)]", "for 10]", "range(10)]"},
                3,
                "List comprehension for iteration"
            },
            {
                "task_05_equality_check",
                "if val ",
                {"= 0:", "== 0:", "is equal 0:", "=== 0:"},
                1,
                "Equality comparison in Python"
            }
        };
    }

    size_t num_tasks() const { return tasks_.size(); }

    const CodeTask& get_task(size_t idx) const {
        return tasks_[idx % tasks_.size()];
    }

    CodeTask sample_task() {
        u32 idx = static_cast<u32>(rng_.next_u64()) % tasks_.size();
        return tasks_[idx];
    }

    // Evaluates a candidate choice index for a task
    f32 evaluate(const CodeTask& task, u32 chosen_candidate_idx) {
        if (chosen_candidate_idx == task.correct_candidate_idx) {
            return +1.0f; // Positive reward for correct completion
        } else {
            return -1.0f; // Negative reward for error
        }
    }

private:
    Rng rng_;
    std::vector<CodeTask> tasks_;
};

} // namespace malefly

#endif // MALEFLY_CODE_ENV_HPP
