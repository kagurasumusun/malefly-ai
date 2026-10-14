#ifndef MALEFLY_CODE_CORPUS_HPP
#define MALEFLY_CODE_CORPUS_HPP

#include "core/types.hpp"
#include <string>
#include <vector>

namespace malefly {

struct CorpusCodeItem {
    std::string id;
    std::string category; // "syntax", "algorithm", "bugfix", "optimization"
    std::string prompt;
    std::vector<std::string> choices;
    u32 correct_choice;
    std::string test_eval_code;

    CorpusCodeItem(std::string i, std::string cat, std::string p,
                   std::vector<std::string> c, u32 cc, std::string t)
        : id(std::move(i)), category(std::move(cat)), prompt(std::move(p)),
          choices(std::move(c)), correct_choice(cc), test_eval_code(std::move(t)) {}
};

class CodeCorpus {
public:
    static std::vector<CorpusCodeItem> load_default_corpus() {
        std::vector<CorpusCodeItem> items;
        // Tier 1: Syntax & Grammar
        items.emplace_back(
            "syn_01_for_loop",
            "syntax",
            "for i in range(10)",
            std::vector<std::string>{":\n    print(i)", " do\n    print(i)", " {\n    print(i);", "()\n    print(i)"},
            0,
            "assert 'for i in range(10):' in code"
        );
        items.emplace_back(
            "syn_02_func_def",
            "syntax",
            "def add(a, b)",
            std::vector<std::string>{" -> int:\n    return a + b", ":\n    return a + b", " => a + b", " {\n    return a + b;}"},
            1,
            "assert 'def add(a, b):' in code"
        );
        items.emplace_back(
            "syn_03_dict_comp",
            "syntax",
            "d = {k: v for k, v in ",
            std::vector<std::string>{"items()}", "zip(keys, vals)}", "enumerate(keys)}", "keys.vals()}"},
            1,
            "assert 'zip(keys, vals)' in code"
        );
        items.emplace_back(
            "syn_04_try_except",
            "syntax",
            "try:\n    val = int(s)\n",
            std::vector<std::string>{"except ValueError:\n    val = 0", "catch Exception:\n    val = 0", "finally Error:\n    val = 0", "else ValueError:\n    val = 0"},
            0,
            "assert 'except ValueError:' in code"
        );

        // Tier 2: Algorithmic Logic
        items.emplace_back(
            "algo_01_fibonacci",
            "algorithm",
            "def fib(n):\n    if n <= 1: return n\n    ",
            std::vector<std::string>{"return fib(n-1) + fib(n-2)", "return fib(n) + fib(n-1)", "return fib(n-1) * n", "return 1"},
            0,
            "assert 'fib(n-1) + fib(n-2)' in code"
        );
        items.emplace_back(
            "algo_02_binary_search",
            "algorithm",
            "while low <= high:\n    mid = (low + high) // 2\n    if arr[mid] == target: return mid\n    elif arr[mid] < target:\n        ",
            std::vector<std::string>{"high = mid - 1", "low = mid + 1", "mid += 1", "return -1"},
            1,
            "assert 'low = mid + 1' in code"
        );
        items.emplace_back(
            "algo_03_factorial",
            "algorithm",
            "def fact(n):\n    if n == 0: return 1\n    ",
            std::vector<std::string>{"return n * fact(n - 1)", "return fact(n - 1)", "return n + fact(n - 1)", "return n * fact(n)"},
            0,
            "assert 'n * fact(n - 1)' in code"
        );

        // Tier 3: Bug Fixing & Auto-Repair
        items.emplace_back(
            "bug_01_off_by_one",
            "bugfix",
            "# Buggy: for i in range(len(arr) + 1):\n# Fix:\n",
            std::vector<std::string>{"for i in range(len(arr)):", "for i in range(len(arr) + 2):", "for i in range(1, len(arr) + 1):", "for i in arr:"},
            0,
            "assert 'range(len(arr))' in code"
        );
        items.emplace_back(
            "bug_02_zero_division",
            "bugfix",
            "# Buggy: avg = total / count\n# Fix:\n",
            std::vector<std::string>{"avg = total / (count if count != 0 else 1)", "avg = total / 0", "avg = count / total", "avg = total / (count + 1)"},
            0,
            "assert 'count if count != 0' in code"
        );

        // Tier 4: Code Optimization
        items.emplace_back(
            "opt_01_set_lookup",
            "optimization",
            "# Optimize list lookup in loop:\n# Unoptimized: if x in my_list:\n# Fix:\n",
            std::vector<std::string>{"my_set = set(my_list)\nif x in my_set:", "my_dict = list(my_list)\nif x in my_dict:", "for x in my_list:", "if x in sorted(my_list):"},
            0,
            "assert 'set(my_list)' in code"
        );

        return items;
    }
};

} // namespace malefly

#endif // MALEFLY_CODE_CORPUS_HPP
