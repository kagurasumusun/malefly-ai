#ifndef MALEFLY_CODE_SANDBOX_HPP
#define MALEFLY_CODE_SANDBOX_HPP

#include "core/types.hpp"
#include <string>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <fstream>

namespace malefly {

struct SandboxResult {
    bool passed = false;
    f32 execution_time_ms = 0.0f;
    std::string stdout_output;
    std::string error_message;
    f32 reward = -1.0f;
};

class CodeSandbox {
public:
    static std::string escape_python_raw_string(const std::string& str) {
        std::string res = "\"\"\"";
        for (char c : str) {
            if (c == '"') res += "\\\"";
            else res += c;
        }
        res += "\"\"\"";
        return res;
    }

    // Evaluates combined prompt + candidate code snippet against assertion test code
    static SandboxResult verify_code_python(const std::string& full_code, const std::string& test_assertions) {
        SandboxResult result;

        std::string py_script = "code = " + escape_python_raw_string(full_code) + "\n" +
                                full_code + "\n" +
                                test_assertions + "\n" +
                                "print('SANDBOX_PASS')\n";

        std::string filename = "build/tmp_sandbox_test.py";
        std::ofstream out(filename);
        if (out.is_open()) {
            out << py_script;
            out.close();
        }

        // Run python3 check
        std::string cmd = "python3 " + filename + " > build/tmp_sandbox_out.txt 2>&1";
        int ret = std::system(cmd.c_str());

        std::ifstream in("build/tmp_sandbox_out.txt");
        std::string output;
        if (in.is_open()) {
            std::string line;
            while (std::getline(in, line)) {
                output += line + "\n";
            }
            in.close();
        }

        result.stdout_output = output;
        if (ret == 0 && output.find("SANDBOX_PASS") != std::string::npos) {
            result.passed = true;
            result.reward = +1.0f;
        } else {
            result.passed = false;
            result.error_message = output;
            result.reward = -1.0f;
        }

        return result;
    }
};

} // namespace malefly

#endif // MALEFLY_CODE_SANDBOX_HPP
