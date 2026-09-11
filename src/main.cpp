// ============================================================================
// main.cpp — malefly command-line interface.
//
//   malefly goal1 [--seed N] [--trials N] [--probes N] [--out FILE] [--csv FILE]
//   malefly calib [--seed N] [--pres N] [--kc-quanta F] [--kc-theta F] [--apl-gain F]
// ============================================================================
#include "circuits/mushroom_body.hpp"
#include "experiments/calib.hpp"
#include "experiments/goal1.hpp"
#include "experiments/goal2.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace malefly;

namespace {

bool flag_f32(char** argv, int argc, int& i, const char* name, f32& out) {
    if (std::strcmp(argv[i], name) == 0 && i + 1 < argc) {
        out = std::strtof(argv[i + 1], nullptr);
        ++i;
        return true;
    }
    return false;
}

bool flag_u64(char** argv, int argc, int& i, const char* name, u64& out) {
    if (std::strcmp(argv[i], name) == 0 && i + 1 < argc) {
        out = std::strtoull(argv[i + 1], nullptr, 10);
        ++i;
        return true;
    }
    return false;
}

bool flag_u32(char** argv, int argc, int& i, const char* name, u32& out) {
    u64 v;
    if (flag_u64(argv, argc, i, name, v)) {
        out = static_cast<u32>(v);
        return true;
    }
    return false;
}

bool flag_str(char** argv, int argc, int& i, const char* name, std::string& out) {
    if (std::strcmp(argv[i], name) == 0 && i + 1 < argc) {
        out = argv[i + 1];
        ++i;
        return true;
    }
    return false;
}

void usage() {
    std::fprintf(stderr,
                 "malefly-ai: MaleCNS-derived brain-based AI\n"
                 "usage:\n"
                 "  malefly goal1 [--seed N] [--trials N] [--probes N] "
                 "[--out results/goal1.json] [--csv results/goal1.csv]\n"
                 "  malefly calib [--seed N] [--pres N] [--kc-quanta F] "
                 "[--kc-theta F] [--apl-gain F]\n");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 2;
    }
    const std::string cmd = argv[1];

    if (cmd == "goal1") {
        Goal1Config cfg;
        for (int i = 2; i < argc; ++i) {
            u64 s = cfg.seed;
            u32 t = cfg.n_train_trials, p = cfg.n_probe_each;
            std::string o = cfg.out_json, c = cfg.out_csv;
            if (flag_u64(argv, argc, i, "--seed", s)) cfg.seed = s;
            else if (flag_u32(argv, argc, i, "--trials", t)) cfg.n_train_trials = t;
            else if (flag_u32(argv, argc, i, "--probes", p)) cfg.n_probe_each = p;
            else if (flag_str(argv, argc, i, "--out", o)) cfg.out_json = o;
            else if (flag_str(argv, argc, i, "--csv", c)) cfg.out_csv = c;
            else { usage(); return 2; }
        }
        return run_goal1(cfg);
    }

    if (cmd == "goal2") {
        Goal2Config cfg;
        for (int i = 2; i < argc; ++i) {
            u64 s = cfg.seed;
            u32 t = cfg.n_train_trials, p = cfg.n_probe_each, e = cfg.n_interfere_trials;
            std::string o = cfg.out_json, c = cfg.out_csv;
            if (flag_u64(argv, argc, i, "--seed", s)) cfg.seed = s;
            else if (flag_u32(argv, argc, i, "--trials", t)) cfg.n_train_trials = t;
            else if (flag_u32(argv, argc, i, "--probes", p)) cfg.n_probe_each = p;
            else if (flag_u32(argv, argc, i, "--interfere", e)) cfg.n_interfere_trials = e;
            else if (flag_str(argv, argc, i, "--out", o)) cfg.out_json = o;
            else if (flag_str(argv, argc, i, "--csv", c)) cfg.out_csv = c;
            else { usage(); return 2; }
        }
        return run_goal2(cfg);
    }

    if (cmd == "calib") {
        // defaults derive from the live circuit configs (single source of truth)
        const MushroomBodyConfig mb_def;
        CalibConfig cfg;
        cfg.kc_quanta = mb_def.kc_quanta;
        cfg.kc_thresh = mb_def.kc_v_thresh;
        cfg.apl_gain = mb_def.apl_gain;
        for (int i = 2; i < argc; ++i) {
            u64 s = cfg.seed;
            u32 p = cfg.n_pres;
            f32 q = cfg.kc_quanta, th = cfg.kc_thresh, ag = cfg.apl_gain;
            if (flag_u64(argv, argc, i, "--seed", s)) cfg.seed = s;
            else if (flag_u32(argv, argc, i, "--pres", p)) cfg.n_pres = p;
            else if (flag_f32(argv, argc, i, "--kc-quanta", q)) cfg.kc_quanta = q;
            else if (flag_f32(argv, argc, i, "--kc-theta", th)) cfg.kc_thresh = th;
            else if (flag_f32(argv, argc, i, "--apl-gain", ag)) cfg.apl_gain = ag;
            else { usage(); return 2; }
        }
        return run_calib(cfg);
    }

    usage();
    return 2;
}
