#!/usr/bin/env python3
# ============================================================================
# tools/plot_results.py — figures from measured results (results/*.json|csv).
# Usage: python3 tools/plot_results.py   ->  results/figures/*.png
# ============================================================================
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = "results/figures"


def fig_goal1() -> None:
    if not os.path.exists("results/goal1.csv"):
        return
    rows = list(csv.DictReader(open("results/goal1.csv")))
    blocks, fracs = [], []
    for b0 in range(0, len(rows), 20):
        ch = rows[b0:b0 + 20]
        blocks.append(b0 // 20)
        fracs.append(sum(int(r["correct"]) for r in ch) / len(ch))
    g = [(int(r["action_approach"])) for r in rows if r["is_good"] == "1"]
    b = [(int(r["action_approach"])) for r in rows if r["is_good"] == "0"]
    ng, nb = len(rows) and sum(1 for r in rows if r["is_good"] == "1"), sum(
        1 for r in rows if r["is_good"] == "0")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(blocks, fracs, "o-", color="#1a6faf")
    ax1.axhline(0.5, ls="--", c="gray", lw=1)
    ax1.axhline(0.8, ls=":", c="#2a8f3f", lw=1)
    ax1.set_ylim(0, 1.02)
    ax1.set_xlabel("training block (20 trials each)")
    ax1.set_ylabel("fraction correct")
    ax1.set_title("Goal 1: learning curve (MaleCNS MB circuit)")

    ax2.plot([t for t, x in enumerate(g)], [None] * 0 or
             [sum(g[max(0, i - 10):i + 1]) / len(g[max(0, i - 10):i + 1])
              for i in range(len(g))], "-", color="#2a8f3f", label="odor A (+)")
    ax2.plot([sum(b[max(0, i - 10):i + 1]) / len(b[max(0, i - 10):i + 1])
              for i in range(len(b))], "-", color="#b03030", label="odor B (-)")
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("trial (per-odor index, moving avg 10)")
    ax2.set_ylabel("p(approach | odor)")
    ax2.legend()
    ax2.set_title(f"approach rate: A vs B  (n_good={ng}, n_bad={nb})")
    fig.tight_layout()
    fig.savefig(f"{FIG}/goal1_learning.png", dpi=140)
    plt.close(fig)


def fig_transfer() -> None:
    if not os.path.exists("results/goal1.json"):
        return
    j = json.load(open("results/goal1.json"))
    pts = j["transfer"]["points"]
    x = [p["jaccard_to_A"] - p["jaccard_to_B"] for p in pts]
    y = [p["p_approach_post"] for p in pts]
    names = [p["odor"] for p in pts]
    r = j["transfer"]["pearson_r_p_post_vs_jacA_minus_jacB"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(x, y, s=90, color="#1a6faf", zorder=3)
    for xi, yi, n in zip(x, y, names):
        ax.annotate(n.replace("_", "\n"), (xi, yi), textcoords="offset points",
                    xytext=(8, -4), fontsize=8)
    ax.axhline(0.5, ls="--", c="gray", lw=1)
    ax.axvline(0.0, ls="--", c="gray", lw=1)
    ax.set_xlabel("KC pattern overlap: jaccard(X,A) - jaccard(X,B)")
    ax.set_ylabel("p(approach | X) after training")
    ax.set_title(f"structural generalization (r = {r:.3f})")
    fig.tight_layout()
    fig.savefig(f"{FIG}/goal1_transfer.png", dpi=140)
    plt.close(fig)


def fig_stm() -> None:
    if not os.path.exists("results/goal2_stm.csv"):
        return
    rows = list(csv.DictReader(open("results/goal2_stm.csv")))
    d = [int(r["delay_ms"]) for r in rows]
    tr = [float(r["acc_trace"]) for r in rows]
    ct = [float(r["acc_count"]) for r in rows]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(d, tr, "o-", color="#7a3fa8", label="circuit state (eligibility trace)")
    ax.plot(d, ct, "s--", color="gray", label="software counters (contrast)")
    ax.axhline(0.5, ls="--", c="#999", lw=1)
    ax.set_xscale("symlog", linthresh=100)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("odor-off delay D (ms)")
    ax.set_ylabel("delayed-choice accuracy")
    ax.set_title("Goal 2: working memory from neural state (decays)\nvs software state (does not)")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIG}/goal2_stm.png", dpi=140)
    plt.close(fig)


def main() -> None:
    os.makedirs(FIG, exist_ok=True)
    fig_goal1()
    fig_transfer()
    fig_stm()
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
