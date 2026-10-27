# -*- coding: utf-8 -*-
"""3 模型池 X-Router 的能力档位扫描 + 帕累托图。

对 localCapabilityLevel ∈ {SIMPLE, MEDIUM, COMPLEX, RESEARCH, REASONING, NONE} 逐一
回放路由，得到 (成本, 准确率) 帕累托前沿，与端侧/云侧单模型、Oracle、Random 对照，
并叠加「启发式分类器」曲线，量化端侧小模型分类器（4B）相对启发式的增益。

用法： python tools/eval_3model_sweep.py [--out results/devllmr_3model_sweep.csv]
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from baselines.data_loader import BaselineDataLoader  # noqa: E402
from eval_device_llm_router_3model import (  # noqa: E402
    LOCAL_MODEL, CLOUD_MODEL, load_questions, load_tiers, run, summarize,
    references, llm_classifier, heur_classifier,
)

CAPABILITIES = ["SIMPLE", "MEDIUM", "COMPLEX", "RESEARCH", "REASONING", "NONE"]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/baseline_config_edge.yaml")
    ap.add_argument("--tiers", default="results/xrouter_classifier_tiers.csv")
    ap.add_argument("--out", default="results/devllmr_3model_sweep.csv")
    ap.add_argument("--plot", default="results/devllmr_3model_pareto.png")
    args = ap.parse_args()

    loader = BaselineDataLoader(config_path=str(ROOT / args.config))
    records = [r for r in loader.load_all_records() if r.model_name in (LOCAL_MODEL, CLOUD_MODEL)]
    questions, meta = load_questions(records)
    tiers = load_tiers(ROOT / args.tiers)
    refs = references(questions)
    print(f"[matrix] {len(questions)} 题 | [tiers] {len(tiers)} 题")

    rows = []
    for cap in CAPABILITIES:
        s_llm = summarize(run(questions, meta, tiers, cap, llm_classifier))
        s_heur = summarize(run(questions, meta, tiers, cap, heur_classifier))
        rows.append({"series": "XRouter-4B分类器", "capability": cap, "acc": s_llm["acc"],
                     "cost": s_llm["cost"], "n": s_llm["n"]})
        rows.append({"series": "XRouter-启发式", "capability": cap, "acc": s_heur["acc"],
                     "cost": s_heur["cost"], "n": s_heur["n"]})

    def ref_point(name, d):
        return {"series": name, "capability": "-", "acc": d["correct"] / d["n"],
                "cost": d["cost"], "n": d["n"]}

    ref_rows = [
        ref_point("端侧 only (9B)", refs["local"]),
        ref_point("云侧 only (DeepSeek)", refs["cloud"]),
        ref_point("Oracle", refs["oracle"]),
        ref_point("Random", refs["random"]),
    ]

    # 打印表格
    print(f"\n{'series':<18} {'capability':<12} {'acc':>7} {'cost':>8} {'n':>5}")
    for r in rows:
        print(f"{r['series']:<18} {r['capability']:<12} {r['acc']*100:6.2f}% {r['cost']:8.4f} {r['n']:>5}")
    for r in ref_rows:
        print(f"{r['series']:<18} {r['capability']:<12} {r['acc']*100:6.2f}% {r['cost']:8.4f} {r['n']:>5}")

    # 落盘 CSV（含参考点）
    out = ROOT / args.out
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["series", "capability", "acc", "cost", "n"])
        w.writeheader()
        for r in rows + ref_rows:
            w.writerow(r)
    print(f"\n[saved] {out}")

    # 帕累托图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        fig, ax = plt.subplots(figsize=(8, 6))
        for series in ("XRouter-4B分类器", "XRouter-启发式"):
            pts = [r for r in rows if r["series"] == series]
            xs = [r["cost"] for r in pts]
            ys = [r["acc"] * 100 for r in pts]
            ax.plot(xs, ys, marker="o", label=series)
            for r in pts:
                ax.annotate(r["capability"], (r["cost"], r["acc"] * 100),
                            textcoords="offset points", xytext=(6, -4), fontsize=7)
        markers = {"端侧 only (9B)": "s", "云侧 only (DeepSeek)": "^", "Oracle": "*", "Random": "x"}
        for r in ref_rows:
            ax.scatter(r["cost"], r["acc"] * 100, marker=markers.get(r["series"], "o"),
                       s=90, label=r["series"], zorder=5)
        ax.set_xlabel("总成本 (¥，含分类器调用)")
        ax.set_ylabel("准确率 (%)")
        ax.set_title("X-Router 端侧路由：质量-成本帕累托（3 模型池，硅基流动真实 API）")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        plot_path = ROOT / args.plot
        fig.savefig(plot_path, dpi=130)
        print(f"[saved] {plot_path}")
    except Exception as e:
        print(f"[warn] 绘图失败: {e}")


if __name__ == "__main__":
    main()