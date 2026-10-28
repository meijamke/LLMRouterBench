# -*- coding: utf-8 -*-
"""XRouter 启发式 vs LLM 分类的逐题回放分析 + 三张图。

产出：
  1) xrouter_heur_classification.png —— 启发式分类：档位分布(端/云) + 分档准确率
  2) xrouter_llm_classification.png   —— LLM 分类：档位分布(端/云) + source/失败 + 分档准确率
  3) xrouter_comparison.png           —— 启发式 / LLM / 各基线 质量-成本对比
  4) devllmr_3model_detail.csv        —— 逐题明细（heuristic/llm 的 tier/side/score）

价格：2026-10-28 查自 https://www.siliconflow.cn/pricing（真实单价）：
  Qwen3.5-4B(分类器)=免费 0/0；Qwen3.5-9B(端侧)=1.5/12；DeepSeek-V4.1-Flash(云侧)=2.0/8.0
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from baselines.data_loader import BaselineDataLoader  # noqa: E402
from baselines.device_llm_router import classify_heuristic, TIER_ORDER  # noqa: E402
from eval_device_llm_router_3model import (  # noqa: E402
    LOCAL_MODEL, CLOUD_MODEL, CLASSIFIER_MODEL,
    load_questions, load_tiers, route_for_tier, references, cost_of,
)

CAPABILITY = "MEDIUM"
TIER_CN = {"SIMPLE": "简单", "MEDIUM": "中等", "COMPLEX": "复杂",
           "RESEARCH": "研究", "REASONING": "推理"}
COLOR_LOCAL = "#2b7bba"
COLOR_CLOUD = "#e08214"


def _score(rec) -> float:
    return 1.0 if rec.score and float(rec.score) > 0 else 0.0


def build_rows(questions, meta, tiers):
    rows = []
    for key in sorted(questions):
        by = questions[key]
        if LOCAL_MODEL not in by or CLOUD_MODEL not in by:
            continue
        text = meta[key]
        # ---- 启发式分类 ----
        h_tier = classify_heuristic([{"role": "user", "content": text}])
        h_model = route_for_tier(h_tier, CAPABILITY)
        h_correct = _score(by[h_model])
        h_cost = cost_of(h_model, int(by[h_model].prompt_tokens or 0),
                         int(by[h_model].completion_tokens or 0))
        # ---- LLM 分类（4B，预打档）----
        t = tiers.get(key)
        if t is not None:
            l_tier, l_src = t["tier"], t["source"]
            cls_cost = cost_of(CLASSIFIER_MODEL, t["prompt_tokens"], t["completion_tokens"])
        else:
            l_tier, l_src = classify_heuristic([{"role": "user", "content": text}]), "heuristic_fallback"
            cls_cost = 0.0
        l_model = route_for_tier(l_tier, CAPABILITY)
        l_correct = _score(by[l_model])
        l_cost = cost_of(l_model, int(by[l_model].prompt_tokens or 0),
                         int(by[l_model].completion_tokens or 0)) + cls_cost
        rows.append({
            "dataset_id": key[0], "record_index": key[1],
            "heur_tier": h_tier, "llm_tier": l_tier, "llm_source": l_src,
            "heur_side": "local" if h_model == LOCAL_MODEL else "cloud",
            "llm_side": "local" if l_model == LOCAL_MODEL else "cloud",
            "heur_correct": int(h_correct), "llm_correct": int(l_correct),
            "heur_cost": round(h_cost, 6), "llm_cost": round(l_cost, 6),
        })
    return rows


def _tier_stack(rows, prefix):
    """档位 -> {local: n, cloud: n, correct: n}；prefix ∈ {heur, llm}。"""
    agg = {t: {"local": 0, "cloud": 0, "correct": 0} for t in TIER_ORDER}
    for r in rows:
        t = r[prefix + "_tier"]
        if t not in agg:
            agg[t] = {"local": 0, "cloud": 0, "correct": 0}
        agg[t][r[prefix + "_side"]] += 1
        agg[t]["correct"] += r[prefix + "_correct"]
    return agg


def _acc_bar(agg):
    return {t: (agg[t]["correct"] / (agg[t]["local"] + agg[t]["cloud"])
                if agg[t]["local"] + agg[t]["cloud"] else 0.0) for t in TIER_ORDER}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/baseline_config_edge.yaml")
    ap.add_argument("--tiers", default="results/xrouter_classifier_tiers.csv")
    args = ap.parse_args()

    loader = BaselineDataLoader(config_path=str(ROOT / args.config))
    records = [r for r in loader.load_all_records() if r.model_name in (LOCAL_MODEL, CLOUD_MODEL)]
    questions, meta = load_questions(records)
    tiers = load_tiers(ROOT / args.tiers)
    refs = references(questions)
    rows = build_rows(questions, meta, tiers)
    n = len(rows)
    print(f"[matrix] {len(questions)} 题 | [tiers] {len(tiers)} 档 | [rows] {n}")

    heur_agg = _tier_stack(rows, "heur")
    llm_agg = _tier_stack(rows, "llm")
    heur_acc = _acc_bar(heur_agg)
    llm_acc = _acc_bar(llm_agg)

    heur_local = sum(1 for r in rows if r["heur_side"] == "local")
    llm_local = sum(1 for r in rows if r["llm_side"] == "local")
    heur_correct = sum(r["heur_correct"] for r in rows)
    llm_correct = sum(r["llm_correct"] for r in rows)
    heur_cost = sum(r["heur_cost"] for r in rows)
    llm_cost = sum(r["llm_cost"] for r in rows)

    # ---- 打印汇总 ----
    print("\n===== 启发式分类（Q1）=====")
    print(f"档位分布: " + ", ".join(f"{t}:{heur_agg[t]['local']+heur_agg[t]['cloud']}" for t in TIER_ORDER))
    print(f"端侧 {heur_local} / 云侧 {n-heur_local} | 准确率 {heur_correct/n*100:.2f}% | 成本 {heur_cost:.4f}")
    print("分档准确率: " + ", ".join(f"{t}:{heur_acc[t]*100:.1f}%" for t in TIER_ORDER))
    print("\n===== LLM 分类（Q2）=====")
    src = defaultdict(int)
    for r in rows:
        src[r["llm_source"]] += 1
    print(f"source: " + ", ".join(f"{k}:{v}" for k, v in sorted(src.items())))
    print(f"档位分布: " + ", ".join(f"{t}:{llm_agg[t]['local']+llm_agg[t]['cloud']}" for t in TIER_ORDER))
    print(f"端侧 {llm_local} / 云侧 {n-llm_local} | 准确率 {llm_correct/n*100:.2f}% | 成本 {llm_cost:.4f}")
    print("分档准确率: " + ", ".join(f"{t}:{llm_acc[t]*100:.1f}%" for t in TIER_ORDER))
    # 失败题分数据集
    fail_ds = defaultdict(int)
    for r in rows:
        if r["llm_source"] == "heuristic_fallback":
            fail_ds[r["dataset_id"]] += 1
    print("LLM 失败(heuristic_fallback) 分数据集: " + ", ".join(f"{k}:{v}" for k, v in sorted(fail_ds.items(), key=lambda x: -x[1])))
    print("\n===== 基线（真实单价）=====")
    for name in ("local", "cloud", "oracle", "random"):
        d = refs[name]
        print(f"{name:<8} acc={d['correct']/d['n']*100:6.2f}%  cost={d['cost']:8.4f}")

    # ---- 落盘逐题明细 ----
    detail = ROOT / "results" / "devllmr_3model_detail.csv"
    with open(detail, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[saved] {detail}")

    # ================= 绘图 =================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    x = range(len(TIER_ORDER))

    # ---- 图1：启发式分类 ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    loc = [heur_agg[t]["local"] for t in TIER_ORDER]
    cl = [heur_agg[t]["cloud"] for t in TIER_ORDER]
    a1.bar(x, loc, color=COLOR_LOCAL, label="端侧 9B")
    a1.bar(x, cl, bottom=loc, color=COLOR_CLOUD, label="云侧 DeepSeek")
    for i, t in enumerate(TIER_ORDER):
        tot = loc[i] + cl[i]
        a1.text(i, tot + 6, f"{tot}", ha="center", fontsize=9)
    a1.set_xticks(list(x)); a1.set_xticklabels([TIER_CN[t] for t in TIER_ORDER])
    a1.set_ylabel("题数"); a1.set_title("档位分布（颜色=路由去向）")
    a1.legend(fontsize=8)
    accs = [heur_acc[t] * 100 for t in TIER_ORDER]
    bars = a2.bar(list(x), accs, color="#5b8ff9")
    a2.axhline(heur_correct / n * 100, color="red", ls="--", lw=1.2,
               label=f"总体 {heur_correct/n*100:.1f}%")
    for i, t in enumerate(TIER_ORDER):
        a2.text(i, accs[i] + 1, f"{accs[i]:.0f}%", ha="center", fontsize=9)
    a2.set_xticks(list(x)); a2.set_xticklabels([TIER_CN[t] for t in TIER_ORDER])
    a2.set_ylim(0, 105); a2.set_ylabel("准确率 (%)"); a2.set_title("分档准确率")
    a2.legend(fontsize=8)
    fig.suptitle(f"X-Router 启发式分类（882 题）：端侧 {heur_local} / 云侧 {n-heur_local}，"
                 f"总体准确率 {heur_correct/n*100:.2f}%，成本 ¥{heur_cost:.2f}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p1 = ROOT / "results" / "xrouter_heur_classification.png"
    fig.savefig(p1, dpi=130); plt.close(fig)
    print(f"[saved] {p1}")

    # ---- 图2：LLM 分类 ----
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    (a1, a2), (a3, a4) = axs
    loc = [llm_agg[t]["local"] for t in TIER_ORDER]
    cl = [llm_agg[t]["cloud"] for t in TIER_ORDER]
    a1.bar(x, loc, color=COLOR_LOCAL, label="端侧 9B")
    a1.bar(x, cl, bottom=loc, color=COLOR_CLOUD, label="云侧 DeepSeek")
    for i, t in enumerate(TIER_ORDER):
        tot = loc[i] + cl[i]
        a1.text(i, tot + 6, f"{tot}", ha="center", fontsize=9)
    a1.set_xticks(list(x)); a1.set_xticklabels([TIER_CN[t] for t in TIER_ORDER])
    a1.set_ylabel("题数"); a1.set_title("档位分布（颜色=路由去向）")
    a1.legend(fontsize=8)

    n_llm = src.get("llm", 0); n_fb = src.get("heuristic_fallback", 0)
    a2.pie([n_llm, n_fb], labels=[f"llm（4B 打档）{n_llm}", f"回退启发式 {n_fb}"],
           colors=["#5b8ff9", "#f95d6a"], autopct="%1.1f%%", startangle=90,
           textprops={"fontsize": 9})
    a2.set_title("分类 source（LLM 失败→回退）")
    a2.text(0, -1.35, "失败原因：硅基流动 4B 限流/慢速（单请求 60–95s）\n"
             "首轮 60s 超时 280 题；重试(120s/并发16)后仍 56 题超时",
            ha="center", fontsize=8, color="#a33")

    accs = [llm_acc[t] * 100 for t in TIER_ORDER]
    a3.bar(list(x), accs, color="#5b8ff9")
    a3.axhline(llm_correct / n * 100, color="red", ls="--", lw=1.2,
               label=f"总体 {llm_correct/n*100:.1f}%")
    for i, t in enumerate(TIER_ORDER):
        a3.text(i, accs[i] + 1, f"{accs[i]:.0f}%", ha="center", fontsize=9)
    a3.set_xticks(list(x)); a3.set_xticklabels([TIER_CN[t] for t in TIER_ORDER])
    a3.set_ylim(0, 105); a3.set_ylabel("准确率 (%)"); a3.set_title("分档准确率")
    a3.legend(fontsize=8)

    dss = sorted(fail_ds.items(), key=lambda kv: -kv[1])
    a4.bar([d[0] for d in dss], [d[1] for d in dss], color="#f95d6a")
    for i, (_, v) in enumerate(dss):
        a4.text(i, v + 1, str(v), ha="center", fontsize=9)
    a4.set_title("LLM 分类失败题分布（按数据集）")
    a4.set_ylabel("题数")
    a4.tick_params(axis="x", labelsize=8)
    fig.suptitle(f"X-Router LLM 分类（882 题）：端侧 {llm_local} / 云侧 {n-llm_local}，"
                 f"总体准确率 {llm_correct/n*100:.2f}%，成本 ¥{llm_cost:.2f}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p2 = ROOT / "results" / "xrouter_llm_classification.png"
    fig.savefig(p2, dpi=130); plt.close(fig)
    print(f"[saved] {p2}")

    # ---- 图3：合并对比 ----
    fig, ax = plt.subplots(figsize=(9, 6.5))
    pts = [
        ("X-Router 启发式(MEDIUM)", heur_correct / n * 100, heur_cost, "o", "#f9a825"),
        ("X-Router LLM分类(MEDIUM)", llm_correct / n * 100, llm_cost, "o", "#d81b60"),
        ("端侧 only (9B)", refs["local"]["correct"] / refs["local"]["n"] * 100, refs["local"]["cost"], "s", COLOR_LOCAL),
        ("云侧 only (DeepSeek)", refs["cloud"]["correct"] / refs["cloud"]["n"] * 100, refs["cloud"]["cost"], "^", COLOR_CLOUD),
        ("Oracle", refs["oracle"]["correct"] / refs["oracle"]["n"] * 100, refs["oracle"]["cost"], "*", "#2e7d32"),
        ("Random", refs["random"]["correct"] / refs["random"]["n"] * 100, refs["random"]["cost"], "x", "#757575"),
    ]
    for name, acc, cost, mk, c in pts:
        ax.scatter(cost, acc, marker=mk, s=160, color=c, zorder=5, edgecolors="black", linewidths=0.6)
        ax.annotate(f"{name}\n(¥{cost:.2f}, {acc:.1f}%)", (cost, acc),
                    textcoords="offset points", xytext=(10, 6), fontsize=8.5)
    ax.set_xlabel("总成本 (¥，882 题，真实单价)")
    ax.set_ylabel("准确率 (%)")
    ax.set_title("X-Router 启发式 vs LLM 分类 vs 基线（真实单价：4B 免费 / 9B 1.5·12 / DeepSeek 2·8）")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p3 = ROOT / "results" / "xrouter_comparison.png"
    fig.savefig(p3, dpi=130); plt.close(fig)
    print(f"[saved] {p3}")


if __name__ == "__main__":
    main()
