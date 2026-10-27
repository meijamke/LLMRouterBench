# -*- coding: utf-8 -*-
"""用 LLMRouterBench 离线回放 DeviceLLMRouter（MiniPC PR#11 XRouter 内核）。

评估对象：DeviceLLMRouter —— 复杂度 5 档路由（启发式），
          SIMPLE/MEDIUM → 端侧 Qwen3.5-4B；COMPLEX/RESEARCH/REASONING → 云侧 DeepSeek-V4.1-Flash。

方法：复用 LLMRouterBench 已采集的「题目 × 模型」全矩阵（results/bench），
      对每道题回放路由决策，查出被路由模型的 score/cost，汇总质量与成本，
      并与 Oracle（每题最优上界）/ 单模型 / Random 对照，得到成本-质量帕累托位置。

用法：
    python tools/eval_device_llm_router.py [--capability MEDIUM] [--out results/device_llm_router.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baselines.data_loader import BaselineDataLoader  # noqa: E402
from baselines.device_llm_router import DeviceLLMRouter  # noqa: E402

LOCAL_MODEL = "Qwen3.5-4B"          # 端侧模型
CLOUD_MODEL = "DeepSeek-V4.1-Flash"  # 云侧模型


def load_questions(records):
    """按 (dataset_id, record_index) 分组，得到 题目 -> {model: record}。"""
    questions = defaultdict(dict)
    meta = {}
    for r in records:
        if r.model_name not in (LOCAL_MODEL, CLOUD_MODEL):
            continue
        key = (r.dataset_id, r.record_index)
        questions[key][r.model_name] = r
        meta.setdefault(key, r.origin_query or r.prompt)
    return questions, meta


def route_model(router: DeviceLLMRouter, question_text: str) -> str:
    """DeviceLLMRouter 决策 → LLMRouterBench 模型名。"""
    decision = router.route_question(question_text)
    return LOCAL_MODEL if decision["side"] == "local" else CLOUD_MODEL


def evaluate(router: DeviceLLMRouter, questions, meta):
    """回放路由，返回按题目聚合的结果 + 汇总统计。"""
    per_question = []  # 每题一条
    stats = defaultdict(lambda: {"n": 0, "correct": 0, "cost": 0.0,
                                 "local": 0, "cloud": 0, "tiers": defaultdict(int)})

    for key, by_model in sorted(questions.items()):
        ds = key[0]
        # 跳过缺失两模型之一的不完整题目
        if LOCAL_MODEL not in by_model or CLOUD_MODEL not in by_model:
            continue
        text = meta[key]
        decision = router.route_question(text)
        side = decision["side"]
        model = LOCAL_MODEL if side == "local" else CLOUD_MODEL
        rec = by_model[model]
        correct = 1.0 if rec.score and float(rec.score) > 0 else 0.0
        cost = float(rec.cost or 0.0)

        per_question.append({
            "dataset_id": ds, "record_index": key[1], "origin_query": text,
            "tier": decision["tier"], "side": side, "routed_model": model,
            "score": correct, "cost": cost, "reason": decision["reason"],
        })
        s = stats[ds]
        s["n"] += 1
        s["correct"] += correct
        s["cost"] += cost
        s["local" if side == "local" else "cloud"] += 1
        s["tiers"][decision["tier"]] += 1

    return per_question, stats


def references(questions):
    """从全矩阵计算对照：单模型 / Oracle / Random。"""
    agg = {"n": 0}
    local = {"n": 0, "correct": 0.0, "cost": 0.0}
    cloud = {"n": 0, "correct": 0.0, "cost": 0.0}
    oracle = {"n": 0, "correct": 0.0, "cost": 0.0}
    random = {"n": 0, "correct": 0.0, "cost": 0.0}

    for key, by_model in questions.items():
        if LOCAL_MODEL not in by_model or CLOUD_MODEL not in by_model:
            continue
        lr, cr = by_model[LOCAL_MODEL], by_model[CLOUD_MODEL]
        ls = 1.0 if lr.score and float(lr.score) > 0 else 0.0
        cs = 1.0 if cr.score and float(cr.score) > 0 else 0.0
        lc, cc = float(lr.cost or 0.0), float(cr.cost or 0.0)

        agg["n"] += 1
        local["n"] += 1
        local["correct"] += ls
        local["cost"] += lc
        cloud["n"] += 1
        cloud["correct"] += cs
        cloud["cost"] += cc
        # Oracle：每题取分高者（平分取更便宜者）
        if cs > ls or (cs == ls and cc < lc):
            oracle["correct"] += cs
            oracle["cost"] += cc
        else:
            oracle["correct"] += ls
            oracle["cost"] += lc
        oracle["n"] += 1
        # Random：0.5/0.5 期望
        random["correct"] += 0.5 * ls + 0.5 * cs
        random["cost"] += 0.5 * lc + 0.5 * cc
        random["n"] += 1

    return {"local": local, "cloud": cloud, "oracle": oracle, "random": random, "n": agg["n"]}


def summarize(per_question):
    n = len(per_question)
    if n == 0:
        return {}
    acc = sum(q["score"] for q in per_question) / n
    cost = sum(q["cost"] for q in per_question)
    local_n = sum(1 for q in per_question if q["side"] == "local")
    return {"n": n, "acc": acc, "cost": cost, "local": local_n, "cloud": n - local_n}


def main() -> None:
    parser = argparse.ArgumentParser(description="离线回放评估 DeviceLLMRouter")
    parser.add_argument("--capability", default="MEDIUM", help="localCapabilityLevel (SIMPLE/MEDIUM/COMPLEX/RESEARCH/REASONING/NONE)")
    parser.add_argument("--config", default="config/baseline_config_edge.yaml")
    parser.add_argument("--out", default="results/device_llm_router.csv")
    args = parser.parse_args()

    loader = BaselineDataLoader(config_path=str(ROOT / args.config))
    records = loader.load_all_records()
    pool = {LOCAL_MODEL, CLOUD_MODEL}
    records = [r for r in records if r.model_name in pool]
    print(f"[load] {len(records)} records (pool = {LOCAL_MODEL}, {CLOUD_MODEL})")

    questions, meta = load_questions(records)
    print(f"[matrix] {len(questions)} 道题（两模型齐全）")

    router = DeviceLLMRouter(local_capability_level=args.capability,
                             tier_models={"COMPLEX": CLOUD_MODEL})
    per_question, stats = evaluate(router, questions, meta)
    refs = references(questions)
    total = summarize(per_question)

    # ---- 总览对照表 ----
    def fmt(name, d):
        return (f"{name:<28} acc={d['acc']*100:6.2f}%  "
                f"cost={d['cost']:8.4f}  n={d['n']}")

    print("\n========== 总览：质量-成本（6 数据集合并）==========")
    print(f"DeviceLLMRouter({args.capability})  " + (f"acc={total['acc']*100:6.2f}%  cost={total['cost']:8.4f}  "
          f"n={total['n']}  端侧 {total['local']} 题 / 云侧 {total['cloud']} 题"))
    print(fmt("端侧 only (Qwen3.5-4B)", summarize_ref(refs["local"])))
    print(fmt("云侧 only (DeepSeek-V4.1-Flash)", summarize_ref(refs["cloud"])))
    print(fmt("Oracle (每题最优上界)", summarize_ref(refs["oracle"])))
    print(fmt("Random (0.5/0.5 期望)", summarize_ref(refs["random"])))

    # 相对「云侧单模型」的节省与质量差
    cloud_acc = refs["cloud"]["correct"] / refs["cloud"]["n"]
    cloud_cost = refs["cloud"]["cost"]
    print(f"\n相对云侧单模型：Δacc={total['acc']-cloud_acc:+.2%}  "
          f"成本节省={cloud_cost-total['cost']:+.4f}")
    if cloud_cost > 0:
        print(f"  → 省掉 {(1-total['cost']/cloud_cost)*100:.1f}% 云侧花销")
    else:
        print("  → 云侧成本为 0，无法计算节省率")

    # ---- 分数据集 ----
    print("\n========== DeviceLLMRouter 分数据集 ==========")
    print(f"{'dataset':<10} {'n':>4} {'acc':>7} {'cost':>8} {'local':>5} {'cloud':>5}  档位分布")
    for ds in sorted(stats):
        s = stats[ds]
        acc = s["correct"] / s["n"]
        tiers = ", ".join(f"{t}:{c}" for t, c in sorted(s["tiers"].items(), key=lambda x: -x[1]))
        print(f"{ds:<10} {s['n']:>4} {acc*100:6.2f}% {s['cost']:8.4f} {s['local']:>5} {s['cloud']:>5}  {tiers}")

    # ---- 全局档位分布 ----
    tier_tot = defaultdict(int)
    for q in per_question:
        tier_tot[q["tier"]] += 1
    print("\n全局档位分布：", ", ".join(f"{t}:{tier_tot[t]}" for t in
                              ["SIMPLE", "MEDIUM", "COMPLEX", "RESEARCH", "REASONING"]))

    # ---- 落盘 CSV ----
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset_id", "record_index", "tier", "side",
                                               "routed_model", "score", "cost", "reason"])
        writer.writeheader()
        for q in per_question:
            row = dict(q)
            row.pop("origin_query", None)
            writer.writerow(row)
    print(f"\n[saved] 逐题决策明细 -> {out}")


def summarize_ref(d):
    n = d["n"]
    return {"n": n, "acc": d["correct"] / n if n else 0.0, "cost": d["cost"],
            "local": 0, "cloud": 0}


if __name__ == "__main__":
    main()
