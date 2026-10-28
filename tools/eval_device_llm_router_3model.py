# -*- coding: utf-8 -*-
"""3 模型池的 X-Router 端侧路由评估（本机 PC + 硅基流动真实 API）。

三种角色（对齐 MiniPC XRouter 语义）：
  - 分类器（端侧小模型 Qwen3.5-4B）：判断难易档位，source=llm
         （由 tools/run_xrouter_classifier.py 预打档，落盘 tiers CSV）
  - 端侧大模型 Qwen3.5-9B：本地回答    —— tier <= localCapabilityLevel(MEDIUM)
  - 云侧模型 DeepSeek-V4.1-Flash：云端回答 —— tier >  MEDIUM

成本 = 分类器调用成本 + 被路由回答模型的成本（token × 硅基流动单价，分钟）。
质量 = 被路由回答模型的得分（复用 results/bench 全矩阵指令离线回放）。

对照：端侧单模型 / 云侧单模型 / Oracle / Random；以及「启发式分类器」
（同名模型池、同规则，仅把分类器换成 classify_heuristic，量化端侧小模型分类器的增益）。

用法：
    python tools/eval_device_llm_router_3model.py [--capability MEDIUM]
        [--tiers results/xrouter_classifier_tiers.csv] [--out results/devllmr_3model.csv]
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baselines.data_loader import BaselineDataLoader  # noqa: E402
from baselines.device_llm_router import (  # noqa: E402
    classify_heuristic, tier_rank, TIER_ORDER,
)

LOCAL_MODEL = "Qwen3.5-9B"            # 端侧大模型（本地回答）
CLOUD_MODEL = "DeepSeek-V4.1-Flash"   # 云侧模型
CLASSIFIER_MODEL = "Qwen3.5-4B"       # 端侧小模型（难度裁判）

# 成本模型（2026-10-28 用户修正）：端侧部署模型本地推理、成本=0（电力不计）；云侧按真实单价。
PRICES = {
    LOCAL_MODEL: {"in": 0.00, "out": 0.00},        # Qwen3.5-9B（端侧，成本0）
    CLOUD_MODEL: {"in": 2.00, "out": 8.00},        # DeepSeek-V4.1-Flash（云侧，0-2点/8-24点档）
    CLASSIFIER_MODEL: {"in": 0.00, "out": 0.00},   # Qwen3.5-4B（端侧，成本0）
}


def cost_of(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p = PRICES[model]
    return prompt_tokens / 1e6 * p["in"] + completion_tokens / 1e6 * p["out"]


def load_questions(records):
    """(dataset, index) -> {model: record}，以及题目文本。"""
    questions = defaultdict(dict)
    meta = {}
    pool = (LOCAL_MODEL, CLOUD_MODEL)
    for r in records:
        if r.model_name not in pool:
            continue
        key = (r.dataset_id, r.record_index)
        questions[key][r.model_name] = r
        meta.setdefault(key, r.prompt or r.origin_query)
    return questions, meta


def load_tiers(path: Path):
    """(dataset, index) -> dict(tier, source, prompt_tokens, completion_tokens)。"""
    tiers = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tiers[(row["dataset_id"], int(row["record_index"]))] = {
                "tier": row["tier"], "source": row["source"],
                "prompt_tokens": int(row["prompt_tokens"]),
                "completion_tokens": int(row["completion_tokens"]),
            }
    return tiers


def route_for_tier(tier: str, capability: str) -> str:
    """XRouter.decideByTier：tier <= capability → local；否则 cloud。"""
    if capability != "NONE" and tier_rank(tier) <= tier_rank(capability):
        return LOCAL_MODEL
    return CLOUD_MODEL


def run(questions, meta, tiers, capability, classifier_fn):
    """回放路由。classifier_fn(text) -> (tier, source, c_pt, c_ct)。"""
    per_question = []
    for key in sorted(questions):
        by_model = questions[key]
        if LOCAL_MODEL not in by_model or CLOUD_MODEL not in by_model:
            continue
        text = meta[key]
        tier, source, c_pt, c_ct = classifier_fn(text, key, tiers)
        model = route_for_tier(tier, capability)
        rec = by_model[model]
        correct = 1.0 if rec.score and float(rec.score) > 0 else 0.0
        ans_cost = cost_of(model, int(rec.prompt_tokens or 0), int(rec.completion_tokens or 0))
        cls_cost = cost_of(CLASSIFIER_MODEL, c_pt, c_ct)
        per_question.append({
            "dataset_id": key[0], "record_index": key[1], "tier": tier, "source": source,
            "routed_model": model, "side": "local" if model == LOCAL_MODEL else "cloud",
            "score": correct, "cost": ans_cost + cls_cost,
            "answer_cost": ans_cost, "classifier_cost": cls_cost,
        })
    return per_question


def llm_classifier(text, key, tiers):
    """分类器 = 预打档的 Qwen3.5-4B（source=llm）；缺档回退启发式。"""
    if key in tiers:
        t = tiers[key]
        return t["tier"], t["source"], t["prompt_tokens"], t["completion_tokens"]
    h = classify_heuristic([{"role": "user", "content": text}])
    return h, "heuristic_fallback", 0, 0


def heur_classifier(text, key, tiers):
    """分类器 = 启发式（对照基线，零成本）。"""
    return classify_heuristic([{"role": "user", "content": text}]), "heuristic", 0, 0


def summarize(per_question):
    n = len(per_question)
    if n == 0:
        return {"n": 0, "acc": 0.0, "cost": 0.0, "local": 0, "cloud": 0}
    acc = sum(q["score"] for q in per_question) / n
    cost = sum(q["cost"] for q in per_question)
    local = sum(1 for q in per_question if q["side"] == "local")
    return {"n": n, "acc": acc, "cost": cost, "local": local, "cloud": n - local}


def references(questions):
    """端侧/云侧单模型、Oracle、Random（用重算后的成本）。"""
    agg = defaultdict(lambda: {"n": 0, "correct": 0.0, "cost": 0.0})
    for key, by_model in questions.items():
        if LOCAL_MODEL not in by_model or CLOUD_MODEL not in by_model:
            continue
        lr, cr = by_model[LOCAL_MODEL], by_model[CLOUD_MODEL]
        ls = 1.0 if lr.score and float(lr.score) > 0 else 0.0
        cs = 1.0 if cr.score and float(cr.score) > 0 else 0.0
        lc = cost_of(LOCAL_MODEL, int(lr.prompt_tokens or 0), int(lr.completion_tokens or 0))
        cc = cost_of(CLOUD_MODEL, int(cr.prompt_tokens or 0), int(cr.completion_tokens or 0))

        agg["n"]["n"] += 1
        agg["local"]["n"] += 1; agg["local"]["correct"] += ls; agg["local"]["cost"] += lc
        agg["cloud"]["n"] += 1; agg["cloud"]["correct"] += cs; agg["cloud"]["cost"] += cc
        if cs > ls or (cs == ls and cc < lc):
            agg["oracle"]["correct"] += cs; agg["oracle"]["cost"] += cc
        else:
            agg["oracle"]["correct"] += ls; agg["oracle"]["cost"] += lc
        agg["oracle"]["n"] += 1
        agg["random"]["correct"] += 0.5 * ls + 0.5 * cs
        agg["random"]["cost"] += 0.5 * lc + 0.5 * cc
        agg["random"]["n"] += 1
    return agg


def fmt(name, d):
    n = d["n"]
    acc = d["correct"] / n if n else 0.0
    return f"{name:<30} acc={acc*100:6.2f}%  cost={d['cost']:8.4f}  n={n}"


def main() -> None:
    ap = argparse.ArgumentParser(description="3 模型池 X-Router 端侧路由评估")
    ap.add_argument("--capability", default="MEDIUM")
    ap.add_argument("--config", default="config/baseline_config_edge.yaml")
    ap.add_argument("--tiers", default="results/xrouter_classifier_tiers.csv")
    ap.add_argument("--out", default="results/devllmr_3model.csv")
    args = ap.parse_args()

    loader = BaselineDataLoader(config_path=str(ROOT / args.config))
    records = loader.load_all_records()
    pool = {LOCAL_MODEL, CLOUD_MODEL}
    records = [r for r in records if r.model_name in pool]
    print(f"[load] {len(records)} answer records (pool = {LOCAL_MODEL}, {CLOUD_MODEL})")
    questions, meta = load_questions(records)
    print(f"[matrix] {len(questions)} 道题（两回答模型齐全）")

    tiers = load_tiers(ROOT / args.tiers)
    print(f"[tiers] {len(tiers)} 题有分类器档位")

    # LLM 分类器（端侧小模型 4B）
    pq_llm = run(questions, meta, tiers, args.capability, llm_classifier)
    # 启发式分类器（对照基线）
    pq_heur = run(questions, meta, tiers, args.capability, heur_classifier)

    refs = references(questions)
    s_llm = summarize(pq_llm)
    s_heur = summarize(pq_heur)

    print("\n========== 总览：质量-成本（6 数据集合并，成本含分类器调用） ==========")
    print(f"X-Router 4B分类器({args.capability})   acc={s_llm['acc']*100:6.2f}%  "
          f"cost={s_llm['cost']:8.4f}  n={s_llm['n']}  端侧{s_llm['local']}/云侧{s_llm['cloud']}")
    print(f"X-Router 启发式分类器({args.capability}) acc={s_heur['acc']*100:6.2f}%  "
          f"cost={s_heur['cost']:8.4f}  n={s_heur['n']}")
    print(fmt("端侧 only (Qwen3.5-9B)", refs["local"]))
    print(fmt("云侧 only (DeepSeek-V4.1-Flash)", refs["cloud"]))
    print(fmt("Oracle (每题最优上界)", refs["oracle"]))
    print(fmt("Random (0.5/0.5 期望)", refs["random"]))

    cloud = refs["cloud"]
    cloud_acc = cloud["correct"] / cloud["n"]
    cloud_cost = cloud["cost"]
    print(f"\n相对云侧单模型：Δacc={s_llm['acc']-cloud_acc:+.2%}  成本节省={cloud_cost-s_llm['cost']:+.4f} "
          f"({(1-s_llm['cost']/cloud_cost)*100:.1f}%)")

    # 分数据集（LLM 分类器）
    ds_stats = defaultdict(lambda: {"n": 0, "correct": 0, "cost": 0.0, "local": 0, "cloud": 0, "tiers": defaultdict(int)})
    for q in pq_llm:
        s = ds_stats[q["dataset_id"]]
        s["n"] += 1; s["correct"] += q["score"]; s["cost"] += q["cost"]
        s["local" if q["side"] == "local" else "cloud"] += 1
        s["tiers"][q["tier"]] += 1
    print("\n========== X-Router(4B分类器) 分数据集 ==========")
    print(f"{'dataset':<10} {'n':>4} {'acc':>7} {'cost':>8} {'local':>5} {'cloud':>5}  档位分布")
    for ds in sorted(ds_stats):
        s = ds_stats[ds]
        acc = s["correct"] / s["n"]
        t = ", ".join(f"{k}:{v}" for k, v in sorted(s["tiers"].items(), key=lambda x: -x[1]))
        print(f"{ds:<10} {s['n']:>4} {acc*100:6.2f}% {s['cost']:8.4f} {s['local']:>5} {s['cloud']:>5}  {t}")

    # 全局档位分布 + source 分布
    tier_tot = defaultdict(int); src_tot = defaultdict(int)
    for q in pq_llm:
        tier_tot[q["tier"]] += 1; src_tot[q["source"]] += 1
    print("\n4B分类器档位分布：", ", ".join(f"{t}:{tier_tot[t]}" for t in TIER_ORDER))
    print("分类器 source 分布：", ", ".join(f"{k}:{v}" for k, v in sorted(src_tot.items())))

    # 落盘
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset_id", "record_index", "tier", "source",
                                           "routed_model", "side", "score", "cost",
                                           "answer_cost", "classifier_cost"])
        w.writeheader()
        for q in sorted(pq_llm, key=lambda x: (x["dataset_id"], x["record_index"])):
            w.writerow(q)
    print(f"\n[saved] 逐题决策明细 -> {out}")


if __name__ == "__main__":
    main()