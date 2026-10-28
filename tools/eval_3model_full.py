# -*- coding: utf-8 -*-
"""XRouter 3 模型池完整评估（端侧成本=0）+ LLMRouterBench 全基线对比。

成本模型（2026-10-28 用户修正）：
  端侧部署模型（Qwen3.5-4B / Qwen3-8B / Qwen3-14B / Qwen3.5-9B）成本 = 0（本地推理，电力不计）
  云侧 DeepSeek-V4.1-Flash = 2.0 输入 / 8.0 输出（¥/M token，真实价，存储成本已按此口径）

基线（LLMRouterBench 语义，全 5 模型池）：
  单模型（5）、AVG（宏观平均）、Random（均匀随机期望）、MaxExpert（每数据集最优静态分表）、
  Category Expert / subjectCategory（每 subject 最优静态分表）、Oracle（每题最优上界）

XRouter（3 角色）：4B 难度分类器 + 14B 端侧回答 + DeepSeek 云侧回答；启发式/LLM 两套分类。

产出：3 张图 + results/devllmr_3model_full.csv（逐题启发式/LLM 决策）
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baselines.data_loader import BaselineDataLoader  # noqa: E402
from baselines.device_llm_router import classify_heuristic, TIER_ORDER  # noqa: E402

EDGE_MODELS = {"Qwen3.5-4B", "Qwen3-8B", "Qwen3-14B", "Qwen3.5-9B"}
CLOUD_MODEL = "DeepSeek-V4.1-Flash"
LOCAL_MODEL = "Qwen3-14B"
ALL_MODELS = sorted(EDGE_MODELS | {CLOUD_MODEL})
CAPABILITY = "MEDIUM"
TIER_CN = {"SIMPLE": "简单", "MEDIUM": "中等", "COMPLEX": "复杂",
           "RESEARCH": "研究", "REASONING": "推理"}
C_LOCAL = "#2b7bba"
C_CLOUD = "#e08214"


def _score(r) -> float:
    return 1.0 if (r.score or 0) > 0 else 0.0


def _cost(model: str, rec) -> float:
    return 0.0 if model in EDGE_MODELS else float(rec.cost or 0.0)


def _rank(t: str) -> int:
    return TIER_ORDER.index(t) if t in TIER_ORDER else -1


def _route(tier: str) -> str:
    return LOCAL_MODEL if _rank(tier) <= _rank(CAPABILITY) else CLOUD_MODEL


def load_tiers(path: Path) -> dict:
    tiers = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tiers[(row["dataset_id"], int(row["record_index"]))] = row
    return tiers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/baseline_config_edge.yaml")
    ap.add_argument("--tiers", default="results/xrouter_classifier_tiers.csv")
    args = ap.parse_args()

    loader = BaselineDataLoader(config_path=str(ROOT / args.config))
    records = [r for r in loader.load_all_records() if r.model_name in ALL_MODELS]

    qs = defaultdict(dict)   # (ds, idx) -> {model: record}
    meta = {}                # (ds, idx) -> origin_query
    cat = {}                 # (ds, idx) -> subject
    for r in records:
        key = (r.dataset_id, r.record_index)
        qs[key][r.model_name] = r
        if r.model_name == CLOUD_MODEL:
            # LLMRouterBench 路由基线统一用 record.prompt（完整题干+选项）作为路由输入；
            # origin_query 对 MCQ 只是题干（缺选项），会低估长度/难度。启发式据此重新评估。
            meta[key] = r.prompt or r.origin_query
        if (r.category or "").strip():
            cat[key] = r.category.strip()

    full = {k: v for k, v in qs.items() if len(v) == len(ALL_MODELS)}
    n = len(full)
    keys = sorted(full)
    print(f"[matrix] {n} 题 × {len(ALL_MODELS)} 模型")

    # ---------------- 基线（端侧成本=0，共享 5 模型池） ----------------
    def single(model):
        corr = sum(_score(full[k][model]) for k in keys)
        cost = sum(_cost(model, full[k][model]) for k in keys)
        return corr / n, cost

    singles = {m: single(m) for m in ALL_MODELS}
    avg_acc = sum(singles[m][0] for m in ALL_MODELS) / len(ALL_MODELS)
    avg_cost = sum(singles[m][1] for m in ALL_MODELS) / len(ALL_MODELS)

    # MaxExpert：每数据集挑均值分最高的模型，全量路由到它
    def max_expert():
        by_ds = defaultdict(list)
        for k in keys:
            by_ds[k[0]].append(k)
        corr = cost = 0.0
        for ds, ks in by_ds.items():
            best = max(ALL_MODELS, key=lambda m: sum(_score(full[k][m]) for k in ks) / len(ks))
            corr += sum(_score(full[k][best]) for k in ks)
            cost += sum(_cost(best, full[k][best]) for k in ks)
        return corr / n, cost

    # CategoryExpert：每 (dataset, subject) 挑最优，无 subject 退数据集最优
    def category_expert():
        scores = defaultdict(lambda: defaultdict(list))
        for k in keys:
            dk = k[0]
            c = cat.get(k, "")
            for m in ALL_MODELS:
                scores[(dk, c)][m].append(_score(full[k][m]))
        def best_of(d):
            return max(d, key=lambda m: sum(d[m]) / len(d[m]))
        cat_best = {g: best_of(d) for g, d in scores.items()}
        ds_best = {}
        for (dk, c), d in scores.items():
            for m, s in d.items():
                ds_best.setdefault(dk, defaultdict(list))[m].extend(s)
        ds_best = {dk: best_of(d) for dk, d in ds_best.items()}
        corr = cost = 0.0
        for k in keys:
            m = cat_best.get((k[0], cat.get(k, ""))) or ds_best[k[0]]
            corr += _score(full[k][m])
            cost += _cost(m, full[k][m])
        return corr / n, cost

    # Oracle：每题任意模型答对即对；成本取「答对者中最低成本」，否则全体最低
    def oracle():
        corr = cost = 0.0
        for k in keys:
            recs = [(m, _score(full[k][m]), _cost(m, full[k][m])) for m in ALL_MODELS]
            ok = [r for r in recs if r[1] > 0]
            corr += 1.0 if ok else 0.0
            cost += min(r[2] for r in ok) if ok else min(r[2] for r in recs)
        return corr / n, cost

    me = max_expert()
    ce = category_expert()
    oc = oracle()
    refs = {
        "端侧 only (14B)": singles[LOCAL_MODEL],
        "端侧 only (4B)": singles["Qwen3.5-4B"],
        "端侧 only (8B)": singles["Qwen3-8B"],
        "端侧 only (9B)": singles["Qwen3.5-9B"],
        "云侧 only (DeepSeek)": singles[CLOUD_MODEL],
        "AVG": (avg_acc, avg_cost),
        "Random": (avg_acc, avg_cost),
        "MaxExpert": me,
        "CategoryExpert": ce,
        "Oracle": oc,
    }

    # ---------------- XRouter（启发式 / LLM 两套分类） ----------------
    tiers = load_tiers(ROOT / args.tiers)
    rows = []
    for k in keys:
        text = meta.get(k, "")
        h_tier = classify_heuristic([{"role": "user", "content": text}])
        h_model = _route(h_tier)
        h_correct = _score(full[k][h_model])
        h_cost = _cost(h_model, full[k][h_model])
        t = tiers.get(k)
        if t is not None:
            l_tier, l_src = t["tier"], t["source"]
        else:
            l_tier, l_src = h_tier, "heuristic_fallback"
        l_model = _route(l_tier)
        l_correct = _score(full[k][l_model])
        l_cost = _cost(l_model, full[k][l_model])
        rows.append({"dataset_id": k[0], "record_index": k[1],
                     "heur_tier": h_tier, "llm_tier": l_tier, "llm_source": l_src,
                     "heur_model": h_model, "llm_model": l_model,
                     "heur_correct": int(h_correct), "llm_correct": int(l_correct),
                     "heur_cost": round(h_cost, 6), "llm_cost": round(l_cost, 6)})

    h_acc = sum(r["heur_correct"] for r in rows) / n
    h_cost = sum(r["heur_cost"] for r in rows)
    l_acc = sum(r["llm_correct"] for r in rows) / n
    l_cost = sum(r["llm_cost"] for r in rows)
    h_local = sum(1 for r in rows if r["heur_model"] == LOCAL_MODEL)
    l_local = sum(1 for r in rows if r["llm_model"] == LOCAL_MODEL)

    def _lc_acc(prefix):
        n_l = n_c = 0
        c_l = c_c = 0
        for r in rows:
            if r[prefix + "_model"] == LOCAL_MODEL:
                n_l += 1; c_l += r[prefix + "_correct"]
            else:
                n_c += 1; c_c += r[prefix + "_correct"]
        return (c_l / n_l if n_l else 0.0), (c_c / n_c if n_c else 0.0)

    h_loc_acc, h_cld_acc = _lc_acc("heur")
    l_loc_acc, l_cld_acc = _lc_acc("llm")

    def tier_agg(prefix):
        a = {t: {"local": 0, "cloud": 0, "correct": 0} for t in TIER_ORDER}
        for r in rows:
            t = r[prefix + "_tier"]
            if t not in a:
                a[t] = {"local": 0, "cloud": 0, "correct": 0}
            a[t]["local" if r[prefix + "_model"] == LOCAL_MODEL else "cloud"] += 1
            a[t]["correct"] += r[prefix + "_correct"]
        return a

    h_agg = tier_agg("heur")
    l_agg = tier_agg("llm")

    def acc_bar(a):
        return {t: (a[t]["correct"] / (a[t]["local"] + a[t]["cloud"])
                    if a[t]["local"] + a[t]["cloud"] else 0.0) for t in TIER_ORDER}

    h_bar = acc_bar(h_agg)
    l_bar = acc_bar(l_agg)
    src = defaultdict(int)
    for r in rows:
        src[r["llm_source"]] += 1
    fail_ds = defaultdict(int)
    for r in rows:
        if r["llm_source"] != "llm":
            fail_ds[r["dataset_id"]] += 1

    # ---------------- 打印 ----------------
    print("\n===== 汇总（端侧成本=0，真实还原） =====")
    print(f"{'方案':<22} {'准确率':>8} {'成本(¥)':>9}  备注")
    print(f"{'X-Router 启发式':<22} {h_acc*100:7.2f}% {h_cost:9.4f}  端侧{h_local}/云侧{n-h_local}")
    print(f"{'X-Router LLM分类':<22} {l_acc*100:7.2f}% {l_cost:9.4f}  端侧{l_local}/云侧{n-l_local}")
    for name in ("端侧 only (14B)", "端侧 only (4B)", "端侧 only (8B)", "端侧 only (9B)",
                 "云侧 only (DeepSeek)", "AVG", "Random", "MaxExpert", "CategoryExpert", "Oracle"):
        a, c = refs[name]
        print(f"{name:<22} {a*100:7.2f}% {c:9.4f}")

    print("\n启发式档位分布:", ", ".join(f"{t}:{h_agg[t]['local']+h_agg[t]['cloud']}" for t in TIER_ORDER))
    print("启发式分档准确率:", ", ".join(f"{t}:{h_bar[t]*100:.1f}%" for t in TIER_ORDER))
    print("LLM 档位分布:", ", ".join(f"{t}:{l_agg[t]['local']+l_agg[t]['cloud']}" for t in TIER_ORDER))
    print("LLM 分档准确率:", ", ".join(f"{t}:{l_bar[t]*100:.1f}%" for t in TIER_ORDER))
    print("LLM source:", ", ".join(f"{k}:{v}" for k, v in sorted(src.items())))
    print(f"启发式 端侧准确率 {h_loc_acc*100:.2f}% / 云侧准确率 {h_cld_acc*100:.2f}%")
    print(f"LLM     端侧准确率 {l_loc_acc*100:.2f}% / 云侧准确率 {l_cld_acc*100:.2f}%")
    if fail_ds:
        print("LLM 失败分数据集:", ", ".join(f"{k}:{v}" for k, v in sorted(fail_ds.items(), key=lambda x: -x[1])))
    failed_rows = [r for r in rows if r["llm_source"] != "llm"]
    if failed_rows:
        print(f"LLM 失败题明细（{len(failed_rows)} 题）：")
        for r in failed_rows[:50]:
            print(f"  {r['dataset_id']} idx={r['record_index']} source={r['llm_source']}")
        if len(failed_rows) > 50:
            print(f"  ... 共 {len(failed_rows)} 题")

    # 落盘逐题明细
    detail = ROOT / "results" / "devllmr_3model_full.csv"
    with open(detail, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[saved] {detail}")

    # ---------------- 绘图 ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    x = range(len(TIER_ORDER))

    # 图1：启发式
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    loc = [h_agg[t]["local"] for t in TIER_ORDER]
    cl = [h_agg[t]["cloud"] for t in TIER_ORDER]
    a1.bar(x, loc, color=C_LOCAL, label="端侧 14B（成本0）")
    a1.bar(x, cl, bottom=loc, color=C_CLOUD, label="云侧 DeepSeek")
    for i, t in enumerate(TIER_ORDER):
        a1.text(i, loc[i] + cl[i] + 6, f"{loc[i]+cl[i]}", ha="center", fontsize=9)
    a1.set_xticks(list(x)); a1.set_xticklabels([TIER_CN[t] for t in TIER_ORDER])
    a1.set_ylabel("题数"); a1.set_title("档位分布（颜色=路由去向）"); a1.legend(fontsize=8)
    accs = [h_bar[t] * 100 for t in TIER_ORDER]
    a2.bar(list(x), accs, color="#5b8ff9")
    a2.axhline(h_acc * 100, color="red", ls="--", lw=1.2, label=f"总体 {h_acc*100:.1f}%")
    for i, t in enumerate(TIER_ORDER):
        a2.text(i, accs[i] + 1, f"{accs[i]:.0f}%", ha="center", fontsize=9)
    a2.set_xticks(list(x)); a2.set_xticklabels([TIER_CN[t] for t in TIER_ORDER])
    a2.set_ylim(0, 105); a2.set_ylabel("准确率 (%)"); a2.set_title("分档准确率"); a2.legend(fontsize=8)
    fig.suptitle(f"X-Router 启发式分类（{n} 题，端侧成本=0）：端侧 {h_local} / 云侧 {n-h_local}，"
                 f"准确率 {h_acc*100:.2f}%，成本 ¥{h_cost:.2f}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p1 = ROOT / "results" / "xrouter_heur_classification.png"
    fig.savefig(p1, dpi=130); plt.close(fig)
    print(f"[saved] {p1}")

    # 图2：LLM 分类
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    (a1, a2), (a3, a4) = axs
    loc = [l_agg[t]["local"] for t in TIER_ORDER]
    cl = [l_agg[t]["cloud"] for t in TIER_ORDER]
    a1.bar(x, loc, color=C_LOCAL, label="端侧 14B（成本0）")
    a1.bar(x, cl, bottom=loc, color=C_CLOUD, label="云侧 DeepSeek")
    for i, t in enumerate(TIER_ORDER):
        a1.text(i, loc[i] + cl[i] + 6, f"{loc[i]+cl[i]}", ha="center", fontsize=9)
    a1.set_xticks(list(x)); a1.set_xticklabels([TIER_CN[t] for t in TIER_ORDER])
    a1.set_ylabel("题数"); a1.set_title("档位分布（颜色=路由去向）"); a1.legend(fontsize=8)

    n_llm = src.get("llm", 0); n_fb = sum(v for k, v in src.items() if k != "llm")
    labels = [f"llm {n_llm}"] + ([f"回退/失败 {n_fb}"] if n_fb else [])
    vals = [n_llm] + ([n_fb] if n_fb else [])
    a2.pie(vals, labels=labels, colors=["#5b8ff9", "#f95d6a"],
           autopct="%1.1f%%", startangle=90, textprops={"fontsize": 9})
    if n_fb == 0:
        a2.set_title("分类 source：全部 llm 成功")
        a2.text(0, -1.3, "4B 限流已彻底解决（长超时+低并发重跑）", ha="center", fontsize=8, color="#2a6")
    else:
        a2.set_title("分类 source（LLM 失败→回退启发式）")
        a2.text(0, -1.3, "失败原因见分数据集面板", ha="center", fontsize=8, color="#a33")

    accs = [l_bar[t] * 100 for t in TIER_ORDER]
    a3.bar(list(x), accs, color="#5b8ff9")
    a3.axhline(l_acc * 100, color="red", ls="--", lw=1.2, label=f"总体 {l_acc*100:.1f}%")
    for i, t in enumerate(TIER_ORDER):
        a3.text(i, accs[i] + 1, f"{accs[i]:.0f}%", ha="center", fontsize=9)
    a3.set_xticks(list(x)); a3.set_xticklabels([TIER_CN[t] for t in TIER_ORDER])
    a3.set_ylim(0, 105); a3.set_ylabel("准确率 (%)"); a3.set_title("分档准确率"); a3.legend(fontsize=8)
    if fail_ds:
        dss = sorted(fail_ds.items(), key=lambda kv: -kv[1])
        a4.bar([d[0] for d in dss], [d[1] for d in dss], color="#f95d6a")
        for i, (_, v) in enumerate(dss):
            a4.text(i, v + 1, str(v), ha="center", fontsize=9)
        a4.set_title("LLM 失败题分布（按数据集）"); a4.set_ylabel("题数"); a4.tick_params(axis="x", labelsize=8)
    else:
        a4.text(0.5, 0.5, "无失败题", ha="center", va="center", fontsize=16, color="#2a6")
        a4.set_title("LLM 失败题分布"); a4.axis("off")
    fig.suptitle(f"X-Router LLM 分类（{n} 题，端侧成本=0）：端侧 {l_local} / 云侧 {n-l_local}，"
                 f"准确率 {l_acc*100:.2f}%，成本 ¥{l_cost:.2f}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p2 = ROOT / "results" / "xrouter_llm_classification.png"
    fig.savefig(p2, dpi=130); plt.close(fig)
    print(f"[saved] {p2}")

    # 图3：合并对比（XRouter + 全基线）
    fig, ax = plt.subplots(figsize=(10, 7))
    style = {
        "X-Router 启发式": ("D", "#d81b60", "XRouter"),
        "X-Router LLM分类": ("D", "#c2185b", "XRouter"),
        "端侧 only (14B)": ("o", C_LOCAL, "单模型"),
        "端侧 only (4B)": ("o", "#90caf9", "单模型"),
        "端侧 only (8B)": ("o", "#64b5f6", "单模型"),
        "端侧 only (9B)": ("o", "#42a5f5", "单模型"),
        "云侧 only (DeepSeek)": ("^", C_CLOUD, "单模型"),
        "AVG": ("s", "#9e9e9e", "朴素基线"),
        "Random": ("x", "#757575", "朴素基线"),
        "MaxExpert": ("p", "#2e7d32", "静态路由上界"),
        "CategoryExpert": ("p", "#388e3c", "静态路由上界"),
        "Oracle": ("*", "#00695c", "静态路由上界"),
    }
    pts = [("X-Router 启发式", h_acc, h_cost), ("X-Router LLM分类", l_acc, l_cost)]
    for name in ("端侧 only (14B)", "端侧 only (4B)", "端侧 only (8B)", "端侧 only (9B)",
                 "云侧 only (DeepSeek)", "AVG", "Random", "MaxExpert", "CategoryExpert", "Oracle"):
        pts.append((name, refs[name][0], refs[name][1]))
    for name, acc, cost in pts:
        mk, c, _ = style[name]
        ax.scatter(cost, acc * 100, marker=mk, s=170, color=c, zorder=5,
                   edgecolors="black", linewidths=0.6)
        label = f"{acc*100:.1f}%"
        if name == "云侧 only (DeepSeek)":
            label = f"¥{cost:.2f}, {acc*100:.1f}%"
        ax.annotate(f"{name}\n{label}", (cost, acc * 100), textcoords="offset points",
                    xytext=(8, 6), fontsize=8)
    ax.set_xlabel("总成本 (¥，" + f"{n}" + " 题；端侧=0，云侧 DeepSeek 2.0/8.0)")
    ax.set_ylabel("准确率 (%)")
    ax.set_title("X-Router 启发式 vs LLM vs LLMRouterBench 全基线（端侧成本=0）")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p3 = ROOT / "results" / "xrouter_comparison.png"
    fig.savefig(p3, dpi=130); plt.close(fig)
    print(f"[saved] {p3}")


if __name__ == "__main__":
    main()