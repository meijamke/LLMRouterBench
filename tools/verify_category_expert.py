"""
verify_category_expert.py — 独立交叉验证 Category Expert（subject 粒度静态分表）基线。

验证思路（三重）：
  1. 数据流正确性：category 是否已从 extra_fields 正确加载到 BaselineRecord.category；
     并抽查与源数据 data/{DS}/test.json 的 question->category 映射是否一致。
  2. 独立重算：用 pandas 走一条与 aggregator 完全不同的代码路径，从原始 records
     重新计算 Category Expert 的每数据集准确率 / 总成本 / 样本级准确率。
  3. 对拍：与 BaselineAggregator.to_summary_table() 的 "Category Expert" 行逐列比对。
"""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from baselines import BaselineDataLoader, BaselineAggregator

loader = BaselineDataLoader(config_path=str(REPO / "config" / "baseline_config_edge.yaml"))
records = loader.load_all_records()

# ---------------------------------------------------------------------------
# 0. 建 DataFrame（dataset_key = dataset_id/split）
# ---------------------------------------------------------------------------
rows = [
    dict(dataset_key=f"{r.dataset_id}/{r.split}", dataset_id=r.dataset_id, split=r.split,
         model=r.model_name, idx=r.record_index, category=(r.category or "").strip(),
         origin_query=r.origin_query, score=r.score, cost=r.cost)
    for r in records
]
df = pd.DataFrame(rows)

print("=" * 70)
print("记录总数:", len(df), "| 模型:", sorted(df.model.unique()))
print("=" * 70)

# ---------------------------------------------------------------------------
# 1. category 数据流正确性
# ---------------------------------------------------------------------------
print("\n[1] category 数据流：每条记录是否带上了 subject")
cov = df.groupby("dataset_key").agg(
    records=("category", "size"),
    with_category=("category", lambda s: int((s != "").sum())),
)
cov["ratio"] = (cov["with_category"] / cov["records"]).round(4)
print(cov.to_string())

# 抽查：与源数据 question->category 映射比对（取 mmlupro 全部 + mmlu 前 5 条）
print("\n[1b] 抽查 record.category 与源数据映射一致性")
for ds in ["mmlu", "mmlupro"]:
    q2c = {}
    with open(REPO / "data" / ds / "test.json", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                if d.get("question"):
                    q2c[d["question"]] = d.get("category", "")
    sub = df[df.dataset_id == ds].drop_duplicates("idx")
    mism = 0
    for _, row in sub.iterrows():
        src = q2c.get(row.origin_query, "")
        if (row.category or "") != src:
            mism += 1
    print(f"  {ds}: {len(sub)} 条唯一样本，category 与源数据不一致: {mism}")

# ---------------------------------------------------------------------------
# 2. 独立重算（pandas 路径，不复用 aggregator 的新方法）
# ---------------------------------------------------------------------------
# 2a. 每个 (dataset, category) 里，均值分数最高的模型
cat_mean = df.groupby(["dataset_key", "category", "model"])["score"].mean().reset_index()
cat_best = cat_mean.loc[cat_mean.groupby(["dataset_key", "category"])["score"].idxmax()] \
    [["dataset_key", "category", "model"]].rename(columns={"model": "cat_best_model"})

# 2b. 每个 dataset（跨 subject 池化）里，均值分数最高的模型（兜底）
ds_mean = df.groupby(["dataset_key", "model"])["score"].mean().reset_index()
ds_best = ds_mean.loc[ds_mean.groupby("dataset_key")["score"].idxmax()] \
    [["dataset_key", "model"]].rename(columns={"model": "ds_best_model"})

# 2c. 每道题只路由一次：优先 subject 最优，缺失则 dataset 最优
samples = df.drop_duplicates(["dataset_key", "idx"])[["dataset_key", "idx", "category"]]
samples = samples.merge(cat_best, on=["dataset_key", "category"], how="left")
samples = samples.merge(ds_best, on="dataset_key", how="left")
samples["route_model"] = samples["cat_best_model"].fillna(samples["ds_best_model"])

# 2d. 取路由模型的真实 score / cost
scored = samples.rename(columns={"route_model": "model"}).merge(
    df[["dataset_key", "idx", "model", "score", "cost"]],
    on=["dataset_key", "idx", "model"], how="left", suffixes=("", "_r")
)
scored["correct"] = scored["score"] > 0

indep_acc = (scored.groupby("dataset_key")["correct"].mean() * 100).round(2)
indep_overall = round(scored["correct"].mean() * 100, 4)
indep_cost = scored.groupby("dataset_key")["cost"].sum()

print("\n[2] 独立重算（pandas 路径）的 Category Expert：")
print(indep_acc.to_string())

# ---------------------------------------------------------------------------
# 3. 对拍 aggregator
# ---------------------------------------------------------------------------
agg = BaselineAggregator(records, data_loader=loader)
perf_df, cost_df = agg.to_summary_table(cost_metric="total_cost", random_seed=42)
agg_acc = (perf_df.loc["Category Expert"] * 100).round(2)   # 每数据集列
agg_overall = round(perf_df.loc["Category Expert", "Sample-Avg"] * 100, 4)
agg_cost = cost_df.loc["Category Expert"]                   # 每数据集列（总成本）

print("\n[3] 对拍 BaselineAggregator 的 'Category Expert' 行：")
print(f"{'dataset':<16} {'agg_acc%':>9} {'indep_acc%':>11} {'match':>6} | {'agg_cost':>9} {'indep_cost':>11} {'match':>6}")
all_ok = True
for ds in agg_acc.index:
    if ds in ("Dataset-Avg", "Sample-Avg"):
        continue
    a = agg_acc[ds]; i = indep_acc.get(ds)
    ac = agg_cost[ds]; ic = indep_cost.get(ds)
    ok_acc = (i is not None) and abs(a - i) < 1e-6
    ok_cost = (ic is not None) and abs(ac - ic) < 1e-6
    if not (ok_acc and ok_cost):
        all_ok = False
    print(f"{ds:<16} {a:>9.2f} {i if i is not None else '-':>11} {str(ok_acc):>6} | "
          f"{ac:>9.4f} {ic if ic is not None else '-':>11} {str(ok_cost):>6}")

print(f"\n样本级准确率:  aggregator={agg_overall}%  independent={indep_overall}%  match={abs(agg_overall-indep_overall)<1e-6}")
print(f"总成本合计:    aggregator={agg_cost[agg_cost.index!='Dataset-Sum'].sum():.4f}  "
      f"independent={indep_cost.sum():.4f}")

# ---------------------------------------------------------------------------
# 4. MMLU-Pro 增益来源：subject 粒度到底把哪些题改路由了
# ---------------------------------------------------------------------------
print("\n[4] MMLU-Pro 各 subject 的最优模型表（解释 CE > MaxExpert 的来源）：")
mp = df[df.dataset_key == "mmlupro/test"]
piv = mp.groupby(["category", "model"])["score"].mean().unstack().round(2)
model_cols = list(piv.columns)
piv_out = piv.copy()
piv_out["best_model"] = piv[model_cols].idxmax(axis=1)
piv_out["best_acc"] = piv[model_cols].max(axis=1).round(2)
piv_out = piv_out.sort_values("best_acc", ascending=False)
print(piv_out.to_string())

# 有多少题被 subject 路由改派到非数据集最优模型，带来多少净收益
mp_samples = scored[scored.dataset_key == "mmlupro/test"].copy()
mp_ds_best = ds_best.loc[ds_best.dataset_key == "mmlupro/test", "ds_best_model"].iloc[0]
mp_samples["changed"] = mp_samples["model"] != mp_ds_best  # scored 中 'model' 即路由到的模型
print(f"\n  MMLU-Pro 数据集最优模型 = {mp_ds_best}")
print(f"  被 subject 路由改派的题数: {int(mp_samples['changed'].sum())} / {len(mp_samples)}")
print(f"  改派后 CE 样本级准确率 = {mp_samples['correct'].mean()*100:.2f}%")

ds_best_scores = df[(df.dataset_key == "mmlupro/test") & (df.model == mp_ds_best)] \
    [["idx", "score"]].rename(columns={"score": "ds_best_score"})
mp_samples = mp_samples.merge(ds_best_scores, on="idx", how="left")
mp_samples["ds_best_correct"] = mp_samples["ds_best_score"] > 0
print(f"  若全部用数据集最优模型 = {mp_samples['ds_best_correct'].mean()*100:.2f}%")


def _changed_rows():
    ch = mp_samples[mp_samples["changed"]]
    for _, r in ch.iterrows():
        yield f"    {r['category']:<16} idx={r['idx']:<3} ds_best={mp_ds_best} -> {r['model']}"


print("  具体改派的题目 subject:")
for line in list(_changed_rows()):
    print(line)

print("\n" + ("=" * 70))
print("结论:", "全部对拍一致 ✓" if all_ok else "存在不一致 ✗（见上表）")
print("=" * 70)
