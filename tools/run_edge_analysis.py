"""
run_edge_analysis.py — 在仓库根目录运行端侧路由 baseline 汇总分析。

产出：性能表(%) 与 成本表(¥)，包含新增的 "Category Expert"（subject 粒度静态分表）行。

用法（仓库根目录）：
    python tools/run_edge_analysis.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from baselines import BaselineDataLoader, BaselineAggregator

loader = BaselineDataLoader(config_path="config/baseline_config_edge.yaml")
records = loader.load_all_records()
print(f"loaded {len(records)} records")

agg = BaselineAggregator(records, data_loader=loader)
perf_df, cost_df = agg.to_summary_table(cost_metric="total_cost", random_seed=42)

perf_pct = perf_df.copy()
for col in perf_pct.columns:
    for idx in perf_pct.index:
        v = perf_pct.loc[idx, col]
        perf_pct.loc[idx, col] = round(v * 100, 2) if isinstance(v, (int, float)) else "-"

print("\n=== 性能表（%）===")
print(perf_pct.to_string())
print("\n=== 成本表（¥）===")
print(cost_df.round(4).to_string())

perf_df.to_csv("results/edge_perf_table.csv", encoding="utf-8-sig")
cost_df.to_csv("results/edge_cost_table.csv", encoding="utf-8-sig")
print("\nEDGE_ANALYSIS_DONE")
