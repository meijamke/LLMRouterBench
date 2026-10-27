# -*- coding: utf-8 -*-
"""绘制 DeviceLLMRouter 的成本-质量帕累托前沿（x=成本, y=准确率）。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_manager.fontManager.addfont(r"C:\Windows\Fonts\msyh.ttc")
font_manager.fontManager.addfont(r"C:\Windows\Fonts\msyhbd.ttc")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

from baselines.data_loader import BaselineDataLoader
from baselines.device_llm_router import DeviceLLMRouter
from eval_device_llm_router import (
    load_questions, evaluate, references, summarize, LOCAL_MODEL, CLOUD_MODEL,
)

loader = BaselineDataLoader(config_path=str(ROOT / "config/baseline_config_edge.yaml"))
records = [r for r in loader.load_all_records() if r.model_name in (LOCAL_MODEL, CLOUD_MODEL)]
questions, meta = load_questions(records)

# DeviceLLMRouter 在不同 capability 下的 (cost, acc)
caps = ["SIMPLE", "MEDIUM", "COMPLEX", "RESEARCH", "REASONING", "NONE"]
router_points = []
for cap in caps:
    router = DeviceLLMRouter(local_capability_level=cap, tier_models={"COMPLEX": CLOUD_MODEL})
    per_q, _ = evaluate(router, questions, meta)
    s = summarize(per_q)
    router_points.append((s["cost"], s["acc"] * 100, cap))

refs = references(questions)

def ref(key, name):
    d = refs[key]
    return (d["cost"], d["correct"] / d["n"] * 100, name)

ref_points = [
    ref("local", "端侧 only (Qwen3.5-4B)"),
    ref("cloud", "云侧 only (DeepSeek-V4.1-Flash)"),
    ref("oracle", "Oracle 上界"),
    ref("random", "Random 期望"),
]

fig, ax = plt.subplots(figsize=(9.2, 6.0))
# 对照点
for cost, acc, name in ref_points:
    ax.scatter(cost, acc, s=110, color="#1a73e8", zorder=3)
    ax.annotate(name, (cost, acc), textcoords="offset points", xytext=(9, 7), fontsize=10)
# DeviceLLMRouter capability 扫描曲线
rc = [p[0] for p in router_points]
ra = [p[1] for p in router_points]
ax.plot(rc, ra, "-o", color="#d93025", lw=2, ms=8, label="DeviceLLMRouter（capability 扫描）", zorder=4)
for cost, acc, cap in router_points:
    ax.annotate(cap, (cost, acc), textcoords="offset points", xytext=(7, -14),
                fontsize=8, color="#d93025")

ax.set_xlabel("成本 (¥)")
ax.set_ylabel("准确率 (%)")
ax.set_title("DeviceLLMRouter 成本-质量帕累托前沿（6 数据集合并 · 882 题）")
ax.grid(alpha=0.3)
ax.legend(loc="lower right")
out = ROOT / "docs/device_llm_router_pareto.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print("saved:", out)
