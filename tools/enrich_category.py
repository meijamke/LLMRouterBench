"""
enrich_category.py — 为已采集的 MMLU / MMLU-Pro 结果回填 subject/category 字段。

原因：历史采集（convert_phase1_to_official.py）生成的 results JSON 里
record.extra_fields 只有 thinking_mode/api_latency_ms/error，没有 category。
evaluator 现已让 evaluate() 返回 category（新采集会自动带上），本脚本负责
一次性给存量结果补齐，供 BaselineAggregator 计算 subject 粒度的
"Category Expert" 静态分表基线。

匹配键：record.origin_query == 源数据 question 文本（唯一、稳定）。
幂等：重复运行安全（仅覆盖 category 字段）。

用法（仓库根目录）：
    python tools/enrich_category.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "bench"

DATASETS = ["mmlu", "mmlupro"]


def load_q2c(dataset_id: str) -> dict:
    path = DATA / dataset_id / "test.json"
    q2c = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            q = d.get("question", "")
            if q:
                q2c[q] = d.get("category", "")
    return q2c


def main():
    total_files = 0
    total_records = 0
    total_matched = 0
    for ds in DATASETS:
        q2c = load_q2c(ds)
        ds_dir = RESULTS / ds
        if not ds_dir.exists():
            print(f"[skip] {ds}: no results dir")
            continue
        for fp in sorted(ds_dir.rglob("*.json")):
            data = json.loads(fp.read_text(encoding="utf-8"))
            records = data.get("records", [])
            matched = 0
            for r in records:
                oq = r.get("origin_query", "")
                cat = q2c.get(oq, "")
                if cat:
                    ef = r.setdefault("extra_fields", {})
                    ef["category"] = cat
                    matched += 1
            fp.write_text(json.dumps(data, ensure_ascii=True), encoding="utf-8")
            total_files += 1
            total_records += len(records)
            total_matched += matched
            print(f"[ok] {fp.relative_to(ROOT)}: {matched}/{len(records)} matched")
    print(f"DONE files={total_files} records={total_records} matched={total_matched}")


if __name__ == "__main__":
    main()
