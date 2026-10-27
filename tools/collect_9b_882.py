# -*- coding: utf-8 -*-
"""定向采集 Qwen3.5-9B 答案矩阵：只采与 DeepSeek 结果对齐的 882 题。

背景：results/bench 里 DeepSeek-V4.1-Flash 的 882 题是上一轮按抽样数据采的，
      而 data/ 现已是全量数据集（gpqa 198、gsm8k 1319、mmlu 14042 等），
      直接跑 collector 会采大全集且 (dataset, record_index) 对齐不上。

做法：复用 LLMRouterBench 的 generator + evaluator + storage，
      逐题读取 DeepSeek 记录的 prompt/ground_truth/index，调 9B 生成并按原判分器打分，
      产出 results/bench/<ds>/test/Qwen3.5-9B/*.json，与 DeepSeek 完全同 (dataset, index) 对齐。

判分对齐性：6 个评估器的 evaluate() 都只用 record_data['answer']（MMLU/MMLUPro 的 category
      仅用于聚合日志），故 record_data={'answer': 存储的 ground_truth} 与原始判分等价。

用法：
    python tools/collect_9b_882.py                    # 全量 882 题
    python tools/collect_9b_882.py --datasets gpqa --limit 2   # 冒烟
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baselines.data_loader import BaselineDataLoader                      # noqa: E402
from evaluation.factory import EvaluatorFactory                          # noqa: E402
from generators.factory import create_generator                          # noqa: E402
from data_collector.storage import ResultsStorage, BenchmarkResult, RecordResult  # noqa: E402

SOURCE_MODEL = "DeepSeek-V4.1-Flash"
TARGET_MODEL = "Qwen3.5-9B"          # 端侧大模型（本地回答）
TARGET_API = "Qwen/Qwen3.5-9B"
BASE_URL = "https://api.siliconflow.cn/v1"

DS_ORDER = ["gpqa", "mmlupro", "math500", "arcc", "gsm8k", "mmlu"]


def load_api_key() -> str:
    env = ROOT / ".env"
    for line in env.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("SILICONFLOW_API_KEY"):
            return line.split("=", 1)[1].strip()
    raise SystemExit("未找到 SILICONFLOW_API_KEY（.env）")


def build_generator(api_key: str):
    return create_generator({
        "name": TARGET_MODEL,
        "api_model_name": TARGET_API,
        "base_url": BASE_URL,
        "api_key": api_key,
        "generator_type": "direct",
        "temperature": 0.0,
        "top_p": 1.0,
        "timeout": 500,
        "extra_body": {"enable_thinking": False},
        "pricing": {"prompt_price_per_million": 0.20, "completion_price_per_million": 0.80},
    }, cache_config=None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="每数据集最多采几题（0=全部），用于冒烟")
    ap.add_argument("--datasets", default="", help="逗号分隔，只采这些数据集")
    args = ap.parse_args()

    loader = BaselineDataLoader(config_path=str(ROOT / "config" / "baseline_config_edge.yaml"))
    records = [r for r in loader.load_all_records() if r.model_name == SOURCE_MODEL]
    by_ds = defaultdict(list)
    for r in records:
        by_ds[r.dataset_id].append(r)
    for ds in by_ds:
        by_ds[ds].sort(key=lambda r: r.record_index)
    print(f"[load] {len(records)} 题 / {len(by_ds)} 数据集（来源 {SOURCE_MODEL}）")

    only = {d.strip() for d in args.datasets.split(",") if d.strip()}

    api_key = load_api_key()
    generator = build_generator(api_key)
    evaluator_factory = EvaluatorFactory()
    storage = ResultsStorage(base_dir=str(ROOT / "results"))

    for ds in DS_ORDER:
        if ds not in by_ds or (only and ds not in only):
            continue
        recs = by_ds[ds]
        if args.limit:
            recs = recs[: args.limit]
        evaluator = evaluator_factory.get_evaluator(ds)
        print(f"[run] {ds}: {len(recs)} 题", flush=True)

        def process(rec):
            gen = generator.generate(rec.prompt)
            ev = evaluator.evaluate({"answer": rec.ground_truth}, gen.output)
            is_correct = ev.get("is_correct", False)
            if is_correct is None:
                score = None
            elif isinstance(is_correct, bool):
                score = 1.0 if is_correct else 0.0
            else:
                score = float(is_correct)
            extra = {k: v for k, v in ev.items()
                     if k not in ("is_correct", "prediction", "ground_truth")}
            return RecordResult(
                index=rec.record_index,          # 与 DeepSeek 同 index，保证对齐
                origin_query=rec.origin_query,
                prompt=rec.prompt,
                prompt_tokens=gen.prompt_tokens,
                completion_tokens=gen.completion_tokens,
                cost=gen.cost,
                score=score,
                prediction=ev.get("prediction", ""),
                ground_truth=ev.get("ground_truth", rec.ground_truth),
                raw_output=gen.output,
                extra_fields=extra,
            )

        results = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(process, r) for r in recs]
            done = 0
            for fut in as_completed(futs):
                results.append(fut.result())
                done += 1
                if done % 50 == 0 or done == len(recs):
                    print(f"  [{ds}] {done}/{len(recs)}", flush=True)

        results.sort(key=lambda r: r.index)
        total_pt = sum(r.prompt_tokens for r in results)
        total_ct = sum(r.completion_tokens for r in results)
        total_cost = sum(r.cost for r in results)
        valid = [r.score for r in results if r.score is not None]
        perf = sum(valid) / len(valid) if valid else 0.0

        result = BenchmarkResult(
            performance=perf, time_taken=0.0,
            prompt_tokens=total_pt, completion_tokens=total_ct, cost=total_cost,
            counts=len(results), model_name=TARGET_MODEL, dataset_name=ds, split="test",
            records=results, demo=False, extra_metrics={},
        )
        storage.save_result(result, ds, "test", TARGET_MODEL, data_fingerprint="")
        print(f"  [saved] {ds}: acc={perf * 100:.2f}%  cost={total_cost:.4f}  n={len(results)}", flush=True)

    print("[all done]")
    # 删除 --limit 冒烟产生的残缺文件（避免与全量文件混淆）
    if args.limit:
        from data_collector.storage import ResultsStorage as _S
        import re
        base = ROOT / "results" / "bench"
        for ds in DS_ORDER:
            d = base / ds / "test" / TARGET_MODEL
            if d.exists():
                for f in d.glob(f"{ds}-test-{TARGET_MODEL}-*.json"):
                    data = None
                    try:
                        data = __import__("json").load(open(f, encoding="utf-8"))
                    except Exception:
                        continue
                    if data and data.get("counts") == args.limit:
                        f.unlink()
                        print(f"  [rm smoke] {f.name}")


if __name__ == "__main__":
    main()