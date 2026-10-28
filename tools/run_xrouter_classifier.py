# -*- coding: utf-8 -*-
"""X-Router 分类器落档脚本：用「端侧小模型」对每道题判断难易档位。

忠实复刻 MiniPC PR#11 的 LLM 分类器分支：
    XRouter.classify()  --backend!=null--> buildClassifierPrompt(问题, 6000)
      -> CloudClassifier.classifyAsync(prompt)
          body: {model, messages:[{role:user,content:prompt}], stream:false,
                 max_tokens:2048, temperature:0, enable_thinking:false}
      -> parseTier(raw) 严格单标签；解析失败 source=parse_failed、
         调用失败 source=heuristic_fallback，档位一律回退 classify_heuristic（故障只降不升）。

分类器模型 = 硅基流动 Qwen/Qwen3.5-4B（端侧小模型，难度裁判）。
只存 token 数，成本由评估端价格表统一计算。

用法：
    python tools/run_xrouter_classifier.py [--concurrency 8] [--limit 0]
    python tools/run_xrouter_classifier.py --retry-failed --concurrency 4   # 只重试失败题
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baselines.data_loader import BaselineDataLoader  # noqa: E402
from baselines.device_llm_router import (  # noqa: E402
    build_classifier_prompt, classify_heuristic, parse_tier,
)

CLASSIFIER_MODEL = "Qwen/Qwen3.5-4B"
ENDPOINT = "https://api.siliconflow.cn/v1/chat/completions"
PREVIEW_CHARS = 6000
MAX_TOKENS = 2048
CANONICAL_MODEL = "DeepSeek-V4.1-Flash"
TIMEOUT = 300
RETRIES = 5


def load_api_key() -> str:
    env = ROOT / ".env"
    for line in env.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("SILICONFLOW_API_KEY"):
            return line.split("=", 1)[1].strip()
    raise SystemExit("未找到 SILICONFLOW_API_KEY（.env）")


def classify_api(prompt: str, api_key: str, timeout: int = TIMEOUT):
    """单次分类调用；返回 (raw_content, prompt_tokens, completion_tokens)；失败抛异常。"""
    body = json.dumps({
        "model": CLASSIFIER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "enable_thinking": False,
    }).encode("utf-8")
    req = urllib.request.Request(ENDPOINT, data=body, method="POST", headers={
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    choice = (data.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content") or ""
    usage = data.get("usage") or {}
    return content, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))


def classify_one(question_text: str, api_key: str) -> dict:
    """对齐 XRouter.classify() 的分段失败语义；API 层带退避重试。"""
    messages = [{"role": "user", "content": question_text}]
    heuristic = classify_heuristic(messages)
    try:
        prompt = build_classifier_prompt(messages, PREVIEW_CHARS)
    except ValueError:
        return {"tier": heuristic, "source": "heuristic_fallback",
                "prompt_tokens": 0, "completion_tokens": 0}

    raw = ""
    pt = ct = 0
    for attempt in range(RETRIES):
        try:
            raw, pt, ct = classify_api(prompt, api_key)
            break
        except Exception:
            if attempt < RETRIES - 1:
                time.sleep(8 + attempt * 20)
    else:
        return {"tier": heuristic, "source": "heuristic_fallback",
                "prompt_tokens": 0, "completion_tokens": 0}

    tier = parse_tier(raw)
    if tier:
        return {"tier": tier, "source": "llm", "prompt_tokens": pt, "completion_tokens": ct}
    return {"tier": heuristic, "source": "parse_failed", "prompt_tokens": pt, "completion_tokens": ct}


def load_questions():
    """(dataset_id, record_index) -> 题目文本（来自 CANONICAL_MODEL 结果）。"""
    loader = BaselineDataLoader(config_path=str(ROOT / "config" / "baseline_config_edge.yaml"))
    records = [r for r in loader.load_all_records() if r.model_name == CANONICAL_MODEL]
    seen = {}
    for r in records:
        seen.setdefault((r.dataset_id, r.record_index), r.origin_query or r.prompt)
    return seen


def write_csv(out: Path, rows: list[dict]) -> None:
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset_id", "record_index", "tier",
                                          "source", "prompt_tokens", "completion_tokens"])
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["dataset_id"], x["record_index"])):
            w.writerow(r)


def run_fresh(args: argparse.Namespace, out: Path, seen: dict, api_key: str) -> None:
    items = sorted(seen.items())
    if args.limit:
        items = items[: args.limit]
    print(f"[load] {len(items)} 道题")

    rows = []

    def work(item):
        (ds, idx), text = item
        r = classify_one(text, api_key)
        r.update({"dataset_id": ds, "record_index": idx})
        return r

    done = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for fut in as_completed([ex.submit(work, it) for it in items]):
            rows.append(fut.result())
            done += 1
            if done % 50 == 0 or done == len(items):
                print(f"[progress] {done}/{len(items)}", flush=True)

    src = {}
    for r in rows:
        src[r["source"]] = src.get(r["source"], 0) + 1
    print("[source]", ", ".join(f"{k}:{v}" for k, v in sorted(src.items())))
    write_csv(out, rows)
    print(f"[saved] {out}")


def run_retry_failed(args: argparse.Namespace, out: Path, seen: dict, api_key: str) -> None:
    existing = {}
    with open(out, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            existing[(row["dataset_id"], int(row["record_index"]))] = row
    failed = {k: v for k, v in existing.items() if v["source"] == "heuristic_fallback"}
    print(f"[retry-failed] {len(failed)} 题待重试（共 {len(existing)} 题）")

    items = sorted(k for k in failed if k in seen)

    def work(key):
        r = classify_one(seen[key], api_key)
        r.update({"dataset_id": key[0], "record_index": key[1]})
        return key, r

    done = 0
    recovered = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for fut in as_completed([ex.submit(work, k) for k in items]):
            key, r = fut.result()
            existing[key].update({"tier": r["tier"], "source": r["source"],
                                  "prompt_tokens": r["prompt_tokens"],
                                  "completion_tokens": r["completion_tokens"]})
            if r["source"] == "llm":
                recovered += 1
            done += 1
            if done % 50 == 0 or done == len(items):
                print(f"[progress] {done}/{len(items)}  recovered={recovered}", flush=True)

    rows = list(existing.values())
    src = {}
    for r in rows:
        src[r["source"]] = src.get(r["source"], 0) + 1
    print("[source]", ", ".join(f"{k}:{v}" for k, v in sorted(src.items())))
    write_csv(out, rows)
    print(f"[saved] {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="X-Router 分类器落档")
    ap.add_argument("--config", default="config/baseline_config_edge.yaml")
    ap.add_argument("--out", default="results/xrouter_classifier_tiers.csv")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args()

    out = ROOT / args.out
    api_key = load_api_key()
    seen = load_questions()
    if args.retry_failed:
        run_retry_failed(args, out, seen, api_key)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        run_fresh(args, out, seen, api_key)


if __name__ == "__main__":
    main()