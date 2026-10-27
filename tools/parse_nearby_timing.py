# -*- coding: utf-8 -*-
"""解析 MiniPC 算力端 hilog `NearbyChat` 计时日志 → 端侧推理成本（方案 C 的测量组件）。

日志来源（apps/compute-node，实测字段）：
    InferenceHost.ets:294    infer output request=<id> epoch=<d> elapsedMs=<d> chars=<d>
                             → 受理→首字 TTFT（Date.now() - request.startedAt），首字即打点
    LinkEnhanceChannel.ets:2043  le-write-start type=<s> request=<id> epoch=<d> id=<frag> count=<d> queueMs=<d>
                             → 应用发送排队耗时
    LinkEnhanceChannel.ets:1810  le-frame-recv epoch=<d> id=<frag> index=<d> count=<d> received=<d> bytes=<d> elapsedMs=<d> gapMs=<d>
                             → 分片接收累计耗时/间隔

抓取：hdc -t <算力端序列号> shell hilog -T NearbyChat > nearby.log
解析：python tools/parse_nearby_timing.py nearby.log [--out results/nearby_timing.csv]

输出：逐 request 的 ttft_ms（主成本指标）+ 发送排队 queue_ms + 传输统计。
只依赖标准库，可 --self-test 用合成日志行验证。
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

INFER_OUTPUT_RE = re.compile(r"infer output request=(\S+) epoch=(\d+) elapsedMs=(\d+) chars=(\d+)")
LE_WRITE_RE = re.compile(r"le-write-start type=(\S+) request=(\S+) epoch=(\d+) id=(\S+) count=(\d+) queueMs=(\d+)")
LE_FRAME_RECV_RE = re.compile(r"le-frame-recv epoch=(\d+) id=(\S+) index=(\d+) count=(\d+) "
                              r"received=(\d+) bytes=(\d+) elapsedMs=(\d+) gapMs=(\d+)")


def parse_line(line: str):
    """解析一行 hilog；返回 (kind, key, fields) 或 None。"""
    idx = line.find("NearbyChat:")
    if idx < 0:
        return None
    msg = line[idx + len("NearbyChat:"):].strip()

    m = INFER_OUTPUT_RE.search(msg)
    if m:
        return ("infer_output", m.group(1), {
            "epoch": int(m.group(2)), "ttft_ms": int(m.group(3)), "chars": int(m.group(4)),
        })
    m = LE_WRITE_RE.search(msg)
    if m:
        return ("le_write_start", m.group(2), {
            "type": m.group(1), "epoch": int(m.group(3)), "queue_ms": int(m.group(6)),
        })
    m = LE_FRAME_RECV_RE.search(msg)
    if m:
        return ("le_frame_recv", m.group(2), {
            "epoch": int(m.group(1)), "index": int(m.group(3)), "count": int(m.group(4)),
            "received": int(m.group(5)), "bytes": int(m.group(6)),
            "elapsed_ms": int(m.group(7)), "gap_ms": int(m.group(8)),
        })
    return None


def parse(stream) -> dict:
    """解析整个日志流，返回按 request 聚合的 dict。"""
    per_request = {}       # request -> {ttft_ms, chars, queue_ms}
    frame_elapsed = []     # 所有分片接收耗时样本

    for line in stream:
        r = parse_line(line)
        if r is None:
            continue
        kind, key, fields = r
        if kind == "infer_output":
            d = per_request.setdefault(key, {})
            d["ttft_ms"] = fields["ttft_ms"]
            d["chars"] = fields["chars"]
        elif kind == "le_write_start":
            d = per_request.setdefault(key, {})
            d["queue_ms"] = fields["queue_ms"]
        elif kind == "le_frame_recv":
            frame_elapsed.append(fields["elapsed_ms"])

    return {"per_request": per_request, "frame_elapsed": frame_elapsed}


def main() -> None:
    ap = argparse.ArgumentParser(description="解析 MiniPC NearbyChat 计时日志")
    ap.add_argument("log", nargs="?", help="hilog 捕获文件；缺省读 stdin")
    ap.add_argument("--out", default=None, help="CSV 输出路径")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(self_test())

    if args.log:
        with open(args.log, encoding="utf-8", errors="replace") as f:
            result = parse(f)
    else:
        result = parse(sys.stdin)

    per_request = result["per_request"]
    frames = result["frame_elapsed"]
    n = len(per_request)
    ttf = [v["ttft_ms"] for v in per_request.values() if "ttft_ms" in v]
    print(f"[parse] {n} 个 request；含 TTFT {len(ttf)} 个；分片接收样本 {len(frames)}")
    if ttf:
        print(f"TTFT(ms): min={min(ttf)} median={sorted(ttf)[len(ttf)//2]} "
              f"avg={sum(ttf)/len(ttf):.0f} max={max(ttf)}")
    if frames:
        print(f"分片传输(ms): avg={sum(frames)/len(frames):.0f} max={max(frames)}")

    # 明细表
    print(f"\n{'request':<24} {'ttft_ms':>8} {'queue_ms':>8} {'chars':>6}")
    for req in sorted(per_request):
        v = per_request[req]
        print(f"{req:<24} {v.get('ttft_ms','-'):>8} {v.get('queue_ms','-'):>8} {v.get('chars','-'):>6}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["request", "ttft_ms", "queue_ms", "chars"])
            w.writeheader()
            for req in sorted(per_request):
                v = per_request[req]
                w.writerow({"request": req, "ttft_ms": v.get("ttft_ms", ""),
                            "queue_ms": v.get("queue_ms", ""), "chars": v.get("chars", "")})
        print(f"\n[saved] {out}")


def self_test() -> int:
    lines = [
        '12-27 10:00:00.000 1234 5678 I 02d00/NearbyChat: le-write-start type=infer.request request=rq-001 epoch=3 id=abc count=4 queueMs=2',
        '12-27 10:00:00.005 1234 5678 I 02d00/NearbyChat: le-frame-recv epoch=3 id=abc index=0 count=4 received=1 bytes=480 elapsedMs=15 gapMs=0',
        '12-27 10:00:00.047 1234 5678 I 02d00/NearbyChat: le-frame-recv epoch=3 id=abc index=3 count=4 received=4 bytes=1920 elapsedMs=47 gapMs=12',
        '12-27 10:00:05.230 1234 5678 I 02d00/NearbyChat: infer output request=rq-001 epoch=3 elapsedMs=5230 chars=14',
        '12-27 10:00:06.100 1234 5678 I 02d00/NearbyChat: infer output request=rq-002 epoch=3 elapsedMs=12345 chars=3',
    ]
    result = parse(lines)
    per = result["per_request"]
    failures = []

    def check(cond, label):
        if not cond:
            failures.append(label)

    check(len(per) == 2, "两个 request")
    check(per["rq-001"]["ttft_ms"] == 5230, "rq-001 TTFT=5230")
    check(per["rq-001"]["queue_ms"] == 2, "rq-001 queueMs=2")
    check(per["rq-001"]["chars"] == 14, "rq-001 chars=14")
    check(per["rq-002"]["ttft_ms"] == 12345, "rq-002 TTFT=12345")
    check("queue_ms" not in per["rq-002"], "rq-002 无 queueMs")
    check(result["frame_elapsed"] == [15, 47], "分片耗时样本 [15,47]")

    if failures:
        print("SELF-TEST FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("SELF-TEST PASS (NearbyChat 计时解析器自检通过)")
    return 0


if __name__ == "__main__":
    main()
