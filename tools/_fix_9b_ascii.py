# -*- coding: utf-8 -*-
"""一次性工具：把 9B 结果文件重写为 ASCII 安全（ensure_ascii=True）。

原因：BaselineDataLoader 用平台默认编码（Windows=GBK）读结果 JSON，
      而 Qwen3.5-9B 的 raw_output 含中文/特殊 Unicode（UTF-8 连续字节在 GBK 下非法），
      导致含非 ASCII 输出的文件被跳过。rewrite as ensure_ascii=True，双编码皆可读。
"""
import glob
import json

files = glob.glob("results/bench/*/test/Qwen3.5-9B/*.json")
for p in files:
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=True, indent=2)
    print("fixed", p)
print("total re-encoded:", len(files))