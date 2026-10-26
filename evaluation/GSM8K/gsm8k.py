"""
GSM8K evaluator for edge-router-bench（LLMRouterBench 扩展，独立类，无 datasets 依赖）。
数据格式：data/GSM8K/test.json（JSONL，每行 {"question","answer"}）。
判分：抽取输出中最后一个数字，与 ground_truth 数字比对。
"""
import json
import os
import re
from typing import Dict, List

DATA_DIR = "data/GSM8K"
PROMPT = "Question: {question}\nAnswer:"


class GSM8KEvaluator:
    def __init__(self):
        self.task = "GSM8K"

    def load_data(self, split: str = "test") -> List[Dict]:
        path = os.path.join(DATA_DIR, f"{split}.json")
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                q = d["question"]
                out.append({
                    "origin_query": q,
                    "prompt": PROMPT.format(question=q),
                    "question": q,
                    "answer": str(d["answer"]).strip(),
                })
        return out

    def get_valid_splits(self) -> List[str]:
        return ["test"]

    def _extract_number(self, text: str) -> str:
        t = (text or "").strip()
        if not t:
            return ""
        # 去掉逗号千分位，取最后一个数字（含小数/负号）
        m = re.findall(r"-?\d[\d,]*(?:\.\d+)?", t.replace(",", ""))
        if not m:
            return ""
        return m[-1].strip()

    def evaluate(self, data: Dict, output_text: str, **kwargs) -> Dict:
        pred = self._extract_number(output_text)
        gt = self._extract_number(str(data.get("answer", "")))
        is_correct = bool(pred) and bool(gt) and pred == gt
        return {"prediction": pred, "ground_truth": gt, "is_correct": is_correct}
