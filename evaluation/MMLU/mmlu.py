"""
MMLU evaluator for edge-router-bench（LLMRouterBench 扩展，独立类，无 datasets 依赖）。
数据格式：data/MMLU/test.json（JSONL，每行 {"question","options"{A-D},"answer"字母,"category"}）。
判分：抽取 "Answer: $LETTER" 与 ground_truth 字母比对。
"""
import json
import os
import re
from typing import Dict, List

DATA_DIR = "data/MMLU"
PROMPT = ("Answer the following multiple choice question. The last line of your response "
          "should be of the format 'Answer: $LETTER' (without quotes) where LETTER is one of "
          "{candidates}.\n\n{question}\n\n{options}\n\nLet's think step by step.")
ANSWER_PATTERN = r"(?i)Answer\s*:\s*\$?([A-D])[.\s\n]?"


class MMLUEvaluator:
    def __init__(self, split: str = "test"):
        self.task = f"MMLU-{split}"
        self.split = split

    def load_data(self, split: str = "test") -> List[Dict]:
        path = os.path.join(DATA_DIR, f"{split}.json")
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                opts = d["options"]
                candidates = "".join(opts.keys())
                options = "\n".join(f"{k}. {v}" for k, v in opts.items())
                out.append({
                    "origin_query": d["question"],
                    "prompt": PROMPT.format(question=d["question"], options=options,
                                            candidates=candidates),
                    "question": d["question"],
                    "options": opts,
                    "answer": d["answer"],
                    "category": d.get("category", "mixed"),
                })
        return out

    def get_valid_splits(self) -> List[str]:
        return ["test"]

    def evaluate(self, data: Dict, output_text: str, **kwargs) -> Dict:
        m = re.findall(ANSWER_PATTERN, output_text or "")
        pred = m[-1].upper() if m else ""
        is_correct = pred == data.get("answer", "")
        return {"prediction": pred, "ground_truth": data.get("answer", ""),
                "is_correct": is_correct, "category": data.get("category", "")}
