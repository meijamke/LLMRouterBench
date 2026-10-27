# -*- coding: utf-8 -*-
"""DeviceLLMRouter —— MiniPC PR#11 中 XRouter 路由决策内核的 Python 严格移植。

源文件（上游）:
    apps/compute-node/entry/src/main/ets/llm/XRouter.ets   (Keanu2/MiniPC, PR#11)
    apps/compute-node/entry/src/main/ets/llm/BackendRouter.ets

导出结论
--------
DeviceLLMRouter（= 代码里的 XRouter）的决策内核是「复杂度分档 + 档位→端/云映射」：
    1. classify: 可选云端 LLM 分类器（缺省走确定性启发式）——本模块复刻启发式分支
       （backend=None），也保留 PROMPT_TEMPLATE 供 LLM 分类器分支离线重现；
    2. decide_by_tier: tier_rank(tier) <= tier_rank(localCapabilityLevel) → 端侧，
       否则 → 该档云模型（未配置档回退 COMPLEX 档模型）。
无训练权重、无状态、无 torch/onnx —— 因此「导出」即逐句翻译本内核。

启发式冻结契约（XRouter.ets:53-98, 389-403）原样保留，不做任何"改进"。
本模块只依赖标准库，可独立运行 ``python -m baselines.device_llm_router --self-test``。
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# 上游常量（python/openjiuwen/x_router 原样移植）
# ---------------------------------------------------------------------------
TIER_ORDER: List[str] = ["SIMPLE", "MEDIUM", "COMPLEX", "RESEARCH", "REASONING"]
FALLBACK_TIER: str = "COMPLEX"
CAPABILITY_NONE: str = "NONE"

MIN_PREVIEW_CHARS: int = 256
DEFAULT_PREVIEW_CHARS: int = 6000
PREVIEW_WINDOW: int = 6
TRUNCATE_MARKER: str = "\n...[truncated]...\n"
QUERY_MAX_TEXT_BYTES: int = 16384

# 决策规则名（上游 RULE_*，进 reason 字段便于遥测比对）
RULE_WITHIN_LOCAL_CAPABILITY = "within_local_capability"
RULE_ESCALATE_CLOUD = "escalate_cloud"
RULE_TARGET_EXCLUDED_DEGRADE = "target_excluded_degrade"

# 档位来源标记（上游 SOURCE_*）
SOURCE_LLM = "llm"
SOURCE_HEURISTIC = "heuristic"
SOURCE_HEURISTIC_FALLBACK = "heuristic_fallback"
SOURCE_PARSE_FAILED = "parse_failed"
SOURCE_UNAVAILABLE = "unavailable"

# 启发式阈值（上游冻结契约：词数 80/400/1200、user 轮次 2/6、问号数 3）
TOKENS_MEDIUM = 80
TOKENS_COMPLEX = 400
TOKENS_RESEARCH = 1200
USER_TURNS_MEDIUM = 2
USER_TURNS_COMPLEX = 6
QUESTIONS_COMPLEX = 3

# 关键词表为上游英文冻结集，不扩充中文（中文由 LLM 分类器主路径覆盖；启发式仅兜底）。
REASONING_KEYWORDS: List[str] = [
    "prove", "proof", "theorem", "lemma", "corollary", "axiom",
    "formal verification", "formal proof", "mathematical proof",
    "by induction", "induction hypothesis", "qed",
    "contrapositive", "bijection", "isomorphism",
    "differential equation", "eigenvalue", "tensor calculus",
    "fourier transform", "laplace transform",
    "np-complete", "np-hard", "sat solver", "model checking",
]
RESEARCH_KEYWORDS: List[str] = [
    "quantum entanglement", "quantum cryptography", "quantum computing",
    "bell's inequality", "epr paradox", "wave function collapse",
    "qubit", "decoherence",
    "genomics", "proteomics", "pharmacokinetics", "pharmacodynamics",
    "pathophysiology", "epidemiology", "clinical trial", "meta-analysis",
    "neurological", "etiology", "comorbidity",
    "jurisprudence", "constitutional law", "case precedent",
    "litigation", "habeas corpus", "estoppel",
    "derivative pricing", "black-scholes", "monte carlo simulation",
    "stochastic process", "martingale",
    "transformer architecture", "attention mechanism", "rlhf",
    "systematic review", "literature review", "survey paper",
]
COMPLEX_KEYWORDS: List[str] = [
    "analyze all", "analyze multiple", "across all files",
    "end-to-end", "architecture design", "root cause analysis",
    "multi-step plan", "step by step plan", "step by step", "step-by-step",
    "compare and contrast", "trade-off analysis",
    "long document", "entire codebase",
]
MEDIUM_KEYWORDS: List[str] = [
    "write a script", "write a function", "generate code", "implement",
    "refactor", "optimize", "rewrite", "debug", "fix this bug",
    "how does", "how do i",
    "summarize", "translate", "convert", "parse", "extract",
]

_LEVEL_RE = re.compile(r"^\s*(SIMPLE|MEDIUM|COMPLEX|RESEARCH|REASONING)\s*$", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")

# 分类器 prompt 模板（上游 PROMPT_TEMPLATE 逐字移植；仅 LLM 分类器分支使用）
PROMPT_TEMPLATE = """Classify the work needed in the next assistant turn as SIMPLE, MEDIUM, COMPLEX, RESEARCH, or REASONING.
Use the overall user goal and recent assistant/tool progress to identify what remains. Ignore system prompts, tool definitions, and completed work.

Levels:
SIMPLE: A direct, single-step response using supplied context, with no substantive transformation or execution.
MEDIUM: Bounded drafting, editing, summarization, calculation, coding, data transformation, or routine tool/file work with clear steps.
COMPLEX: Substantial planning, implementation, debugging, design, or analysis requiring broad context or multiple interdependent steps or artifacts.
RESEARCH: Investigative work requiring gathering, evaluation, comparison, and synthesis across multiple sources.
REASONING: A hard problem where rigorous multi-hop inference, formal proof, derivation, or verification is the central work.

Locality: SIMPLE and MEDIUM are local-only; COMPLEX, RESEARCH, and REASONING are cloud tiers. If the next step requires internet access, external sources, or a remote API, choose among the three cloud tiers by the definitions above; external access alone does not distinguish among them. Looking up current or live data, or using a remote service through a tool or CLI, requires a cloud tier. Writing code or instructions that may use an API later does not itself require cloud access.
Classify the remaining step, not the entire original task. The mere availability of tools must not affect the level.

Conversation and recent progress:
{content}

Reply with ONLY the tier name."""


# ---------------------------------------------------------------------------
# 上游纯函数（XRouter.ets 静态方法逐句移植）
# ---------------------------------------------------------------------------
def tier_rank(tier: str) -> int:
    """档位秩（上游 ComplexityLevel IntEnum 序）；未知档 -1。"""
    if tier not in TIER_ORDER:
        return -1
    return TIER_ORDER.index(tier)


def parse_tier(value: Optional[str]) -> str:
    """严格单标签解析（上游 parse_complexity）：整串必须恰为一个档名；散文/多标签一律拒绝。"""
    match = _LEVEL_RE.match(value or "")
    if match is None:
        return ""
    return match.group(1).upper()


def is_scaffolding(role: str, content: str) -> bool:
    """上游 _is_scaffolding：system 消息，以及 agent 框架注入 user 轮的 system-reminder 块。"""
    if role == "system":
        return True
    stripped = re.sub(r"^\s+", "", content)
    return stripped.startswith("<system-reminder>") and "<prompt-attachment" in stripped


def conversation_view(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """对话视图（上游 conversation_messages）：过滤脚手架与空消息。"""
    view: List[Dict[str, str]] = []
    for message in messages or []:
        role = (message.get("role") or "user").lower()
        content = str(message.get("content") or "")
        if is_scaffolding(role, content):
            continue
        trimmed = content.strip()
        if len(trimmed) == 0:
            continue
        view.append({"role": role, "content": trimmed})
    return view


def _has_keyword(text: str, keywords: List[str]) -> bool:
    for keyword in keywords:
        if keyword in text:
            return True
    return False


def _count_char(text: str, ch: str) -> int:
    return text.count(ch)


def classify_heuristic(messages: List[Dict[str, str]]) -> str:
    """免模型分类器（上游 classify_heuristic 逐句移植）。

    注意：tool_call_depth 在本移植的 wire 协议不可得，恒为 0（与上游 PR#11 一致）。
    """
    view = conversation_view(messages)
    text = " ".join(entry["content"] for entry in view).lower()
    stripped = text.strip()
    token_count = len(_WHITESPACE_RE.split(stripped)) if len(stripped) > 0 else 0
    user_turns = sum(1 for entry in view if entry["role"] == "user")
    question_count = _count_char(text, "?") + _count_char(text, "？")
    if _has_keyword(text, REASONING_KEYWORDS):
        return "REASONING"
    if _has_keyword(text, RESEARCH_KEYWORDS) or token_count > TOKENS_RESEARCH:
        return "RESEARCH"
    if (_has_keyword(text, COMPLEX_KEYWORDS) or token_count > TOKENS_COMPLEX
            or user_turns > USER_TURNS_COMPLEX or question_count > QUESTIONS_COMPLEX):
        return "COMPLEX"
    if (_has_keyword(text, MEDIUM_KEYWORDS) or token_count > TOKENS_MEDIUM
            or user_turns > USER_TURNS_MEDIUM):
        return "MEDIUM"
    return "SIMPLE"


def model_for(tier: str, tier_models: Dict[str, str]) -> str:
    """档位模型（上游 model_for）：未配置的档位回退 FALLBACK_TIER 的模型。"""
    mapped = tier_models.get(tier)
    if mapped is not None and len(mapped) > 0:
        return mapped
    return tier_models.get(FALLBACK_TIER, "")


def pick(preferred: str, targets: List[str]) -> str:
    """上游 _pick：首选被排除时落到目录首个目标。"""
    if preferred in targets:
        return preferred
    return targets[0] if len(targets) > 0 else ""


def decide_by_tier(
    tier: str,
    local_capability_level: str,
    targets: List[str],
    tier_models: Dict[str, str],
) -> Dict[str, str]:
    """档位映射（上游 decide_by_tier 逐句移植）：tier <= 本地能力 → 本地；
    超过 → 档位云模型；云模型不在目标目录 → 降级本地。
    上游 quirk：本地自身被排除时 _pick 落到目录首个目标，规则名不变。"""
    capability = local_capability_level
    if capability != CAPABILITY_NONE and tier_rank(tier) <= tier_rank(capability):
        return {"model": pick("local", targets), "rule": RULE_WITHIN_LOCAL_CAPABILITY}
    model = model_for(tier, tier_models)
    if model not in targets:
        return {"model": pick("local", targets), "rule": RULE_TARGET_EXCLUDED_DEGRADE}
    return {"model": model, "rule": RULE_ESCALATE_CLOUD}


def conversation_preview(messages: List[Dict[str, str]], max_chars: int) -> str:
    """有界对话预览（上游 conversation_preview）：取最近 WINDOW 条；最新 user 轮被挤出时重新置顶。"""
    view = conversation_view(messages)
    recent = view[-PREVIEW_WINDOW:]
    latest_user = None
    for index in range(len(view) - 1, -1, -1):
        if view[index]["role"] == "user":
            latest_user = view[index]
            break
    if latest_user is not None and latest_user not in recent:
        recent = [latest_user] + view[-(PREVIEW_WINDOW - 1):]
    parts = ["[" + entry["role"] + "]: " + entry["content"] for entry in recent]
    return _bounded_text("\n".join(parts), max_chars)


def _bounded_text(text: str, max_chars: int) -> str:
    """上游 _bounded_text：超预算时保留开头与最近文本，中段以标记截断。"""
    if len(text) <= max_chars:
        return text
    available = max_chars - len(TRUNCATE_MARKER)
    head = available // 2
    return text[:head] + TRUNCATE_MARKER + text[len(text) - (available - head):]


def build_classifier_prompt(messages: List[Dict[str, str]], max_chars: int) -> str:
    """上游 build_classifier_request：空对话抛错（→ heuristic_fallback）。"""
    content = conversation_preview(messages, max_chars)
    if len(content) == 0:
        raise ValueError("classifier requires a non-empty conversation")
    return PROMPT_TEMPLATE.replace("{content}", content)


# ---------------------------------------------------------------------------
# 路由决策入口
# ---------------------------------------------------------------------------
class DeviceLLMRouter:
    """XRouter 决策内核的离线可回放版本。

    仅复刻启发式分类（backend=None）。LLM 分类器分支请用
    :func:`build_classifier_prompt` 生成 prompt 后自行调用云端模型，
    再以 :func:`parse_tier` 解析；bandit 反馈环不在离线范围（Stage 1 关闭）。
    """

    def __init__(
        self,
        local_capability_level: str = "MEDIUM",
        tier_models: Optional[Dict[str, str]] = None,
        shadow: bool = False,
    ) -> None:
        if tier_rank(local_capability_level) < 0 and local_capability_level != CAPABILITY_NONE:
            raise ValueError(f"unknown localCapabilityLevel={local_capability_level}")
        self.local_capability_level = local_capability_level
        self.tier_models = dict(tier_models or {})
        self.shadow = shadow

    def targets_of(self, local_ready: bool, cloud_ready: bool, exclusions: Optional[List[str]] = None) -> List[str]:
        """目标目录（上游 targetsOf）：本地模型（就绪）+ 各档云模型（云就绪），剔除状态排除项。"""
        declared: List[str] = []
        if local_ready:
            declared.append("local")
        if cloud_ready:
            for model in self.tier_models.values():
                if model not in declared:
                    declared.append(model)
        excluded = set(exclusions or [])
        return [model for model in declared if model not in excluded]

    def classify(self, messages: List[Dict[str, str]]) -> Dict[str, str]:
        """分类（上游 classify 的启发式-only 分支：backend=None）。"""
        try:
            return {"tier": classify_heuristic(messages), "source": SOURCE_HEURISTIC}
        except Exception:
            return {"tier": "MEDIUM", "source": SOURCE_UNAVAILABLE}

    def decide(
        self,
        messages: List[Dict[str, str]],
        local_ready: bool = True,
        cloud_ready: bool = True,
        exclusions: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """返回 RouteDecision：{side, reason, tier, model}；side ∈ {'local','cloud'}。"""
        targets = self.targets_of(local_ready, cloud_ready, exclusions)
        if len(targets) == 0:
            raise RuntimeError("no available target")  # 上游 RouterError::NoTarget
        pair = self.classify(messages)
        mapped = decide_by_tier(pair["tier"], self.local_capability_level, targets, self.tier_models)
        reasoning = "x-router: rule=" + mapped["rule"] + " tier=" + pair["tier"] + " source=" + pair["source"]
        side = "local" if mapped["model"] == "local" else "cloud"
        if self.shadow:
            effective = "cloud" if cloud_ready else "local"
            return {
                "side": effective,
                "reason": reasoning + " shadow=true would=" + side,
                "tier": pair["tier"],
                "model": mapped["model"],
            }
        return {"side": side, "reason": reasoning, "tier": pair["tier"], "model": mapped["model"]}

    def route_question(self, question: str) -> Dict[str, str]:
        """便捷入口：单条 user 消息（无历史上下文）的路由决策。"""
        return self.decide([{"role": "user", "content": question}])


# ---------------------------------------------------------------------------
# 自测（对齐 XRouter.test.cjs 关键断言）
# ---------------------------------------------------------------------------
def _self_test() -> int:
    failures: List[str] = []

    def check(cond: bool, label: str) -> None:
        if not cond:
            failures.append(label)

    # tier_rank / parse_tier
    check(tier_rank("SIMPLE") == 0, "tier_rank SIMPLE")
    check(tier_rank("MEDIUM") == 1, "tier_rank MEDIUM")
    check(tier_rank("REASONING") == 4, "tier_rank REASONING")
    check(tier_rank("BOGUS") == -1, "tier_rank unknown")
    check(parse_tier("RESEARCH") == "RESEARCH", "parse RESEARCH")
    check(parse_tier("  simple  ") == "SIMPLE", "parse simple")
    check(parse_tier("you should use COMPLEX") == "", "parse strict-reject prose")
    check(parse_tier("SIMPLE MEDIUM") == "", "parse strict-reject multi")
    check(parse_tier("") == "", "parse empty")

    # classify_heuristic
    check(classify_heuristic([{"role": "user", "content": "你好"}]) == "SIMPLE", "heuristic CJK -> SIMPLE")
    check(classify_heuristic([{"role": "user", "content": "Prove that sqrt(2) is irrational"}]) == "REASONING",
          "heuristic prove -> REASONING")
    check(classify_heuristic([{"role": "user", "content": "write a script to parse json"}]) == "MEDIUM",
          "heuristic write a script -> MEDIUM")
    check(classify_heuristic([{"role": "user", "content": " ".join(["word"] * 500)}]) == "COMPLEX",
          "heuristic 500 words -> COMPLEX")
    check(classify_heuristic([{"role": "user", "content": " ".join(["word"] * 1300)}]) == "RESEARCH",
          "heuristic 1300 words -> RESEARCH")
    check(classify_heuristic([{"role": "system", "content": "sys"}, {"role": "user", "content": "hi there"}]) == "SIMPLE",
          "heuristic skips system scaffolding")

    # decide（档位映射）
    cloud = {"COMPLEX": "deepseek-v4.1-flash"}
    r = DeviceLLMRouter(local_capability_level="NONE", tier_models=cloud)
    d = r.route_question("你好")
    check(d["side"] == "cloud" and d["tier"] == "SIMPLE", "NONE -> escalate SIMPLE to cloud")

    r2 = DeviceLLMRouter(local_capability_level="SIMPLE", tier_models=cloud)
    d = r2.route_question("你好")
    check(d["side"] == "local" and d["model"] == "local", "SIMPLE capability -> local")

    r3 = DeviceLLMRouter(local_capability_level="MEDIUM", tier_models=cloud)
    d = r3.route_question("write a script to parse json")
    check(d["side"] == "local", "MEDIUM capability, MEDIUM tier -> local")
    d = r3.route_question("Prove that sqrt(2) is irrational")
    check(d["side"] == "cloud" and d["model"] == "deepseek-v4.1-flash", "MEDIUM capability, REASONING -> cloud")

    if failures:
        print("SELF-TEST FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("SELF-TEST PASS (DeviceLLMRouter 决策内核移植自检通过)")
    return 0


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    # 交互示例
    router = DeviceLLMRouter(local_capability_level="MEDIUM",
                             tier_models={"COMPLEX": "deepseek-v4.1-flash"})
    for q in ["你好", "write a function to sort a list", "prove Fermat's last theorem",
              "summarize this paragraph for me"]:
        print(repr(q), "->", router.route_question(q))
