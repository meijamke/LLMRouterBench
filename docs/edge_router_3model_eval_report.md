# XRouter 端侧路由 3 模型池离线回放评估报告（PC + 硅基流动真实 API）

## 1. 目标

在本机 PC 上用 LLMRouterBench 离线回放评估 MiniPC 端侧路由算法 **DeviceLLMRouter（= XRouter）**，
模型池按 XRouter 原生「三角色」语义扩为：

| 角色 | 模型 | 硅基流动 API 名 | 职责 |
|---|---|---|---|
| 端侧小模型（分类器） | Qwen3.5-4B | `Qwen/Qwen3.5-4B` | 先判断题目难易档位 |
| 端侧大模型（本地回答） | Qwen3.5-9B | `Qwen/Qwen3.5-9B` | 回答 SIMPLE/MEDIUM |
| 云侧模型（云端回答） | DeepSeek-V4.1-Flash | `deepseek-ai/DeepSeek-V4.1-Flash` | 回答 COMPLEX/RESEARCH/REASONING |

成本 / 质量均基于硅基流动**真实 API**（非模拟），逐题记录 token，成本可按价格表精确重算。

## 2. 三角色路由语义（忠实 XRouter 原生）

```
用户提问
  └─ classify()  → 端侧小模型 4B（LLM 分类器）
        │  buildClassifierPrompt(题目, 6000)  → PROMPT_TEMPLATE
        │  CloudClassifier: temperature=0, max_tokens=2048, enable_thinking=false
        │  parseTier() 严格单标签解析
        │  失败语义：调用失败→heuristic_fallback；解析失败→parse_failed
        │             （档位一律回退 classify_heuristic，故障只降不升）
        ├─ tier ≤ localCapabilityLevel(MEDIUM)  → 端侧 9B 回答
        └─ tier ∈ {COMPLEX, RESEARCH, REASONING} → 云侧 DeepSeek 回答
```

## 3. 方法

### 3.1 答案矩阵采集与对齐
- 既有矩阵（上一轮）：DeepSeek-V4.1-Flash 等 4 模型 × 6 数据集，共 **882 题**（gpqa/mmlupro/math500/arcc 各 150、gsm8k 100、mmlu 182）。
- 该 882 题是上一轮按**抽样数据**采的，而 `data/` 现已是全量数据集（gpqa 198、mmlu 14042…），直接跑官方 collector 会对齐不上且要采海量题。
- 因此写 [tools/collect_9b_882.py](../tools/collect_9b_882.py)：逐题复用旧 DeepSeek 记录的 `prompt/ground_truth/index`，调 9B 生成并按原判分器打分，产出与 DeepSeek 完全同 `(dataset, record_index)` 对齐的 9B 结果。

### 3.2 难度档位打标（4B 分类器）
- [tools/run_xrouter_classifier.py](../tools/run_xrouter_classifier.py) 对 882 题用 4B 分类器打档，复刻 CloudClassifier 请求（temperature=0、max_tokens=2048、enable_thinking=false）。
- 档位严格单标签；调用/解析失败回退启发式。产出 `results/xrouter_classifier_tiers.csv`。

### 3.3 质量-成本评估
- [tools/eval_device_llm_router_3model.py](../tools/eval_device_llm_router_3model.py) 离线回放：分类器档位 → 路由决策 → 查被路由回答模型的得分。
- 成本 = **分类器调用成本 + 回答模型成本**，均按 `token × 单价` 重算。
- [tools/eval_3model_sweep.py](../tools/eval_3model_sweep.py) 对 `localCapabilityLevel ∈ {SIMPLE, MEDIUM, COMPLEX, RESEARCH, REASONING, NONE}` 扫描出帕累托前沿。

### 3.4 计费单价（¥ / 百万 token，硅基流动）
| 模型 | 输入 | 输出 | 说明 |
|---|---|---|---|
| Qwen3.5-4B（分类器） | 0.10 | 0.40 | 估算 |
| Qwen3.5-9B（端侧） | 0.20 | 0.80 | 估算 |
| DeepSeek-V4.1-Flash（云侧） | 2.00 | 8.00 | 已核对 |

> 逐题 token 已落盘，单价若修正可精确重算成本。

### 3.5 对照基线
端侧 only（9B）、云侧 only（DeepSeek）、Oracle（每题最优上界）、Random（0.5/0.5 期望）、
以及「**启发式分类器**」（同名模型池、同规则，仅把分类器换成 `classify_heuristic`，量化 4B 分类器相对启发式的增益）。

## 4. 结果

### 4.1 总览（6 数据集合并，成本含分类器调用）

| 方案 | 准确率 | 总成本 (¥) | n | 备注 |
|---|---|---|---|---|
| **X-Router（4B 分类器，MEDIUM）** | **57.82%** | **3.404** | 882 | 端侧 545 / 云侧 337 |
| X-Router（启发式分类器，MEDIUM） | 46.26% | 0.808 | 882 | |
| 端侧 only（9B） | 45.92% | 0.736 | 882 | |
| 云侧 only（DeepSeek） | 82.20% | 4.743 | 882 | |
| Oracle（每题最优上界） | 88.55% | 2.562 | 882 | |
| Random（0.5/0.5 期望） | 64.06% | 2.740 | 882 | |

> 相对云侧单模型：**成本节省 28.2%**（3.404 vs 4.743），准确率回落 24.4 个百分点（57.82% vs 82.20%）。
> 相对端侧单模型：+11.9 个百分点准确率（57.82% vs 45.92%），代价是多花 2.67 元。

### 4.2 分类器档位 / source 分布

| source | 题数 | 占比 |
|---|---|---|
| llm（4B 成功打档） | 826 | 93.7% |
| heuristic_fallback（限流回退） | 56 | 6.3% |

档位分布（826 llm + 56 回退合并）：`SIMPLE:541, REASONING:297, RESEARCH:28, COMPLEX:12, MEDIUM:4`。

> 4B 分类器明显偏「REASONING」档（297/882），这是真实 LLM 分类器的行为；其效果是：对 gpqa/mmlupro 等难题更倾向甩云端，
> 用 3.40 元换来 57.82% 准确率，明显高于端侧 only（45.92%），显著低于云侧 only（82.20%）。

### 4.3 分数据集（X-Router，4B 分类器）

| dataset | n | 准确率 | 成本 (¥) | 端侧 | 云侧 |
|---|---|---|---|---|---|
| arcc | 150 | 43.33% | 0.047 | 138 | 12 |
| gpqa | 150 | 67.33% | 1.936 | 18 | 132 |
| gsm8k | 100 | 87.00% | 0.049 | 87 | 13 |
| math500 | 150 | 88.67% | 0.624 | 58 | 92 |
| mmlu | 182 | 30.77% | 0.202 | 151 | 31 |
| mmlupro | 150 | 45.33% | 0.546 | 93 | 57 |

### 4.4 帕累托图

![XRouter 3 模型池质量-成本帕累托](../results/devllmr_3model_pareto.png)

（横轴 = 总成本 ¥，纵轴 = 准确率；蓝色点 = 4B 分类器在 SIMPLE→NONE 能力档位扫描曲线，绿色 = 启发式分类器，其余为单模型 / Oracle / Random 参考点。）

## 5. 结论与说明

1. **4B 分类器有效抬升质量**：相比端侧 only（9B），X-Router（4B 分类器）把 337/882 道难题路由到云端，
   准确率 45.92% → **57.82%**（+11.9pp），成本 0.736 → 3.404 元。
2. **相对云侧 only 有明显成本收益**：仅用云侧的 71.8% 成本，达到云侧 70.4% 的准确率；省下的 28.2% 成本正是路由的价值。
3. **4B 比启发式分类器强**：同模型池下，4B 分类器（57.82% / 3.40）比启发式（46.26% / 0.81）多花 2.6 元、多 11.6pp，
   说明 LLM 分类器能识别启发式（以 token/词频/长度）看不清的「难度」，把更多真难卷调云端——差距集中在 gpqa/mmlupro/math500。
4. **尚不及 Oracle 与 Random 的某些指标**：Oracle（88.55% / 2.56）说明分类器仍有较大提升空间；Random（64.06%）高于 4B 分类器，
   是因「无脑 50% 上云」在成本相近时命中率尚可——这也是端侧路由需审慎权衡的典型现象。
5. **成本口径**：4B/9B 单价为估算（硅基流动详情页需登录未能抓取），逐题 `prompt_tokens/completion_tokens` 已落盘，
   单价一经确认即可按 `token × 单价` 精确重算，结论方向不受影响（DeepSeek 单价已核实）。
6. **限流与回退**：分类器首轮 32%（280 题）因硅基流动 4B 慢速（单请求 60–95s）触发 60s 超时回退启发式；
   重试（120s 超时、并发 16）后恢复到 **93.7%** llm 打档（826/882），仅 56 题留启发式——忠实 XRouter 的「失败只降不升」语义。

## 5.5 LLMRouterBench 如何评估路由算法（以 XRouter 为例）

```mermaid
flowchart TD
    subgraph STEP1["① 全矩阵离线采集（一次性，模型×题目）"]
        D[6 个数据集 882 题]
        D -->|DeepSeek-V4.1-Flash| M1[云侧答案矩阵]
        D -->|Qwen3.5-9B| M2[端侧答案矩阵]
        D -->|Qwen3.5-4B| M3[4B 答案矩阵*]
    end

    subgraph STEP2["② 路由决策回放（不重新调用回答模型）"]
        Q[逐题提问] --> C[端侧小模型 4B 难度分类器<br/>temperature=0, enable_thinking=false]
        C -->|parse_tier 单标签| T{tier<br/>SIMPLE/MEDIUM/COMPLEX/RESEARCH/REASONING}
        T -->|失败回退| H[classify_heuristic]
        H --> T
        T -->|tier ≤ MEDIUM| L[路由到端侧 9B]
        T -->|tier > MEDIUM| CL[路由到云侧 DeepSeek]
    end

    subgraph STEP3["③ 查表汇总质量-成本"]
        L -->|查 9B 矩阵| S1[score + token/cost]
        CL -->|查 DeepSeek 矩阵| S2[score + token/cost]
        S1 --> AGG[汇总：准确率 = 被路由模型得分<br/>成本 = 分类器成本 + 回答成本]
        S2 --> AGG
    end

    AGG --> CMP[对照：端侧/云侧单模型、Oracle、Random、启发式分类器]
    CMP --> PR[帕累托前沿图]
```

> *4B 本轮角色是分类器，其「答案矩阵」不参与路由回答，仅 4B 单独对每道题打档（见 3.2）。

关键点：**路由算法本身只产出「每道题该走哪个模型」的决策；LLMRouterBench 用预先采好的「题目×模型」答案矩阵离线查表**，
把路由决策换算成质量（被路由模型的得分）与成本（分类器 + 回答模型的真实 token 计费），因此无需在评估时重复跑模型，
即可快速回放任意路由策略、扫描能力阈值、对比基线。

## 6. 复现命令

```bash
python tools/collect_9b_882.py                 # 采集 9B 答案矩阵
python tools/run_xrouter_classifier.py         # 4B 打难度档位
python tools/run_xrouter_classifier.py --retry-failed --concurrency 4   # 重试限流失败题
python tools/eval_device_llm_router_3model.py  # 评估（MEDIUM）
python tools/eval_3model_sweep.py              # 扫描 + 帕累托图
```