# 任务二：两台鸿蒙真机的 LLMRouterBench 评估 —— 实施计划

> 前置结论：DeviceLLMRouter 的路由**质量**评估已在任务一离线完成（可复现、可审计）。
> 真机的增量价值 = **端侧真实成本轴**（延迟/能耗）。本文梳理 MiniPC 框架、规划评估映射、给出两条可执行路径。

---

## 0. 结论速览

| 事实 | 结论 |
|---|---|
| 算力端（compute-node）**无 HTTP/TCP 接口** | PC 不能直连；聊天端↔算力端走 HarmonyOS 软总线（BLE 发现 + `linkEnhance`） |
| 路由内核 `XRouter` 跑在**算力端**（非聊天端） | 决策逻辑已导出（任务一 `baselines/device_llm_router.py`），可离线回放 |
| 本机 PATH **无 hdc** | 需提供 `hdc.exe` 全路径或确认运行机器 |
| MiniPC 自带**推理计时 + 分片耗时日志**（hilog `NearbyChat`） | 方案 C 无需改代码即可测端侧延迟 |
| 云侧模型/分类器 = OpenAI 兼容 HTTP（`cloud_config.json` baseUrl） | 云侧可被 PC 直接调 |

---

## 1. MiniPC 框架梳理

### 1.1 角色与工程

| 工程 | 角色 | 关键模块 |
|---|---|---|
| `apps/agent-client` | **聊天端**：用户交互、Agent 工具调用、结果汇总 | `agent-core`、`nearby/LinkEnhanceChannel`、`model/OpenAiCompatibleModel`（云） |
| `apps/compute-node` | **算力端**：本地模型推理 + 路由决策，`IS_SERVER=true` | `llm/XRouter`(路由)、`llm/InferenceHost`、`llm/LocalChat`、`cpp/gewu_probe.cpp` |
| `packages/inference-protocol` | 推理通信协议 JS 参考实现 | `protocol='nearby-chat-v1'`、分片 480B/Base64 |

### 1.2 主流程（一次请求）

```
聊天端 agent-client                          算力端 compute-node                        云侧
   │  用户输入                                    │                                    │
   │── BLE 发现 / linkEnhance 软总线 ───────────▶│  BackendRouter(policy='xrouter')     │
   │                                             │    └─ XRouter.decideAsync            │
   │                                             │        ├─ classify(启发式/云分类器)   │
   │                                             │        └─ tier ≤ 本地能力?            │
   │                                             │              ├─ 是 → GEWU/KNPU 本地推理│
   │                                             │              └─ 否 → cloud_config.json │── HTTP ─▶ DeepSeek...
   │◀────────────── 结果（分片回传）──────────────│                                             │
```

- **传输**：`@kit.DistributedServiceKit`（`linkEnhance.createServer('nearby-compute')`），非 HTTP/TCP，PC 无法接入。
- **计时**：推理计时从执行端受理开始（含模型加载），日志 tag `NearbyChat`：
  - `le-write-start`：请求编号、片数、应用发送排队耗时；
  - `le-frame-recv`：累计收片数/字节数、从首片起的耗时、新分片间隔。

### 1.3 路由内核

`XRouter.ets`（= DeviceLLMRouter）：
- 复杂度 5 档 `SIMPLE/MEDIUM/COMPLEX/RESEARCH/REASONING`（秩 0–4）；
- `tier ≤ localCapabilityLevel`（默认 MEDIUM）→ 本地，否则 → 档位云模型（回退 COMPLEX）；
- 分档来源：启发式（确定性，已导出）或云分类器（`CloudClassifier` 调 OpenAI 兼容端点）。

---

## 2. 评估目标 → LLMRouterBench 映射

| 轴 | 真机数据 | 落到 LLMRouterBench |
|---|---|---|
| 质量 | 端侧/云侧每题是否正确 | 复用任务一 `results/bench` 分数（已采） |
| 成本 | 端侧真实**延迟**(或能耗) | 覆写 `record.cost`（`tools/recode_cost.py` 思路） |
| 路由决策 | `XRouter` 决策 | 离线回放（`baselines/device_llm_router.py`） |

帕累托前沿 = 复用 `BaselineAggregator` 汇总 + `tools/eval_device_llm_router.py` 的对照逻辑。

---

## 3. 方案 C（推荐先行）：真机只测成本轴

**原理**：不改 MiniPC 代码，用 `hdc shell hilog` 抓算力端的计时日志，得到端侧延迟；质量轴用任务一分数。

**步骤**：
1. 接入：`hdc list targets` 拿到两台序列号（聊天端、算力端）。
2. 起日志：`hdc -t <算力端序列号> shell hilog -T NearbyChat` 实时抓取。
3. 灌题：通过聊天端逐题输入（或 `hdc shell uinput` 脚本化点击/输入），每题记录：
   - 端侧**首字 TTFT**（`infer output ... elapsedMs`，`tools/parse_nearby_timing.py` 已可解析）；
   - 该题是否走端侧（`XRouter.reason` 日志 `rule=within_local_capability` / `escalate_cloud`）。
4. 汇总：延迟中位数（多次取，抗抖动）→ 覆写对应题目的 `record.cost`。
5. 出图：`tools/eval_device_llm_router.py` 用新 cost 重算帕累托。

**局限**：成本=延迟（能耗需 `hidumper` 电池统计，属进阶）；灌题靠 UI 自动化，吞吐低；仅测端侧成本，云侧仍用 API 计费或真机无云侧成本。

---

## 4. 方案 A：算力端加 HTTP 桥（完整端到端）

**设计**：在 `compute-node` 加一个本地 TCP/HTTP server（ArkTS `@ohos.net.socket` 的 `TCPSocket.bind+listen`，手写最小 HTTP/1.1 或自定义行协议），把本地模型包成 OpenAI 兼容 `/v1/chat/completions`。PC 经 `hdc fport tcp:LOCAL tcp:PORT` 转发直连。

**改动点**（需 DevEco 重编译 + 签名，用户配合）：
1. 新增 `entry/src/main/ets/bridge/HttpBridge.ets`：监听 `127.0.0.1:<port>`，解析请求 → 调 `LocalChat`/`InferenceHost` → 回 JSON（含 `latency_ms`）。
2. `entry/src/main/module.json5` 声明网络/后台权限。
3. PC 侧：`config/edge_router_config_harmony.yaml` 把 `base_url` 指向 `http://127.0.0.1:<local>/v1`。

**优点**：自动化、精确、可跑全矩阵；**缺点**：改 ArkTS + 重编译（本机无 HarmonyOS SDK，需在你机器做）。

---

## 5. hdc 接入清单（需要你提供）

- [ ] `hdc.exe` 的**绝对路径**（或把它加入 PATH 后重开会话）；
- [ ] `hdc list targets` 输出（两台序列号 + 哪台是聊天端/算力端）；
- [ ] 算力端 `cloud_config.json` 是否已配置云侧 `baseUrl/model/apiKey`；
- [ ] 决定走 **C** 还是 **A**。

## 6. 分阶段执行清单

| 阶段 | 动作 | 产出 |
|---|---|---|
| P0 | 你提供 hdc 路径 + 双机序列号 + 路线选择 | 环境就绪 |
| P1(C) | hdc 抓 `NearbyChat` 计时，灌题测端侧延迟 | 端侧成本样本 |
| P1(A) | 我写 ArkTS HTTP 桥，你 DevEco 编译安装 | 算力端 OpenAI 端点 |
| P2 | cost 覆写 + 离线回放 + 帕累托 | 真机评估结果 |
| P3 | 文档化 + 提交 | 闭环 |
