# 对话上下文：AiScientist 代码阅读 — MLE 数据流完成

> **生成时间**：2026-04-19
> **状态**：MLE Track 数据流分析完成，Paper Track 部分细节待探索
> **项目路径**：`/Users/admin/PycharmProjects/AiScientist`

---

## 📋 问题背景

### 项目信息
- **项目**：AiScientist —— 一个自动化科研 AI 系统
- **仓库**：`/Users/admin/PycharmProjects/AiScientist`（本地，git 分支 `main`）
- **目的**：纯代码阅读（无修改），理解整个系统架构、数据流、设计模式

### 已完成阅读层级
| 层级 | 内容 | 状态 |
|------|------|------|
| L1 | 项目整体结构、README、CLAUDE.md | ✅ 完成 |
| L2 | 六大 package 职责划分、Module Map | ✅ 完成 |
| L3-A | CLI 入口（aisci_app） | ✅ 完成 |
| L3-B | JobService / Worker / JobRunner | ✅ 完成 |
| L3-C | PaperDomainAdapter + SubagentCoordinator (DAG) | ✅ 完成 |
| L3-D | EmbeddedPaperEngine 主循环 + prompt 体系 | ✅ 完成 |
| L3-E | aisci_core 数据模型（models.py 全读） | ✅ 完成 |
| L3-F | Tool ABC + build_* 函数 + dispatch 数据流 | ✅ 完成 |
| L3-G | MLE Track 完整数据流（adapter.py + orchestrator.py 全读） | ✅ 完成（本次新增）|

### 涉及组件
所有 6 个 package 均已覆盖：
- `aisci_app` — CLI / TUI / JobService / worker_main
- `aisci_core` — 数据模型 / SQLite store / JobRunner / paths
- `aisci_agent_runtime` — LLM client / Tool ABC / retry / tracing
- `aisci_runtime_docker` — 容器生命周期 / workspace mounting
- `aisci_domain_paper` — Paper track adapter / EmbeddedPaperEngine / subagents / Jinja prompts
- `aisci_domain_mle` — MLE track adapter / EmbeddedMLEEngine / candidate registry

---

## 🎯 实现目标

**本 session 为纯阅读，无代码变更。**
目标：理解 AiScientist 的完整系统架构，包括：
1. 两条 track（paper / mle）的数据流
2. Agent 主循环与子 agent 协调机制
3. LLM client 实现（openai SDK 封装层）
4. Tool ABC 设计与 PaperBench 渊源
5. 评分/验证机制差异

---

## ✅ 关键技术发现

### 1. 设计来源：直接移植自 PaperBench
- `aisci_agent_runtime/llm_client.py` 注释：**"Mirrors PaperBench's dual-completer architecture"**
- `aisci_agent_runtime/tools/base.py` 注释：**"mirrors PaperBench's `basicagent.tools.base.Tool`"**
- PaperBench 仓库：`https://github.com/openai/preparedness`（`project/paperbench/` 子目录）
- MLE-Bench 仓库：`https://github.com/openai/mle-bench`（75 个 Kaggle 竞赛评测）

### 2. LLM Client 架构
```
openai Python SDK (底层 HTTP)
  ├── CompletionsLLMClient  → Chat Completions API（glm-5, gemini-3-flash）
  └── ResponsesLLMClient    → OpenAI Responses API（gpt-5.4）

LLMConfig 关键字段：
  provider, model, api_mode, reasoning_effort, reasoning_summary,
  web_search, context_window, use_phase, temperature, clear_thinking

Retry 策略：
  - 总预算：2 小时
  - 退避：1s → 300s（指数）
  - 不重试：ContentPolicyError, ContextLengthError
```

### 3. Tool ABC（`aisci_agent_runtime/tools/base.py`）
```python
class Tool(ABC):
    def name(self) -> str: ...
    def execute(self, shell, **kwargs) -> str: ...
    def get_tool_schema(self) -> dict: ...
    def execute_with_constraints(...): ...  # paper track 专用
```

### 4. Paper Track vs MLE Track 核心差异

| 维度 | Paper Track | MLE Track |
|------|-------------|-----------|
| 主循环 | DAG + ThreadPoolExecutor（并行读论文） | Flat sequential loop |
| 最大步数 | 80 步 | 500 步 |
| 子 agent 协调 | `SubagentCoordinator`（有向图） | 直接工具调用 |
| ImplementTool 模式 | `full` / `fix` | `full` / `fix` / `explore` / `refine` / `ensemble` |
| 评分机制 | `reproduce.sh` exit code + rubric 核查 | `mlebench grade.py`（Kaggle 评分函数）|
| 网络策略 | `--network host`（允许下载论文依赖） | `--network bridge`（数据隔离，防作弊）|
| 提交验证 | 论文复现完整性 | CSV 格式/列名/行数硬检查 + NaN 软警告 |

---

## 📊 MLE Track 完整数据流（本次新增）

```
MLESpec (competition_zip / data_dir / bundle)
  ↓
MLEDomainAdapter.run()
  ↓
_stage_inputs()
  ├─ 解压竞赛数据 → workspace/data/
  ├─ 安全过滤：SENSITIVE_DATA_MARKERS
  │    ("answer", "gold_submission", "solution",
  │     "test_with_solutions", "verification_label", "verification_set")
  ├─ grading_config_path 显式屏蔽（ValueError）
  └─ 只保留 public：train.csv, test.csv, description.md, sample_submission.csv
  ↓
git init workspace/code/
  ↓
docker run -d --network bridge -v workspace:/home aisci-mle:latest sleep infinity
  ↓
EmbeddedMLEEngine.run()  ← 主循环 max 500步
  ├─ analyze_data()     → DataAnalysisSubagent → agent/analysis/summary.md
  ├─ prioritize_tasks() → PrioritizationSubagent → agent/prioritized_tasks.md
  ├─ implement(mode)    → ImplementationSubagent [5 modes]
  │    ← 注入：exp_log.md 最后一个 Session（grep+sed）
  │    → git commit 到 code/
  │    → _snapshot_submission() 快照 candidates/<timestamp>.csv
  ├─ run_experiment()   → ExperimentSubagent
  │    ← 注入：impl_log.md 最后一个 Session（grep+sed）
  │    → 训练+推理 → code/submission.csv
  │    → _snapshot_submission() 快照 candidates/<timestamp>.csv
  ├─ [每5步] _build_reminder()
  │    ├─ 4阶段时间提醒（0-50% / 50-70% / 70-85% / 85%+）
  │    ├─ submission.csv 存在性检查
  │    └─ impl/exp 调用差≥2 警告，≥4 强制警告
  └─ submit()
       ├─ 硬检查：列名/行数不匹配 → 拒绝（不计次数）
       ├─ 软警告：NaN / 时间<50% 且首次提交 → 警告可绕过
       └─ 通过 → 跳出主循环
  ↓
_finalize()
  → summary.json: runtime_seconds, total_tokens, impl_calls, exp_calls, submission_exists
  → 复制 impl_log/exp_log/prioritized_tasks/analysis_summary → logs/
  ↓
_materialize_registry()  ← finally 块，crash 也执行
  → submission_registry.jsonl（所有 candidates/ 快照 + 得分占位）
  → champion_report.md（最优 submission 摘要）
  ↓
_collect_artifacts()
  → 13 种 ArtifactRecord 写入 SQLite
  (analysis_summary, prioritized_tasks, impl_log, exp_log,
   submission, submission_registry, champion_report, conversation_log, ...)
  ↓
_maybe_validate()
  ├─ legacy_grade  → host 执行 vendored_mlebench_lite/mlebench/grade.py
  ├─ docker_exec   → 容器内执行 validation_command
  └─ none          → 跳过
  ↓
JobStatus.SUCCEEDED
```

### File-as-Bus 文件表

| 文件 | 写入方 | 读取方 |
|------|--------|--------|
| `agent/analysis/summary.md` | DataAnalysisSubagent | 主 agent system prompt |
| `agent/prioritized_tasks.md` | PrioritizationSubagent | ImplementTool (full mode) |
| `agent/impl_log.md` | ImplementationSubagent（追加） | ExperimentTool（注入上下文）|
| `agent/exp_log.md` | ExperimentSubagent（追加） | ImplementTool（注入上下文）|
| `agent/candidates/<ts>.csv` | `_snapshot_submission()` | champion_report 生成时排序 |

---

## 📁 关键文件清单（已全读）

### aisci_core
- `src/aisci_core/models.py` — 全部数据模型（JobType, JobStatus, RunPhase, PaperSpec, MLESpec, JobSpec, RuntimeProfile, JobRecord, JobPaths 等）

### aisci_domain_paper
- `src/aisci_domain_paper/tools/basic_tool.py` — Tool ABC 实现 + build_* 工厂函数
- `src/aisci_domain_paper/tools/spawn_subagent_tool.py` — build_main_tools() 入口
- `src/aisci_domain_paper/tools/implementation_tool.py` — ImplementationTool, SpawnEnvSetupTool, SpawnResourceDownloadTool
- `src/aisci_domain_paper/tools/__init__.py` — 所有 tool exports

### aisci_domain_mle
- `src/aisci_domain_mle/contracts.py` — MLE 内部 dataclasses（RuntimeWorkspaceLayout, ValidationPlan, RuntimeOrchestrationPlan 等）
- `src/aisci_domain_mle/adapter.py` — MLEDomainAdapter（1078 行，完整数据流）
- `src/aisci_domain_mle/orchestrator.py` — EmbeddedMLEEngine（1451 行，主循环+所有工具）

### aisci_agent_runtime
- `src/aisci_agent_runtime/llm_client.py` — 前 180 行（双 client + LLMConfig + retry 策略）
- `src/aisci_agent_runtime/tools/base.py` — 前 60 行（Tool ABC）

---

## 🔧 系统关键常量

```python
# MLE 主循环
AISCI_MAX_STEPS = 500          # 环境变量可覆盖
REMINDER_FREQ = 5              # 每 5 步触发 reminder

# Retry 预算
TOTAL_RETRY_BUDGET = 7200s     # 2 小时
BACKOFF_MIN = 1s, MAX = 300s

# 敏感数据过滤
SENSITIVE_DATA_MARKERS = (
    "answer", "gold_submission", "solution",
    "test_with_solutions", "verification_label", "verification_set"
)

# SubmitTool 早提交阈值
EARLY_SUBMIT_THRESHOLD = 0.5   # 时间使用率 < 50% 触发软警告

# impl/exp 平衡警告阈值
BALANCE_WARN_GAP = 2
BALANCE_HARD_GAP = 4
```

---

## 🚀 下一步可探索方向

1. **Paper Track 数据流**（对应 MLE 侧的深度）
   - `src/aisci_domain_paper/adapter.py` — Paper adapter 全读
   - `src/aisci_domain_paper/engine.py` 或 `orchestrator.py` — Paper 主循环
   - SubagentCoordinator DAG 具体实现（`coordinator.py`）

2. **Jinja Prompt 体系**
   - `src/aisci_domain_paper/prompts/` — 所有 `.j2` 模板文件
   - system prompt 的动态注入机制

3. **aisci_agent_runtime 深读**
   - `llm_client.py` 后半部分（context summarization 机制）
   - `tracing.py` — 追踪实现
   - `subagents/base.py` — SubagentOutput, SubagentStatus

4. **eval 工具补全**
   - `vendored_mlebench_lite/mlebench/grade.py` — Kaggle 评分函数
   - `src/aisci_domain_mle/candidate_registry.py` — 候选提交注册

5. **Docker 运行时**
   - `src/aisci_runtime_docker/` — 容器生命周期细节

---

## 💡 使用方法（新对话恢复上下文）

在新对话开始时，将以下内容发给 Claude：

```
请读取这个文件来恢复我们的对话上下文：
/Users/admin/PycharmProjects/AiScientist/对话上下文-AiScientist代码阅读-MLE数据流完成.md

这是一个纯代码阅读 session，项目在 /Users/admin/PycharmProjects/AiScientist，
不需要修改任何代码。
```

---

## 📝 备注

### 设计理念提炼
1. **File-as-Bus**：所有 agent 间通信通过磁盘文件，天然持久化、可审计
2. **抗崩溃设计**：`_materialize_registry()` 在 `finally` 块，即使 agent 崩溃也能恢复候选提交
3. **安全优先**：MLE track 三重保障防止私有评分数据泄露
4. **从 benchmark 来**：整套 agent 基础设施直接移植自 PaperBench，不是从零长出来的

### 参考资料
- PaperBench: https://github.com/openai/preparedness （project/paperbench/）
- MLE-Bench: https://github.com/openai/mle-bench

### 本地运行:
uv run aisci --env-file .env mle run \
  --zip /abs/path/to/competition.zip \
  --name <slug> \
  --llm-profile gpt-5.4 \
  --time-limit 12h \
  --local \
  --wait