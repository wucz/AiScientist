# 对话上下文：AiScientist 代码深度阅读

> 会话名称：read_code
> 生成时间：2026-04-18
> 状态：L2 + L3 全部完成（含 L3-E Prompt注入 + L3-F 工具定义与注册）

---

## 📋 问题背景

### 项目信息
- **项目路径**：`/Users/admin/PycharmProjects/AiScientist`
- **项目性质**：AI 自动化科研系统，可自动复现 ML 论文（paper 轨道）或优化 Kaggle 竞赛（mle 轨道）
- **运行环境**：Mac M3，无 CUDA GPU，Docker 必须可用
- **包管理**：uv，Python 3.12+，Pydantic v2

### 当前状态
- CLAUDE.md 已创建（项目根目录）
- 代码阅读已完成 L2 + L3 全部深入点
- 没有做任何代码修改，仅阅读分析

### 涉及组件
| 包名 | 职责 |
|------|------|
| `aisci_app` | CLI（Typer）、JobService、TUI、worker 入口 |
| `aisci_core` | 数据模型、SQLite store、JobRunner、路径工具 |
| `aisci_agent_runtime` | LLM 客户端、token 计数、重试、tracing |
| `aisci_runtime_docker` | 容器生命周期、workspace 挂载、shell exec |
| `aisci_domain_paper` | Paper 轨道 adapter、Engine、subagents、Jinja prompts |
| `aisci_domain_mle` | MLE 轨道（结构类似 paper，未深入） |

---

## 🎯 实现目标

本次会话目标为**纯阅读理解**，不修改代码：
1. 理解系统整体行为和数据流
2. 准备修改/扩展该系统
3. 学习设计模式，用于改造自己的项目

---

## ✅ 已完成的代码阅读

### L2 数据流（完整走读）

```
paper.md（文件）
  ↓ build_paper_job_spec()          [presentation.py:329]
JobSpec（Pydantic，不可变）
  ↓ store.create_job()              [store.py:101]
JobRecord + SQLite 持久化（含 job_id = 时间戳+hex）
  ↓ service.spawn_worker()          [service.py]
subprocess: python -m aisci_app.worker_main <job_id>
  ↓ adapter._stage_inputs()         复制 paper.md 到 workspace/paper/
  ↓ runtime.start_session()         docker run -d ... sleep infinity
Docker 容器（运行中）+ workspace 磁盘挂载
  ↓ render_main_agent_system_prompt(capabilities)
system_prompt（按运行时能力动态生成）+ tools（schema 列表）
  ↓ 主循环（最多80步）
  LLM.chat(messages, tools) → tool_calls → docker exec bash -lc "cmd"
  → 结果追加 messages，文件写入磁盘（File-as-Bus）
  ↓ submit() → SubagentCompleteSignal
artifacts 收集 + 最终验证 + zip 打包
JobStatus.SUCCEEDED
```

### L3-D：LLM Profile 解析

**文件**：`src/aisci_agent_runtime/llm_profiles.py`，`config/llm_profiles.yaml`

```yaml
# yaml 结构
defaults:
  paper: glm-5
  mle: gpt-5.4
backends:
  openai: {type: openai, env: {api_key: {var: OPENAI_API_KEY}}}
  azure-openai: {type: azure-openai, env: {endpoint/api_key/api_version}}
profiles:
  gpt-5.4: {backend: openai, model: gpt-5.4, api: responses, limits: {...}}
  glm-5:   {backend: azure-openai, api: completions, features: {clear_thinking: true}}
```

**解析链**：`resolve_llm_profile("gpt-5.4")` → `load_llm_registry()` → `_merged_profile_map()`（支持 `extends` 继承）→ `LLMProfile`（frozen dataclass）

**关键**：
- `api_mode=responses` → `ResponsesLLMClient`（OpenAI Responses API）
- `api_mode=completions` → `CompletionsLLMClient`（Chat Completions API）
- env var 名称全在 YAML 配置，代码不硬编码

---

### L3-A：Docker 容器生命周期

**文件**：`src/aisci_runtime_docker/agent_session.py`（`DockerRuntimeManager` 只是空壳，继承 `AgentSessionManager`）

```
prepare_image()   → docker image inspect / pull（按 pull_policy）
create_session_spec() → 组装挂载表：
    workspace/        → /home/          (整体挂载)
    logs/             → /workspace/logs
start_session()   → docker run -d --name aisci-<job_id>-<hex> \
                       -w /home/submission \
                       --network host \
                       -v workspace:/home \
                       aisci-paper:latest sleep infinity
exec()            → docker exec -w /home/submission <name> bash -lc "<cmd>"
cleanup()         → docker rm -f <name>
```

**关键**：
- 容器用 `sleep infinity` 保活，所有执行通过 `docker exec`
- `-lc` = login shell，确保 `~/.bashrc`/venv/PATH 生效
- paper 轨道默认 `--network host`，mle 轨道默认 `--network bridge`

---

### L3-B：主循环 messages 管理

**文件**：`src/aisci_domain_paper/engine.py:run_main_loop()`

```python
messages = [
    {"role": "system", "content": system_prompt},   # 固定
    {"role": "user",   "content": initial_prompt},  # 固定
    # 每步追加：
    {"role": "assistant", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": ..., "content": "<bash 结果>"},
    {"role": "user", "content": "<reminder>"},      # 每5步插一条
]
```

**Context 溢出两级处理**：
1. `summarize_messages()` — 调用 LLM 压缩历史为摘要（保留语义）
2. `_reduce_messages()`/`prune_messages()` — 直接截断老消息（最后手段）

**Reminder 机制**：每 `reminder_freq=5` 步插入一条 user 消息，提醒当前步数/剩余时间/reproduce.sh 状态。

---

### L3-C：Subagent 执行流程（两级结构）

**文件**：`src/aisci_domain_paper/engine.py`，`src/aisci_domain_paper/subagents/coordinator.py`

#### 类型 1：read_paper 并行分析（SubagentCoordinator）

```
read_paper()
  └─ SubagentCoordinator.run(tasks)
       └─ _group_by_level() 拓扑排序：
          Level 0（并行）: StructureSubagent       → structure.md
          Level 1（并行）: AlgorithmSubagent        → algorithm.md
                           ExperimentsSubagent      → experiments.md
                           BaselineSubagent         → baseline.md
          Level 2（串行）: SynthesisSubagent        → summary.md
       └─ ThreadPoolExecutor 同层并发执行
```

每个 Subagent 内部也是完整的 LLM tool_call 循环（独立 messages）。

#### 类型 2：implement/experiment/spawn_subagent（顺序执行）

```
implement(task="...", mode="full")
  └─ engine.execute_named_subagent(subagent_type="implementation")
       └─ state_manager.create_session("implementation")  # 独立日志目录
       └─ ImplementationSubagent(engine, shell, llm, config)
            └─ subagent.run(context=build_context())
                 └─ 完整 LLM tool_call 循环（独立 messages）
                 └─ bash 写代码、git commit
       └─ 返回 SubagentOutput(content, status, log_path)
  tool_result = 格式化字符串 → 追加到 main messages
```

**关键**：subagent 有独立 messages，主 agent 只看最终字符串，context 不被子任务污染。

---

### L3-E：Prompt 注入机制（新手向详解）

#### 背景：LLM API 的消息格式

调用 LLM 时传入的是一个 **消息列表**，不是单个字符串：

```python
messages = [
    {"role": "system",    "content": "你是谁、有什么工具、怎么工作"},
    {"role": "user",      "content": "具体任务是什么"},
    {"role": "assistant", "content": "..."},  # LLM 上一轮回复（含 tool_calls）
    {"role": "tool",      "content": "工具执行结果"},
    # 每步追加，历史全保留
]
```

Agent 的主循环就是：往 messages 追加内容 → 调 LLM → 执行工具 → 追加结果 → 重复。

#### 三层注入结构

```
Layer 1: system 消息      → 角色定义 + 工具列表 + 工作流程 + 路径表（固定，只写一次）
Layer 2: 首条 user 消息   → 本次任务目标 + capabilities JSON + 路径表（固定，只写一次）
Layer 3: 后续 user 消息   → 每 N 步插入的时间/步数/行为纠偏 reminder（动态）
```

#### 能力条件渲染（最重要的设计）

**文件**：`src/aisci_domain_paper/prompts/templates.py`（103KB，11 个 system prompt）

```python
def render_main_agent_system_prompt(capabilities) -> str:
    if capabilities["online_research"]["available"]:
        research_lines = "- **web_search** — 搜索互联网\n- **link_summary** — 访问 URL"
    else:
        research_lines = ""   # 没能力就不注入描述

    return f"""## Your Tools
...{research_lines}..."""
```

**为什么要这样？** system prompt 里描述的工具必须和 `tools=` 参数里注册的 schema 完全一致。
若 prompt 里描述了 `web_search` 但 tools 里没注册，LLM 会幻想调用不存在的工具导致报错。
`render_*()` 函数就是用来保证两者同步的。

`_capabilities()` 检查运行时条件（env var `AISCI_WEB_SEARCH`、llm config 的 `web_search` 字段）
决定哪些能力"开着"，再把结果传给所有 `render_*()` 函数。

| 常量 | Agent 角色 |
|------|-----------|
| `render_main_agent_system_prompt` | 主 orchestrator |
| `STRUCTURE_SYSTEM_PROMPT` | 论文结构提取器 |
| `ALGORITHM_SYSTEM_PROMPT` | 算法/架构提取器 |
| `EXPERIMENTS_SYSTEM_PROMPT` | 实验参数提取器 |
| `BASELINE_SYSTEM_PROMPT` | 基线方法提取器 |
| `render_implementation_system_prompt` | 写代码 agent |
| `render_experiment_system_prompt` | 跑实验 agent |
| `render_prioritization_system_prompt` | 任务优先级排序 |
| `render_explore/plan/general_system_prompt` | 通用辅助 |

#### Subagent 的 context 注入

Subagent（如 ImplementationSubagent）有**独立的、全新的 messages 列表**：

```python
# aisci_agent_runtime/subagents/base.py:run()
messages = [
    {"role": "system", "content": self.system_prompt()},  # subagent 自己的角色定义
]
if context:
    messages.append({"role": "user", "content": context}) # 主 Agent 传来的任务描述
```

`context` 字符串由主 Agent 调用工具时传入（task + context 参数），内容大致是：

```
Objective: 实现 VAE encoder

Canonical workspace contract:
- /home/paper 包含论文输入
- /home/submission 是实现仓库

Additional context:
上一次实验 loss 发散，请把 lr 降到 1e-4 并添加 gradient clipping
```

主 Agent 的 80 步历史，Subagent 完全看不到，只看到这段文字。
Subagent 完成后返回一段文字，追加到主 Agent 的 tool result 消息里。
**Context 不被子任务历史污染**。

#### Paper Reader 的级联注入

5 个 reader subagent 形成 DAG，后面的 agent 把前面的输出作为自己的 context：

```
StructureSubagent  context = reader_context()（目标 + 路径表）
       ↓ 输出 structure.md
AlgorithmSubagent  context = reader_context() + "\n\n" + structure.md 内容
ExperimentsSubagent context = reader_context() + "\n\n" + structure.md 内容
BaselineSubagent   context = reader_context() + "\n\n" + structure.md 内容
       ↓ 三个并发执行
SynthesisSubagent  context = structure.md + algorithm.md + experiments.md + baseline.md 拼接
```

实现见 `paper_reader.py:_stage_context()` 和 `_synthesis_context()`。

#### WORKSPACE_REFERENCE 常量

`constants.py` 定义 4 个路径表常量，嵌入进各 Agent 的 system prompt 末尾，
告诉 LLM"哪个路径放什么文件、该文件何时存在"：

| 常量 | 注入位置 |
|------|---------|
| `MAIN_AGENT_WORKSPACE_REFERENCE` | 主 agent system_prompt 末尾 + initial_user_prompt |
| `IMPLEMENTATION_WORKSPACE_REFERENCE` | impl subagent system_prompt |
| `EXPERIMENT_WORKSPACE_REFERENCE` | experiment subagent system_prompt |
| `SUBAGENT_WORKSPACE_REFERENCE` | explore/plan/general subagent system_prompt |

#### Reminder 的行为纠偏逻辑

`_build_reminder()` 不只是报时，还包含**根据行为计数动态注入警告**：

```python
exp_count = sequence.count("exp")
impl_count = sequence.count("impl")

if exp_count - impl_count >= 4:
    # 注入：实验次数远超实现次数，不要重复跑同样实验
if impl_count - exp_count >= 3:
    # 注入：实现多次但没验证，先跑一次 validate
if time_ratio >= 0.85:
    # 注入：时间快用完，停止新任务，确保 reproduce.sh 已 commit
```

---

### Prompt 体系原则（可借鉴）

**原则 1：Prompt 与 tools 必须同步**
用 `render_*(capabilities)` 函数按运行时能力动态生成 prompt，而不是写死字符串。
若 prompt 描述了某工具，tools 参数里就必须注册它，反之亦然。

**原则 2：system prompt 放角色+流程，user 消息放任务+状态**
system prompt 是不变的"你是谁"，首条 user 消息是变化的"这次要做什么"。
这样 system prompt 可复用，任务信息通过 user 消息替换。

**原则 3：Subagent 独立 messages，主 Agent 只看结果字符串**
每个 subagent 开一个全新的 messages 列表，避免历史污染主 Agent 的 context 窗口。
主 Agent 通过 tool_result 消息看到 subagent 的最终输出文本。

**原则 4：用文件（而非 context）传递大块信息**
Subagent 的输出写到磁盘文件（`/home/agent/paper_analysis/*.md`），
主 Agent 通过 `read_file_chunk` 工具按需读取，而不是把全部内容塞进 context。

**原则 5：Reminder 承担行为约束职责**
除了计时，reminder 还检查行为计数器（implement/experiment 比例），
在 LLM 偏离正确工作流时动态注入纠偏指令。

---

### L3-F：工具（Tool）定义与注册机制

**文件**：`src/aisci_agent_runtime/tools/base.py`，`src/aisci_agent_runtime/tools/shell_tools.py`，`src/aisci_domain_paper/tools/`

#### Tool 抽象基类

所有工具必须继承 `Tool` ABC，实现三个抽象方法：

```python
class Tool(ABC):
    @abstractmethod
    def name(self) -> str:
        """工具名称，与 LLM 返回的 tool_call.name 完全匹配"""

    @abstractmethod
    def execute(self, shell, **kwargs) -> str:
        """工具执行逻辑，返回字符串（追加为 tool role 消息）"""

    @abstractmethod
    def get_tool_schema(self) -> dict:
        """返回 OpenAI function calling 格式的 schema，传给 LLM API"""
```

**特殊工具**：`SubagentCompleteTool` — 每个 subagent 必须有这个工具，调用时抛出 `SubagentCompleteSignal` 异常，以此跳出 subagent 的 tool_call 循环。

#### 两类工具

**类型 A：直接 Shell 工具**（在 `shell_tools.py`）

调用 `shell.send_shell_command()` 执行 docker exec，直接操作容器文件系统：

| 工具 | 功能 | 关键细节 |
|------|------|----------|
| `BashToolWithTimeout` | 在容器内执行 bash 命令 | 两层超时（壁钟+进程），50K 字符截断，exit-137 检测 |
| `PythonTool` | 执行 python3 -c | 快速脚本，不写文件 |
| `ReadFileChunkTool` | 读文件（分块） | sed -n 带行号，单次 ≤2000 行、≤50K 字符 |
| `SearchFileTool` | grep -rn 搜索 | 返回带行号的匹配结果 |
| `FileEditTool` | 创建/替换/插入文件 | 三种 mode：create、str_replace、insert |
| `GitCommitTool` | git add -A + commit | 自动处理 .gitignore，含 commit message |
| `ExecCommandTool` | 长时间运行实验 | 输出重定向到日志文件，轮询检查完成状态 |
| `AddImplLogTool` | 追加实现日志 | append-only，写 impl_log.md |
| `AddExpLogTool` | 追加实验日志 | append-only，写 exp_log.md |

**类型 B：Subagent 触发工具**（在 `tools/` 下各文件）

不执行 shell，而是调用 `engine.run_subagent_output()` 启动子 Agent：

| 工具 | Subagent 类型 | 说明 |
|------|--------------|------|
| `build_implement_tool` → `ImplementationTool` | `implementation` | 写代码；注入 recent_exp_history 作 context |
| `build_run_experiment_tool` | `experiment` | 跑实验；返回结果摘要 |
| `build_read_paper_tool` | PaperReaderCoordinator | 5 个 reader subagent DAG 并发 |
| `build_prioritize_tasks_tool` | `prioritization` | 对任务列表排优先级 |
| `build_spawn_subagent_tool` | explore/plan/general | 只读探索/规划/通用辅助 |
| `build_clean_validation_tool` | `clean_validation` | 干净环境验证 reproduce.sh |

#### build_* 函数：按角色组合工具

每种 Agent 角色获得不同的工具子集，由 `build_*_tools(capabilities)` 函数组装：

```
build_main_direct_tools(capabilities)       → bash + python + read_file + search + [web_search]
build_implementation_tools(capabilities)   → 以上 + edit_file + git_commit + add_impl_log + linter
build_experiment_tools(capabilities)       → 以上 + exec_command + add_exp_log
build_reader_tools(capabilities)           → bash + python + read_file + search（只读）
build_explore_tools / build_plan_tools / build_general_tools → 类似 reader，轻量子集
```

`capabilities` 参数控制是否加入 `web_search`/`link_summary`，与 prompt 里的描述保持同步。

#### 主 Agent 工具注册（完整入口）

`build_main_tools(engine)` 是主 Agent 的唯一注册入口，返回完整工具列表：

```python
def build_main_tools(engine):
    return [
        *build_main_direct_tools(engine._capabilities()),  # 直接 shell 工具
        build_read_paper_tool(engine),       # 读论文（DAG 并发）
        build_prioritize_tasks_tool(engine), # 任务优先级
        build_implement_tool(engine),        # 实现代码
        build_run_experiment_tool(engine),   # 运行实验
        build_spawn_subagent_tool(engine),   # 通用辅助子 Agent
        build_clean_validation_tool(engine), # 干净环境验证
        SubmitTool(),                        # 提交完成信号
    ]
```

#### 工具运行时分发（数据流）

```
engine.run_main_loop()
  ├─ tools = build_main_tools(self)
  ├─ tool_map    = {t.name(): t for t in tools}      # 运行时分发表
  ├─ tool_schemas = [t.get_tool_schema() for t in tools]  # 传给 LLM API
  │
  ├─ llm.chat(messages, tools=tool_schemas)
  │     LLM 返回 tool_calls: [{name: "bash", arguments: {cmd: "ls"}}]
  │
  └─ for call in tool_calls:
         tool = tool_map[call.name]           # 按名字查表
         result = tool.execute(shell, **call.arguments)  # 执行
         messages.append({role:"tool", content: result}) # 追加结果
```

#### 工具体系设计原则

**原则 6：工具 schema 是给 LLM 看的，execute() 是给程序看的**
`get_tool_schema()` 里的字段描述影响 LLM 如何填参数；`execute()` 才是实际运行逻辑。两者解耦，LLM 只看 schema，永远不执行代码。

**原则 7：按 Agent 角色最小化工具集**
实现 Agent 有 `edit_file`+`git_commit`，读论文 Agent 只有只读工具。最小权限原则：LLM 看不到的工具不会被误用。

**原则 8：Subagent 触发工具屏蔽内部细节**
`ImplementationTool.execute()` 启动子 Agent，等待完成后返回格式化字符串。主 Agent 不知道子 Agent 跑了多少步、用了什么工具——复杂性被封装在工具接口后面。

---

## 📝 关键文件清单（已读）

```
src/aisci_app/
  cli.py                    # CLI 入口，paper/mle run 命令
  service.py                # JobService：create_job + spawn_worker
  worker_main.py            # Worker 进程入口
  presentation.py           # build_paper/mle_job_spec()，视图辅助

src/aisci_core/
  models.py                 # JobSpec/JobRecord/PaperSpec/MLESpec/RunPhase 等数据模型
  store.py                  # SQLite 操作（jobs/events/artifacts 三张表）
  runner.py                 # JobRunner：dispatch → collect artifacts → export

src/aisci_domain_paper/
  adapter.py                # PaperDomainAdapter：stage_inputs + run_real_loop
  engine.py                 # EmbeddedPaperEngine：主循环 + subagent 调度
  prompts/templates.py      # 11个 system prompt + render 函数
  subagents/coordinator.py  # SubagentCoordinator：拓扑排序 + ThreadPoolExecutor
  subagents/base.py         # PaperSubagent 基类
  subagents/paper_reader.py # PaperReaderCoordinator：级联 context 注入
  tools/basic_tool.py       # build_*_tools() 按角色组合工具
  tools/implementation_tool.py  # ImplementationTool（subagent 触发类）
  tools/spawn_subagent_tool.py  # build_main_tools()：主 Agent 工具注册入口
  orchestrator.py           # 已废弃（仅保留错误提示）

src/aisci_runtime_docker/
  runtime.py                # DockerRuntimeManager（空壳）
  agent_session.py          # AgentSessionManager：完整 Docker 操作实现
  shell_interface.py        # DockerShellInterface（工具调用 → docker exec）

src/aisci_agent_runtime/
  llm_profiles.py           # YAML profile 解析 → LLMProfile
  llm_client.py             # ResponsesLLMClient / CompletionsLLMClient
  tools/base.py             # Tool ABC + SubagentCompleteTool
  tools/shell_tools.py      # BashToolWithTimeout + 所有直接 shell 工具

config/
  llm_profiles.yaml         # LLM 配置（backends + profiles）
  image_profiles.yaml       # Docker 镜像配置（未深入读）
  paper_subagents.yaml      # 各 subagent 步数/时间预算（未深入读）
```

---

## 🏗️ 核心设计模式总结（可借鉴）

### 1. File-as-Bus 协调模式
Agent 之间通过磁盘文件通信，不通过内存/消息队列。好处：持久化、可调试、可重放。

### 2. Prompt 与工具注册同步
`render_*_prompt(capabilities)` 按运行时能力动态生成，确保 prompt 中的工具描述和实际 tools 列表一致。

### 3. Subagent 独立 Context
每个 subagent 有独立 messages 列表，主 agent 只看到 subagent 的最终输出字符串，context 不被子任务历史污染。

### 4. 依赖图并发调度
`SubagentCoordinator._group_by_level()` 拓扑排序后分层执行，同层 ThreadPoolExecutor 并发，下层依赖上层输出。

### 5. Context 溢出两级处理
`ContextLengthError` → 先尝试 LLM 摘要压缩（保留语义）→ 失败再截断剪枝。

### 6. 进程隔离 + SQLite 通信
CLI 进程与 Worker 进程完全隔离，通过 SQLite 传递状态。Worker 崩溃不影响 CLI，支持 `--no-wait` 异步运行。

### 7. JobSpec 不可变 + 序列化为 JSON 列
`RuntimeProfile` 和 `PaperSpec` 整体序列化成两列 JSON 字段入库，schema 随字段演进不需要 migration。

### 8. Tool ABC + build_* 函数 + 角色最小权限
每个工具继承 `Tool` ABC，`build_*_tools(capabilities)` 按 Agent 角色组合最小工具子集。LLM 只能看到（也只能调用）分配给它的工具，防止越权或误用。

### 9. Subagent 触发工具封装复杂性
`ImplementationTool` 等 subagent 触发类工具对主 Agent 完全透明：主 Agent 调用工具 → 等待 → 拿到字符串结果。子 Agent 内部的多步循环、工具调用、错误处理全部被封装。

---

## 🚀 下一步计划（用户尚未决定）

用户的目标是**参考该设计改造自己的项目**。可能的切入方向：

1. **借鉴 File-as-Bus**：在自己的多 Agent 系统中用磁盘文件替代内存传递
2. **借鉴 Subagent 独立 context**：每个子任务 Agent 独立 messages，主 Agent 只看摘要结果
3. **借鉴 prompt/tools 同步机制**：按能力动态渲染 prompt，防止 LLM 幻想不存在的工具
4. **借鉴 依赖图并发**：用拓扑排序 + ThreadPoolExecutor 调度有依赖关系的并行任务
5. **借鉴 Tool ABC + build_* 函数**：统一工具接口，按 Agent 角色最小化工具集
6. **可能想修改/扩展的点**：
   - 新增 LLM profile（新模型）：改 `config/llm_profiles.yaml`
   - 新增工具：在 `tools/` 下新建，继承 `Tool` ABC，在 `build_main_tools()` 注册
   - 新增 subagent 类型：在 `subagents/` 下新建，在 `subagent_class_for_kind()` 注册
   - 修改主 prompt：改 `render_main_agent_system_prompt()`

---

## 💡 在新对话中使用本文档

在新对话开始时，粘贴以下内容：

```
我正在阅读 /Users/admin/PycharmProjects/AiScientist 项目的代码。
请先读取这个上下文文档：
/Users/admin/PycharmProjects/AiScientist/对话上下文-AiScientist代码阅读-L3完成.md

我们已经完成了 L2 数据流走读和 L3 深入分析（Docker生命周期、LLM Profile解析、
主循环messages管理、Subagent执行流程、Prompt体系、工具定义与注册）。

接下来我想 [你的目标]。
```

---

## 📝 备注

- `orchestrator.py` 已废弃，历史上 Agent 是在容器内运行的，现在改为宿主机运行 Agent + Docker 只作为代码执行沙箱
- `DockerRuntimeManager` 是空壳，真正实现在 `AgentSessionManager`，这是一个常见的"接口类继承实现类"模式
- `glm-5` 是默认 paper profile，`gpt-5.4` 是默认 mle profile（见 `config/llm_profiles.yaml`）
- paper 轨道用 `--network host`（需要访问 HuggingFace 等），mle 轨道用 `--network bridge`（隔离网络）
- 最大步数：`AISCI_MAX_STEPS=80`（env var 可覆盖），reminder 频率：`AISCI_REMINDER_FREQ=5`
