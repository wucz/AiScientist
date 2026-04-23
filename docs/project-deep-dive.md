# AiScientist 项目深度文档

## 目录

1. [项目概览](#1-项目概览)
2. [目录结构](#2-目录结构)
3. [环境配置](#3-环境配置)
4. [常用命令](#4-常用命令)
5. [架构设计](#5-架构设计)
6. [核心执行链路](#6-核心执行链路)
7. [模块详解](#7-模块详解)
8. [数据模型](#8-数据模型)
9. [LLM 客户端](#9-llm-客户端)
10. [Docker 运行时](#10-docker-运行时)
11. [配置文件体系](#11-配置文件体系)
12. [工作空间布局（File-as-Bus）](#12-工作空间布局file-as-bus)
13. [Paper 工作流详解](#13-paper-工作流详解)
14. [MLE 工作流详解](#14-mle-工作流详解)
15. [SQLite 数据库结构](#15-sqlite-数据库结构)
16. [日志与工件体系](#16-日志与工件体系)

---

## 1. 项目概览

**AiScientist** 是一个为长期 ML 研究工程设计的虚拟研究实验室，核心创新是 **File-as-Bus** 协调模式：用磁盘文件作为 Agent 之间的协调总线，而不是依赖内存消息传递。

**两个工作轨道：**

| 轨道 | 输入 | 优化目标 | 验证端点 |
|------|------|---------|---------|
| `paper` | `--paper-md` / `--zip` | 将论文转换为可运行的端到端复现 | `validation_report.json` |
| `mle` | `--zip` / `--name` / `--data-dir` | 通过迭代改进目标指标 | 提交格式或评分验证 |

**核心设计原则：**
- **对话很便宜，展示你的文件** — 所有计划、代码、实验、日志均写入磁盘
- **薄控制 / 厚状态** — 宿主机轻量级协调，容器内专注执行
- **可恢复** — 任何阶段中断均可从文件状态恢复
- **可审计** — 完整的 conversation.jsonl 和各阶段工件

**技术栈：**
- Python 3.12+，uv 包管理
- Typer CLI，Pydantic v2 数据模型，SQLite 存储
- OpenAI / Azure OpenAI LLM，Docker 沙箱
- Rich TUI 实时仪表盘

---

## 2. 目录结构

```
AiScientist/
├── pyproject.toml              # 项目配置，Python >=3.12，入口 aisci
├── uv.lock                     # uv 依赖锁文件
├── .env.example                # 环境变量模板
│
├── config/                     # 共享配置注册表
│   ├── llm_profiles.yaml       # LLM 后端、模型、API 模式配置
│   ├── image_profiles.yaml     # Docker 镜像配置
│   └── paper_subagents.yaml    # Paper 子 Agent 步数/时间预算
│
├── docker/                     # Docker 构建
│   ├── build_paper_image.sh    → aisci-paper:latest
│   ├── build_mle_image.sh      → aisci-mle:test
│   ├── paper-agent.Dockerfile
│   └── mle-agent.Dockerfile
│
├── src/
│   ├── aisci_app/              # CLI 应用层
│   │   ├── cli.py              # Typer 命令注册
│   │   ├── service.py          # JobService（create_job / spawn_worker）
│   │   ├── worker_main.py      # Worker 子进程入口
│   │   ├── presentation.py     # JobSpec 构建、doctor 报告
│   │   └── tui.py              # Rich TUI 仪表盘
│   │
│   ├── aisci_core/             # 核心基础设施
│   │   ├── models.py           # 所有数据模型
│   │   ├── store.py            # SQLite JobStore
│   │   ├── runner.py           # JobRunner（worker 主循环）
│   │   ├── paths.py            # 路径解析与目录初始化
│   │   ├── exporter.py         # 作业导出打包
│   │   ├── env_config.py       # 环境变量加载
│   │   └── logging_utils.py    # 日志工具
│   │
│   ├── aisci_agent_runtime/    # LLM Agent 运行时
│   │   ├── llm_client.py       # LLM 客户端（OpenAI/Azure 双 API）
│   │   ├── llm_profiles.py     # LLM 配置解析
│   │   ├── subagents/base.py   # 子 Agent 基类（消息剪枝、重试）
│   │   ├── tools/base.py       # 工具基类
│   │   ├── summary_utils.py    # 上下文摘要（防超长）
│   │   ├── log_utils.py        # conversation.jsonl 写入
│   │   └── trace.py            # AgentTraceWriter
│   │
│   ├── aisci_runtime_docker/   # Docker 运行时管理
│   │   ├── runtime.py          # DockerRuntimeManager
│   │   ├── agent_session.py    # AgentSessionManager（容器生命周期）
│   │   ├── shell_interface.py  # DockerShellInterface（docker exec 封装）
│   │   ├── profiles.py         # Docker 镜像配置解析
│   │   └── models.py           # ContainerSession, SessionSpec 等
│   │
│   ├── aisci_domain_paper/     # Paper 复现工作流
│   │   ├── adapter.py          # PaperDomainAdapter（Docker 准备 + 运行）
│   │   ├── engine.py           # EmbeddedPaperEngine（Agent 主循环）
│   │   ├── state_manager.py    # PaperStateManager（会话状态）
│   │   ├── configs.py          # 子 Agent 默认配置
│   │   ├── constants.py        # 工作空间路径常量
│   │   ├── runtime.py          # 工作空间工具函数
│   │   ├── prompts/            # Jinja2 系统提示模板
│   │   ├── subagents/          # 专项子 Agent
│   │   │   ├── base.py
│   │   │   ├── implementation.py
│   │   │   ├── experiment.py
│   │   │   ├── prioritization.py
│   │   │   └── ...
│   │   └── tools/              # 主 Agent 可调用工具
│   │       ├── paper_reader_tool.py
│   │       ├── implementation_tool.py
│   │       ├── experiment_tool.py
│   │       ├── prioritization_tool.py
│   │       ├── clean_validation_tool.py
│   │       └── ...
│   │
│   ├── aisci_domain_mle/       # MLE 竞赛工作流
│   │   ├── adapter.py          # MLEDomainAdapter
│   │   ├── orchestrator.py     # EmbeddedMLEEngine 主循环
│   │   ├── candidate_registry.py # 候选方案注册与比较
│   │   ├── input_resolver.py   # 竞赛输入源解析
│   │   ├── preflight.py        # 启动预检
│   │   ├── mlebench_compat.py  # MLE-Bench 兼容层
│   │   ├── subagents/          # MLE 专项子 Agent
│   │   └── prompts/
│   │
│   └── logid.py                # 日志 ID 生成工具
│
├── tests/                      # 测试套件
├── docs/                       # 文档
└── examples/                   # 示例脚本
```

---

## 3. 环境配置

### 必需依赖

```bash
# 1. Python 3.12+
# 2. uv 包管理器
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Docker Desktop（需运行中）

# 4. 安装 Python 依赖
uv sync --dev
```

### 环境变量（.env）

```bash
cp .env.example .env
```

**必须配置（二选一）：**

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# 或 Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=...
OPENAI_API_VERSION=2025-01-01-preview
```

**可选配置：**

```bash
# 网络代理
HTTP_PROXY=http://proxy:port
HTTPS_PROXY=http://proxy:port
NO_PROXY=localhost,127.0.0.1

# Hugging Face
HF_TOKEN=hf_...

# 运行时路径（默认 ./jobs/）
AISCI_OUTPUT_ROOT=/abs/path/to/aisci-runtime

# Agent 主循环调优
AISCI_MAX_STEPS=80          # 最大步数（默认 80）
AISCI_REMINDER_FREQ=5       # 每 N 步注入进度提醒（默认 5）

# 覆盖配置文件路径
AISCI_LLM_PROFILE_FILE=/abs/path/to/llm_profiles.yaml
AISCI_IMAGE_PROFILE_FILE=/abs/path/to/image_profiles.yaml
```

### Docker 镜像构建

```bash
bash docker/build_paper_image.sh   # → aisci-paper:latest
bash docker/build_mle_image.sh     # → aisci-mle:test
```

**Paper 镜像**（`paper-agent.Dockerfile`）：
- 基础：`ubuntu:24.04` + Python 3.12
- 包含：Julia 1.10 LTS、Docker CLI、Jupyter、git/vim/tmux
- 容器内环境变量：`PAPER_DIR=/home/paper`，`SUBMISSION_DIR=/home/submission`，`AGENT_DIR=/home/agent`

**MLE 镜像**（`mle-agent.Dockerfile`）：
- 基础：`ubuntu:24.04` + Python 3.12
- 动态从 `pyproject.toml` 安装项目依赖
- 工作目录：`/home/code`

---

## 4. 常用命令

```bash
# 健康检查
AISCI_PAPER_DOCTOR_PROFILE=gpt-5.4 uv run aisci paper doctor
uv run aisci mle doctor

# 运行 Paper 复现作业
uv run aisci --env-file .env paper run \
  --paper-md /abs/path/to/paper.md \
  --image aisci-paper:latest \
  --llm-profile gpt-5.4 \
  --gpu-ids 0 \
  --time-limit 24h \
  --wait --tui

# 运行 MLE 竞赛作业
uv run aisci --env-file .env mle run \
  --zip /abs/path/to/competition.zip \
  --name <competition-slug> \
  --image aisci-mle:test \
  --llm-profile gpt-5.4 \
  --gpu-ids 0 \
  --time-limit 12h \
  --wait --tui

# 作业管理
uv run aisci jobs list
uv run aisci jobs show <job_id>

# 日志查看（kind: main / conversation / agent / subagent / validation / all）
uv run aisci logs tail <job_id> --kind main
uv run aisci logs tail <job_id> --kind conversation
uv run aisci logs tail <job_id> --kind subagent --subagent implementation

# 工件与导出
uv run aisci artifacts ls <job_id>
uv run aisci export <job_id>

# TUI 实时仪表盘
uv run aisci tui job <job_id>

# 恢复中止的作业
uv run aisci paper resume <job_id> --wait
uv run aisci mle resume <job_id> --wait

# 单独运行最终验证
uv run aisci paper validate <job_id> --wait

# 测试
uv run pytest tests/
uv run pytest tests/test_store.py
uv run pytest tests/test_store.py::test_create_job
```

---

## 5. 架构设计

### 分层架构图

```
┌─────────────────────────────────────────────────────────┐
│                    宿主机（Host）                          │
│                                                           │
│  CLI (aisci)  →  JobService  →  JobStore (SQLite)        │
│                      │                                    │
│               spawn_worker()                              │
│                      │ subprocess                         │
│                      ▼                                    │
│               worker_main.py                              │
│                      │                                    │
│               JobRunner.run_job()                         │
│                   /       \                               │
│    PaperDomainAdapter   MLEDomainAdapter                  │
│           │                    │                          │
│    DockerRuntimeManager ───────┘                          │
│           │                                               │
└───────────┼───────────────────────────────────────────────┘
            │  docker run（绑定 workspace/）
┌───────────▼───────────────────────────────────────────────┐
│                    Docker 容器（Sandbox）                   │
│                                                           │
│    EmbeddedPaperEngine / EmbeddedMLEEngine                │
│           │                                               │
│    LLM.chat()  ←→  Tool执行（bash / python / file I/O）   │
│           │                                               │
│    /home/paper/        ← 输入论文                          │
│    /home/submission/   ← 输出代码（git 跟踪）              │
│    /home/agent/        ← File-as-Bus 协调文件              │
│    /home/logs/         ← 对话日志                          │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

### File-as-Bus 协调原理

Agent 的所有中间状态均写入磁盘文件，而非保存在内存或通过消息传递：

```
论文分析  → agent/paper_analysis/{summary,structure,algorithm,experiments,baseline}.md
任务规划  → agent/prioritized_tasks.md
实现计划  → agent/plan.md
实现日志  → agent/impl_log.md
实验日志  → agent/exp_log.md
最终自检  → agent/final_self_check.{md,json}
输出代码  → submission/（git 仓库）
复现脚本  → submission/reproduce.sh
```

这使得：
- 任务可以从任意阶段恢复（`paper resume`）
- TUI 通过轮询文件实时展示进度
- 操作员可以随时检查中间状态（`logs tail`、`artifacts ls`）

---

## 6. 核心执行链路

### 完整流程（以 Paper 轨道为例）

#### 第一层：CLI → 创建作业 → 启动 Worker

```python
# cli.py: run_paper()
spec = build_paper_job_spec(...)         # 组装 JobSpec
job  = service.create_job(spec)          # 写 SQLite，创建 jobs/<id>/ 目录
pid  = service.spawn_worker(job.id)      # subprocess.run("python -m aisci_app.worker_main <id>")
#                                          stdout/stderr → logs/worker.log
```

CLI 与 Worker **完全进程隔离**：
- `--wait`：CLI 阻塞等待 Worker 退出码
- `--detach`：CLI 立即返回 PID，Worker 后台运行
- `--tui`：CLI 启动 Rich TUI 轮询 SQLite 状态

#### 第二层：Worker → JobRunner → 分发

```python
# worker_main.py
status = JobRunner().run_job(job_id)
return 0 if status == SUCCEEDED else 1

# runner.py: run_job()
store.mark_running(pid)
_write_job_spec()          # 写 jobs/<id>/job_spec.json（File-as-Bus 起点）
result = _dispatch(job)    # → PaperDomainAdapter.run(job) 或 MLEDomainAdapter.run(job)
# 收集 artifacts，写 validation_report.json，打包 export zip
store.complete_job(SUCCEEDED/FAILED)
```

#### 第三层：PaperDomainAdapter → Docker 准备 → 启动 Engine

```python
# adapter.py: run()
ensure_layout()                # 宿主机创建 workspace/paper、submission、agent/
_stage_inputs()                # 复制 paper.md 到 workspace/paper/
git init submission/           # 为输出代码建立 git 仓库

profile = resolve_llm_profile(job.llm_profile)   # 解析 llm_profiles.yaml
image_tag = prepare_image(docker_profile)          # 检查/拉取 Docker 镜像
session = runtime.start_session(spec, image_tag)   # docker run，挂载 workspace/

# _run_real_loop()
shell = DockerShellInterface(session)   # 封装 docker exec
llm   = create_llm_client(profile)     # OpenAI/Azure 客户端
engine = EmbeddedPaperEngine(config, shell, llm, ...)
engine.run()
```

#### 第四层：EmbeddedPaperEngine 主循环（核心 Agent）

```python
# engine.py: run_main_loop()

messages = [system_prompt, initial_user_prompt]
tools = build_main_tools(self)     # read_paper, implement, run_experiment,
                                   # prioritize_tasks, submit, bash, ...

for step in range(1, max_steps+1):   # 默认 max_steps=80
    # 1. 超时检查（elapsed >= time_limit）
    # 2. 每 reminder_freq 步注入进度提醒（剩余时间、任务完成率）

    response = llm.chat(messages, tools=tool_schemas)
    #          ↑ 失败自动重试（RateLimitError等），最多 2 小时

    if not response.tool_calls:
        messages.append("continue instruction")
        continue

    for call in response.tool_calls:
        if call.name == "submit":
            # 前置检查：reproduce.sh 必须存在且 git tracked
            precheck = _handle_submit_precheck()
            if precheck:  # 有阻断项则拒绝，提示修复
                continue

        tool_result = tool_map[call.name].execute(shell, **call.arguments)
        # execute() 内部通过 DockerShellInterface 在容器内运行 bash

        messages.append(tool_result)
        log_to(conversation.jsonl)     # 记录每步对话

    # 子 Agent 完成时抛出 SubagentCompleteSignal → 正常退出

# 达到步数/时间上限 → auto_finalize_summary() → 正常退出
```

#### Agent 工具调用 → 子 Agent（两层 Agent 架构）

```
主 Agent 调用 implement(task="实现 Transformer 编码器")
    │
    ▼ implementation_tool.py: execute()
    │
    ▼ engine.execute_named_subagent("implementation", objective=task)
    │
    ▼ ImplementationSubagent.run()
         独立 LLM 对话（最多 500 步，8 小时预算）
         通过同一个 DockerShellInterface 在容器内执行 bash
         写入 impl_log.md、提交 git commit
         返回执行摘要给主 Agent
```

#### 退出路径汇总

| 情况 | 触发点 | 最终状态 |
|------|--------|---------|
| Agent 调用 `submit()` | `SubagentCompleteSignal` | `SUCCEEDED` |
| 达到 `max_steps` | 循环结束 | `SUCCEEDED` |
| 达到 `time_limit` | elapsed 检查 | `SUCCEEDED` |
| `ContentPolicyError` | LLM 响应 | `SUCCEEDED`（提前终止） |
| 任何未捕获异常 | `except Exception` in `run_job` | `FAILED` |

---

## 7. 模块详解

### aisci_app（CLI 应用层）

**cli.py** — Typer 命令树：
```
aisci
├── paper run / doctor / validate / resume
├── mle   run / doctor / validate / resume
├── jobs  list / show
├── logs  tail / list
├── artifacts ls
├── tui   job
└── export
```

全局 `--env-file`、`--output-root`、`--llm-profile-file`、`--image-profile-file` 在 `@app.callback()` 中处理，通过 `os.environ` 透传给子进程。

**service.py** — `JobService`：
```python
create_job(spec)      # 写 SQLite + 创建 jobs/<id>/ 目录结构
spawn_worker(job_id)  # subprocess.run/Popen，stdout/stderr 重定向到 worker.log
export_bundle(job_id) # 调用 export_job_bundle() 打包 zip
```

### aisci_core（核心基础设施）

**runner.py** — `JobRunner`：
- 是 Worker 进程的核心类
- 持有 `DockerRuntimeManager`、`PaperDomainAdapter`、`MLEDomainAdapter` 的实例
- 负责作业生命周期管理、工件收集、对话日志回放到 SQLite events 表

**paths.py** — 路径体系：
```python
resolve_job_paths(job_id) → JobPaths(
    root         = OUTPUT_ROOT/jobs/<id>/
    workspace_dir = .../workspace/
    logs_dir      = .../logs/
    artifacts_dir = .../artifacts/
    state_dir     = .../state/
    export_dir    = .../export/
    input_dir     = .../input/
)
```

### aisci_agent_runtime（LLM 运行时）

**llm_client.py** — 双 API 支持：
- `ResponsesLLMClient`：OpenAI Responses API（`gpt-5.4`，支持 `web_search_preview`）
- `CompletionsLLMClient`：Chat Completions API（`glm-5`、`gemini-3-flash`）

**subagents/base.py** — 子 Agent 基础能力：
- `prune_messages()`：超过上下文窗口时删除旧消息
- `prune_messages_individual()`：按单条消息截断
- `fix_message_consistency()`：修复 tool_call / tool_result 不匹配

---

## 8. 数据模型

**JobSpec**（不可变，定义作业）：
```python
class JobSpec(BaseModel):
    job_type: JobType                   # "paper" | "mle"
    objective: str
    llm_profile: str                    # e.g. "gpt-5.4"
    runtime_profile: RuntimeProfile
    mode_spec: PaperSpec | MLESpec
```

**RuntimeProfile**（Docker 资源配置）：
```python
class RuntimeProfile(BaseModel):
    gpu_count: int = 0
    gpu_ids: list[str] = []
    cpu_limit: str | None = None
    memory_limit: str | None = None
    time_limit: str = "24h"
    image: str | None = None
    image_profile: str | None = None
    pull_policy: PullPolicy | None = None
    network_policy: NetworkPolicy = NetworkPolicy.HOST
    run_final_validation: bool = False
    validation_strategy: ValidationStrategy = ValidationStrategy.FRESH_CONTAINER
    keep_container_on_failure: bool = False
    checkpoint_interval_seconds: int = 300
```

**PaperSpec**：
```python
class PaperSpec(BaseModel):
    paper_md_path: str | None = None
    paper_zip_path: str | None = None
    rubric_path: str | None = None        # 评分标准（可选）
    blacklist_path: str | None = None     # 禁用 API/方法列表（可选）
    addendum_path: str | None = None      # 补充说明（可选）
    submission_seed_repo_zip: str | None  # 预置代码仓库（可选）
    enable_online_research: bool = True
```

**MLESpec**：
```python
class MLESpec(BaseModel):
    competition_name: str | None = None
    competition_zip_path: str | None = None
    workspace_bundle_zip: str | None = None
    data_dir: str | None = None
    validation_command: str | None = None
    metric_direction: Literal["maximize", "minimize"] | None = None
```

**JobRecord**（SQLite 中的作业记录）：
```python
class JobRecord(BaseModel):
    id: str                         # UUID
    job_type: JobType
    status: JobStatus               # pending→running→succeeded|failed|cancelled
    phase: RunPhase                 # ingest→analyze→prioritize→implement→validate→finalize
    objective: str
    llm_profile: str
    runtime_profile: RuntimeProfile
    mode_spec: PaperSpec | MLESpec
    created_at / updated_at / started_at / ended_at: datetime
    worker_pid: int | None
    error: str | None
```

---

## 9. LLM 客户端

### 配置（LLMConfig）

```python
class LLMConfig(BaseModel):
    provider: str           # "openai" | "azure-openai"
    model: str
    api_mode: str           # "responses" | "completions"
    max_tokens: int
    context_window: int
    reasoning_effort: str | None
    reasoning_summary: str | None
    web_search: bool = False
    use_phase: bool = False
    clear_thinking: bool = False
    temperature: float | None = None
    # 认证
    api_key: str | None = None
    base_url: str | None = None
    azure_endpoint: str | None = None
    api_version: str | None = None

    @property
    def prune_context_window(self) -> int:
        # 为 system prompt 保留缓冲区
        return context_window - max_tokens
```

### 重试策略

```
可重试错误：RateLimitError, APIConnectionError, APITimeoutError, InternalServerError
不可重试：ContentPolicyError（内容政策）, ContextLengthError（超长）

退避策略：指数退避 1s ~ 300s
总时间预算：最多 2 小时（7200s）
```

### 上下文长度处理

当触发 `ContextLengthError` 时：
1. 尝试用 `SummaryConfig` 摘要历史消息（若启用）
2. 调用 `prune_messages()` 删除最旧的轮次
3. 调用 `prune_messages_individual()` 截断超长单条消息

---

## 10. Docker 运行时

### DockerRuntimeManager 关键方法

```python
can_use_docker() → bool              # 检查 Docker daemon 可达性
image_exists(ref) → bool
pull_image(ref) → None
prepare_image(profile, runtime_profile) → str   # 按 pull_policy 决定是否拉取
ensure_layout(job_paths, layout) → None          # 创建 workspace 目录结构
create_session_spec(...) → SessionSpec           # 组装容器挂载、环境、资源限制
start_session(spec, image_tag) → ContainerSession # docker run
execute_in_session(session, cmd) → Result        # docker exec
inspect_session(session) → dict
run_validation(spec, image, cmd) → ValidationReport  # 独立容器验证
cleanup(session) → None                              # docker stop + rm
```

### WorkspaceLayout

```python
class WorkspaceLayout(StrEnum):
    PAPER = "paper"   # 挂载: paper/, submission/, agent/
    MLE   = "mle"     # 挂载: data/, code/, submission/, agent/
```

### ContainerSession

```python
@dataclass
class ContainerSession:
    container_id: str
    container_name: str
    image_id: str
    image_tag: str
    started_at: datetime
    workdir: str
    run_as_user: str
    labels: list[tuple[str, str]]
    mounts: list[Mount]
    profile: DockerProfile
    runtime_profile: RuntimeProfile
```

---

## 11. 配置文件体系

### llm_profiles.yaml

```yaml
version: 1

defaults:
  default: glm-5
  paper: glm-5
  mle: gpt-5.4

backends:
  openai:
    type: openai
    env:
      api_key: { var: OPENAI_API_KEY, required: true }
      base_url: { var: OPENAI_BASE_URL }

  azure-openai:
    type: azure-openai
    env:
      endpoint:   { var: AZURE_OPENAI_ENDPOINT, required: true }
      api_key:    { var: AZURE_OPENAI_API_KEY, required: true }
      api_version:{ var: OPENAI_API_VERSION, required: true }

profiles:
  gpt-5.4:
    backend: openai
    model: gpt-5.4
    api: responses
    limits:
      max_completion_tokens: 131072
      context_window: 1000000
    features:
      use_phase: true

  glm-5:
    backend: azure-openai
    model: glm-5
    api: completions
    limits:
      max_completion_tokens: 65536
      context_window: 202752
    features:
      clear_thinking: true

  gemini-3-flash:
    backend: openai
    model: gemini-3-flash-preview
    api: completions
    limits:
      max_completion_tokens: 20480
      context_window: 1000000
    reasoning:
      effort: high
```

### image_profiles.yaml

```yaml
version: 1

defaults:
  paper: paper-default
  mle: mle-default

profiles:
  paper-default:
    image: aisci-paper:latest
    pull_policy: never

  mle-default:
    image: hub.example.com/team/aisci-mle:latest
    pull_policy: if-missing
```

### paper_subagents.yaml

```yaml
subagents:
  implementation:
    max_steps: 500
    time_limit: 28800   # 8h
    reminder_freq: 20

  experiment:
    max_steps: 500
    time_limit: 36000   # 10h
    reminder_freq: 30

  env_setup:
    max_steps: 500
    time_limit: 7200    # 2h
    reminder_freq: 15

timeouts:
  main_agent_bash_default: 36000
  experiment_command_timeout: 36000
  experiment_validate_time_limit: 18000
```

---

## 12. 工作空间布局（File-as-Bus）

```
jobs/<job_id>/
├── job_spec.json              ← 作业定义快照（File-as-Bus 起点）
│
├── workspace/
│   ├── paper/                 # [paper] 输入论文
│   │   ├── paper.md
│   │   ├── rubric.json        # 可选评分标准
│   │   ├── blacklist.txt      # 可选禁用列表
│   │   └── addendum.md        # 可选补充说明
│   │
│   ├── data/                  # [mle] 竞赛数据
│   │
│   ├── submission/            # [paper] 输出代码（git 仓库）
│   │   ├── reproduce.sh       # ← 必须存在才能 submit
│   │   └── requirements.txt
│   │
│   ├── code/                  # [mle] 解决方案代码（git 仓库）
│   │
│   └── agent/                 ← File-as-Bus 核心协调目录
│       ├── paper_analysis/    # 论文理解分析
│       │   ├── summary.md
│       │   ├── structure.md
│       │   ├── algorithm.md
│       │   ├── experiments.md
│       │   └── baseline.md
│       ├── analysis/          # [mle] 数据分析
│       │   └── summary.md
│       ├── prioritized_tasks.md  # 任务优先级（P0/P1/P2）
│       ├── plan.md               # 实现计划
│       ├── impl_log.md           # 实现进度日志
│       ├── exp_log.md            # 实验结果日志
│       └── final_self_check.{md,json}  # 最终自检报告
│
├── logs/
│   ├── job.log                # 宿主机作业日志
│   ├── worker.log             # Worker 进程输出
│   ├── agent.log              # Agent 完整消息历史
│   ├── conversation.jsonl     # 结构化对话记录（每步一行 JSON）
│   ├── traceback.log          # 异常堆栈（失败时）
│   └── subagent_logs/         # 各子 Agent 独立日志
│       ├── implementation_001/
│       ├── experiment_001/
│       └── ...
│
├── artifacts/
│   └── validation_report.json # 最终验证报告
│
├── state/
│   ├── capabilities.json       # Agent 能力声明
│   ├── resolved_llm_config.json # 实际使用的 LLM 配置
│   ├── sandbox_session.json    # Docker 容器信息
│   └── paper_main_prompt.md    # 实际使用的系统提示
│
├── input/
│   └── paper.zip              # 原始输入文件备份
│
└── export/
    └── <job_id>.zip           # 最终导出包
```

---

## 13. Paper 工作流详解

### 五阶段工作流

```
ingest      → 准备工作空间，复制输入文件，初始化 git
analyze     → PaperDomainAdapter 启动，Docker 准备
prioritize  → Agent 调用 read_paper + prioritize_tasks
implement   → 循环调用 implement（子 Agent：最多 500 步 / 8h）
validate    → 调用 run_experiment + clean_reproduce_validation
finalize    → Agent 调用 submit，收集工件，打包导出
```

### Agent 工具集（主 Agent 可用）

| 工具 | 作用 |
|------|------|
| `read_paper` | 调用 PaperReader 子 Agent 分析论文 |
| `prioritize_tasks` | 调用 Prioritization 子 Agent 生成任务列表 |
| `implement` | 调用 Implementation 子 Agent 编写代码 |
| `run_experiment` | 调用 Experiment 子 Agent 运行实验 |
| `clean_reproduce_validation` | 在新鲜容器中验证 reproduce.sh |
| `bash` | 直接在容器内执行 shell 命令 |
| `submit` | 触发提交前检查，通过后结束主循环 |

### Submit 前置检查

`_handle_submit_precheck()` 阻断条件：
1. `reproduce.sh` 不存在 → **硬阻断**
2. `reproduce.sh` 未被 git 追踪 → **硬阻断**

警告（不阻断但提示再次调用才允许提交）：
- 时间使用不足 50% 时过早提交
- `reproduce.sh` 有语法错误
- `requirements.txt` 缺失
- 从未调用过 `clean_reproduce_validation()`

### 进度提醒机制

每 `reminder_freq` 步（默认 5 步）注入进度提醒，内容根据时间消耗比例动态调整：
- < 50%：聚焦 P0 任务，提示创建 reproduce.sh
- 50-70%：持续实现，不要提早 submit
- 70-85%：如未验证则立即验证，定期 commit
- > 85%：收尾，确保 reproduce.sh 最新，不开新任务

---

## 14. MLE 工作流详解

### 迭代优化循环

```
ingest      → 解析竞赛源（zip/name/data-dir）
analyze     → DataAnalysisSubagent 理解竞赛任务
prioritize  → PrioritizationSubagent 制定解决方案策略
implement   → ImplementationSubagent 编写解决方案
experiment  → ExperimentSubagent 运行并评分
     ↑______↓  （循环至时间/步数耗尽）
finalize    → 选最优候选，生成 submission.csv，导出
```

### 候选方案注册表

MLE 轨道维护候选方案注册表（`submission_registry.jsonl`），记录每次实验的指标得分，选最优的提交。

```
workspace/submission/
├── candidates/
│   ├── candidate_001/
│   ├── candidate_002/
│   └── ...
├── submission.csv             # 最优方案提交
└── submission_registry.jsonl  # 候选方案历史
```

### 输入源优先级

```
--competition-bundle-zip  > --workspace-zip  > --zip + --name
--data-dir                                    > MLE-Bench 缓存
```

---

## 15. SQLite 数据库结构

数据库位于 `OUTPUT_ROOT/.aisci/jobs.db`（默认 `.aisci/jobs.db`）。

```sql
CREATE TABLE jobs (
    id                   TEXT PRIMARY KEY,
    job_type             TEXT NOT NULL,       -- "paper" | "mle"
    status               TEXT NOT NULL,       -- pending/running/succeeded/failed/cancelled
    phase                TEXT NOT NULL,       -- ingest/analyze/.../finalize
    objective            TEXT NOT NULL,
    llm_profile          TEXT NOT NULL,
    runtime_profile_json TEXT NOT NULL,
    mode_spec_json       TEXT NOT NULL,
    created_at           TEXT NOT NULL,
    updated_at           TEXT NOT NULL,
    started_at           TEXT,
    ended_at             TEXT,
    worker_pid           INTEGER,
    error                TEXT
);

CREATE TABLE events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id       TEXT NOT NULL,
    event_type   TEXT NOT NULL,   -- status/artifact/error/tool_result/model_response/...
    phase        TEXT NOT NULL,
    message      TEXT NOT NULL,
    payload      TEXT DEFAULT '{}',
    created_at   TEXT NOT NULL
);

CREATE TABLE artifacts (
    job_id        TEXT NOT NULL,
    artifact_type TEXT NOT NULL,  -- paper_analysis/impl_log/validation_report/...
    path          TEXT NOT NULL,
    phase         TEXT NOT NULL,
    size_bytes    INTEGER DEFAULT 0,
    created_at    TEXT NOT NULL,
    metadata      TEXT DEFAULT '{}'
);
```

### JobStore 关键方法

```python
create_job(spec)        → JobRecord
get_job(job_id)         → JobRecord
list_jobs()             → list[JobRecord]
mark_running(id, pid)
update_phase(id, phase)
complete_job(id, status, error=None)
append_event(id, type, phase, message, payload)
list_events(id)         → list[EventRecord]
add_artifact(id, record)
list_artifacts(id)      → list[ArtifactRecord]
```

---

## 16. 日志与工件体系

### 日志类型

| kind 参数 | 文件 | 内容 |
|-----------|------|------|
| `main` | `logs/job.log` + `logs/worker.log` | 宿主机层面事件 |
| `conversation` | `logs/conversation.jsonl` | 每步 LLM 对话（结构化 JSON） |
| `agent` | `logs/agent.log` | 完整消息历史（含 tool_calls） |
| `subagent` | `logs/subagent_logs/*/` | 各子 Agent 独立对话 |
| `validation` | `artifacts/validation_report.json` | 最终验证结果 |

### conversation.jsonl 格式

每行一个 JSON 事件，事件类型：
- `model_response`：LLM 返回（含 text_content、tool_calls、usage、reasoning）
- `tool_result`：工具执行结果（含 tool name、call_id、output）

### 工件类型（Paper 轨道）

```
paper_analysis      → agent/paper_analysis/summary.md
paper_structure     → agent/paper_analysis/structure.md
paper_algorithm     → agent/paper_analysis/algorithm.md
paper_experiments   → agent/paper_analysis/experiments.md
paper_baseline      → agent/paper_analysis/baseline.md
prioritized_tasks   → agent/prioritized_tasks.md
plan                → agent/plan.md
impl_log            → agent/impl_log.md
exp_log             → agent/exp_log.md
self_check_report   → agent/final_self_check.md
reproduce_script    → submission/reproduce.sh
validation_report   → artifacts/validation_report.json
export_bundle       → export/<job_id>.zip
```

### validation_report.json 结构

```json
{
  "status": "passed" | "failed" | "skipped",
  "summary": "...",
  "details": { ... },
  "runtime_profile_hash": "...",
  "container_image": "aisci-paper:latest"
}
```
