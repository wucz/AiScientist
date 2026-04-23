# 对话上下文：VSCode Step-by-Step 调试本地 MLE 执行链路

> 生成时间：2026-04-19
> 状态：已配置完成，可立即使用

---

## 📋 问题背景

**项目**：AiScientist — 本地运行 MLE（机器学习竞赛）Agent 的系统

**当前状态**：
- 已成功修复 MLE 本地模式（`--local` 参数）的一系列 bug（见历史上下文文档）
- Job `20260419-171515-05d5dd77` 正在跑（detecting-insults 竞赛，glm-5 模型）
- 用户希望 step-by-step 理解执行链路

**涉及组件**：
- `src/aisci_app/cli.py` — CLI 入口（Typer）
- `src/aisci_app/service.py` — `spawn_worker()` 启动子进程
- `src/aisci_app/worker_main.py` — Worker 子进程入口
- `src/aisci_domain_mle/adapter.py` — MLE 域适配器，`_run_real_loop()`
- `src/aisci_domain_mle/orchestrator.py` — `EmbeddedMLEEngine.run()` agent 主循环
- `src/aisci_agent_runtime/local_shell.py` — 本地 shell 执行（无 Docker）

---

## 🔴 核心问题

**问题表现**：VSCode 用 `launch.json` 启动 `aisci_app.cli` 后，debugger 没有停在断点，进程直接返回。

**根源分析**：
```
VSCode debugger
  └─ attach → aisci_app.cli (主进程)
                  └─ service.spawn_worker()
                       └─ subprocess.Popen/run(["python", "-m", "aisci_app.worker_main", job_id])
                            ↑ stdout/stderr 重定向到 worker.log 文件
                            ↑ VSCode debugger 无法自动跟进此子进程
                            ↑ 断点在这里 → 永远不会触发
```

**关键代码**（`src/aisci_app/service.py:23`）：
```python
def spawn_worker(self, job_id: str, wait: bool = False) -> int:
    command = [sys.executable, "-m", "aisci_app.worker_main", job_id]
    # stdout/stderr 重定向到文件，与 debugger 完全隔离
    with worker_log.open("ab") as handle:
        completed = subprocess.run(command, stdout=handle, stderr=handle)
```

---

## 🎯 实现目标

- 在 VSCode 中 step-by-step 调试 MLE 本地执行链路
- 在三个关键位置打断点：
  1. `adapter.py._run_real_loop()` — job 启动入口
  2. `orchestrator.py.EmbeddedMLEEngine.run()` — agent 主循环
  3. `local_shell.py.send_shell_command()` — shell 命令执行

---

## 🔧 技术约束

- macOS 环境，`.venv` 在项目根目录
- 使用 `uv` 管理依赖
- Python interpreter：`.venv/bin/python`
- 环境变量从 `.env` 加载（GLM-5 API key）
- Worker 是独立子进程，主进程只是调度器

---

## 🚫 已尝试的失败方案

### 方案 A：直接调试 `aisci_app.cli` + `--wait`
```json
{
  "module": "aisci_app.cli",
  "args": ["mle", "run", "--wait", ...]
}
```
**失败原因**：实际执行逻辑在子进程 `worker_main` 中，`--wait` 只是让主进程等待子进程结束，debugger attach 在主进程，断点在子进程里，永远不触发。

### 方案 B：`"subProcess": true`
`launch.json` 加 `"subProcess": true` 让 debugpy 跟进子进程。
**问题**：macOS 上对 `subprocess.Popen` 的跟踪不稳定，且 stdout/stderr 被重定向到文件，交互输出无法显示在 VSCode 终端。

---

## ✅ 最终方案：两步调试法

### 设计思路
跳过 CLI 层，**直接调试 `worker_main`**。先用 CLI 创建 job（不启动 worker），再用 VSCode 直接以 debug 模式运行 `worker_main <job_id>`。

### 执行流程
```
Step 1: CLI 创建 job（不带 --wait，不启动 worker）
         → 获得 job_id
Step 2: VSCode 直接调试 worker_main <job_id>
         → debugger 直接在 worker 进程里
         → 所有断点正常触发
```

---

## 📝 关键文件变更

### 1. `.vscode/launch.json`（新建）
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "MLE Worker (direct debug)",
      "type": "debugpy",
      "request": "launch",
      "module": "aisci_app.worker_main",
      "args": ["${input:jobId}"],
      "python": "${workspaceFolder}/.venv/bin/python",
      "cwd": "${workspaceFolder}",
      "envFile": "${workspaceFolder}/.env",
      "justMyCode": false,
      "console": "integratedTerminal"
    },
    {
      "name": "MLE Run CLI (creates job only)",
      "type": "debugpy",
      "request": "launch",
      "module": "aisci_app.cli",
      "args": [
        "--env-file", ".env",
        "mle", "run",
        "--zip", "/Users/admin/Downloads/detecting-insults-in-social-commentary.zip",
        "--name", "detecting-insults-in-social-commentary",
        "--llm-profile", "glm-5",
        "--time-limit", "2h",
        "--local"
      ],
      "python": "${workspaceFolder}/.venv/bin/python",
      "cwd": "${workspaceFolder}",
      "envFile": "${workspaceFolder}/.env",
      "justMyCode": false,
      "console": "integratedTerminal"
    }
  ],
  "inputs": [
    {
      "id": "jobId",
      "type": "promptString",
      "description": "Job ID to debug (e.g. 20260419-171515-05d5dd77)",
      "default": "20260419-171515-05d5dd77"
    }
  ]
}
```

### 2. 三处断点（`breakpoint()` 调用）

**`src/aisci_domain_mle/adapter.py:626`**
```python
def _run_real_loop(self, job: JobRecord, job_paths, llm_profile) -> None:
    breakpoint()  # DEBUG: job 启动入口 — 查看 job / job_paths / llm_profile
```

**`src/aisci_domain_mle/orchestrator.py:909`**
```python
def run(self) -> str:
    breakpoint()  # DEBUG: agent 主循环入口 — 查看 self.config / self.shell / self.llm
```

**`src/aisci_agent_runtime/local_shell.py:64`**
```python
def send_shell_command(self, cmd: str, timeout: int = 300) -> ShellResult:
    breakpoint()  # DEBUG: shell 命令执行 — 查看 cmd / rewritten / 执行结果
```

---

## 💡 使用方法

### 完整调试流程

**第一步**：终端创建 job（不带 `--wait`）
```bash
uv run aisci --env-file .env mle run \
  --zip /Users/admin/Downloads/detecting-insults-in-social-commentary.zip \
  --name detecting-insults-in-social-commentary \
  --llm-profile glm-5 --time-limit 2h --local
# 输出中找到类似：job_id=20260419-XXXXXX-XXXXXXXX
```

**第二步**：VSCode 启动调试
- 左侧点 **Run and Debug** (`⇧⌘D`)
- 下拉选 **"MLE Worker (direct debug)"**
- 点绿色三角运行
- 弹出输入框，填入上一步的 job_id
- 自动停在断点①（`_run_real_loop`）

### 三个断点的观察重点

| 断点 | 位置 | 观察变量 | 说明 |
|------|------|---------|------|
| ① `_run_real_loop` | `adapter.py:626` | `job`, `job_paths`, `llm_profile` | 确认路径映射、LLM配置是否正确 |
| ② `EmbeddedMLEEngine.run` | `orchestrator.py:909` | `self.config.paths`, `self.llm`, `self.shell` | 确认 OrchestratorPaths 使用的是真实主机路径 |
| ③ `send_shell_command` | `local_shell.py:64` | `cmd`(原始), `rewritten`(路径替换后), `exit_code` | 观察 agent 发出的每条 shell 命令及结果 |

### 提示：`send_shell_command` 断点过于频繁时

在 VSCode 断点上右键 → **Edit Breakpoint** → 加条件：
```python
"pip" in cmd  # 只在安装包时停
"python" in cmd and "train" in cmd  # 只在训练时停
```

---

## 🐛 已知问题

1. **`send_shell_command` 断点太密集**：agent 每条 shell 命令都触发，调试时建议加条件断点或临时禁用该断点，只在需要时启用。

2. **`breakpoint()` 在非调试模式下会阻塞**：正式运行时记得删除或注释掉三处 `breakpoint()`，否则 `uv run aisci mle run` 会卡住等待 pdb 输入。

3. **job 状态复用**：`worker_main <job_id>` 会在已有 job 目录上继续运行。如果 job 已经跑过一半，部分文件已存在，agent 可能跳过某些阶段。如需从头调试，需要创建新 job。

---

## 🚀 下一步计划

- [ ] 理解 `OrchestratorPaths` 的完整数据流
- [ ] 理解 `EmbeddedMLEEngine` 的 tool dispatch 机制（`analyze_data` / `implement` / `run_experiment` 是如何调用子 agent 的）
- [ ] 了解 `candidate_registry` 和 submission 评分流程
- [ ] （可选）把 `breakpoint()` 改为条件断点，减少调试噪音

---

## 📝 备注

**执行链路总结**：
```
CLI (aisci mle run)
  → JobService.create_job()          # 写入 SQLite，生成 job_id
  → JobService.spawn_worker(job_id)  # subprocess: python -m aisci_app.worker_main
      → JobRunner.run_job()
          → MLEDomainAdapter.run()
              → _run_real_loop()           ← 断点①
                  → EmbeddedMLEEngine(config, shell, llm)
                      → engine.run()       ← 断点②
                          → LLM call (GLM-5)
                          → tool dispatch
                              → LocalShellInterface.send_shell_command()  ← 断点③
```

**GLM-5 API 配置**（`.env`）：
```
OPENAI_API_KEY=<token>
OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
```

**macOS 特有问题记录**（已修复）：
- `/tmp` → `/private/tmp` 导致路径安全检查误报
- macOS 无 `timeout` 命令（GNU coreutils），`LocalShellInterface` 绕过处理
- `/home/` 目录在 macOS 受保护，`OrchestratorPaths` 需用真实主机路径替换
