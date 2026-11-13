# 上游代码审查与本地迁移说明

- 上游：`https://github.com/1998x-stack/Nebula-Notebook`
- 调研时仓库树 SHA：`e8e7e8e38afa5c414b70f3323fca98475287c2fb`
- 许可：MIT；保留了上游 Nebula Notebook contributors 的许可文本。
- 迁移策略：重新实现本地 Python 工具链，采用 Cell ID、CAS、操作收据、不可变历史快照、三方 Cell 合并、MCP tool/resource、内核执行与科研记录等设计模式；不直接复制上游 FastAPI/SQLAlchemy/Docker 服务。

## 源码映射

| 上游 | 本地 | 决策 |
| --- | --- | --- |
| `backend/app/validation.py` | `notebook_builder.py` | nbformat 校验、大小限制 |
| `backend/app/cell_ops.py` | `notebook_builder.py` | 原子 Cell 编辑、expected_version、operation_id 幂等 |
| `backend/app/history.py` | `notebook_ops.py` | 不可变旧版本、恢复/Fork |
| `backend/app/merge.py` | `notebook_ops.py` | 保守三方 Cell 合并、冲突显式选择 |
| `backend/app/runtime.py` | `runtime_manager.py` | 用本地 Jupyter Kernel 取代 Docker；仅可信机器 |
| `mcp/server.py`, `mcp/nebula_client.py` | `mcp_tools.py`, `mcp_protocol.py`, `mcp_server.py` | 一体化 stdio 56 tool、无需 Nebula REST 服务 |
| 托管执行 Run 状态 | `run_jobs.py`, `run_worker.py` | SQLite + 受控子进程，未知执行结果拒绝重放 |
| `workspace_*` | `workspace_ops.py` | 每 Notebook 逻辑独立的受限文本文件工作区 |
| Web UI / RBAC / SSE / GPU | 无 | 当前运行环境不具备前置基础设施，不声称实现 |
| Research Trial/evaluator | `research_ledger.py` | 证据链接的轻量研究账本；不保证隔离评估或数据源认证 |

## 一致性说明

工具名称有意贴近上游，但操作模型不完全兼容：上游使用 Notebook ID + 用户 API 会话，本地使用 `NB_ROOT` 内的相对 `.ipynb` 路径。上游 `runtime_*` 受 Docker 管理，本地 Kernel 存活期依赖 MCP 进程；上游 `run_submit` 通过 API 管理任务，本地使用本机 SQLite 和 subprocess。对外称 **Nebula-inspired local port**，不是 Nebula Server 的官方 SDK 或 API 仿制品。

阅读上游：
- [原仓库说明](https://github.com/1998x-stack/Nebula-Notebook/blob/main/README.md)
- [MCP 功能](https://github.com/1998x-stack/Nebula-Notebook/blob/main/mcp/server.py)
- [版本/三方合并](https://github.com/1998x-stack/Nebula-Notebook/blob/main/backend/app/merge.py)
- [架构与安全模型](https://github.com/1998x-stack/Nebula-Notebook/blob/main/docs/ARCHITECTURE.md)
