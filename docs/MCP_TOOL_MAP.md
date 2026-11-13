# MCP 工具映射清单

本地已经实现 **57 个** MCP 工具。所有方法通过 `mcp_tools.invoke` 检查参数与权限。

权限：`read` 始终可用；`write/destructive` 需要 `NB_MCP_READONLY=0`；`execute` 需要 `NB_MCP_READONLY=0 NB_MCP_ALLOW_EXEC=1`。

> 这一迁移版本使用 NB_ROOT 下的相对文件路径，不是上游使用 Notebook ID 的 FastAPI/RBAC Server。

## 基础发现

| 工具 | 权限 | 作用 |
|---|---|---|
| `nebula_health` | `read` | Check local notebook-builder and MCP environment availability. |
| `nebula_capabilities` | `read` | List verified local capabilities; not a hosted Nebula server. |
| `runtime_diagnose` | `read` | Inspect dependency and runtime capabilities without executing Notebook code. |

## Notebook 生命周期

| 工具 | 权限 | 作用 |
|---|---|---|
| `notebook_list` | `read` | List Notebook files and their document versions under NB_ROOT. |
| `notebook_get` | `read` | Get Notebook metadata and complete sources with bounded output previews. |
| `notebook_quality` | `read` | Validate saved output: executed Cells, stdout, valid PNG and error-free status. |
| `notebook_create` | `write` | Create a standard .ipynb from a title and optional cell specifications. |
| `notebook_create_file` | `write` | Alias for creating a real on-disk .ipynb document. |
| `notebook_import` | `write` | Import a validated notebook from a file already contained inside NB_ROOT. |
| `notebook_export` | `write` | Export a notebook to a separate .ipynb, .html or .md file under NB_ROOT. |
| `notebook_rename` | `write` | Update the Notebook title metadata without renaming its stable file path. |
| `notebook_delete` | `destructive` | Soft-delete a notebook into the hidden recoverable trash folder. |
| `notebook_validate` | `read` | Validate nbformat schema and report document metadata. |
| `notebook_render_html` | `write` | Render a Notebook as local standalone HTML. Outputs may contain untrusted HTML. |
| `notebook_fingerprint` | `read` | Produce reproducibility hash of full Notebook bytes and each Cell source. |
| `notebook_audit` | `read` | Report recent embedded idempotent operations and preserved version snapshots. |

## 历史/三方合并

| 工具 | 权限 | 作用 |
|---|---|---|
| `notebook_versions` | `read` | List historical immutable versions and the current document version. |
| `notebook_version` | `read` | Read a particular retained Notebook revision by integer version. |
| `notebook_fork` | `write` | Fork a retained document version into an independent Notebook file. |
| `notebook_restore` | `write` | Restore a prior snapshot as a new version; preserve history. |
| `notebook_diff` | `read` | Compare cells by stable ID across two Notebook revisions. |
| `notebook_merge_preview` | `read` | Preview cell-level three-way merge; conflicting cells need explicit choices. |
| `notebook_merge_commit` | `write` | Commit a conflict-resolved draft against an exact latest Notebook version. |

## 目录

| 工具 | 权限 | 作用 |
|---|---|---|
| `folder_list` | `read` | List actual nonhidden directories containing local Notebook documents. |
| `folder_ensure` | `write` | Idempotently create a local Notebook folder under NB_ROOT. |

## Cell

| 工具 | 权限 | 作用 |
|---|---|---|
| `cell_list` | `read` | List stable Cell IDs, types and short source summaries. |
| `cell_get` | `read` | Read the full Cell source by stable ID or zero-based index. |
| `cell_add` | `write` | Insert a Cell with optional CAS version and idempotent operation identifier. |
| `cell_insert` | `write` | Insert a Cell at a zero-based index (alias for cell_add). |
| `cell_update` | `write` | Replace Cell source/type and invalidate stale execution outputs. |
| `cell_delete` | `destructive` | Delete a selected Cell while retaining immutable previous Notebook version. |
| `cell_move` | `write` | Move an existing Cell to a zero-based index. |

## 执行/Kernel

| 工具 | 权限 | 作用 |
|---|---|---|
| `cell_run` | `execute` | Run one Cell with fresh kernel and prior code cells as bootstrap dependencies. |
| `cell_run_all` | `execute` | Run every code Cell in a fresh Python Kernel and persist outputs on success. |
| `notebook_run_cells` | `execute` | Execute a bounded window with optionally executed prerequisite Cells. |
| `code_execute` | `execute` | Execute trusted Python. Live mode shares memory; fresh mode can persist Cell. |
| `runtime_start` | `execute` | Start a local stateful Python Kernel for the lifetime of this MCP process. |
| `runtime_status` | `read` | Get state of the local in-process MCP Python Kernel. |
| `runtime_stop` | `execute` | Stop the in-process Kernel and discard its in-memory Python variables. |
| `kernel_variables` | `execute` | Read bounded variable metadata from an already-running Kernel. |

## 本地任务

| 工具 | 权限 | 作用 |
|---|---|---|
| `run_submit` | `execute` | Submit one trusted Cell as an idempotent local subprocess job, not a cloud task. |
| `run_status` | `read` | Read a durable SQLite run record; dead processes become interrupted/cancelled. |
| `run_wait` | `read` | Wait at most 30 seconds for a local job to reach a terminal state. |
| `run_cancel` | `execute` | Signal a local job's isolated process group to cancel running code. |

## 工作区

| 工具 | 权限 | 作用 |
|---|---|---|
| `workspace_list` | `read` | List stored files in the isolated per-notebook logical workspace directory. |
| `workspace_read` | `read` | Read bounded UTF-8 content of a per-notebook workspace file. |
| `workspace_write` | `write` | Write UTF-8 file within that Notebook's guarded workspace (2 MB cap). |
| `workspace_delete` | `destructive` | Delete a per-notebook workspace file, not the Notebook document. |

## 科研 Trial 和工件

| 工具 | 权限 | 作用 |
|---|---|---|
| `research_list` | `read` | List evidence-linked research experiments for one Notebook. |
| `research_create` | `write` | Freeze evaluator Cell SHA and user-asserted dataset/environment SHA in a bounded protocol. |
| `research_get` | `read` | Read trial history, remaining budget and verified best metric. |
| `research_trial_register` | `write` | Register a Trial backed by a successful non-evaluator local candidate Run ID. |
| `research_trial_finalize` | `write` | Score an actual evaluator Run with frozen source and NB_METRIC_V1 JSON output. |
| `research_close` | `write` | Close experiment enrollment while retaining its immutable trial records. |
| `research_artifacts` | `read` | List experiment-associated content-addressed SHA-256 artifacts. |
| `research_artifact_upload` | `write` | Snapshot a per-notebook workspace file into immutable 2MB SHA-256 storage. |
| `research_artifact_download` | `write` | Verify and copy an experiment artifact back into a per-notebook workspace. |

## MCP Resources

- `nbuild://notebooks` — Notebook 列表；`resources/list` 可发现。
- `nbuild://notebooks/{path}` — 单个 Notebook 的完整 Cell 源码（经过输出截断）。

## 没有移植或不可等价的功能

没有 Web UI、Docker/GPU 隔离、FastAPI/多用户 RBAC、SSE、CRDT、远程分布式调度。`research_*` 只是本地可追溯实验记录：评估 Cell 源码被冻结、指标来自 Run 的保存版本，但数据/环境哈希由调用者断言，评估与候选运行的 Python 环境不互相隔离；不能称为第三方独立审计或不可篡改的评测。原生 stdio 没有支持 MCP 的所有扩展。
