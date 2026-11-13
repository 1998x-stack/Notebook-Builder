# Notebook Builder 2.1 验收报告（2026-11-13）

## 验收环境

- 本地项目 `/mnt/notebook-builder/`；Python 3.13.5。
- 参考项目 `1998x-stack/Nebula-Notebook`，参考 tree `e8e7e8e38afa5c414b70f3323fca98475287c2fb`；本地代码是精选功能迁移/重构，并非上游整个 Web 服务。
- 教学源码 `karpathy/nanoGPT` 固定 `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`；教程为 CPU 教学适配，非 GPT-2 基准再现。

## 实测结果

| 验收项 | 实测 | 结论 |
|---|---|---|
| 完整 pytest 单进程 | `python -m pytest -q /mnt/notebook-builder/tests` | **44 passed, 1 warning, 64.44s；退出码 0** |
| 分组回归 A | `tests/test_builder.py tests/test_ops.py tests/test_hardening.py tests/test_optimizations.py` | 25 passed |
| 分组回归 B | `tests/test_mcp.py tests/test_runtime.py tests/test_workspace.py tests/test_jobs.py` | 11 passed |
| 分组回归 C | `tests/test_research.py tests/test_nanogpt_tutorials.py` | 8 passed |
| Python 语法 | `python -m compileall -q ...` | 通过 |
| 包安装 | `pip install -e . --no-deps --no-build-isolation` | 2.1.0 |
| MCP 工具目录 | `nbuild mcp-tools` | 57 tools |
| nanoGPT 教程枚举 | `nbuild tutorial nanogpt --list` | 8 chapters |
| nanoGPT 已保存输出质量 | `notebook_quality.inspect` 遍历 8 本 | 8/8 passed，29 code cells，19 embedded valid PNG images |
| 质量门禁命令 | `nbuild quality tutorials/nanogpt/notebooks/04_training_loop_checkpoint.ipynb` | `passed=true`；未达标时退出码 2 |

pytest warning：环境中 ddtrace/nbconvert 对 bleach <5 的支持弃用提醒；不是本项目的执行失败。

## 本次具体优化

1. **Kernel 执行锁释放与 CAS**：完整 Notebook、分段 Cell 以及临时代码执行在锁内读取快照、锁外执行、锁内以版本和内容 SHA-256 检测冲突，避免覆盖并发改动。
2. **Source Spec 漂移检测**：教程构建遇到现有 `.ipynb` 时比较原始 Cell 顺序、类型和源码；源变更需 `--force` 显式重建，生成新的 Notebook 保存 `spec_sha256`。
3. **质量检查**：只读工具 `notebook_quality` 与 `nbuild quality`，检查 schema、执行计数、stdout、PNG 签名/尺寸、error 输出，且发现缺失教程文件立即失败。
4. **Run Worker 日志**：单任务独立日志 `.notebook-runs/logs/<uuid>.log`；专用文件以 0600 创建，失败 Traceback 不再全部丢弃。
5. **测试**：新加并发执行与版本冲突、手工编辑未更新版本、规格漂移、损坏 PNG、质量 CLI 返回码和任务日志回归。

## 仍存在的限制

- 本地 Python Kernel **无 Docker/VM 沙箱**，不应运行不可信第三方代码。任何 Python 副作用无法因 CAS 冲突而回滚。
- `nbclient` 的 timeout 是按 Cell 的超时；没有另外实现跨整个多 Cell 的硬性进程级 watchdog，长时间/挂死的 Kernel 仍值得继续压测。
- 已上传的 `.ipynb` 预览含图片，但源代码依赖 `nanogpt_edu.py`；在 Colab 运行前必须将工作目录切换到同目录并确保模块可访问。Drive 当前报告其 MIME 为 `application/octet-stream`，未验证直接用 Colab 打开。已上传 `START_HERE.md` 辅助说明。
- 无正式的自动 Reflection/LLM 修复闭环、独立评估沙箱或 GPU/DDP 调度。本次版本重点修复 P0 稳定性、审计和 QA，而不是虚构已存在这些功能。
- GitHub 远程 CI 尚未触发；本次测试仅在当前运行环境完成。

详细设计与后续建议见 `docs/OPTIMIZATION_2026-11-13.md`。完整 pytest 日志保留在 `/mnt/data/notebook-builder-2.1-full-suite.log`。
