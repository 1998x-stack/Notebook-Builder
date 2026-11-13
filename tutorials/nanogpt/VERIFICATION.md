# nanoGPT × Notebook Builder — 验收报告

**日期：** 2026-11-13。**源码固定：** `karpathy/nanoGPT@3adf61e154c3fe3fca428ad6bc3818b27a3b8291`；**实现类型：** 教学型微型改写（不是原版训练代码完整复制）。运行环境 Python 3.13.5 / PyTorch 2.10.0+cpu / nbformat 5.10.4 / nbclient 0.10.4。

## 已实际执行

- 8 本有效 `.ipynb`，共 29 个执行完成的代码 Cell。
- 29 个代码 Cell 含真实 stdout 输出，19 个代码 Cell 含内嵌 PNG 图，另存 19 张 PNG。
- 全部 8 本 `nbformat.validate` 通过；0 个代码错误输出。
- 8 本独立 HTML 成功导出，课程导航 `index.html` 已生成，支持离线打开相对资源。
- `nbuild tutorial nanogpt --list` 与 `nbuild tutorial nanogpt --html` 已调用成功。
- 50 步 CPU 实训合成字符实验：抽样验证 loss 从第 1 步约 **3.240** nats/token 下降至第 50 步约 **2.212** nats/token。训练产生 `outputs/nanogpt_tiny_checkpoint.pt`，采样教程使用此 checkpoint。该数值不对应 GPT-2/OpenWebText/莎士比亚真实 benchmark。

## 代码测试

| 运行分组 | 结果 |
| --- | --- |
| `tests/test_nanogpt_tutorials.py` | 6 passed |
| `tests/test_builder.py tests/test_ops.py tests/test_hardening.py` | 18 passed |
| `tests/test_mcp.py tests/test_nanogpt_tutorials.py` | 11 passed |
| `tests/test_jobs.py tests/test_research.py tests/test_runtime.py tests/test_workspace.py` | 8 passed |

以上测试分组覆盖全部 **37 个不同的测试案例**（`test_nanogpt_tutorials.py` 在两个分组中重复运行了一次）。各个分组均通过。

**重要说明：** 单进程运行 `python -m pytest -q tests` 的一次完整 37 项批量执行，在已完成 25 项后超时退出，因此没有把整套单进程测试认定为通过。这不等于所有测试在一次完整连续运行中成功，仍建议后续追踪 Kernel/子进程间的资源释放或测试次序影响。

## 防伪与限制

- 代码内真实 print、图片、loss 均来自本地计算；未使用示意图代替实测图。只有第 07 章的注意力分数矩阵图明确标注为理论内存公式，而不是实际 VRAM 占用。
- 只使用原创合成字符语料；没有下载数据、预训练模型权重或执行外部项目代码。
- 没有 GPU / Docker / DDP / checkpoint 下载验证；本地 Python Kernel 没有安全沙箱隔离。
- 上游 nanoGPT README 将项目标记为 deprecated 并建议关注 nanochat；本课程只展示其核心透明架构思想。

## 复现

```bash
cd /mnt/notebook-builder
nbuild tutorial nanogpt --list
nbuild tutorial nanogpt --build --execute --html
python -m pytest -q tests/test_nanogpt_tutorials.py
```

注：默认不覆盖已存在的教学 Notebook；加 `--force` 才重新生成原文档（Builder 会保留版本历史）。
