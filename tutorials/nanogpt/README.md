# nanoGPT · Notebook Builder 交互式教程

> 八章循序渐进的中文 Jupyter 教学系列：**每章有 Markdown 原理、`print` 中间过程、至少两幅真实运行图表、可执行断言和课后练习**。通过本地 `notebook-builder` 创建、执行并导出 `.ipynb` / HTML；全程 CPU、离线、小模型。

## 源码与准确性

- 原仓库：[karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)，锁定源提交 [`3adf61e154c3fe3fca428ad6bc3818b27a3b8291`](https://github.com/karpathy/nanoGPT/tree/3adf61e154c3fe3fca428ad6bc3818b27a3b8291)。源代码阅读范围：`model.py`, `train.py`, `sample.py`, `data/shakespeare_char/prepare.py`, `config/train_shakespeare_char.py`。
- `nanogpt_edu.py` 为**教学化改写**，保留 GPTConfig / CausalSelfAttention / MLP / Block / GPT 的核心结构，包括 causal SDPA、LayerNorm、residual、权重共享、next-token CE，以及温度/top-k 采样，**不是 upstream `model.py` 的逐字复制**；未包含 GPT-2 checkpoint 导入、DDP 等功能。
- 上游 README 于 2025-11 标记 nanoGPT 已旧版/弃用，并建议关注 [`nanochat`](https://github.com/karpathy/nanochat)。本项目用于代码教学，而非宣称最新模型或现代规模训练实践。
- 数据为 `nanogpt_edu.training_text()` 中的原创微型合成文本。没有下载莎士比亚、OpenWebText、模型权重，没有依赖 `tiktoken` / `transformers`。
- 模型的验证 loss 只在这些合成字符上测得；图表和性能都不应被描述为 GPT-2 复现结果。

## 教学大纲

| 章节 | 核心代码对照 | 可视化与 print |
|---|---|---|
| [00 全景与张量流](notebooks/00_map_environment.ipynb) | `model.py/GPT.forward` | token/embedding/logits 形状与参数分布 |
| [01 字符 Tokenization](notebooks/01_char_tokenizer_batches.ipynb) | `data/shakespeare_char/prepare.py` | 字符频次、移位 X/Y、基线熵 |
| [02 Causal Attention](notebooks/02_causal_self_attention.ipynb) | `model.py/CausalSelfAttention` | 两个 head 的注意力热图、entropy、SDPA 数值对照 |
| [03 GPT Block 与 loss](notebooks/03_transformer_gpt_forward.ipynb) | `model.py/Block` / `GPT` | 参数统计、梯度范数与 weight tying |
| [04 实际训练](notebooks/04_training_loop_checkpoint.ipynb) | `train.py` | 真实 train/val loss 曲线、checkpoint |
| [05 自回归生成](notebooks/05_autoregressive_sampling.ipynb) | `sample.py` / `GPT.generate` | 不同温度 token 概率、top-k 分布和文本输出 |
| [06 优化器与学习率](notebooks/06_optimizer_accumulation_schedule.ipynb) | `train.py` | warmup + cosine、微批梯度等价、AdamW 参数组 |
| [07 Scaling 与不变量](notebooks/07_scaling_causality_checks.ipynb) | `model.py` 与设计约束 | 模型参数 heatmap、未来信息泄漏测试、T² 理论尺度 |

## 一条命令：真正使用 Notebook Builder

前提：Python 3.11+，`torch`、`numpy`、`matplotlib` 和本仓库已有的 `nbformat/nbclient/nbconvert/ipykernel`。在 `/mnt`：

```bash
nbuild tutorial nanogpt --list
nbuild tutorial nanogpt --build --execute --html  # 创建、执行并生成 HTML
# 重建现有教材（留历史版本；新 Cell ID 会更新）：
nbuild tutorial nanogpt --build --execute --html --force
# 单节再现：
nbuild tutorial nanogpt --build --execute --html --only 02
# 项目级复现测试：
cd /mnt/notebook-builder && python -m pytest -q tests/test_nanogpt_tutorials.py
```

等价的独立脚本：`python tutorials/nanogpt/build_tutorials.py --build --execute --html`。默认遇到已有 Notebook 不会覆盖；`--force` 才显式覆盖。执行已有 Notebook 会创建新版本历史；建议将个人修改的 Cell 先保存/备份。`--execute` 使用本地普通 Python Kernel，**并不具有容器安全隔离**，只能运行可信 Notebook 代码。

## Google Drive / Colab 使用说明

8 个教学 `.ipynb` 归档于 `Colab Notebooks/02_LLM_Generative_AI/nanoGPT/tutorials/`。**阅读**时输出图表已嵌入 Notebook 文件，无需 GPU 或训练数据；**再次执行**时，`nanogpt_edu.py` 必须可从 Kernel 的工作目录访问（有网络的 Colab 默认工作目录一般是 `/content`，不会自动切换到 Drive Notebook 所属文件夹）。

打开 Colab 后，在教程原有的 `SETUP` Code Cell **之前**执行一次：

```python
from google.colab import drive
from pathlib import Path
import os

drive.mount('/content/drive')
folder = Path('/content/drive/MyDrive/Colab Notebooks/02_LLM_Generative_AI/nanoGPT/tutorials')
assert (folder / 'nanogpt_edu.py').is_file(), '请先把 nanogpt_edu.py 同步到该目录'
os.chdir(folder)
print('Notebook tutorial working directory:', Path.cwd())
```

随后执行原 Notebook 各 Cell。运行时的 PNG 与 checkpoint 会输出到教程文件夹的 `outputs/` 中；生成文本是原创教学语料的 CPU 微模型结果，不代表 GPT-2 的效果。更换 Drive 账号或路径时，按真实文件夹修改 `folder`。原代码包含本地文件读写；仅运行自己信任的 Notebook。

## 目录说明

```text
tutorials/nanogpt/
├── README.md                  # 学习大纲 / 复现指令
├── index.html                 # 浏览器课程导航
├── nanogpt_edu.py             # nanoGPT 核心设计的教学型微型实现
├── build_tutorials.py         # Notebook Builder 驱动脚本
├── specs/                     # 可审阅、可修改的 JSON Cell specs
├── notebooks/                 # 8 份含真实输出和嵌入图片的 .ipynb
├── outputs/                   # 8 份 HTML、训练 checkpoint、manifest
│   └── figures/               # 独立 PNG 图
└── THIRD_PARTY_LICENSE.txt    # 上游 MIT 许可及说明
```

每份 Notebook 都带源提交 SHA、引用路径、限制和课后练习。各章节默认独立运行，但 05 会优先加载 04 的本地可信 checkpoint；若 04 尚未运行，05 会自行短训一个小模型。

## 正确性检查与安全

1. `nbformat.validate` 验证全部 `.ipynb`，并检查 Cell 输出没有错误。
2. 每章至少两段真实 stdout print、至少一份嵌入 PNG figure，独立保存 PNG 至 `outputs/figures/`。
3. 核验 token shift、causal mask、权重共享、梯度累积等核心数值不变量。
4. 所有模型为微型 CPU 模型。**不承诺** GPT-2/OpenWebText 训练精度、多 GPU 性能或最新 nanochat 代码等价。
5. 不加载来自未知用户的 PyTorch `.pt` 文件；本地 checkpoint 用 `torch.load(weights_only=True)`。
6. `outputs/` 中的图像/数据来自实际执行，非外部的训练曲线占位图。

**后续扩展**：增加外部小语料分词、BPE、正规实验元数据管理、可复现评估、Docker/VM 隔离和更完整的 GPT2 权重对齐测试；不应直接照搬对主机安全敏感的上游训练默认参数。
