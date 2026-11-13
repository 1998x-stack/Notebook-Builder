# Notebook Builder — Scientific Paper Reproduction Contract v2.3

> **IMPORTANT:** A runnable Notebook is NOT a completed reproduction. A real-dataset smoke test is NOT a complete experiment. A completed experiment is NOT equivalent to reproducing an entire paper.

## Status gates

| Stage | What it proves | What it does NOT prove |
|---|---|---|
| `implementation_only` | Architecture source exists, passes unit tests | Any data or training has run |
| `protocol_ready` | A reproducible experiment plan with real dataset loader exists | The real dataset is present or tested |
| `real_data_partial` | Authentic dataset loaded and some training steps executed | Publication training budget and held-out evaluation complete |
| `single_real_experiment_complete` | One model finished its full declared optimizer steps; all original test samples evaluated; evidence recorded | That the paper benchmark suite or reported accuracy is reproduced |
| `scope_protocol_complete_not_independently_reproduced` | All required model/split/seed/metric evidence for one named scope passes deterministic checks | Independent confirmation of source data, exact protocol fidelity, comparable results |
| `external_independent_review_required_for_full_paper` | No automated full-paper certification exists | No machine can infer a scientific result from self-reported JSON alone |

`nbuild quality <ipynb>` checks saved Cell execution and figures **only**.

`nbuild reproduce-quality <paper_evidence.json> --scope cifar10_paper` checks declared scientific protocol coverage. Incomplete or synthetic experiments produce **exit status 2**, even when Jupyter cells execute correctly. `--checkpoint-root` independently hashes locally accessible trained checkpoint bytes. The run manifest is still SELF-REPORTED by the producer and must undergo external inspection for scientific conclusions.

MCP: `notebook_quality` and `paper_reproduction_quality` are separate, read-only tools. All file references for the MCP evidence check must remain inside `NB_ROOT`.

### Original paper and fidelity

Source: [He et al., *Deep Residual Learning for Image Recognition*, arXiv:1512.03385](https://arxiv.org/pdf/1512.03385), §3.4, §4 and Appendix A.

| Workstream | Original benchmark | Scope implemented | Reality / execution state |
|---|---|---|---|
| CIFAR-10 plain vs residual | 50k train, 10k test, SGD 64k steps, batch 128; LR 0.1 -> 0.01 at 32k -> 0.001 at 48k | Full real-data train/resume/testing for Plain-20/32/44/56 and ResNet-20/32/44/56/110/1202; 5 independent 110 seeds | **Not executed on full CIFAR-10 in this environment**; CPU pipeline unit-tested without claiming metric |
| ImageNet/ILSVRC2012 | ~1.28m train, 50k val; 18/34 plain, 18/34/50/101/152 ResNet; 256 global batch; up to 600k steps | Real data folder loader, model definitions, resumable SGD, 10-crop Top-1/Top-5 evaluation | **Not run**. Single-machine memory and data needs; deviations in batchnorm, augmentation, LR plateau mean NOT exact paper fidelity |
| VOC 2007/2012 + MS COCO 2014 detection | Faster R-CNN 2015 baseline (VGG16 vs ResNet101); 8 GPU, 240k+80k RPN and Fast R-CNN COCO stages | Licensed data loaders, checkpointed modern Faster R-CNN, VOC 11-point mAP and official COCO evaluation | **Not run**. Modern FPN, ROIAlign, unified detector train is a **deliberate approximation**, NOT original 2015 detector |
| Paper's competition/ensemble appendix | ImageNet detection/localization, COCO competition ensemble, context, box refinement | Requirements and gaps documented | **Not implemented/reproduced** |

**Very important:** Neither the new CIFAR Notebook nor its research evidence should ever be described as *entire paper reproduced*. The paper also analyzes ImageNet, VOC/COCO, and benchmark ensembles. Full-paper gate cannot automatically return `passed=True`; manual independent audit is mandatory.

## Colab deliverables

Built from source via `nbuild paper-resnet` (all standalone .ipynb; no `pip install notebook-builder` required to run the built notebooks):

```bash
python nbuild.py paper-resnet --dataset cifar10 \
  --output /mnt/data/notebooks/ResNet_2015_CIFAR10_FULL_PAPER_Colab.ipynb
python nbuild.py paper-resnet --dataset imagenet \
  --output /mnt/data/notebooks/ResNet_2015_ImageNet_FULL_Protocol_Colab.ipynb
python nbuild.py paper-resnet --dataset detection \
  --output /mnt/data/notebooks/ResNet_2015_VOC_COCO_Detection_RealData_Colab.ipynb
```

1. Open in Colab; select **T4 GPU** or better. CPU fallback exists but full training may be impractical.
2. For CIFAR10, official `torchvision.datasets.CIFAR10(download=True)` obtains the licensed public dataset and checks official per-batch MD5. The code is **fail-closed**: missing download gives an error, never synthetic fallback.
3. Mount Drive before training; use default `MyDrive/Notebook-Builder/ResNet-2015/...` for checkpoints. Do not replace known checkpoints with untrusted weights. If Colab disconnects, reopen and run again to resume with saved optimizer and sampler state.
4. The default CIFAR session budget is **2000 optimizer steps** (not the full paper's 64000); increase it at your own discretion, accounting for GPU time and storage. **All 14 CIFAR experiment seeds** must complete for that scope to be finished.
5. Real ImageNet and COCO/VOC require dataset archives/permissions and large amounts of time. Do not claim completion because a Notebook exists or a unit test passed.
6. After running, download `paper_evidence.json` and optionally checkpoint files; use Builder's `reproduce-quality` tool. For the comprehensive paper, independently audit methodology and metric differences before scientific reporting.

## Required authoring rules for future Agent sessions

- Notebook specs **must declare** `reproduction: {stage, paper, scope, dataset, fidelity}`; Builder forbids declaring `protocol_complete` at document creation.
- Default `from_spec` never executes code, claims dataset authenticity or writes scientific metrics.
- Run original official datasets, log source fingerprints, environment, seeds, training steps, evaluation set sizes, checkpoints, hyperparameters, and all modifications to paper protocol.
- Do NOT use FakeData or synthetic images for benchmark training/validation. Unit tests may use small mocked fixtures, but those tests must NEVER be presented as scientific benchmark experiments.
- A checkpointless loss graph or an accuracy figure without full real data is not paper evidence.
- Always compare declared `steps_required` to actual optimizer updates. `SESSION_STEP_BUDGET` is a pause threshold, not a substitute for paper's `steps_required`.
- Fail on missing, incomplete, corrupt or wrong dataset, missing evaluation, missing seeds, and unsupported paper deviations. Report remaining unverifiable details and scientific uncertainty explicitly.
