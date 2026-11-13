# ResNet 2015 real-data Colab protocol — implementation validation (2026-11-13)

**DO NOT INTERPRET THIS AS THE PAPER HAVING BEEN REPRODUCED.**

- Source: arXiv 1512.03385 original; these protocols were authored with Notebook Builder 2.3, not executed against their full official datasets.
- CIFAR-10: 18 Notebook cells, 10 executable; official torchvision CIFAR10 download and MD5 checks; 14 full run configurations (plain 20/32/44/56; ResNet 20/32/44/56/110/1202; five independent seeds for ResNet110). All 64,000-step runs and full 10,000-example test evaluation are required before evidence is accepted. No synthetic training-data fallback. Per-session training budget of 2000 is a **pause** and does not change the 64k criterion.
- ImageNet-1k: 15 cells, 8 executable; requires user-supplied authorized 1.28m-class dataset and 50k val images; seven classification comparisons, global 256 batch, 600k-step target, 10-crop evaluation. Original PCA color jitter / LR plateau and multi-GPU BN are *not* exactly implemented; `paper_deviations` will block strict scientific validation.
- VOC/COCO: 13 cells, 6 executable; real official annotations and mAP evaluators. **Modern torchvision FPN/ROIAlign** is not the original 2015 two-stage Faster R-CNN, so cannot be certified as exact paper replication.
- All three Notebook files are valid nbformat v4 and have zero execution counts and zero stored output images, as they should before authentic scientific execution.
- Unit test: CIFAR model trained two optimizer steps on a **tiny synthetic unit fixture only for software validation**, checkpoint saved, loaded, then trained two further steps. Verified step=4 and restored data fingerprint. This is not CIFAR scientific data; no metrics are written as scientific evidence.
- QA: synthetic-origin manifest, incomplete 24 steps, wrong test sample count, single experiment lacking paper matrix, missing five ResNet110 seeds, missing checkpoint, changed dataset fingerprint all fail; `full_paper` never auto-passes even with self-reported metadata.
- Local test groups after additions: core 22 pass, optimizations+scientific gates 19 pass, MCP/runtime 8 pass, research/tutorials 17 pass. **66 unique tests passed in isolated groups**. Previously a combined process timed out in an order-sensitive test; there is no claim that a single-process run passed.
- Remote GitHub Actions status must be checked against the final commit, not earlier runs. Matrix uses Python 3.11 / 3.12.

## Scientific omissions and costs

1. Real CIFAR10 14 full 64k-step runs are not yet executed. Original paper calibration on 45k/5k, full training schedule, stochastic seeding and GPU numerical choices can differ; final numbers must be analyzed with confidence intervals / multiple seeds, not only compared by a single number.
2. Licensed ImageNet tarballs are not present. 1.28m training and 50k validation with 600k steps for multiple models need multi-GPU compute and storage, not a conventional free Colab session. Dataset metadata hashes do **not** constitute trusted content authentication by themselves.
3. COCO2014 80k/40k and VOC2007/2012 data are not present. The modern Faster R-CNN pipeline is an explicit method deviation. The original paper's competition ensemble and auxiliary methods are not implemented.
4. No actual Colab CUDA/GPU run has been performed by this session. CPU/GPU branch handling exists, but GPU performance/precision and memory behavior are unverified.
5. Manuscript replication requires method fidelity, true source integrity, complete experiments, baselines, statistical replicates and independent review. Builder's deterministic QA is necessary but not sufficient.

## Machine-readable audits

```bash
nbuild quality <notebook.ipynb>
nbuild reproduce-quality <paper_evidence.json> --scope cifar10_single
nbuild reproduce-quality <paper_evidence.json> --scope cifar10_paper \
    --checkpoint-root /path/to/checkpoints
nbuild reproduce-quality <paper_evidence.json> --scope full_paper
```

Structural `quality` fails for intentionally unexecuted scientific notebooks. `reproduce-quality` fails on absent/unverified scientific evidence (exit=2) and never auto-certifies whole manuscript.
