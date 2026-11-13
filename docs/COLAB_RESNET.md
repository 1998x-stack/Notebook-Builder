# ResNet Paper Tutorial — Google Colab deployment

The self-contained notebook lives at `tutorials/resnet/outputs/ResNet_Deep_Residual_Learning_Complete_Tutorial.ipynb`.

1. Upload the `.ipynb` to your Google Drive or open the file in Colab.
2. In Colab choose **Runtime → Change runtime type → T4 GPU** if an accelerator is offered. CPU works too.
3. Run **Runtime → Run all**. The first code Cell prints `torch.cuda.is_available()`, the device name if CUDA is available, selected `DEVICE`, and whether AMP was enabled.
4. Automatic selection: `RESNET_DEVICE=auto` (default) selects `cuda` if available, otherwise `cpu`. `RESNET_AMP=1` (default) enables GPU autocast; `RESNET_AMP=0` disables it. Model architecture audits stay on CPU to avoid GPU out-of-memory; the *toy training/validation/checkpoint* step uses the selected device.
5. Outputs go to `/content/resnet-paper-lab/{figures,artifacts}` on Colab. They are ephemeral; after execution, optionally mount Drive and copy the folder using the final Markdown instructions.
6. Real CIFAR-10/ImageNet experiments are **opt-in**, need appropriate datasets and resources, and are **not** demonstrated by toy results. Enable with `RUN_REAL_CIFAR10=1` (existing local dataset required) or `RUN_IMAGENET_TRAIN=1` (ImageFolder root required). Never confuse toy metrics with paper accuracy.

## Verify without Colab GPU

```bash
cd /mnt/notebook-builder
RESNET_DEVICE=cpu python3 tutorials/resnet/build_tutorial.py --force --timeout 150
python3 nbuild.py quality tutorials/resnet/outputs/ResNet_Deep_Residual_Learning_Complete_Tutorial.ipynb --min-prints 14 --min-figures 9
python3 -m pytest -q tests/test_resnet_tutorial.py tests/test_colab_resnet.py
```

The present verification checks CPU fallback and source contracts. It does not claim a live CUDA/T4 execution where no GPU is available.
