# ResNet Paper → Notebook Builder tutorial

Reference: **Deep Residual Learning for Image Recognition**, arXiv:1512.03385.

This is the repository-to-notebook test of `notebook-builder` using 2015 ResNet paper specifications.

### Rebuild and execute

```bash
cd /path/to/notebook-builder
python nbuild.py tutorial resnet --list
python nbuild.py tutorial resnet --build --execute --html
# or run explicit generator (configure RESNET_TUTORIAL_OUT to control destination)
python tutorials/resnet/build_tutorial.py --output ./tutorials/resnet/outputs --force
```

- `build_tutorial.py`: Paper-structured tutorial cells; writes `spec.json`, constructs via `notebook_builder.from_spec`, executes via `notebook_builder.execute`, gates via `notebook_quality.inspect`, then exports via `notebook_builder.export_html`.
- `resnet_educational.py`: Standalone reusable PyTorch ResNet-v1 implementation. Its full content is **embedded** in the generated Notebook.
- `outputs/`: One executed Notebook, HTML, 13 figures, tiny trained model checkpoint, manifest, notebook quality and reproducible source spec.
- `../../tests/test_resnet_tutorial.py`: CPU tests checking all five ImageNet parameter counts, original stride, CIFAR formulas and Option A, gradients, output validity and CLI.

The Notebook is offline and CPU-executable; optional real CIFAR10 and ImageNet pipelines require explicitly supplied datasets and compute. Synthetic losses are not original paper benchmark reproduction. Only run trusted notebooks: local kernels have no sandbox.

### Google Colab / optional GPU

The `.ipynb` is **self-contained**. It selects `torch.device('cuda')` automatically when PyTorch reports CUDA, otherwise CPU, with optional AMP for the toy training loop. Configuration defaults: `RESNET_DEVICE=auto`, `RESNET_AMP=1`, `RESNET_WORKDIR=/content/resnet-paper-lab` inside Colab; no weight/data download by default. See [Colab guide](../../docs/COLAB_RESNET.md). CUDA behavior is source-reviewed, whereas automated end-to-end acceptance was run on CPU.
