# Fresh ChatGPT session → `/mnt/notebook-builder`

The ChatGPT execution container may be temporary. A successful installation in one chat **does not** ensure the same `/mnt` contents exist in another chat. Do not assume network access, GPU, globally installed MCP plugins, or persistent secrets.

## A. GitHub network available

```bash
mkdir -p /mnt
# Refuse to overwrite an existing checkout. Review its changes first.
git clone https://github.com/1998x-stack/Notebook-Builder.git /mnt/notebook-builder
bash /mnt/notebook-builder/scripts/install-to-mnt.sh --quick
python3 /mnt/notebook-builder/nbuild.py doctor
```

To validate full test fixtures, run `bash /mnt/notebook-builder/scripts/install-to-mnt.sh --test` after dependencies are installed.

## B. No shell network / no `git clone`

1. In ChatGPT, use the connected GitHub app to obtain repository code, or upload a portable ZIP downloaded from the repository/previous deliverable.
2. Unpack it with `unzip -q Notebook-Builder.zip -d /mnt` and locate the source root with `pyproject.toml`.
3. Run `bash <unpacked-source>/scripts/install-to-mnt.sh --quick`. The script copies the source to `/mnt/notebook-builder` **only if that target is empty**, installs editable Python entry points without dependency downloads, compiles core modules and runs `doctor`.
4. If prerequisites are missing, install them with network access or use an environment already containing them. The installer will not silently fetch packages.

The script never needs a GitHub token, does not delete existing workspaces, and does not enable execution access for MCP clients.

## Starter prompt for a new ChatGPT session

> Use the connected GitHub repository `https://github.com/1998x-stack/Notebook-Builder`. Inspect `AGENTS.md` and `docs/NEW_SESSION.md`, install the repository to `/mnt/notebook-builder` using `scripts/install-to-mnt.sh`, run `nbuild doctor`, and report any unavailable network/dependencies. If shell GitHub access is unavailable, use a downloadable source archive from the conversation. Do not invent installation success. Keep the MCP read-only and execution disabled unless specifically requested.

## Common commands

```bash
cd /mnt/notebook-builder
python3 nbuild.py doctor
python3 nbuild.py mcp-tools
python3 nbuild.py tutorial nanogpt --list
python3 nbuild.py tutorial resnet --list
python3 nbuild.py tutorial resnet --build --execute --html --force
python3 nbuild.py quality tutorials/resnet/outputs/ResNet_Deep_Residual_Learning_Complete_Tutorial.ipynb --min-prints 14 --min-figures 9
python3 -m pytest -q tests
```

## Environment and safety notes

- `torch` is optional for the Builder core, but required for ML tutorial execution and scientific model tests; CPU fallback exists when CUDA is unavailable.
- An `ipynb` includes runnable arbitrary Python; local execution is **not sandboxed**. Run only trusted code.
- `NB_ROOT` limits paths exposed to MCP; `NB_READONLY` and `NB_ALLOW_EXEC` are distinct gates. Start read-only.
- `RUN_REAL_CIFAR10`, `RUN_IMAGENET_TRAIN`, `ALLOW_WEIGHT_DOWNLOAD` default to off; no model/data download is implied.
- CLI can write `/mnt/data`, but the new session's filesystem and linked application permissions are not guaranteed.
