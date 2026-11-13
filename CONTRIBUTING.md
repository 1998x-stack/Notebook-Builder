# Contributing to Notebook Builder

Please read [AGENTS.md](AGENTS.md) and [docs/NEW_SESSION.md](docs/NEW_SESSION.md). Use short, tested changes; preserve stable MCP tool names and explicit read/write/execute gates. Never use untrusted code execution as a test fixture. Keep generated HTML, training images and model weights out of commits unless a deliberate release artifact. Self-contained educational `.ipynb` files with embedded output may be committed as reference fixtures.

Before a PR, run:

```bash
python -m compileall -q notebook_builder.py notebook_ops.py mcp_tools.py mcp_protocol.py mcp_server.py
python -m pytest -q tests/test_builder.py tests/test_hardening.py tests/test_ops.py tests/test_optimizations.py tests/test_colab_resnet.py
python -m pytest -q tests/test_mcp.py tests/test_jobs.py tests/test_runtime.py
python -m pytest -q tests/test_research.py
python -m pytest -q tests/test_nanogpt_tutorials.py tests/test_resnet_tutorial.py tests/test_workspace.py
```

A single-process whole-suite run may be sensitive to hosted Jupyter process cleanup; CI shards test families into separate invocations. Log failures explicitly rather than claiming a full run passed.
