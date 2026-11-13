#!/usr/bin/env bash
set -euo pipefail
ROOT="$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")"
cd "$ROOT"
python -m compileall -q notebook_builder.py notebook_ops.py mcp_tools.py mcp_protocol.py mcp_server.py runtime_manager.py run_jobs.py run_worker.py workspace_ops.py research_ledger.py
python -m pytest -q tests
python nbuild.py doctor
python nbuild.py mcp-tools > /dev/null
