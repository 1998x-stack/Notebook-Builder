#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
python - <<'PY'
import importlib.util
required=['nbformat','nbclient','nbconvert','ipykernel','jupyter_client','pydantic','setuptools']
missing=[m for m in required if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit('Missing prerequisites (install requirements.txt first): '+', '.join(missing))
PY
python -m pip install -e . --no-build-isolation --no-deps
python -m pytest -q tests
nbuild doctor
printf '\nNotebook Builder installed and verified. MCP default is read-only.\n'
