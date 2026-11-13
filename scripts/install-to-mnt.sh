#!/usr/bin/env bash
# Portable, conservative setup from a checked-out or extracted Notebook-Builder source tree.
set -euo pipefail
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
TARGET="${NB_TARGET:-/mnt/notebook-builder}"
TARGET="$(python3 -c 'import os,sys;print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$TARGET")"
RUN_TESTS=0
for a in "$@"; do
  case "$a" in
    --test) RUN_TESTS=1;;
    --quick) RUN_TESTS=0;;
    *) echo "Unknown option: $a" >&2; exit 2;;
  esac
done
if [[ ! -f "$SELF/pyproject.toml" || ! -f "$SELF/nbuild.py" ]]; then
  echo 'Not a Notebook-Builder source tree' >&2; exit 2
fi
if [[ "$SELF" != "$TARGET" ]]; then
  if [[ -e "$TARGET" && -n "$(ls -A "$TARGET" 2>/dev/null)" ]]; then
    echo "Target not empty: $TARGET; refusing to overwrite; choose NB_TARGET or move your existing directory" >&2;exit 3
  fi
  mkdir -p "$TARGET"
  cp -a "$SELF/." "$TARGET/"
fi
cd "$TARGET"
PY="${PYTHON_BIN:-python3}"
"$PY" - <<'PY'
import importlib.util
req=('nbformat','nbclient','nbconvert','ipykernel','jupyter_client','pydantic','setuptools')
missing=[x for x in req if importlib.util.find_spec(x) is None]
if missing:raise SystemExit('Missing prerequisites: '+', '.join(missing)+'; install requirements.txt before retrying')
PY
"$PY" -m pip install -e . --no-build-isolation --no-deps --disable-pip-version-check
"$PY" -m compileall -q notebook_builder.py notebook_ops.py mcp_tools.py mcp_protocol.py mcp_server.py runtime_manager.py run_jobs.py run_worker.py workspace_ops.py research_ledger.py notebook_quality.py
"$PY" nbuild.py doctor
if [[ "$RUN_TESTS" == 1 ]]; then
  "$PY" -m pytest -q tests
fi
if [[ -w /usr/local/bin ]]; then
  ln -sfn "$TARGET/nbuild" /usr/local/bin/nbuild || true
  ln -sfn "$TARGET/nbuild-mcp" /usr/local/bin/nbuild-mcp || true
fi
printf '\nReady at %s\nUse: python3 %s/nbuild.py doctor\n' "$TARGET" "$TARGET"
