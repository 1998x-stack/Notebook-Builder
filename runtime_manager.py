"""Opt-in, trusted-process Jupyter kernel runtime for MCP stdio sessions.

This is not Docker or a sandbox. Each kernel belongs to the MCP server process
and is cleaned on exit; no durable run_submit or cross-process persistence.
"""
from __future__ import annotations

import atexit
import json
import re
import threading
import time
from pathlib import Path
from typing import Any

from jupyter_client import KernelManager

_MAX_STREAM = 20000
_IDENT = re.compile(r'^[A-Za-z_][a-zA-Z0-9_]*$')


class RuntimeManager:
    def __init__(self):
        self._runtimes: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        atexit.register(self.close_all)

    def start(self, notebook_path: str | Path) -> dict:
        key = str(Path(notebook_path).resolve())
        with self._lock:
            if key in self._runtimes:
                return self.status(key)
            if len(self._runtimes) >= 4:
                raise RuntimeError('Maximum 4 local kernels per MCP process')
            manager = KernelManager(kernel_name='python3')
            manager.start_kernel(cwd=str(Path(key).parent))
            client = manager.blocking_client()
            client.start_channels()
            try:
                client.wait_for_ready(timeout=30)
            except Exception:
                client.stop_channels()
                manager.shutdown_kernel(now=True)
                raise
            self._runtimes[key] = {'manager': manager, 'client': client,
                                   'started_at': time.time()}
            return self.status(key)

    def status(self, notebook_path: str | Path) -> dict:
        key = str(Path(notebook_path).resolve())
        with self._lock:
            session = self._runtimes.get(key)
            if session is None:
                return {'status': 'stopped', 'notebook': key, 'persistent_across_mcp_restarts': False}
            alive = bool(session['manager'].is_alive())
            return {'status': 'ready' if alive else 'dead', 'notebook': key,
                    'started_at': session['started_at'],
                    'persistent_across_mcp_restarts': False}

    def stop(self, notebook_path: str | Path) -> dict:
        key = str(Path(notebook_path).resolve())
        with self._lock:
            info = self._runtimes.pop(key, None)
            if info:
                try:
                    info['client'].stop_channels()
                finally:
                    info['manager'].shutdown_kernel(now=True)
            return self.status(key)

    def close_all(self):
        for key in list(self._runtimes):
            try:
                self.stop(key)
            except Exception:
                pass

    def execute(self, notebook_path: str | Path, code: str, *, timeout: int = 60) -> dict:
        if not 1 <= timeout <= 600:
            raise ValueError('timeout must be 1..600 seconds')
        key = str(Path(notebook_path).resolve())
        with self._lock:
            session = self._runtimes.get(key)
            if not session or not session['manager'].is_alive():
                raise RuntimeError('Runtime is stopped; start it first')
            client = session['client']
            msg_id = client.execute(code, allow_stdin=False, stop_on_error=True)
            outputs = []
            reply = None
            idle = False
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline and (reply is None or not idle):
                remaining = min(1.0, max(0.01, deadline-time.monotonic()))
                if reply is None:
                    try:
                        msg = client.get_shell_msg(timeout=0.05)
                        if msg.get('parent_header',{}).get('msg_id') == msg_id:
                            reply = msg.get('content',{})
                    except Exception as ex:
                        if type(ex).__name__ != 'Empty':
                            raise
                try:
                    msg = client.get_iopub_msg(timeout=remaining)
                except Exception as ex:
                    if type(ex).__name__ == 'Empty':
                        continue
                    raise
                if msg.get('parent_header',{}).get('msg_id') != msg_id:
                    continue
                mtype = msg.get('msg_type')
                value = msg.get('content', {})
                if mtype == 'status' and value.get('execution_state') == 'idle':
                    idle = True
                if mtype in ('stream', 'execute_result','display_data','error') and len(outputs) < 40:
                    item = {'output_type': mtype}
                    if mtype == 'stream':
                        item.update(name=value.get('name','stdout'),text=str(value.get('text',''))[:_MAX_STREAM])
                    elif mtype == 'error':
                        item.update(ename=value.get('ename','Error'),evalue=value.get('evalue',''))
                    else:
                        item['data'] = {mime: (str(data)[:_MAX_STREAM] if mime.startswith('text/')
                                                else f'<{mime} omitted>')
                                        for mime, data in value.get('data',{}).items()}
                    outputs.append(item)
            if not idle or reply is None:
                # Cannot confirm whether side effects finished. Require intentional restart.
                self.stop(key)
                raise TimeoutError('Kernel response incomplete; runtime stopped, state is uncertain')
            return {'status': reply.get('status','unknown'), 'outputs': outputs,
                    'execution_count': reply.get('execution_count'), 'notebook': key}

    def variables(self, notebook_path: str | Path, names: list[str] | None = None,
                  limit: int = 50) -> dict:
        if names is not None and (len(names) > 100 or not all(isinstance(n,str) and _IDENT.fullmatch(n) for n in names)):
            raise ValueError('names must be a list of <=100 Python identifiers')
        if not 1 <= limit <= 100:
            raise ValueError('limit must be 1..100')
        # No repr/str of arbitrary values: metadata-only reporting avoids accidentally
        # serializing huge arrays, arbitrary object methods and obvious secrets.
        payload = json.dumps({'names': names, 'limit': limit})
        source = '''\nimport json as __nb_json\n__nb_spec = __nb_json.loads(%r)\n__nb_ns = get_ipython().user_ns\n__nb_keys = sorted(k for k in __nb_ns if isinstance(k, str) and not k.startswith('_'))\nif __nb_spec['names'] is not None:\n    __nb_keys = [k for k in __nb_keys if k in __nb_spec['names']]\n__nb_rows = []\nfor __nb_name in __nb_keys[:__nb_spec['limit']]:\n    __nb_obj = __nb_ns[__nb_name]\n    __nb_row = {'name': __nb_name, 'type': type(__nb_obj).__name__}\n    try:\n        if type(__nb_obj) in (list, tuple, dict, set, str):\n            __nb_row['length'] = len(__nb_obj)\n        elif hasattr(__nb_obj, 'shape') and type(__nb_obj).__module__.startswith(('numpy','pandas','torch')):\n            __nb_row['shape'] = [int(i) for i in __nb_obj.shape][:8]\n    except Exception:\n        pass\n    __nb_rows.append(__nb_row)\nprint('NB_VARIABLES_V2:' + __nb_json.dumps(__nb_rows, ensure_ascii=True))\n''' % payload
        result = self.execute(notebook_path, source, timeout=30)
        if result['status'] != 'ok':
            raise ValueError('Kernel variable inspection failed')
        text = ''.join(item.get('text','') for item in result['outputs'] if item['output_type']=='stream')
        for line in reversed(text.splitlines()):
            if line.startswith('NB_VARIABLES_V2:'):
                return {'variables': json.loads(line[len('NB_VARIABLES_V2:'):])}
        raise ValueError('Variable probe returned no structured result')


runtime_manager = RuntimeManager()
