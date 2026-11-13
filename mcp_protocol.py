"""Small dependency-light MCP stdio transport (JSON-RPC 2.0).

Covers initialize/ping, tools/list, tools/call, resources/list,
resources/templates/list and resources/read; it is not a full implementation of
MCP features (prompts, elicitation, sampling, subscriptions, progress, auth).
This transport is locally tested, not certified against all MCP clients.
"""
from __future__ import annotations

import json
import sys
from urllib.parse import quote, unquote

from mcp_tools import REGISTRY, ROOT, invoke, notebook_list, notebook_get

SERVER_VERSION = '2.1.0'
SUPPORTED_VERSIONS = ('2025-06-18', '2025-03-26', '2024-11-05')
MAX_INPUT_BYTES = 3 * 1024 * 1024
MAX_OUTPUT_BYTES = 2 * 1024 * 1024


def _err(id, code: int, message: str) -> dict:
    return {'jsonrpc': '2.0', 'id': id, 'error': {'code': code, 'message': message}}


def _result(id, result) -> dict:
    return {'jsonrpc': '2.0', 'id': id, 'result': result}


def _tool_list() -> dict:
    return {'tools': [{'name': s.name, 'description': s.description, 'inputSchema': s.schema,
                       'annotations': {'readOnlyHint': s.permission == 'read',
                                       'destructiveHint': s.permission == 'destructive'}}
                      for s in REGISTRY.values()]}


def _resources():
    return {'resources': [{'uri': 'nbuild://notebooks', 'name': 'Notebook index',
                           'mimeType': 'application/json'}]}


def handle(request: dict) -> dict | None:
    if not isinstance(request, dict) or request.get('jsonrpc') != '2.0' or not isinstance(request.get('method'), str):
        return _err(request.get('id') if isinstance(request,dict) else None, -32600, 'Invalid JSON-RPC request')
    method = request['method']
    if 'id' not in request:
        # MCP notifications do not have a response.
        return None
    id = request['id']
    params = request.get('params') or {}
    if not isinstance(params, dict):
        return _err(id, -32602, 'params must be an object')
    try:
        if method == 'initialize':
            version = params.get('protocolVersion')
            chosen = version if version in SUPPORTED_VERSIONS else SUPPORTED_VERSIONS[0]
            return _result(id, {'protocolVersion': chosen,
                                'capabilities': {'tools': {'listChanged': False},
                                                 'resources': {'listChanged': False}},
                                'serverInfo': {'name': 'notebook-builder-local', 'version': SERVER_VERSION}})
        if method == 'ping':
            return _result(id, {})
        if method == 'tools/list':
            return _result(id, _tool_list())
        if method == 'tools/call':
            name, arguments = params.get('name'), params.get('arguments') or {}
            if not isinstance(name, str) or not isinstance(arguments, dict):
                return _err(id, -32602, 'Tool name and argument object required')
            try:
                output = invoke(name, arguments)
                serialized = json.dumps(output, ensure_ascii=False, default=str)
                if len(serialized.encode()) > MAX_OUTPUT_BYTES:
                    raise ValueError('Tool result exceeds MCP response size cap')
                return _result(id, {'content': [{'type': 'text', 'text': serialized}],
                                    'isError': False})
            except Exception as exc:
                return _result(id, {'content': [{'type': 'text', 'text':
                              f'{type(exc).__name__}: {exc}'[:3000]}], 'isError': True})
        if method == 'resources/list':
            return _result(id, _resources())
        if method == 'resources/templates/list':
            return _result(id, {'resourceTemplates': [{
                'uriTemplate': 'nbuild://notebooks/{path}',
                'name': 'Notebook document', 'mimeType': 'application/json'}]})
        if method == 'resources/read':
            uri = params.get('uri')
            if uri == 'nbuild://notebooks':
                data = notebook_list()
            elif isinstance(uri, str) and uri.startswith('nbuild://notebooks/'):
                data = notebook_get(unquote(uri[len('nbuild://notebooks/'):]), include_outputs=False)
            else:
                return _err(id, -32602, 'Unknown resource URI')
            content = json.dumps(data, ensure_ascii=False, default=str)
            if len(content.encode()) > MAX_OUTPUT_BYTES:
                raise ValueError('Resource exceeds response size cap')
            return _result(id, {'contents': [{'uri': uri, 'mimeType': 'application/json', 'text': content}]})
        return _err(id, -32601, f'Method not implemented: {method}')
    except Exception as exc:
        return _err(id, -32603, f'{type(exc).__name__}: {exc}'[:1500])


def main() -> None:
    for raw in sys.stdin.buffer:
        if len(raw) > MAX_INPUT_BYTES:
            response = _err(None, -32600, 'Request exceeds message size limit')
        else:
            try:
                request = json.loads(raw)
                response = handle(request)
            except json.JSONDecodeError:
                response = _err(None, -32700, 'Parse error')
            except Exception as exc:
                response = _err(None, -32603, f'{type(exc).__name__}: {exc}'[:1500])
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False, separators=(',', ':')) + '\n')
            sys.stdout.flush()


if __name__ == '__main__':
    main()
