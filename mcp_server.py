#!/usr/bin/env python3
"""MCP stdio server for Notebook Builder.

Uses the official Python MCP SDK when installed; otherwise runs a small
self-contained subset MCP stdio server verified by local protocol tests.

Default: read-only and execution-disabled. All file paths are scoped to NB_ROOT.
"""
from __future__ import annotations

import os
import functools
import inspect

from mcp_tools import REGISTRY, invoke


def main():
    # Official adapter (optional). The small fallback works without the MCP SDK.
    if os.getenv('NB_MCP_TRANSPORT', 'native') == 'sdk':
        try:
            from mcp.server.fastmcp import FastMCP
        except ImportError as exc:
            raise SystemExit('Official MCP SDK unavailable; select NB_MCP_TRANSPORT=native') from exc
        server = FastMCP('notebook-builder-local')
        for item in REGISTRY.values():
            # Register a signature-preserving wrapper: never bypass read-only /
            # execution gates by registering the unguarded business function.
            def bound(*args, __name=item.name, **kwargs):
                return invoke(__name, kwargs)
            guarded = functools.wraps(item.fn)(bound)
            guarded.__signature__ = inspect.signature(item.fn)
            server.tool(name=item.name, description=item.description)(guarded)
        server.run(transport='stdio')
    else:
        from mcp_protocol import main as native_main
        native_main()


if __name__ == '__main__':
    main()
