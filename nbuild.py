#!/usr/bin/env python3
"""Notebook Builder CLI: reproducible local Jupyter authoring and MCP inspection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import notebook_builder as core
import notebook_ops as ops
import workspace_ops as ws


def source_arg(group):
    src = group.add_mutually_exclusive_group()
    src.add_argument('--source', help='Inline UTF-8 source')
    src.add_argument('--source-file', help='Read text file (- reads stdin)')


def get_source(args, *, required=True):
    if getattr(args, 'source', None) is not None:
        return args.source
    sf = getattr(args, 'source_file', None)
    if sf == '-':
        return sys.stdin.read()
    if sf is not None:
        return Path(sf).read_text(encoding='utf-8')
    if required:
        raise ValueError('Provide --source or --source-file')
    return None


def build_parser():
    p = argparse.ArgumentParser(prog='nbuild',
        description='Create, inspect, edit, execute and render standard Jupyter notebooks')
    subs = p.add_subparsers(dest='cmd', required=True)
    c = subs.add_parser('create', help='Create .ipynb from title or JSON spec')
    c.add_argument('path'); c.add_argument('--title'); c.add_argument('--spec')
    c.add_argument('--overwrite', action='store_true')
    for name in ('add', 'update', 'delete', 'move'):
        a = subs.add_parser(name, help=f'{name} a Cell')
        a.add_argument('path')
        a.add_argument('--expected-version', type=int)
        a.add_argument('--operation-id', help='Idempotency key for safe retries')
        if name != 'add':
            a.add_argument('--cell', required=True, help='Cell ID or index')
        if name in ('add','update'):
            a.add_argument('--type', choices=['code','markdown','raw'],
                           default='code' if name == 'add' else None)
            source_arg(a)
        if name == 'add':
            where = a.add_mutually_exclusive_group()
            where.add_argument('--at',type=int)
            where.add_argument('--after')
        if name == 'move':
            a.add_argument('--to',required=True,type=int)
    a = subs.add_parser('quality', help='Read-only Notebook execution and figure QA')
    a.add_argument('path')
    a.add_argument('--allow-unexecuted', action='store_true')
    a.add_argument('--min-prints', type=int, default=2)
    a.add_argument('--min-figures', type=int, default=1)
    for name in ('inspect','validate','history','get','versions','audit','fingerprint'):
        a = subs.add_parser(name)
        a.add_argument('path')
        if name == 'get': a.add_argument('--include-outputs',action='store_true')
    for name in ('cell-get','title','fork','restore','diff','merge-preview','merge-commit'):
        a = subs.add_parser(name)
        a.add_argument('path')
        if name == 'cell-get':
            a.add_argument('--cell',required=True);a.add_argument('--include-outputs',action='store_true')
        if name == 'title':
            a.add_argument('--title',required=True)
        if name in ('fork','restore'):
            a.add_argument('--version',required=True,type=int)
        if name == 'fork': a.add_argument('--to',required=True)
        if name == 'diff':
            a.add_argument('--from-version',required=True,type=int)
            a.add_argument('--to-version',type=int)
        if name in ('merge-preview','merge-commit'):
            a.add_argument('--base-version',required=True,type=int)
            a.add_argument('--draft',required=True,help='Unsaved .ipynb JSON draft')
            a.add_argument('--choices',help='Optional JSON map of cell ID -> mine/theirs')
        if name in ('title','restore','merge-commit'):
            a.add_argument('--expected-version',type=int,required=name in ('restore','merge-commit'))
        if name == 'merge-commit': a.add_argument('--operation-id')
    a = subs.add_parser('import',help='Import validated .ipynb from file')
    a.add_argument('source');a.add_argument('destination');a.add_argument('--overwrite',action='store_true')
    a = subs.add_parser('export',help='Export ipynb, markdown or html')
    a.add_argument('source');a.add_argument('destination');a.add_argument('--format',choices=['ipynb','html','markdown'],default='ipynb')
    a.add_argument('--overwrite',action='store_true')
    a = subs.add_parser('list',help='List .ipynb files in a workspace')
    a.add_argument('--root',default='/mnt/data/notebooks')
    a.add_argument('--flat',action='store_true')
    a = subs.add_parser('execute',help='Run trusted notebook code in fresh local kernel; NO sandbox')
    a.add_argument('path');a.add_argument('--timeout',type=int,default=120)
    a.add_argument('--kernel',default='python3');a.add_argument('--cwd');a.add_argument('--expected-version',type=int)
    a = subs.add_parser('run-cells',help='Execute a window, optionally running earlier code as bootstrap')
    a.add_argument('path');a.add_argument('--start',type=int,default=0)
    a.add_argument('--count',type=int,default=10);a.add_argument('--no-bootstrap',action='store_true')
    a.add_argument('--timeout',type=int,default=120);a.add_argument('--expected-version',type=int)
    a = subs.add_parser('code',help='Execute trusted code in fresh kernel (no persistent state)')
    a.add_argument('path');source_arg(a);a.add_argument('--persist',action='store_true')
    a.add_argument('--timeout',type=int,default=60);a.add_argument('--expected-version',type=int)
    for name,ext in (('html','.html'),('markdown','.md')):
        a = subs.add_parser(name,help=f'Export {ext}')
        a.add_argument('path');a.add_argument('--output',required=True)
    a = subs.add_parser('work-list',help='List files in a Notebook workspace')
    a.add_argument('notebook');a.add_argument('--root',default='/mnt/data/notebooks')
    a.add_argument('--recursive',action='store_true')
    for name in ('work-read','work-write','work-delete'):
        a = subs.add_parser(name)
        a.add_argument('notebook');a.add_argument('relative')
        a.add_argument('--root',default='/mnt/data/notebooks')
        if name == 'work-write':
            source_arg(a);a.add_argument('--overwrite',action='store_true')
        if name == 'work-read': a.add_argument('--max-chars',type=int,default=20000)
    a=subs.add_parser('run-submit',help='Submit a trusted Cell as local background job')
    a.add_argument('notebook');a.add_argument('--root',default='/mnt/data/notebooks')
    a.add_argument('--cell',required=True);a.add_argument('--key',required=True)
    a.add_argument('--timeout',type=int,default=120);a.add_argument('--expected-version',type=int)
    for name in ('run-status','run-wait','run-cancel'):
        a=subs.add_parser(name)
        a.add_argument('run_id');a.add_argument('--root',default='/mnt/data/notebooks')
        if name=='run-wait':a.add_argument('--seconds',type=int,default=10)
    tut = subs.add_parser('tutorial', help='Build/execute illustrated tutorials (nanoGPT, ResNet)')
    tut.add_argument('topic', choices=['nanogpt','resnet'])
    tut.add_argument('--list', action='store_true')
    tut.add_argument('--build', action='store_true')
    tut.add_argument('--execute', action='store_true', help='Run trusted local Python cells (no sandbox)')
    tut.add_argument('--html', action='store_true', help='Render standalone HTML')
    tut.add_argument('--force', action='store_true', help='Overwrite existing notebooks and retain history')
    tut.add_argument('--only', nargs='*', help='Chapter prefixes: 00 01 etc')
    tut.add_argument('--timeout', type=int, default=120)
    subs.add_parser('mcp-tools',help='List available MCP tools and permissions')
    subs.add_parser('doctor',help='Inspect real runtime capabilities')
    return p


def run(args):
    c = args.cmd
    if c == 'create':
        return core.from_spec(args.path,args.spec,overwrite=args.overwrite) if args.spec else core.create(args.path,title=args.title,overwrite=args.overwrite)
    if c == 'add':
        return core.add(args.path,args.type,get_source(args),at=args.at,after=args.after,
                        expected_version=args.expected_version,operation_id=args.operation_id)
    if c == 'update':
        return core.update(args.path,args.cell,source=get_source(args,required=False),
                           cell_type=args.type,expected_version=args.expected_version,operation_id=args.operation_id)
    if c == 'delete':
        return core.delete(args.path,args.cell,expected_version=args.expected_version,operation_id=args.operation_id)
    if c == 'move':
        return core.move(args.path,args.cell,args.to,expected_version=args.expected_version,operation_id=args.operation_id)
    if c in ('inspect','validate'): return core.describe(args.path)
    if c == 'get': return ops.notebook_get(args.path,include_outputs=args.include_outputs)
    if c == 'cell-get': return ops.cell_get(args.path,args.cell,include_outputs=args.include_outputs)
    if c == 'history': return core.history(args.path)
    if c == 'versions':
        from mcp_tools import REGISTRY
        p=Path(args.path).resolve()
        return {'current_version':core.describe(p)['version'], 'snapshots':core.history(p)['snapshots']}
    if c == 'audit':
        import notebook_builder
        nb = notebook_builder._read(notebook_builder._path(args.path))
        return {'version':notebook_builder._version(nb),'receipts':nb.metadata.get(core.META_KEY,{}).get('operation_receipts',[]),
                'snapshots':core.history(args.path)['snapshots']}
    if c == 'fingerprint':
        import hashlib
        p=Path(args.path)
        n=core._read(core._path(p))
        return {'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'version':core._version(n),
                'cells':[{'id':cell.id,'source_sha256':hashlib.sha256(cell.source.encode()).hexdigest()} for cell in n.cells]}
    if c == 'title': return ops.title(args.path,args.title,expected_version=args.expected_version)
    if c == 'fork': return ops.fork_version(args.path,args.version,args.to)
    if c == 'restore': return ops.restore_version(args.path,args.version,expected_version=args.expected_version)
    if c == 'diff': return ops.diff_versions(args.path,args.from_version,args.to_version)
    if c in ('merge-preview','merge-commit'):
        draft=json.loads(Path(args.draft).read_text(encoding='utf-8'))
        choices=json.loads(Path(args.choices).read_text(encoding='utf-8')) if args.choices else None
        if c=='merge-preview':return ops.merge_preview(args.path,args.base_version,draft,choices)
        return ops.merge_commit(args.path,args.base_version,draft,choices=choices,
                                expected_version=args.expected_version,operation_id=args.operation_id)
    if c == 'import':return ops.import_notebook(args.source,args.destination,overwrite=args.overwrite)
    if c == 'export':
        if args.format=='ipynb':return ops.export_ipynb(args.source,args.destination,overwrite=args.overwrite)
        if args.format=='html':return core.export_html(args.source,args.destination)
        return ops.export_markdown(args.source,args.destination)
    if c == 'list':return ops.list_notebooks(args.root,recursive=not args.flat)
    if c == 'quality':
        from notebook_quality import inspect as inspect_quality
        return inspect_quality(args.path, require_executed=not args.allow_unexecuted,
                               min_print_cells=args.min_prints, min_figure_cells=args.min_figures)
    if c == 'execute':return core.execute(args.path,timeout=args.timeout,kernel=args.kernel,
                                          cwd=args.cwd,expected_version=args.expected_version)
    if c == 'run-cells':return ops.execute_cells(args.path,start=args.start,count=args.count,
                                                 bootstrap=not args.no_bootstrap,timeout=args.timeout,
                                                 expected_version=args.expected_version)
    if c == 'code':return ops.code_execute(args.path,get_source(args),persist=args.persist,
                                           timeout=args.timeout,expected_version=args.expected_version)
    if c == 'html':return core.export_html(args.path,args.output)
    if c == 'markdown':return ops.export_markdown(args.path,args.output)
    if c == 'work-list':return ws.list_files(args.root,args.notebook,recursive=args.recursive)
    if c == 'work-read':return ws.read(args.root,args.notebook,args.relative,max_chars=args.max_chars)
    if c == 'work-write':return ws.write(args.root,args.notebook,args.relative,get_source(args),overwrite=args.overwrite)
    if c == 'work-delete':return ws.delete(args.root,args.notebook,args.relative)
    if c.startswith('run-') and c in ('run-submit','run-status','run-wait','run-cancel'):
        import run_jobs as jobs
        if c=='run-submit':return jobs.submit(args.root,args.notebook,args.cell,
                        idempotency_key=args.key,timeout=args.timeout,
                        expected_version=args.expected_version)
        if c=='run-status':return jobs.status(args.root,args.run_id)
        if c=='run-wait':return jobs.wait(args.root,args.run_id,wait_seconds=args.seconds)
        return jobs.cancel(args.root,args.run_id)
    if c == 'tutorial':
        import runpy
        if args.topic=='nanogpt':
            path = Path(__file__).resolve().parent / 'tutorials' / 'nanogpt' / 'build_tutorials.py'
            if not path.is_file():
                raise FileNotFoundError(f'Tutorial asset missing from this installation: {path}; use the source checkout')
            return runpy.run_path(str(path))['run_series'](list_only=args.list,build=args.build,
                    execute=args.execute,html=args.html,force=args.force,only=args.only,timeout=args.timeout)
        path = Path(__file__).resolve().parent / 'tutorials' / 'resnet' / 'build_tutorial.py'
        if not path.is_file():
            raise FileNotFoundError(f'ResNet tutorial missing: {path}')
        return runpy.run_path(str(path))['run_from_cli'](list_only=args.list,build=args.build,
                execute=args.execute,html=args.html,force=args.force,only=args.only,timeout=args.timeout)
    if c == 'mcp-tools':
        from mcp_tools import REGISTRY
        return {'count':len(REGISTRY),'tools':[{'name':t.name,'permission':t.permission,'description':t.description} for t in REGISTRY.values()]}
    if c == 'doctor':
        return ops.execution_diagnostics() | {'builder':str(Path(__file__).absolute())}
    raise ValueError(f'Unsupported command: {c}')


def main(argv=None):
    args=build_parser().parse_args(argv)
    try:
        result = run(args)
        print(json.dumps(result,indent=2,ensure_ascii=False,default=str))
        # A QA command must fail CI on unmet contracts, not just print false.
        return 2 if args.cmd == 'quality' and not result['passed'] else 0
    except Exception as exc:
        print(json.dumps({'error':type(exc).__name__,'message':str(exc)},ensure_ascii=False),file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
