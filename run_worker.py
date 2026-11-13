#!/usr/bin/env python3
"""Dedicated job worker invoked by run_jobs; never run as a public daemon."""
from __future__ import annotations
import sys
from run_jobs import run_worker

if __name__=='__main__':
    if len(sys.argv)!=5:
        raise SystemExit('usage: run_worker.py ROOT RELATIVE_NOTEBOOK CELL_ID RUN_ID')
    raise SystemExit(run_worker(*sys.argv[1:]))
