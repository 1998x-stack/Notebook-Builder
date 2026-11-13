"""Deterministic, read-only quality gates for Jupyter tutorial artifacts.

Counts are evidence in saved notebook output, not claims about model correctness.
User-defined semantic or numerical assertions must run in the code cells.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import notebook_builder as core

PNG_MAGIC = b'\x89PNG\r\n\x1a\n'


def _check_png(value: Any) -> tuple[bool, int, int]:
    if not isinstance(value, str):
        return False, 0, 0
    try:
        raw = base64.b64decode(''.join(value.split()), validate=True)
        if len(raw) < 24 or raw[:8] != PNG_MAGIC or raw[12:16] != b'IHDR':
            return False, 0, 0
        w, h = int.from_bytes(raw[16:20], 'big'), int.from_bytes(raw[20:24], 'big')
        return (w > 0 and h > 0 and w <= 32768 and h <= 32768), w, h
    except (ValueError, base64.binascii.Error):
        return False, 0, 0


def inspect(path: str | Path, *, require_executed: bool = True,
            min_print_cells: int = 2, min_figure_cells: int = 1) -> dict:
    """Validate saved Notebook execution and print/figure contracts.

    This is deliberately side-effect free. It does not run untrusted Python.
    A `passed` result means structural output checks pass, NOT that the
    tutorial's scientific claims or source dependencies were independently vetted.
    """
    if min_print_cells < 0 or min_figure_cells < 0:
        raise ValueError('Output minima cannot be negative')
    p = core._path(path)
    nb = core._read(p)
    code = [c for c in nb.cells if c.cell_type == 'code']
    print_count = 0
    image_count = 0
    figure_cells = 0
    errors = []
    bad_images = []
    missing = []
    for index, cell in enumerate(nb.cells):
        if cell.cell_type != 'code':
            continue
        if cell.execution_count is None:
            missing.append(index)
        has_print = False
        has_figure = False
        for output in cell.get('outputs', []):
            if output.output_type == 'error':
                errors.append({'index': index, 'ename': str(output.get('ename', '?'))[:100]})
            if output.output_type == 'stream' and output.get('name') == 'stdout' and str(output.get('text', '')).strip():
                has_print = True
            if output.output_type in ('display_data', 'execute_result'):
                png = output.get('data', {}).get('image/png')
                if png is not None:
                    valid, width, height = _check_png(png)
                    if valid:
                        image_count += 1
                        has_figure = True
                    else:
                        bad_images.append(index)
        print_count += int(has_print)
        figure_cells += int(has_figure)
    reasons = []
    if not code:
        reasons.append('No executable code cells')
    if require_executed and missing:
        reasons.append(f'{len(missing)} code cells lack execution_count')
    if errors:
        reasons.append(f'{len(errors)} error outputs present')
    if bad_images:
        reasons.append(f'{len(bad_images)} PNG outputs invalid')
    if print_count < min_print_cells:
        reasons.append(f'Only {print_count} stdout cells; require {min_print_cells}')
    if figure_cells < min_figure_cells:
        reasons.append(f'Only {figure_cells} figure cells; require {min_figure_cells}')
    return {
        'path': str(p), 'passed': not reasons, 'violations': reasons,
        'metrics': {'total_cells': len(nb.cells), 'code_cells': len(code),
                    'executed_code_cells': len(code) - len(missing),
                    'stdout_cells': print_count, 'figure_cells': figure_cells,
                    'valid_png_images': image_count, 'invalid_png_cells': bad_images,
                    'error_outputs': errors},
        'version': core._version(nb),
        'spec_sha256': nb.metadata.get(core.META_KEY, {}).get('spec_sha256'),
        'scope': 'saved-output structural checks; not scientific validation or sandboxing',
    }
