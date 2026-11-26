#!/usr/bin/env python3
"""CPU-only diagnostic and safe repair of Demo 4 confidence checkpoint fields.

Dry-run by default: print parse statistics and the raw outputs for unparsed
confidence stages. With --apply, patch ONLY previously-null confidence values
recoverable from raw text. Existing non-null confidences, all answer stages,
all raw strings, and all frozen inputs remain unchanged. Requires no GPU.

Never run while a Demo 4 collector is active.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile

from demo4_prompts import confidence_from_raw

DATASETS = ('sciq', 'sciq_bias_incorrect', 'qasper')


def inspect_file(path: Path, *, do_apply: bool, stamp: str, raw_limit: int) -> tuple[int, int, int]:
    if not path.exists():
        print(f'MISSING {path}')
        return 0, 0, 0
    rows = []
    with path.open(encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as err:
                raise RuntimeError(f'Malformed JSON in {path} at line {line_number}: {err}') from err

    total = 0
    recoverable = 0
    still_missing = 0
    existing = 0
    for row in rows:
        if row.get('stage') != 'confidence':
            continue
        total += 1
        old = row.get('confidence')
        if old is not None:
            existing += 1
            continue
        raw = row.get('raw', '')
        new = confidence_from_raw(raw)
        if new is None:
            still_missing += 1
            print(f'  UNPARSED {row.get("method")} idx={row.get("global_idx")} '
                  f'tokens={row.get("output_tokens")}: {raw[:raw_limit]!r}')
        else:
            recoverable += 1
            print(f'  RECOVERABLE {row.get("method")} idx={row.get("global_idx")} '
                  f'score={new}: {raw[:raw_limit]!r}')
            row['confidence'] = new

    print(f'{path.name}: {total} confidence stages; '
          f'{existing} already parsed; {recoverable} recoverable on CPU; '
          f'{still_missing} still missing')

    if do_apply and recoverable:
        backup = path.with_name(path.name + '.backup-confidence-' + stamp)
        if backup.exists():
            raise FileExistsError(f'Backup already exists: {backup}')
        shutil.copy2(path, backup)
        tempname = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                             dir=path.parent, prefix='.' + path.name + '.',
                                             suffix='.tmp', delete=False) as fh:
                tempname = fh.name
                for row in rows:
                    fh.write(json.dumps(row, ensure_ascii=False,
                                        separators=(',', ':')) + '\n')
                fh.flush()
                os.fsync(fh.fileno())
            os.chmod(tempname, path.stat().st_mode & 0o777)
            os.replace(tempname, path)
        finally:
            if tempname and os.path.exists(tempname):
                os.unlink(tempname)
        print(f'  APPLIED: original backed up at {backup.name}')

    return total, recoverable, still_missing


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parent,
                    help='Root of the 447_demo2 project')
    ap.add_argument('--model', default='qwen_4b',
                    choices=('qwen_1p7b', 'qwen_4b', 'llama_1b', 'llama_3b'))
    ap.add_argument('--dataset', choices=('all', *DATASETS), default='all')
    ap.add_argument('--apply', action='store_true',
                    help='Apply recoverable fixes, with backups; default is report-only')
    ap.add_argument('--raw-limit', type=int, default=600)
    args = ap.parse_args()
    out_dir = args.root / 'results' / 'demo4' / 'responses'
    if not out_dir.is_dir():
        ap.error(f'No checkpoint directory: {out_dir}')
    lock_path = out_dir / f'.{args.model}.lock'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as ex:
            raise SystemExit(f'Another process is collecting {args.model}; stop before inspecting/repairing') from ex
        totals = [0, 0, 0]
        datasets = DATASETS if args.dataset == 'all' else (args.dataset,)
        for ds in datasets:
            path = out_dir / f'{args.model}_{ds}.jsonl'
            counts = inspect_file(path, do_apply=args.apply, stamp=stamp,
                                  raw_limit=args.raw_limit)
            totals = [a+b for a, b in zip(totals, counts)]
    print(f'TOTAL: {totals[0]} confidence stages; '
          f'{totals[1]} recoverable; {totals[2]} still missing after reparse')
    if not args.apply:
        print('DRY RUN. No files changed. To apply recoverable fixes, rerun with --apply.')


if __name__ == '__main__':
    main()
