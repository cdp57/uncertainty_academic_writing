#!/usr/bin/env python3
"""CPU-only, safe deletion of selected Demo 4 confidence stages for regeneration.

By default, DRY RUN. Requires --apply for edits; creates timestamped backups.
Never modify checkpoints while a Demo 4 Slurm job for that model is running.
Retains every answer stage. Use --first-n 2 to restrict to the pilot questions.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile
from datetime import datetime, timezone

from demo4_collect import read_checkpoint, response_file, MODEL_METHODS

DATASETS = ('sciq', 'sciq_bias_incorrect', 'qasper')


def run(model, datasets, first_n, mode, apply, root):
    results = root / 'results' / 'demo4' / 'responses'
    if not results.is_dir():
        raise FileNotFoundError(results)
    lock_path = results / f'.{model}.lock'
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as ex:
            raise RuntimeError(f'Demo 4 {model} collector is running; stop it first') from ex
        total = 0
        for ds in datasets:
            path = results / f'{model}_{ds}.jsonl'
            if not path.exists():
                print(f'MISSING {path.name}; skipped')
                continue
            # Full validation, not just line-by-line JSON parsing.
            read_checkpoint(path, model, ds, repair=False)
            all_rows = []
            with path.open(encoding='utf-8') as f:
                for line in f:
                    all_rows.append(json.loads(line))
            removed = []
            kept = []
            for row in all_rows:
                target = (row['stage'] == 'confidence' and
                          row['global_idx'] < first_n and
                          row['method'] in MODEL_METHODS[model] and
                          (mode == 'all' or row.get('confidence') is None))
                if target:
                    removed.append(row)
                else:
                    kept.append(row)
            print(f'{path.name}: keep {len(kept)} stages; reset {len(removed)} confidence stages')
            for row in removed:
                print(f'  RESET {row["method"]} idx={row["global_idx"]} '
                      f'previous_score={row.get("confidence")!r}')
            total += len(removed)
            if apply and removed:
                stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                backup = path.with_name(path.name + '.backup-before-prefill-' + stamp)
                shutil.copy2(path, backup)
                tmp = None
                try:
                    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                          dir=path.parent, prefix='.'+path.name+'.', suffix='.tmp',
                          delete=False) as fh:
                        tmp = fh.name
                        for row in kept:
                            fh.write(json.dumps(row, ensure_ascii=False,
                                                separators=(',', ':'))+'\n')
                        fh.flush()
                        os.fsync(fh.fileno())
                    os.chmod(tmp, path.stat().st_mode & 0o777)
                    os.replace(tmp, path)
                finally:
                    if tmp and os.path.exists(tmp):
                        os.unlink(tmp)
                read_checkpoint(path, model, ds)
                print(f'  APPLIED; backup: {backup.name}')
        print(f'TOTAL: {total} confidence stages selected for reset. '
              + ('Changes applied.' if apply else 'DRY RUN, no changes.'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, choices=('qwen_1p7b', 'qwen_4b'))
    p.add_argument('--dataset', default='all', choices=('all', *DATASETS))
    p.add_argument('--first-n', type=int, default=2,
                   help='Only reset first N question indices in each dataset (default: pilot N=2)')
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--all', action='store_true', help='Reset all selected confidence stages, including parsed')
    modes.add_argument('--missing-only', action='store_true', help='Reset only confidence stages without a parsed score')
    p.add_argument('--apply', action='store_true', help='Modify checkpoints after backing them up; default dry run')
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    a = p.parse_args()
    if a.first_n < 1:
        p.error('--first-n must be >= 1')
    datasets = DATASETS if a.dataset == 'all' else (a.dataset,)
    run(a.model, datasets, a.first_n, 'all' if a.all else 'missing', a.apply, a.root)


if __name__ == '__main__':
    main()
