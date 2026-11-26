#!/usr/bin/env python3
"""CPU-only Demo 4 answer parser repair and selective confidence-stage reset.

Restores answers from raw Qwen XML-like <answer> outputs. When this changes the
proposed answer the original collector passed to the confidence stage, remove
that confidence checkpoint so a resumed job regenerates ONLY that stage.
Never regenerates an answer and never changes its raw text.

Dry-run by default. --apply makes a timestamped backup and updates checkpoints
atomically. NEVER run while a Demo 4 collector is running.
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

from demo4_prompts import parse_answer


def original_proposed_answer(parsed_answer, raw_answer):
    """Exact old fallback logic, for deciding whether stage 2 saw a dirty input."""
    if parsed_answer:
        return parsed_answer
    raw = (raw_answer or '').strip()
    return raw[:1500] if raw else 'Unparseable answer'


def process(path: Path, apply: bool, stamp: str) -> tuple[int, int, int]:
    if not path.exists():
        print(f'Skip missing {path}')
        return 0, 0, 0
    records = []
    with path.open(encoding='utf-8') as fh:
        for n, ln in enumerate(fh, 1):
            if not ln.strip():
                continue
            try:
                records.append(json.loads(ln))
            except json.JSONDecodeError as e:
                raise RuntimeError(f'Malformed JSON in {path} line {n}; repair torn checkpoints before running this utility') from e

    changed = set()
    answer_updates = 0
    for row in records:
        if row.get('stage') != 'answer':
            continue
        old = row.get('answer')
        clean = parse_answer(row.get('raw'))
        if clean is None or old == clean:
            continue
        old_proposed = original_proposed_answer(old, row.get('raw'))
        if old_proposed != clean:
            changed.add((row['global_idx'], row['method']))
        row['answer'] = clean
        answer_updates += 1

    invalidated = sum(1 for row in records if row.get('stage') == 'confidence'
                      and (row.get('global_idx'), row.get('method')) in changed)
    if invalidated:
        records = [row for row in records if not (
            row.get('stage') == 'confidence'
            and (row.get('global_idx'), row.get('method')) in changed
        )]
    print(f'{path.name}: {answer_updates} improved answers; '
          f'{invalidated} confidence stages to regenerate; '
          f'{len(records)} retained rows')
    if not apply or not (answer_updates or invalidated):
        return answer_updates, invalidated, len(records)

    backup = path.with_name(path.name + '.backup-' + stamp)
    if backup.exists():
        raise FileExistsError(f'Backup already exists: {backup}')
    shutil.copy2(path, backup)
    tempname = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                         dir=path.parent, prefix='.' + path.name + '.',
                                         suffix='.tmp', delete=False) as tmp:
            tempname = tmp.name
            for row in records:
                tmp.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
            tmp.flush()
            os.fsync(tmp.fileno())
        os.chmod(tempname, path.stat().st_mode & 0o777)
        os.replace(tempname, path)
    finally:
        if tempname and os.path.exists(tempname):
            os.unlink(tempname)
    print(f'  Applied. Original checkpoint backed up at {backup}')
    return answer_updates, invalidated, len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['qwen_1p7b', 'qwen_4b'],
                        default='qwen_1p7b', help='Model to repair; default: pilot Qwen 1.7B')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent,
                        help='Project root (default: location of script)')
    parser.add_argument('--apply', action='store_true',
                        help='Apply changes after creating a backup; default is dry-run')
    args = parser.parse_args()
    response_dir = args.root / 'results' / 'demo4' / 'responses'
    if not response_dir.is_dir():
        parser.error(f'Response directory not found: {response_dir}')

    lock_path = response_dir / f'.{args.model}.lock'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError(f'Collector currently running for {args.model}; do not repair live checkpoints') from e
        sum_answers = sum_conf = 0
        for ds in ('sciq', 'sciq_bias_incorrect', 'qasper'):
            a, c, n = process(response_dir / f'{args.model}_{ds}.jsonl', args.apply, stamp)
            sum_answers += a
            sum_conf += c
    print(f'\nTOTAL: {sum_answers} repaired answer fields; '
          f'{sum_conf} affected confidence stages to regenerate.')
    if not args.apply:
        print('DRY RUN ONLY. Re-run with --apply to update backed-up checkpoints.')
    elif sum_conf:
        print('Now re-submit the two-question pilot; it will generate only missing confidence stages.')


if __name__ == '__main__':
    main()
