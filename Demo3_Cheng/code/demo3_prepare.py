#!/usr/bin/env python3
"""CPU-only, one-time dataset preparation for Demo 3.

Run on a login/CPU node, never under a GPU allocation. Freeze 3005 input
records (SciQ test 1000, SciQ incorrect cue 1000, QASPER val 1005).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path

from demo3_data import COUNTS, load_records

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'results' / 'demo3' / 'inputs'


def record_file(dataset):
    return DATA_DIR / f'{dataset}.jsonl'


def file_sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_frozen(dataset):
    path = record_file(dataset)
    if not path.is_file():
        raise FileNotFoundError(path)
    indices = set()
    with path.open(encoding='utf-8') as fh:
        for line_no, line in enumerate(fh, 1):
            r = json.loads(line)
            if r.get('dataset') != dataset or not isinstance(r.get('global_idx'), int):
                raise ValueError(f'{path}:{line_no}: bad dataset or global_idx')
            if r['global_idx'] in indices:
                raise ValueError(f'{path}:{line_no}: duplicate index {r["global_idx"]}')
            if not r.get('question') or not isinstance(r.get('support'), str):
                raise ValueError(f'{path}:{line_no}: missing question/support')
            indices.add(r['global_idx'])
    if indices != set(range(COUNTS[dataset])):
        raise ValueError(f'{path}: expected indices 0..{COUNTS[dataset]-1}, found {len(indices)}')
    return file_sha256(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--validate-only', action='store_true',
                    help='Check frozen files without loading datasets or network access')
    args = ap.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = DATA_DIR / 'manifest.json'
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest = {}

    if args.validate_only and not all(record_file(d).is_file() for d in COUNTS):
        ap.error('Frozen input files missing. Run python demo3_prepare.py first.')

    for dataset in COUNTS:
        path = record_file(dataset)
        if path.exists():
            digest = validate_frozen(dataset)
            if old.get(dataset, {}).get('sha256') not in (None, digest):
                raise RuntimeError(f'Frozen {dataset} input changed since manifest was written')
            print(f'Reusing {dataset:20} {COUNTS[dataset]} questions SHA256={digest}', flush=True)
            manifest[dataset] = old.get(dataset) or {'count': COUNTS[dataset], 'sha256': digest}
            continue
        if args.validate_only:
            continue
        print(f'Preparing {dataset}: downloading/loading CPU dataset data...', flush=True)
        records, source_digest = load_records(dataset, ROOT)
        if len(records) != COUNTS[dataset]:
            raise ValueError(f'{dataset}: wrong record count {len(records)}')
        temp = path.with_suffix('.jsonl.tmp')
        with temp.open('w', encoding='utf-8') as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n')
        os.replace(temp, path)
        digest = validate_frozen(dataset)
        manifest[dataset] = {
            'count': len(records), 'sha256': digest,
            'source': source_digest, 'max_context_chars': 12000,
        }
        print(f'Created {path}, SHA256={digest}', flush=True)

    if not args.validate_only:
        if old and any(old.get(k, {}).get('sha256') not in (None, manifest[k]['sha256'])
                       for k in COUNTS):
            raise RuntimeError('Dataset fingerprint mismatch: refusing to change frozen inputs')
        tmp = manifest_path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        os.replace(tmp, manifest_path)
    print('Dataset preparation/validation complete (0 GPU required).')


if __name__ == '__main__':
    main()
