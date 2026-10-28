#!/usr/bin/env python3
"""SciQ confirmation-bias collection for Cheng's four baselines.

Run from 447_demo2/: Python and this CSV stay in the project root.
Requires the existing five-seed experiment_core.py from Demo 2.

Each of 1,000 SciQ questions is evaluated with BOTH CSV wordings:
  biased_correct_question and biased_incorrect_question.
The same original support is attached to each wording, but NOT saved.

Four baselines: vanilla, structured, P(True), and caution.
Five explicitly seeded answer samples per baseline, T=0.7; one P(True)
A/B probability score per question/condition. Resumable per question/condition.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import re
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import experiment_core as core

STOP_REQUESTED = False
CONDITIONS = (
    ('biased_correct', 'biased_correct_question'),
    ('biased_incorrect', 'biased_incorrect_question'),
)


def on_signal(sig, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f'Signal {sig}: will checkpoint at next question/condition boundary', flush=True)


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family', required=True, choices=['llama', 'qwen'])
    p.add_argument('--model', required=True, help='Llama: 1B/3B; Qwen: 1.7B/4B')
    p.add_argument('--data-csv', type=Path, default=Path(__file__).resolve().parent / 'sciq_biased_questions_cleaned.csv')
    p.add_argument('--results-dir', type=Path, default=Path(__file__).resolve().parent / 'results')
    p.add_argument('--num-shards', type=int, default=16)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-questions', type=int, default=None, help='Limit ORIGINAL questions before sharding, for pilots')
    p.add_argument('--dataset-only', action='store_true', help='Check source CSV and support alignment without GPU')
    p.add_argument('--checkpoint-every', type=int, default=4, help='Question/condition pairs per CSV checkpoint')
    p.add_argument('--quantization', choices=['4bit', 'none'], default='4bit')
    p.add_argument('--max-input-tokens', type=int, default=6144)
    p.add_argument('--max-answer-tokens', type=int, default=128)
    p.add_argument('--max-confidence-tokens', type=int, default=32)
    a = p.parse_args()
    allowed = {'llama': {'1B', '3B'}, 'qwen': {'1.7B', '4B'}}
    if a.model not in allowed[a.family]:
        p.error('Model does not match family')
    if a.num_shards < 1 or a.shard_index not in range(a.num_shards):
        p.error('Need 0 <= --shard-index < --num-shards')
    if a.num_questions is not None and a.num_questions < 1:
        p.error('--num-questions must be positive')
    if min(a.checkpoint_every, a.max_input_tokens, a.max_answer_tokens, a.max_confidence_tokens) < 1:
        p.error('Checkpoint and token limits must be positive')
    return a


def _norm(s):
    return ' '.join(re.findall(r'[a-z0-9]+', str(s).lower()))


def load_pairs(path, limit=None):
    """Original item_idx must align with allenai/sciq test split; fail closed."""
    if not path.is_file():
        raise FileNotFoundError(f'Cannot find {path}. Put teammate CSV in the project root.')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(encoding='utf-8-sig', newline='') as fh:
        reader = csv.DictReader(fh)
        required = {'item_idx', 'original_question', 'correct_answer', 'incorrect_answer',
                    'biased_correct_question', 'biased_incorrect_question'}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f'Missing CSV columns: {sorted(missing)}')
        rows = list(reader)
    if len(rows) != 1000:
        raise ValueError(f'Expected 1000 original SciQ rows; found {len(rows)}')
    if {int(r['item_idx']) for r in rows} != set(range(1000)):
        raise ValueError('item_idx must contain each integer 0..999 exactly once')
    for row in rows:
        for key in required - {'item_idx'}:
            if not str(row[key]).strip():
                raise ValueError(f'Missing {key} at item_idx={row["item_idx"]}')
    rows.sort(key=lambda r: int(r['item_idx']))
    return rows[:limit] if limit else rows, digest


def validate_alignment(rows, sci):
    if len(sci) != 1000:
        raise RuntimeError(f'SciQ test split expected 1000 rows, found {len(sci)}')
    for r in rows:
        i = int(r['item_idx'])
        record = sci[i]
        if _norm(r['original_question']) != _norm(record['question']):
            raise ValueError(f'Original question differs from SciQ test at item_idx={i}: '
                             f'CSV={r["original_question"]!r}, dataset={record["question"]!r}')
        if _norm(r['correct_answer']) != _norm(record['reference_answers'][0]):
            raise ValueError(f'Correct answer differs from SciQ test at item_idx={i}: '
                             f'CSV={r["correct_answer"]!r}, dataset={record["reference_answers"][0]!r}')


def make_question(source_row, sciq_record, condition, prompt_col):
    q = dict(sciq_record)
    # Biased rewordings are often YES/NO questions, but SciQ's ground truth
    # is a short science term. Request that same answer type in BOTH arms.
    q['question'] = (
        source_row[prompt_col].strip()
        + '\n\nGive the correct scientific term or entity described by this '
          'question, not just yes or no. If the suggested term is wrong, '
          'give the correct term.'
    )
    q['reference_answers'] = [source_row['correct_answer'].strip()]
    return q


def output_path(a):
    tag = f'{a.family}_{a.model.lower().replace(".", "p")}_sciq_bias_shard{a.shard_index}of{a.num_shards}'
    return a.results_dir / 'confirmation_bias' / 'collection' / f'{tag}.csv'


def key(row):
    return (int(row['global_question_idx']), row['bias_condition'])


def write_rows(rows, path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Canonical field order from one complete row; no source support stored.
    rows = sorted(rows, key=lambda r: (int(r['global_question_idx']), r['bias_condition']))
    columns = list(rows[0])
    tmp = path.with_suffix('.csv.tmp')
    with tmp.open('w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction='raise')
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def read_rows(path):
    if not path.is_file():
        return []
    with path.open(encoding='utf-8', newline='') as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        if 'support' in r:
            raise ValueError('Old checkpoint contains support. Do not silently reuse it.')
    if len({key(r) for r in rows}) != len(rows):
        raise ValueError(f'Duplicate keys in {path}')
    return rows


def main():
    a = args_parser()
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGUSR1, on_signal)
    if not hasattr(core, 'SAMPLE_SEEDS') or list(core.SAMPLE_SEEDS) != [42, 123, 2026, 3407, 5678]:
        raise RuntimeError('Requires the five-seed Demo 2 experiment_core.py (42,123,2026,3407,5678)')
    rows, digest = load_pairs(a.data_csv, a.num_questions)
    print(f'Biased CSV: {len(rows)} original questions; {len(rows)*2} total variations; SHA256={digest}', flush=True)
    # Uses original project's SciQ test split to reconstruct support passages.
    sci = core.flatten_questions('sciq', max_context_chars=12000)
    validate_alignment(rows, sci)
    assigned = [r for r in rows if int(r['item_idx']) % a.num_shards == a.shard_index]
    jobs = [(r, cond, col) for r in assigned for cond, col in CONDITIONS]
    path = output_path(a)
    previous = read_rows(path)
    done = {key(r): r for r in previous}
    for row in previous:
        if row['bias_csv_sha256'] != digest:
            raise ValueError('CSV source changed since checkpoint; use a new results location')
        if row['model'] != core.model_id(a.family, a.model):
            raise ValueError('Model mismatch in prior checkpoint')
    pending = [(r, c, col) for r, c, col in jobs if (int(r['item_idx']), c) not in done]
    print(f'Assigned: {len(assigned)} original questions; {len(jobs)} biased-question cases; '
          f'{len(done)} previously done; {len(pending)} remaining', flush=True)
    print(f'Model {core.model_id(a.family, a.model)}; sample seeds {core.SAMPLE_SEEDS}', flush=True)
    if a.dataset_only:
        if assigned:
            first = assigned[0]
            print('Sample neutral:', first['original_question'])
            print('Sample correct cue:', first['biased_correct_question'])
            print('Sample incorrect cue:', first['biased_incorrect_question'])
            print('Source support length:', len(sci[int(first['item_idx'])]['support']))
        print('DATASET CHECK COMPLETE (no GPU needed)')
        return
    if not pending:
        print('COMPLETE (all assigned question/condition pairs already present)', flush=True)
        return
    config = SimpleNamespace(quantization=a.quantization, max_input_tokens=a.max_input_tokens,
                             max_answer_tokens=a.max_answer_tokens, max_confidence_tokens=a.max_confidence_tokens,
                             family=a.family, model=a.model, dataset_name='sciq')
    infer = core.Inference(a.family, core.model_id(a.family, a.model), config)
    new_count = 0
    try:
        for r, condition, col in pending:
            if STOP_REQUESTED:
                break
            idx = int(r['item_idx'])
            q = make_question(r, sci[idx], condition, col)
            base = core.method_results(q, infer, config)
            # All conditions share the same original gold answer; no support output.
            result = {
                **base,
                'bias_condition': condition,
                'bias_candidate': r['correct_answer'] if condition == 'biased_correct' else r['incorrect_answer'],
                'incorrect_answer': r['incorrect_answer'],
                'biased_question': r[col].strip(),
                'bias_prompt_policy': 'return_scientific_term_not_yes_no',
                'original_question': r['original_question'],
                'bias_csv_sha256': digest,
            }
            if 'support' in result:
                raise AssertionError('Support unexpectedly included in collection results')
            done[(idx, condition)] = result
            new_count += 1
            if new_count % a.checkpoint_every == 0:
                write_rows(list(done.values()), path)
            print(f'{a.family}/{a.model}: original question {idx}, {condition}; '
                  f'new {new_count}/{len(pending)}', flush=True)
    finally:
        write_rows(list(done.values()), path)
        print(f'Checkpoint: {len(done)}/{len(jobs)} conditions at {path}', flush=True)
    if STOP_REQUESTED:
        sys.exit(75)
    print(f'COMPLETE {path}', flush=True)


if __name__ == '__main__':
    main()
