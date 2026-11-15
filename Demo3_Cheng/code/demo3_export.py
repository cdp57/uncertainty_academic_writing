#!/usr/bin/env python3
"""CPU-only: export lossless Demo 3 JSONL checkpoints into wide CSVs.

One row per question and model per condition, five methods. No model load,
GPU, retries or LLM-as-judge. Parsing failures remain visible as empty cells.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

from demo3_collect import MODEL_IDS, response_file, read_checkpoint
from demo3_data import COUNTS
from demo3_prepare import DATA_DIR, validate_frozen
from demo3_prompts import METHODS

ROOT = Path(__file__).resolve().parent
EXPORT_DIR = ROOT / 'results' / 'demo3' / 'csv'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--allow-incomplete', action='store_true', help='Allow pilots/incomplete models')
    args = ap.parse_args()
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for ds, count in COUNTS.items():
        validate_frozen(ds)
        with (DATA_DIR / f'{ds}.jsonl').open(encoding='utf-8') as f:
            inputs = {r['global_idx']: r for r in map(json.loads, f)}
        for model in MODEL_IDS:
            file = response_file(model, ds)
            if not file.is_file():
                if args.allow_incomplete:
                    print(f'Skip missing {file}')
                    continue
                raise FileNotFoundError(file)
            keys = read_checkpoint(file, model, ds)
            expected = count * len(METHODS)
            if len(keys) != expected and not args.allow_incomplete:
                raise RuntimeError(f'{model}/{ds}: {len(keys)}/{expected} responses. '
                                   'Use --allow-incomplete for pilot data.')
            rows = {}
            parse_counts = {m: {'answer': 0, 'score': 0} for m in METHODS}
            with file.open(encoding='utf-8') as f:
                for item in map(json.loads, f):
                    idx = item['global_idx']
                    src = inputs[idx]
                    if idx not in rows:
                        rows[idx] = {
                            'model': model, 'dataset': ds,
                            'global_idx': idx,
                            'paper_idx': src.get('paper_idx'),
                            'question_idx': src.get('question_idx'),
                            'question': src['question'],
                            'original_question': src.get('original_question', ''),
                            'incorrect_cue': src.get('incorrect_answer', ''),
                            'reference_answers_json': json.dumps(src['reference_answers'], ensure_ascii=False),
                        }
                    method = item['method']
                    for fld in ('answer', 'confidence', 'bet_usd', 'raw', 'input_tokens',
                                'output_tokens', 'elapsed_seconds', 'input_truncated',
                                'panel_json_valid'):
                        rows[idx][f'{method}_{fld}'] = item.get(fld)
                    parse_counts[method]['answer'] += item['answer'] is not None
                    parse_counts[method]['score'] += (
                        item['bet_usd'] if method == 'money_betting' else item['confidence']) is not None
            path = EXPORT_DIR / f'{model}_{ds}.csv'
            columns = ['model', 'dataset', 'global_idx', 'paper_idx', 'question_idx',
                       'question', 'original_question', 'incorrect_cue', 'reference_answers_json']
            columns += [f'{m}_{k}' for m in METHODS for k in (
                'answer', 'confidence', 'bet_usd', 'raw', 'input_tokens',
                'output_tokens', 'elapsed_seconds', 'input_truncated', 'panel_json_valid')]
            with path.open('w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows[idx] for idx in sorted(rows))
            for method in METHODS:
                summary.append({
                    'model': model, 'dataset': ds, 'method': method,
                    'expected_questions': count,
                    'completed_responses': sum((i, method) in keys for i in range(count)),
                    'parsed_answers': parse_counts[method]['answer'],
                    'parsed_scores': parse_counts[method]['score'],
                })
            print(f'Exported {path}: {len(rows)} questions, {len(keys)}/{expected} methods')
    if summary:
        out = EXPORT_DIR / 'collection_coverage.csv'
        with out.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        print('Coverage report:', out)


if __name__ == '__main__':
    main()
