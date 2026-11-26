#!/usr/bin/env python3
"""CPU-only Demo 4 checkpoint export to per-model per-condition wide CSVs.

One row per question; two-stage structured-rerun and academic-system columns.
Raw outputs and parsed answers/confidences retained. No accuracy computation.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

from demo3_data import COUNTS
from demo3_prepare import DATA_DIR
from demo4_collect import MODEL_IDS, MODEL_METHODS, load_inputs, read_checkpoint, response_file
from demo4_prompts import METHODS

ROOT = Path(__file__).resolve().parent
CSV_DIR = ROOT / 'results' / 'demo4' / 'csv'
STAGE_FIELDS = ('raw', 'input_tokens', 'output_tokens', 'elapsed_seconds', 'input_truncated')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--allow-incomplete', action='store_true', help='Export partial pilot/progress')
    args = ap.parse_args()
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    coverage = []
    for model in MODEL_IDS:
        for dataset, count in COUNTS.items():
            source = load_inputs(dataset)
            cp = read_checkpoint(response_file(model, dataset), model, dataset)
            methods = MODEL_METHODS[model]
            completed = sum('confidence' in cp.get((rec['global_idx'], m), {})
                            for rec in source for m in methods)
            expected = count * len(methods)
            if completed != expected and not args.allow_incomplete:
                raise RuntimeError(f'{model}/{dataset}: {completed}/{expected} methods complete. '
                                   'Use --allow-incomplete for pilot data.')
            rows = []
            for rec in source:
                idx = rec['global_idx']
                row = {
                    'model': model, 'dataset': dataset, 'global_idx': idx,
                    'question_idx': rec.get('question_idx'),
                    'paper_idx': rec.get('paper_idx'),
                    'question': rec['question'],
                    'original_question': rec.get('original_question', ''),
                    'incorrect_cue': rec.get('incorrect_answer', ''),
                    'reference_answers_json': json.dumps(rec['reference_answers'], ensure_ascii=False),
                }
                for method in METHODS:
                    p = method + '_'
                    stage_pair = cp.get((idx, method), {})
                    a = stage_pair.get('answer', {})
                    c = stage_pair.get('confidence', {})
                    row[p+'answer'] = a.get('answer')
                    row[p+'confidence'] = c.get('confidence')
                    row[p+'raw_answer'] = a.get('raw')
                    row[p+'raw_confidence'] = c.get('raw')
                    row[p+'answer_input_tokens'] = a.get('input_tokens')
                    row[p+'answer_output_tokens'] = a.get('output_tokens')
                    row[p+'confidence_input_tokens'] = c.get('input_tokens')
                    row[p+'confidence_output_tokens'] = c.get('output_tokens')
                    row[p+'answer_elapsed_seconds'] = a.get('elapsed_seconds')
                    row[p+'confidence_elapsed_seconds'] = c.get('elapsed_seconds')
                    row[p+'input_truncated'] = bool(a.get('input_truncated') or c.get('input_truncated')) if a or c else None
                rows.append(row)
            filename = CSV_DIR / f'{model}_{dataset}.csv'
            with filename.open('w', encoding='utf-8', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]), extrasaction='ignore')
                w.writeheader()
                w.writerows(rows)
            print(f'Exported {filename}: {len(rows)} rows, {completed}/{expected} completed pairs')
            for method in methods:
                coverage.append({
                    'model': model, 'dataset': dataset, 'method': method,
                    'expected_questions': count,
                    'completed_responses': sum('confidence' in cp.get((i, method), {}) for i in range(count)),
                    'answer_stage_saved': sum('answer' in cp.get((i, method), {}) for i in range(count)),
                    'parsed_answers': sum(cp.get((i, method), {}).get('answer', {}).get('answer') is not None
                                          for i in range(count)),
                    'parsed_scores': sum(cp.get((i, method), {}).get('confidence', {}).get('confidence') is not None
                                         for i in range(count)),
                })
    with (CSV_DIR / 'collection_coverage.csv').open('w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(coverage[0]))
        w.writeheader()
        w.writerows(coverage)
    print('Coverage:', CSV_DIR / 'collection_coverage.csv')


if __name__ == '__main__':
    main()
