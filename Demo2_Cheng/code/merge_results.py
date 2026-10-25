#!/usr/bin/env python3
"""Merge the four question shards per model/dataset (one pass, five sample seeds).

Example:
  python merge_results.py --results-dir results
  python merge_results.py --results-dir results --allow-incomplete   # pilots

Previous *_seed42_shard0of4.csv files are intentionally NOT merged: they used
one seed for num_return_sequences=5 and are incompatible with the new design.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

EXPECTED_QUESTIONS = {'sciq': 1000, 'qasper': 1005}
SAMPLE_SEEDS = [42, 123, 2026, 3407, 5678]
SHARD_COUNT = 4
CONFIDENCE_COLS = {
    'vanilla': 'vanilla_confidence_json',
    'structured': 'structured_confidence_json',
    'caution': 'initial_confidence_json',
}
ARRAY_COLS = [
    'vanilla_answers_json', 'vanilla_confidence_json',
    'structured_answers_json', 'structured_confidence_json',
    'p_true_answers_json', 'initial_answers_json', 'initial_confidence_json',
]


def json_list(value, column):
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Invalid JSON in {column}: {str(value)[:120]}') from exc
    if not isinstance(parsed, list) or len(parsed) != len(SAMPLE_SEEDS):
        raise ValueError(f'{column} must contain exactly {len(SAMPLE_SEEDS)} values; got {parsed!r}')
    return parsed


def parse_confidence(value):
    if value is None or value == '':
        return None
    try:
        val = float(value)
    except (ValueError, TypeError):
        return None
    return val if 1 <= val <= 10 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=Path(__file__).resolve().parent/'results')
    parser.add_argument('--allow-incomplete', action='store_true', help='Allow pilot/partial coverage')
    args = parser.parse_args()
    files = sorted(args.results_dir.glob('*.csv'))
    pattern = re.compile(r'^(llama|qwen)_(1b|3b|1p7b|4b)_(sciq|qasper)_shard(\d+)of(\d+)\.csv$')
    legacy_pattern = re.compile(r'^(llama|qwen)_.+_seed\d+_shard\d+of\d+\.csv$')
    legacy = [p.name for p in files if legacy_pattern.match(p.name)]
    if legacy:
        print(f'Ignoring {len(legacy)} old-seed-format checkpoint(s) from the earlier experiment design.')
    groups = {}
    for file in files:
        match = pattern.match(file.name)
        if match:
            family, variant, dataset, shard, total = match.groups()
            groups.setdefault((family, variant, dataset), []).append((file, int(shard), int(total)))
    if not groups:
        raise SystemExit(f'No new-format shard CSV files found in {args.results_dir}')

    combined_dir = args.results_dir / 'merged'
    combined_dir.mkdir(parents=True, exist_ok=True)
    seed_records = []
    ptrue_records = []

    for (family, variant, dataset), group in sorted(groups.items()):
        if any(total != SHARD_COUNT for _, _, total in group):
            raise RuntimeError(f'{family} {variant} {dataset}: expected shard count {SHARD_COUNT}')
        shard_ids = [s for _, s, _ in group]
        if len(shard_ids) != len(set(shard_ids)):
            raise RuntimeError(f'{family} {variant} {dataset}: duplicate shard index')
        if not args.allow_incomplete and sorted(shard_ids) != list(range(SHARD_COUNT)):
            raise RuntimeError(f'{family} {variant} {dataset}: missing shards: {set(range(SHARD_COUNT)) - set(shard_ids)}')
        df = pd.concat([pd.read_csv(path, keep_default_na=False) for path, _, _ in group], ignore_index=True)
        if df.empty:
            raise RuntimeError(f'{family} {variant} {dataset}: no questions in shard files')
        if 'support' in df.columns:
            df = df.drop(columns=['support'])
        if 'seed' in df.columns:
            raise RuntimeError(f'{family} {variant} {dataset}: found old run-level seed column')
        if df.duplicated('global_question_idx').any():
            raise RuntimeError(f'{family} {variant} {dataset}: duplicate question IDs across shards')
        indices = set(pd.to_numeric(df['global_question_idx'], errors='raise').astype(int).tolist())
        if not args.allow_incomplete and indices != set(range(EXPECTED_QUESTIONS[dataset])):
            missing = set(range(EXPECTED_QUESTIONS[dataset])) - indices
            raise RuntimeError(f'{family} {variant} {dataset}: expected {EXPECTED_QUESTIONS[dataset]} unique questions, got {len(indices)}; missing {len(missing)}')
        # Validate the new experiment design in every saved question.
        for _, row in df.iterrows():
            if json_list(row['sample_seeds_json'], 'sample_seeds_json') != SAMPLE_SEEDS:
                raise RuntimeError('sample_seeds_json differs from the expected five seeds')
            for column in ARRAY_COLS:
                json_list(row[column], column)
        df = df.sort_values('global_question_idx').reset_index(drop=True)
        label = f'{family}_{variant}_{dataset}'
        df.to_csv(combined_dir / f'{label}_combined.csv', index=False)
        df.to_excel(combined_dir / f'{label}_combined.xlsx', index=False, sheet_name='Questions')

        for pos, seed in enumerate(SAMPLE_SEEDS):
            record = dict(model=f'{family}_{variant}', dataset=dataset, sample_seed=seed,
                          question_count=len(df))
            for key, column in CONFIDENCE_COLS.items():
                vals = [parse_confidence(json.loads(raw)[pos]) for raw in df[column]]
                observed = [value for value in vals if value is not None]
                record[f'{key}_mean_confidence_1to10'] = sum(observed)/len(observed) if observed else float('nan')
                record[f'{key}_parsed_confidence_rate'] = len(observed)/len(df)
            seed_records.append(record)
        scores = pd.to_numeric(df['p_true_probability_A'], errors='coerce')
        ptrue_records.append(dict(model=f'{family}_{variant}', dataset=dataset,
                                  question_count=len(df), ptrue_mean_prob_A=scores.mean(),
                                  ptrue_std_across_questions=scores.std(),
                                  ptrue_valid_questions=int(scores.notna().sum())))
        print(f'Merged {len(df)} unique questions: {label}')

    per_seed = pd.DataFrame(seed_records).sort_values(['model', 'dataset', 'sample_seed'])
    per_seed.to_csv(combined_dir/'per_sample_seed_summary.csv', index=False)
    per_seed.to_excel(combined_dir/'per_sample_seed_summary.xlsx', index=False)
    metric_cols = [c for c in per_seed if c.endswith('_1to10') or c.endswith('_rate')]
    across = per_seed.groupby(['model', 'dataset'])[metric_cols].agg(['mean', 'std'])
    across.columns = ['_'.join(tup) for tup in across.columns]
    across.to_csv(combined_dir/'across_sample_seed_mean_std.csv')
    across.to_excel(combined_dir/'across_sample_seed_mean_std.xlsx')
    ptrue_df = pd.DataFrame(ptrue_records).sort_values(['model','dataset'])
    ptrue_df.to_csv(combined_dir/'ptrue_question_summary.csv', index=False)
    ptrue_df.to_excel(combined_dir/'ptrue_question_summary.xlsx', index=False)
    print('Means and SD for self-report confidence are across five per-answer seeds.')
    print('P(True) is computed once per question: its SD is ACROSS QUESTIONS, not across seeds.')
    print('Accuracy/LLM-as-a-Judge labels have not yet been collected.')


if __name__ == '__main__':
    main()
