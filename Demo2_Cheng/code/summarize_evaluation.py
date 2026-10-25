#!/usr/bin/env python3
"""Merge evaluation shards and summarize exact accuracy, token F1, self-judge.

Run only after all eight model/dataset evaluation job arrays complete:
    python summarize_evaluation.py --results-dir results

Outputs in results/evaluation/:
  merged/{model}_{dataset}_evaluated.csv  # one row per answer sample
  summary/overall_summary.csv             # baseline x dataset x model
  summary/per_seed_summary.csv            # seed-specific performance
  summary/first_sample_summary.csv        # first-seed only
  summary/evaluation_summary.xlsx         # three summary sheets + definitions
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from copy import copy

BASELINES = ['vanilla', 'structured', 'p_true', 'initial_test']
EXPECTED = {
    ('llama', '1b', 'sciq'): 1000,
    ('llama', '3b', 'sciq'): 1000,
    ('qwen', '1p7b', 'sciq'): 1000,
    ('qwen', '4b', 'sciq'): 1000,
    ('llama', '1b', 'qasper'): 1005,
    ('llama', '3b', 'qasper'): 1005,
    ('qwen', '1p7b', 'qasper'): 1005,
    ('qwen', '4b', 'qasper'): 1005,
}
RX = re.compile(r'^(llama|qwen)_(1b|3b|1p7b|4b)_(sciq|qasper)_shard(\d+)of(\d+)\.csv$')
KEYS = ['dataset', 'model', 'global_question_idx', 'baseline', 'sample_index']


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir', type=Path, default=Path(__file__).resolve().parent / 'results')
    p.add_argument('--allow-incomplete', action='store_true',
                   help='Allow partial runs, e.g. for testing the pipeline')
    return p.parse_args()


def summaries(frame):
    """Aggregate over evaluable reference responses; unknown judge verdicts excluded."""
    f = frame.copy()
    f['exact_match'] = pd.to_numeric(f['exact_match'], errors='coerce')
    f['token_f1'] = pd.to_numeric(f['token_f1'], errors='coerce')
    f['judge_confidence'] = pd.to_numeric(f['judge_confidence'], errors='coerce')
    f['evaluable'] = pd.to_numeric(f['is_evaluable'], errors='coerce').eq(1)
    f['verdict'] = f['judge_verdict'].fillna('').astype(str).str.upper()
    f['valid_verdict'] = (
        f['verdict'].isin(['TRUE', 'FALSE'])
        & f['judge_status'].fillna('').astype(str).eq('ok')
    )
    f['judge_true'] = f['verdict'].eq('TRUE').where(f['valid_verdict'])
    f['judge_matches_exact'] = (
        f['verdict'].eq('TRUE') == f['exact_match'].eq(1)
    ).where(f['valid_verdict'] & f['evaluable'])
    f['sample_seed'] = pd.to_numeric(f['sample_seed'], errors='raise').astype(int)
    f['sample_index'] = pd.to_numeric(f['sample_index'], errors='raise').astype(int)
    return f


def summarize_group(group):
    ev = group[group['evaluable']]
    judged = group[group['valid_verdict']]
    compare = group[group['valid_verdict'] & group['evaluable']]
    seed_count = int(group['sample_seed'].nunique())
    return pd.Series({
        'questions': int(group['global_question_idx'].nunique()),
        'answer_samples': len(group),
        'distinct_sample_seeds': seed_count,
        'evaluable_answer_samples': len(ev),
        'accuracy_exact_match': ev['exact_match'].mean(),
        'token_f1': ev['token_f1'].mean(),
        'judge_valid_count': len(judged),
        'judge_parse_rate': len(judged) / len(group) if len(group) else float('nan'),
        'judge_true_rate': judged['judge_true'].mean(),
        'judge_false_rate': 1 - judged['judge_true'].mean() if len(judged) else float('nan'),
        'mean_judge_confidence_1to10': judged['judge_confidence'].mean(),
        'judge_agreement_with_exact_match': compare['judge_matches_exact'].mean(),
    })


def aggregate(frame, columns):
    """Version-independent pandas groupby aggregation (keeps grouping columns)."""
    output = []
    for group_key, subframe in frame.groupby(columns, sort=True, dropna=False):
        keys = group_key if isinstance(group_key, tuple) else (group_key,)
        output.append({**dict(zip(columns, keys)), **summarize_group(subframe).to_dict()})
    return pd.DataFrame(output)


def main():
    args = parse_args()
    base = args.results_dir / 'evaluation'
    metrics_dir = base / 'metrics'
    judge_dir = base / 'judgments'
    detail_dir = base / 'merged'
    summary_dir = base / 'summary'
    detail_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    files = {}
    for file in sorted(metrics_dir.glob('*.csv')):
        match = RX.fullmatch(file.name)
        if match:
            family, variant, dataset, shard, total = match.groups()
            files.setdefault((family, variant, dataset), {})[int(shard)] = (file, int(total))
    if not files:
        raise SystemExit(f'No metrics CSVs in {metrics_dir}. Run evaluate_demo2.py first.')
    missing_groups = sorted(set(EXPECTED) - set(files))
    if missing_groups and not args.allow_incomplete:
        raise RuntimeError(f'Missing evaluation configurations: {missing_groups}')

    combined_frames = []
    for group, shards in sorted(files.items()):
        if group not in EXPECTED:
            raise RuntimeError(f'Unexpected model/dataset combination: {group}')
        if any(count != 4 for _, count in shards.values()):
            raise RuntimeError(f'{group}: expected 4 shards')
        if not args.allow_incomplete and set(shards) != {0, 1, 2, 3}:
            raise RuntimeError(f'{group}: missing metric shards {set(range(4)) - set(shards)}')
        score_dfs = []
        judge_dfs = []
        for shard, (metric_file, _) in sorted(shards.items()):
            judge_file = judge_dir / metric_file.name
            if not judge_file.exists() and not args.allow_incomplete:
                raise RuntimeError(f'Missing judge CSV: {judge_file}')
            score_dfs.append(pd.read_csv(metric_file, keep_default_na=False))
            if judge_file.exists():
                judge_dfs.append(pd.read_csv(judge_file, keep_default_na=False))
        score = pd.concat(score_dfs, ignore_index=True)
        if score.duplicated(KEYS).any():
            raise RuntimeError(f'{group}: duplicate metric (question,baseline,slot) rows')
        if judge_dfs:
            judge = pd.concat(judge_dfs, ignore_index=True)
            if judge.duplicated(KEYS).any():
                raise RuntimeError(f'{group}: duplicate judgment rows')
            judge = judge[KEYS + ['judge_model', 'judge_verdict', 'judge_confidence', 'judge_status', 'judge_raw']]
            score = score.merge(judge, on=KEYS, how='left', validate='one_to_one')
        else:
            for name in ('judge_model', 'judge_verdict', 'judge_confidence', 'judge_status', 'judge_raw'):
                score[name] = ''
        expected_answers = EXPECTED[group] * len(BASELINES) * 5
        if not args.allow_incomplete and len(score) != expected_answers:
            raise RuntimeError(f'{group}: expected {expected_answers} answer rows, found {len(score)}')
        if not args.allow_incomplete and score['judge_status'].isna().any():
            raise RuntimeError(f'{group}: missing judgments; resubmit unfinished shards')
        if not args.allow_incomplete and (score['judge_status'] == '').any():
            raise RuntimeError(f'{group}: blank judgment statuses')
        score = score.sort_values(['global_question_idx', 'baseline', 'sample_index'])
        label = '_'.join(group)
        score.to_csv(detail_dir / f'{label}_evaluated.csv', index=False)
        combined_frames.append(score)
        print(f'{label}: {len(score)} answer evaluations', flush=True)

    all_rows = summaries(pd.concat(combined_frames, ignore_index=True))
    grouping = ['model', 'dataset', 'baseline']
    overall = aggregate(all_rows, grouping)
    by_seed = aggregate(all_rows, grouping + ['sample_seed'])
    first = all_rows.loc[all_rows['sample_index'] == 1]
    first_summary = aggregate(first, grouping)

    sd = by_seed.groupby(grouping)[['accuracy_exact_match', 'token_f1']].std(ddof=1).reset_index()
    sd = sd.rename(columns={
        'accuracy_exact_match': 'sample_seed_sd_accuracy',
        'token_f1': 'sample_seed_sd_f1',
    })
    overall = overall.merge(sd, on=grouping, how='left', validate='one_to_one')

    overall.to_csv(summary_dir / 'overall_summary.csv', index=False)
    by_seed.to_csv(summary_dir / 'per_seed_summary.csv', index=False)
    first_summary.to_csv(summary_dir / 'first_sample_summary.csv', index=False)

    methodology = pd.DataFrame([
        ['Accuracy', 'SQuAD-style normalized exact answer match against ANY available reference; range 0-1'],
        ['Token F1', 'Word-overlap precision/recall harmonic mean, BEST across multiple references; range 0-1'],
        ['QASPER note', 'This is answer token F1 only (not the complete official QASPER evaluation or evidence F1)'],
        ['Missing gold', 'Missing references excluded from reference metrics; coverage reported separately'],
        ['LLM self-judge', 'Same model as answering model; supplied context/question/answer; no gold references'],
        ['Judge TRUE rate', 'Proportion of valid judge verdicts labeled TRUE; NOT an independent accuracy estimate'],
        ['Judge confidence', 'One 1-10 confidence score per judgment; missing scores reduce valid parse rate'],
        ['Agreement', 'Judge TRUE/FALSE decision versus normalized exact-match correctness proxy'],
        ['Sample seeds', '42, 123, 2026, 3407, 5678: one answer per seed for each baseline and question'],
        ['Support', 'Provided to generation and reconstructed for judging; NEVER written to CSV or XLSX'],
        ['P(True)', 'All five brainstorm answers scored individually here; original P(True) probability is only for first answer'],
    ], columns=['Metric', 'Definition / caveat'])
    with pd.ExcelWriter(summary_dir / 'evaluation_summary.xlsx', engine='openpyxl') as writer:
        overall.to_excel(writer, sheet_name='Overall', index=False)
        by_seed.to_excel(writer, sheet_name='Per_Seed', index=False)
        first_summary.to_excel(writer, sheet_name='First_Sample', index=False)
        methodology.to_excel(writer, sheet_name='Definitions', index=False)
        for sheet in writer.sheets.values():
            sheet.freeze_panes = 'D2'
            sheet.auto_filter.ref = sheet.dimensions
            for cell in sheet[1]:
                new_font = copy(cell.font)
                new_font.bold = True
                cell.font = new_font
            for col in sheet.columns:
                column = col[0].column_letter
                header = str(col[0].value or '')
                sheet.column_dimensions[column].width = min(46, max(15, len(header) + 3))
    print(f'Wrote summaries and workbook to {summary_dir}')


if __name__ == '__main__':
    main()
