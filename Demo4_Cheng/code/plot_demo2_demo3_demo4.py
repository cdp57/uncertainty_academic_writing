#!/usr/bin/env python3
"""Demo 2 vs Demo 3 vs Demo 4 matched-question study plots. CPU only; no error bars.

Usage (from 447_demo2/):
    python plot_demo2_demo3_demo4.py --results-dir results

Required existing files:
    results/merged/{llama_1b,llama_3b,qwen_1p7b,qwen_4b}_{sciq,qasper}_combined.csv
    results/confirmation_bias/merged/{model}_sciq_bias_combined.csv
    results/demo3/csv/{model}_{sciq,sciq_bias_incorrect,qasper}.csv
    results/demo4/csv/{model}_{sciq,sciq_bias_incorrect,qasper}.csv

Demo 2: first item in the five-answer JSON arrays (seed 42, when recorded).
Demo 3: one response per method. Demo 4: one answer/confidence pair per method.
All metrics are computed on matched question IDs. Demo 4 structured rerun exists only for Qwen models.
Exact-match and token F1 use the Demo 2 evaluation normalization convention.
Missing answers score zero; rows without gold answers do not enter accuracy metrics.
Missing scores are omitted only from score-based analyses.
Only bar charts and box plots. Default charts focus on study metrics; --extended adds per-model views.
"""
from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_KEYS = ('llama_1b', 'llama_3b', 'qwen_1p7b', 'qwen_4b')
MODEL_LABELS = {
    'llama_1b': 'Llama 1B', 'llama_3b': 'Llama 3B',
    'qwen_1p7b': 'Qwen 1.7B', 'qwen_4b': 'Qwen 4B',
}
DATASETS = ('sciq', 'sciq_bias_incorrect', 'qasper')
DATA_LABELS = {
    'sciq': 'SciQ',
    'sciq_bias_incorrect': 'SciQ: Incorrect Cue',
    'qasper': 'QASPER',
}
D2 = ('vanilla', 'structured', 'p_true', 'caution')
D3 = ('speculative_editor', 'opposing_counsel', 'compliance_officer',
      'persona_panel', 'money_betting')
METHODS = D2 + D3
METHOD_LABELS = {
    'vanilla': 'Vanilla', 'structured': 'Structured', 'p_true': 'P(True)',
    'caution': 'Caution', 'speculative_editor': 'Speculative Editor',
    'opposing_counsel': 'Opposing Counsel', 'compliance_officer': 'Compliance Officer',
    'persona_panel': 'Persona Panel', 'money_betting': 'Money Betting',
}
METHOD_COLORS = dict(zip(METHODS, plt.get_cmap('tab10').colors[:len(METHODS)]))
D2_ANSWER_COLS = {
    'vanilla': 'vanilla_answers_json', 'structured': 'structured_answers_json',
    'p_true': 'p_true_answers_json', 'caution': 'initial_answers_json',
}
D2_CONF_COLS = {
    'vanilla': 'vanilla_confidence_json', 'structured': 'structured_confidence_json',
    'caution': 'initial_confidence_json',
}
EXPECTED = {'sciq': 1000, 'sciq_bias_incorrect': 1000, 'qasper': 1005}


def nonempty(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    value = str(value).strip()
    return value if value else None


def first_json_item(value, col, file):
    try:
        values = json.loads(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f'{file}: invalid JSON array in {col}: {str(value)[:100]!r}') from e
    if not isinstance(values, list) or len(values) < 1:
        raise ValueError(f'{file}: {col} must be a nonempty JSON array')
    return values[0]


def score_float(value, lower, upper):
    try:
        num = float(value)
        return num if np.isfinite(num) and lower <= num <= upper else np.nan
    except (TypeError, ValueError):
        return np.nan


def get_refs(cell, path):
    try:
        refs = json.loads(cell)
    except (TypeError, ValueError) as e:
        raise ValueError(f'{path}: malformed reference_answers_json') from e
    if not isinstance(refs, list):
        raise ValueError(f'{path}: reference_answers_json must be a JSON array')
    return [str(x) for x in refs if isinstance(x, str) and x.strip()]


def normalize_answer(value):
    # Consistent with Demo 2 evaluate_demo2.py: lowercase, remove punctuation/articles.
    s = str(value or '').lower()
    s = ''.join(c for c in s if c not in string.punctuation)
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())


def token_f1(prediction, reference):
    p, r = normalize_answer(prediction), normalize_answer(reference)
    special = {'yes', 'no', 'unanswerable'}
    if (p in special or r in special) and p != r:
        return 0.0
    a, b = p.split(), r.split()
    if not a or not b:
        return float(a == b)
    overlap = sum((Counter(a) & Counter(b)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(a), overlap / len(b)
    return 2 * precision * recall / (precision + recall)


def score_answer(answer, references):
    if not references:
        return np.nan, np.nan
    if not nonempty(answer):
        return 0.0, 0.0
    em = float(any(normalize_answer(answer) == normalize_answer(r) for r in references))
    f1 = max(token_f1(answer, r) for r in references)
    return em, f1


def model_old_path(root, model, dataset):
    if dataset == 'sciq_bias_incorrect':
        path = root / 'confirmation_bias' / 'merged' / f'{model}_sciq_bias_combined.csv'
        if not path.exists():
            raise FileNotFoundError(
                f'{path}\nRun: python summarize_bias_sciq.py --phase collection --results-dir results'
            )
    else:
        path = root / 'merged' / f'{model}_{dataset}_combined.csv'
        if not path.exists():
            raise FileNotFoundError(
                f'{path}\nRun: python merge_results.py --results-dir results'
            )
    return path


def read_old(root, model, dataset):
    path = model_old_path(root, model, dataset)
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig',
                     usecols=lambda c: c in set(D2_ANSWER_COLS.values()) |
                     set(D2_CONF_COLS.values()) |
                     {'global_question_idx', 'reference_answers_json',
                      'sample_seeds_json', 'p_true_probability_A',
                      'bias_condition', 'question'})
    required = {'global_question_idx', 'reference_answers_json',
                'p_true_probability_A', *D2_ANSWER_COLS.values(), *D2_CONF_COLS.values()}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'{path}: missing {sorted(missing)}')
    if 'sample_seeds_json' in df.columns:
        samples = df['sample_seeds_json'].unique()
        for sample in samples:
            first = first_json_item(sample, 'sample_seeds_json', path)
            if int(first) != 42:
                raise ValueError(f'{path}: first Demo 2 sample is not seed 42: {first}')
    if dataset == 'sciq_bias_incorrect':
        if 'bias_condition' not in df:
            raise ValueError(f'{path}: missing bias_condition')
        df = df.loc[df['bias_condition'].str.lower() == 'biased_incorrect'].copy()
    df['global_idx'] = pd.to_numeric(df['global_question_idx'], errors='raise').astype(int)
    if df['global_idx'].duplicated().any():
        raise ValueError(f'{path}: duplicate selected question IDs')
    if len(df) != EXPECTED[dataset]:
        raise ValueError(f'{path}: expected {EXPECTED[dataset]} matched questions, got {len(df)}')

    records = []
    for row in df.to_dict('records'):
        refs = get_refs(row['reference_answers_json'], path)
        for method in D2:
            answer = first_json_item(row[D2_ANSWER_COLS[method]], D2_ANSWER_COLS[method], path)
            if method == 'p_true':
                rank = score_float(row['p_true_probability_A'], 0, 1)
                score10 = np.nan  # P(True) is NOT a 1-10 self-report.
            else:
                score10 = score_float(first_json_item(
                    row[D2_CONF_COLS[method]], D2_CONF_COLS[method], path), 1, 10)
                rank = score10 / 10 if np.isfinite(score10) else np.nan
            records.append({
                'model': model, 'dataset': dataset, 'global_idx': row['global_idx'],
                'method': method, 'demo': 'Demo 2', 'answer': nonempty(answer),
                'refs': refs, 'score10': score10, 'rank_score': rank,
                'question': row.get('question', ''),
            })
    return pd.DataFrame(records)


def read_new(root, model, dataset):
    path = root / 'demo3' / 'csv' / f'{model}_{dataset}.csv'
    if not path.exists():
        raise FileNotFoundError(f'{path}\nRun: python demo3_export.py')
    needed = {'global_idx', 'reference_answers_json', 'question', 'incorrect_cue'}
    for method in D3:
        needed.update({f'{method}_answer', f'{method}_{"bet_usd" if method == "money_betting" else "confidence"}'})
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig',
                     usecols=lambda c: c in needed)
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f'{path}: missing {sorted(missing)}')
    df['global_idx'] = pd.to_numeric(df['global_idx'], errors='raise').astype(int)
    if df['global_idx'].duplicated().any():
        raise ValueError(f'{path}: duplicate question IDs')
    if len(df) != EXPECTED[dataset]:
        raise ValueError(f'{path}: expected {EXPECTED[dataset]} questions, got {len(df)}')
    records = []
    for row in df.to_dict('records'):
        refs = get_refs(row['reference_answers_json'], path)
        for method in D3:
            answer = nonempty(row[f'{method}_answer'])
            bound = (0, 10) if method == 'money_betting' else (1, 10)
            field = 'bet_usd' if method == 'money_betting' else 'confidence'
            score10 = score_float(row[f'{method}_{field}'], *bound)
            records.append({
                'model': model, 'dataset': dataset, 'global_idx': row['global_idx'],
                'method': method, 'demo': 'Demo 3', 'answer': answer,
                'refs': refs, 'score10': score10,
                'rank_score': score10 / 10 if np.isfinite(score10) else np.nan,
                'question': row['question'], 'incorrect_cue': row['incorrect_cue'],
            })
    return pd.DataFrame(records)


def load_comparisons(root, allow_partial=False):
    chunks = []
    for model in MODEL_KEYS:
        for ds in DATASETS:
            try:
                old = read_old(root, model, ds)
                new = read_new(root, model, ds)
            except (FileNotFoundError, ValueError) as e:
                if not allow_partial:
                    raise
                print(f'WARNING: skipping {model}/{ds}: {e}')
                continue
            # Use original SciQ index for biased and unbiased questions; QASPER global order is fixed.
            ids_old = set(old.global_idx)
            ids_new = set(new.global_idx)
            if ids_old != ids_new:
                raise ValueError(f'{model}/{ds}: Demo 2 and Demo 3 question IDs differ; '
                                 f'{len(ids_old ^ ids_new)} unmatched IDs')
            gold_old = old.drop_duplicates('global_idx').set_index('global_idx')['refs']
            gold_new = new.drop_duplicates('global_idx').set_index('global_idx')['refs']
            mismatch = [idx for idx in ids_old if
                        {normalize_answer(v) for v in gold_old.loc[idx]} !=
                        {normalize_answer(v) for v in gold_new.loc[idx]}]
            if mismatch:
                raise ValueError(f'{model}/{ds}: references differ at IDs {mismatch[:10]}. '
                                 'Do not compare mismatched questions.')
            if ds != 'sciq_bias_incorrect':
                old_questions = old.drop_duplicates('global_idx').set_index('global_idx')['question']
                new_questions = new.drop_duplicates('global_idx').set_index('global_idx')['question']
                mismatch_q = [idx for idx in ids_old if
                              normalize_answer(old_questions.loc[idx]) !=
                              normalize_answer(new_questions.loc[idx])]
                if mismatch_q:
                    raise ValueError(f'{model}/{ds}: question texts differ at IDs '
                                     f'{mismatch_q[:10]}; refusing unmatched comparison.')
            else:
                # Bias prompt versions differ by design; verify both mention the same original index
                # and use the same reference. The two collection scripts append the same instruction.
                pass
            combo = pd.concat([old, new], ignore_index=True)
            for method in METHODS:
                n = len(combo.loc[combo.method == method])
                if n != EXPECTED[ds]:
                    raise ValueError(f'{model}/{ds}/{method}: expected {EXPECTED[ds]}, got {n}')
            chunks.append(combo)
            print(f'Matched {model}/{ds}: {len(ids_old):,} questions × 9 methods')
    if not chunks:
        raise RuntimeError('No matched Demo 2 and Demo 3 collection files found')
    all_rows = pd.concat(chunks, ignore_index=True)
    all_rows['model_label'] = all_rows['model'].map(MODEL_LABELS)
    all_rows['method_label'] = all_rows['method'].map(METHOD_LABELS)
    all_rows['dataset_label'] = all_rows['dataset'].map(DATA_LABELS)
    metrics = [score_answer(row.answer, row.refs) for row in all_rows.itertuples(index=False)]
    all_rows['exact_match'] = [v[0] for v in metrics]
    all_rows['token_f1'] = [v[1] for v in metrics]
    all_rows['evaluable'] = all_rows['exact_match'].notna()
    all_rows['score_valid'] = all_rows['rank_score'].notna()
    # Only SciQ incorrect-cue wording: exact match to the wrong suggested term.
    # This is a lexical proxy for cue adoption, NOT proof of confirmation bias.
    mask = all_rows.dataset.eq('sciq_bias_incorrect')
    all_rows['cue_adopted'] = np.nan
    # The old exports don't include the cue, so attach it from the matched Demo 3 records.
    cue_by_key = (all_rows.loc[mask & all_rows.demo.eq('Demo 3'),
                               ['model', 'global_idx', 'incorrect_cue']]
                  .drop_duplicates(['model', 'global_idx']))
    lookup = {(r.model, r.global_idx): r.incorrect_cue for r in cue_by_key.itertuples()}
    biased_rows = all_rows.loc[mask, ['model', 'global_idx', 'answer']]
    all_rows.loc[mask, 'cue_adopted'] = [
        float(bool(nonempty(ans)) and
              normalize_answer(ans) == normalize_answer(lookup.get((m, idx), '')))
        for m, idx, ans in biased_rows.itertuples(index=False, name=None)
    ]
    return all_rows


# ============================================================================
# Demo 4 extension (stand-alone; uses the Demo 2/3 loaders defined above)
# ============================================================================

D4 = ('structured_rerun', 'academic_system')
ALL_METHODS = METHODS + D4
METHOD_LABELS.update({
    'structured_rerun': 'Structured Rerun (D4)',
    'academic_system': 'Academic System (D4)',
})
METHOD_COLORS.update({
    'structured_rerun': '#17becf',
    'academic_system': '#1a1a1a',
})


def demo4_methods_for(model):
    return D4 if model.startswith('qwen_') else ('academic_system',)


def read_demo4(root, model, dataset):
    path = root / 'demo4' / 'csv' / f'{model}_{dataset}.csv'
    if not path.is_file():
        raise FileNotFoundError(f'{path}\nRun: python demo4_export.py')
    methods = demo4_methods_for(model)
    required = {'global_idx', 'reference_answers_json', 'question'}
    for m in methods:
        required.update((f'{m}_answer', f'{m}_confidence',
                         f'{m}_answer_output_tokens', f'{m}_confidence_output_tokens'))
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig',
                     usecols=lambda c: c in required)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'{path}: missing required fields: {sorted(missing)}')
    df['global_idx'] = pd.to_numeric(df.global_idx, errors='raise').astype(int)
    if len(df) != EXPECTED[dataset]:
        raise ValueError(f'{path}: expected {EXPECTED[dataset]} questions, got {len(df)}')
    if df.global_idx.duplicated().any():
        raise ValueError(f'{path}: duplicate global_idx')
    # Empty output text is possible, but generated-stage token counts are present
    # only after stage completion. Refuse incomplete runs by default.
    for method in methods:
        for stage in ('answer', 'confidence'):
            missing_stage = (df[f'{method}_{stage}_output_tokens'].astype(str).str.strip() == '')
            if missing_stage.any():
                first = df.loc[missing_stage, 'global_idx'].head(4).tolist()
                raise ValueError(f'{path}: {method} has {missing_stage.sum()} missing '
                                 f'{stage} stages, e.g. IDs {first}. Complete/export Demo 4 first.')
    records = []
    for row in df.to_dict('records'):
        refs = get_refs(row['reference_answers_json'], path)
        for method in methods:
            score = score_float(row[f'{method}_confidence'], 1, 10)
            records.append(dict(
                model=model, dataset=dataset, global_idx=row['global_idx'],
                method=method, demo='Demo 4', answer=nonempty(row[f'{method}_answer']),
                refs=refs, score10=score,
                rank_score=score/10 if np.isfinite(score) else np.nan,
                question=row['question'],
            ))
    return pd.DataFrame(records)


def append_demo4(root, base):
    chunks = [base]
    for model in MODEL_KEYS:
        for ds in DATASETS:
            d4 = read_demo4(root, model, ds)
            d3 = base[(base.model == model) & (base.dataset == ds) &
                      (base.method == 'vanilla')].set_index('global_idx')
            indices = set(d3.index)
            actual = set(d4.global_idx)
            if indices != actual:
                raise ValueError(f'{model}/{ds}: Demo 4 mismatches Demo 2/3 question IDs: '
                                 f'{len(indices ^ actual)} differences')
            q4 = d4.drop_duplicates('global_idx').set_index('global_idx')
            sample_d3 = base[(base.model == model) & (base.dataset == ds) &
                             (base.method == 'speculative_editor')].set_index('global_idx')
            gold_bad, q_bad = [], []
            for idx in indices:
                refs3 = {normalize_answer(r) for r in d3.loc[idx, 'refs']}
                refs4 = {normalize_answer(r) for r in q4.loc[idx, 'refs']}
                if refs3 != refs4:
                    gold_bad.append(idx)
                # Bias question wording in D2 differs by design; D3/D4 frozen
                # inputs should match; D3 question is in base via other methods.
                if normalize_answer(q4.loc[idx, 'question']) != normalize_answer(sample_d3.loc[idx, 'question']):
                    q_bad.append(idx)
            if gold_bad or q_bad:
                raise ValueError(f'{model}/{ds}: inconsistent Demo 4 references {gold_bad[:5]}'
                                 f' or questions {q_bad[:5]}; cannot compare')
            print(f'Matched Demo 4 {model}/{ds}: {len(indices):,} questions, '
                  f'{len(demo4_methods_for(model))} method(s)', flush=True)
            chunks.append(d4)
    long = pd.concat(chunks, ignore_index=True)
    long['model_label'] = long.model.map(MODEL_LABELS)
    long['dataset_label'] = long.dataset.map(DATA_LABELS)
    long['method_label'] = long.method.map(METHOD_LABELS)
    # Scored again from the same original evaluator for all demos.
    metrics = [score_answer(row.answer, row.refs) for row in long.itertuples(index=False)]
    long['exact_match'] = [m[0] for m in metrics]
    long['token_f1'] = [m[1] for m in metrics]
    long['evaluable'] = long.exact_match.notna()
    long['score_valid'] = long.score10.notna()
    return long


def summarize_all(long):
    parts = []
    for (model, ds, method), g in long.groupby(['model', 'dataset', 'method'], sort=False):
        ok = g[g.evaluable]
        conf = g[g.score10.notna()]
        correct_conf = ok[(ok.exact_match == 1) & ok.score10.notna()]
        wrong_conf = ok[(ok.exact_match == 0) & ok.score10.notna()]
        avg_em = ok.exact_match.mean() if len(ok) else np.nan
        avg_conf = conf.score10.mean() if len(conf) else np.nan
        parts.append(dict(
            model=model, model_label=MODEL_LABELS[model],
            dataset=ds, dataset_label=DATA_LABELS[ds],
            method=method, method_label=METHOD_LABELS[method],
            demo='Demo 2' if method in D2 else ('Demo 3' if method in D3 else 'Demo 4'),
            questions=len(g), evaluable_questions=len(ok),
            exact_match=avg_em,
            token_f1=ok.token_f1.mean() if len(ok) else np.nan,
            confidence=avg_conf,
            confidence_n=len(conf),
            mean_conf_correct=correct_conf.score10.mean() if len(correct_conf) else np.nan,
            mean_conf_incorrect=wrong_conf.score10.mean() if len(wrong_conf) else np.nan,
            confidence_minus_accuracy=(avg_conf/10-avg_em) if
                np.isfinite(avg_em) and np.isfinite(avg_conf) else np.nan,
        ))
    return pd.DataFrame(parts)


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_group_bars(data, xkey, groupkey, valuekey, order_x, order_groups,
                    path, title, ylabel, y_bounds=None, percent=False, large=False):
    fig, ax = plt.subplots(figsize=(19 if large else 14, 6.5))
    x = np.arange(len(order_x))
    n = len(order_groups)
    width = .84/n
    for j, group in enumerate(order_groups):
        subset = data[data[groupkey] == group].set_index(xkey)
        vals = []
        for label in order_x:
            if label in subset.index:
                v = subset.loc[label, valuekey]
                # If duplicate keys appear, caller must aggregate first.
                if isinstance(v, pd.Series):
                    raise ValueError(f'Duplicate plotting keys: {label}, {group}')
                vals.append(float(v) * (100 if percent else 1))
            else:
                vals.append(np.nan)
        color = METHOD_COLORS.get(group) if groupkey == 'method' else None
        ax.bar(x - .42 + width * (j+.5), vals, width=width*.95,
               color=color, label=METHOD_LABELS.get(group, MODEL_LABELS.get(group, DATA_LABELS.get(group, group))))
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(k, MODEL_LABELS.get(k, DATA_LABELS.get(k, k))) for k in order_x],
                       rotation=0 if xkey == 'model' else 30, ha='center' if xkey == 'model' else 'right')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=.23)
    ax.set_axisbelow(True)
    if y_bounds is not None:
        ax.set_ylim(*y_bounds)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.13), ncol=5, frameon=False, fontsize=9)
    save_figure(fig, path)


def plot_box(data, cat_col, val_col, labels, path, title, ylabel, y_bounds=None):
    vals = []
    names = []
    for item in labels:
        vec = pd.to_numeric(data.loc[data[cat_col] == item, val_col], errors='coerce').dropna().to_numpy()
        if len(vec):
            vals.append(vec)
            names.append(METHOD_LABELS.get(item, MODEL_LABELS.get(item, DATA_LABELS.get(item, item))) +
                         f'\n(n={len(vec):,})')
    if not vals:
        return
    fig, ax = plt.subplots(figsize=(max(9, 1.45*len(vals)), 6.5))
    box = ax.boxplot(vals, tick_labels=names, patch_artist=True,
                     showmeans=True, showfliers=False)
    for i, patch in enumerate(box['boxes']):
        patch.set_facecolor(plt.get_cmap('tab20')(i))
        patch.set_alpha(.48)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=.23)
    if y_bounds is not None:
        ax.set_ylim(*y_bounds)
    save_figure(fig, path)


def make_bias_delta(long):
    ori = long[long.dataset == 'sciq'][['model', 'global_idx', 'method',
                                      'exact_match', 'token_f1', 'score10']]
    biased = long[long.dataset == 'sciq_bias_incorrect'][['model', 'global_idx', 'method',
                                                         'exact_match', 'token_f1', 'score10']]
    paired = ori.merge(biased, on=['model', 'global_idx', 'method'], how='inner',
                       validate='one_to_one', suffixes=('_sciq', '_biased'))
    for value in ('exact_match', 'token_f1', 'score10'):
        paired[f'{value}_delta'] = paired[f'{value}_biased'] - paired[f'{value}_sciq']
    return paired


def run_plots(root, long, summary, extended=False):
    out = root / 'demo4' / 'plots' / 'demo2_demo3_demo4'
    out.mkdir(parents=True, exist_ok=True)
    conf_methods = [m for m in ALL_METHODS if m != 'p_true']

    # 9 charts: each dataset's four-model comparison (all available methods).
    for ds in DATASETS:
        s = summary[summary.dataset == ds]
        for metric, title, ylabel, lim, pct, methods in [
            ('exact_match', 'Exact-match accuracy', 'Exact match (%)', (0, 105), True, ALL_METHODS),
            ('token_f1', 'Answer token F1', 'Token F1 (%)', (0, 105), True, ALL_METHODS),
            ('confidence', 'Mean confidence / hypothetical bet', 'Mean score (0–10)', (0, 10.5), False, conf_methods),
        ]:
            plot_group_bars(s, 'model', 'method', metric, MODEL_KEYS, methods,
                out / f'bar_{metric}_by_model_{ds}.png',
                f'{title}: {DATA_LABELS[ds]} — all methods', ylabel, lim, pct, True)

    if extended:
        # 12 charts: within each model, compare across all three datasets.
        for model in MODEL_KEYS:
            s = summary[summary.model == model]
            model_methods = [m for m in ALL_METHODS if m != 'structured_rerun' or model.startswith('qwen_')]
            for metric, title, ylabel, lim, pct, methods in [
                ('exact_match', 'Exact-match accuracy', 'Exact match (%)', (0, 105), True, model_methods),
                ('token_f1', 'Token F1', 'Token F1 (%)', (0, 105), True, model_methods),
                ('confidence', 'Mean confidence / bet', 'Mean score (0–10)', (0, 10.5), False,
                    [m for m in model_methods if m != 'p_true']),
            ]:
                plot_group_bars(s, 'dataset', 'method', metric, DATASETS, methods,
                    out / f'bar_{metric}_by_dataset_{model}.png',
                    f'{title}: {MODEL_LABELS[model]} — three conditions', ylabel, lim, pct, True)

    # 3 focused bars: Demo 2 structured vs Demo 4 rerun vs system intervention.
    s = summary[summary.method.isin(['structured', 'structured_rerun', 'academic_system'])].copy()
    s['model_ds'] = s.model.map(MODEL_LABELS) + '\n' + s.dataset.map(DATA_LABELS)
    model_ds = [MODEL_LABELS[m] + '\n' + DATA_LABELS[d] for m in MODEL_KEYS for d in DATASETS]
    for metric, ylabel, bounds, pct in [
        ('exact_match', 'Exact match (%)', (0, 105), True),
        ('token_f1', 'Token F1 (%)', (0, 105), True),
        ('confidence', 'Mean score (0–10)', (0, 10.5), False),
    ]:
        plot_group_bars(s, 'model_ds', 'method', metric, model_ds,
            ['structured', 'structured_rerun', 'academic_system'],
            out / f'bar_d2_structured_vs_d4_{metric}.png',
            f'Demo 2 Structured vs Demo 4: {ylabel}', ylabel, bounds, pct, True)

    # 3 Qwen-only paired deltas: identical questions, both scores required for
    # confidence comparison. Missing confidence on either side is excluded.
    q = long[(long.model.isin(['qwen_1p7b', 'qwen_4b'])) &
             (long.method.isin(D4))]
    key = ['model', 'dataset', 'global_idx']
    q_stru = q[q.method == 'structured_rerun'][key + ['exact_match', 'token_f1', 'score10']]
    q_acad = q[q.method == 'academic_system'][key + ['exact_match', 'token_f1', 'score10']]
    qpair = q_stru.merge(q_acad, on=key, how='inner', validate='one_to_one',
                         suffixes=('_structured', '_academic'))
    for metric in ('exact_match', 'token_f1', 'score10'):
        qpair[metric + '_delta'] = (qpair[metric + '_academic'] -
                                   qpair[metric + '_structured'])
    qdelta = qpair.groupby(['model', 'dataset'], as_index=False).agg(
        exact_match_delta=('exact_match_delta', 'mean'),
        token_f1_delta=('token_f1_delta', 'mean'),
        confidence_delta=('score10_delta', 'mean'),
        paired_questions=('global_idx', 'size'),
        paired_valid_confidence=('score10_delta', 'count'),
    )
    qdelta['model_ds'] = qdelta.model.map(MODEL_LABELS) + '\n' + qdelta.dataset.map(DATA_LABELS)
    qorder = [MODEL_LABELS[m] + '\n' + DATA_LABELS[d]
              for m in ('qwen_1p7b', 'qwen_4b') for d in DATASETS]
    for metric, ylabel, scale in [
        ('exact_match_delta', 'Accuracy change (percentage points)', 100),
        ('token_f1_delta', 'Token F1 change (percentage points)', 100),
        ('confidence_delta', 'Paired confidence change (0–10 points)', 1),
    ]:
        changes = pd.Series({row.model_ds: getattr(row, metric)*scale
                             for row in qdelta.itertuples(index=False)})
        fig, ax = plt.subplots(figsize=(12, 5.8))
        ax.bar(range(len(qorder)), [changes.get(k, np.nan) for k in qorder], color='#1a1a1a')
        ax.axhline(0, color='gray', linewidth=1)
        ax.set_xticks(range(len(qorder)))
        ax.set_xticklabels(qorder, rotation=30, ha='right')
        ax.set_title(f'Academic System − Structured Rerun: {metric.replace("_", " ")} (Qwen only)')
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        save_figure(fig, out / f'bar_qwen_academic_minus_rerun_{metric}.png')

    # 3 bar charts: confirmation bias impact (incorrect cue minus original SciQ), by model.
    paired = make_bias_delta(long)
    delta_stats = paired.groupby(['model', 'method'])[['exact_match_delta', 'token_f1_delta', 'score10_delta']].mean().reset_index()
    for col, ylabel, scale, ms in [
        ('exact_match_delta', 'Accuracy change (percentage points)', 100, ALL_METHODS),
        ('token_f1_delta', 'Token F1 change (percentage points)', 100, ALL_METHODS),
        ('score10_delta', 'Score change (0–10 points)', 1, conf_methods),
    ]:
        scaled_col = col + '_plot'
        delta_stats[scaled_col] = delta_stats[col] * scale
        plot_group_bars(delta_stats, 'model', 'method', scaled_col, MODEL_KEYS, ms,
            out / f'bar_bias_delta_{col}.png',
            f'Incorrect cue minus original SciQ: {ylabel}', ylabel, None, False, True)

    # 3 descriptive gap bars: mean self-reported score/10 minus exact-match accuracy.
    # Money betting dollars are shown on 0–10 scale; the gap is NOT formal calibration.
    for ds in DATASETS:
        s = summary[(summary.dataset == ds) & (summary.method != 'p_true')]
        plot_group_bars(s, 'model', 'method', 'confidence_minus_accuracy',
            MODEL_KEYS, conf_methods,
            out / f'bar_confidence_minus_accuracy_{ds}.png',
            f'Mean confidence/10 minus exact-match accuracy: {DATA_LABELS[ds]}',
            'Descriptive gap (percentage points)', None, True, True)

    # 3 token F1 boxes and 3 confidence boxes: each dataset, per-question distributions.
    for ds in DATASETS:
        g = long[long.dataset == ds]
        plot_box(g, 'method', 'token_f1', ALL_METHODS,
            out / f'box_token_f1_by_method_{ds}.png',
            f'Token F1 per question by method: {DATA_LABELS[ds]}',
            'Token F1 (0–1)', (0, 1.05))
        plot_box(g[g.method != 'p_true'], 'method', 'score10', conf_methods,
            out / f'box_confidence_by_method_{ds}.png',
            f'Confidence / hypothetical bet by method: {DATA_LABELS[ds]}',
            'Reported score (0–10)', (-0.3, 10.5))

        # Two boxes in each dataset (correct vs incorrect), matched to accuracy.
        good = g[g.method != 'p_true'].copy()
        good = good[good.evaluable & good.score10.notna()]
        good['correctness'] = np.where(good.exact_match == 1,
                                        'Exact-match correct', 'Exact-match incorrect')
        plot_box(good, 'correctness', 'score10',
            ['Exact-match correct', 'Exact-match incorrect'],
            out / f'box_confidence_correct_vs_incorrect_{ds}.png',
            f'Confidence by exact-match correctness: {DATA_LABELS[ds]}',
            'Reported score / bet (0–10)', (-0.3, 10.5))

    # 4 method-wise confidence boxes within individual models.
    for model in MODEL_KEYS:
        g = long[(long.model == model) & (long.method != 'p_true')]
        model_methods = [m for m in conf_methods if m != 'structured_rerun' or model.startswith('qwen_')]
        plot_box(g, 'method', 'score10', model_methods,
                 out / f'box_confidence_by_method_{model}.png',
                 f'Confidence distribution: {MODEL_LABELS[model]} (all conditions)',
                 'Reported score / bet (0–10)', (-0.3, 10.5))

    if extended:
        # 4 per-model token F1 boxes.
        for model in MODEL_KEYS:
            g = long[long.model == model]
            methods = [m for m in ALL_METHODS if m != 'structured_rerun' or model.startswith('qwen_')]
            plot_box(g, 'method', 'token_f1', methods,
                     out / f'box_token_f1_by_method_{model}.png',
                     f'Token F1: {MODEL_LABELS[model]} (all conditions)',
                     'Token F1 (0–1)', (0, 1.05))

    # Data exports for audit.
    summary.sort_values(['model', 'dataset', 'method']).to_csv(out / 'metrics_summary.csv', index=False)
    paired.to_csv(out / 'sciq_bias_question_deltas.csv', index=False)
    qdelta.to_csv(out / 'qwen_system_minus_structured_rerun.csv', index=False)
    print(f'Generated {len(list(out.glob("*.png")))} PNG charts and 3 audit CSVs in:\n  {out}', flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=Path('results'))
    parser.add_argument('--extended', action='store_true',
                        help='Add 16 supplementary per-model charts')
    args = parser.parse_args()
    if not args.results_dir.exists():
        parser.error(f'Not found: {args.results_dir}')
    root = args.results_dir.resolve()
    print('Loading and verifying matched Demo 2 and Demo 3 questions...', flush=True)
    d23 = load_comparisons(root, allow_partial=False)
    print('Loading and verifying Demo 4 exports...', flush=True)
    combined = append_demo4(root, d23)
    summary = summarize_all(combined)
    run_plots(root, combined, summary, extended=args.extended)


if __name__ == '__main__':
    main()
