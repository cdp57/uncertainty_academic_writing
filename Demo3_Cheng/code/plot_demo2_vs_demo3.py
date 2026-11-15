#!/usr/bin/env python3
"""Demo 2 vs Demo 3 matched-question study plots. CPU only; no error bars.

Usage (from 447_demo2/):
    python plot_demo2_vs_demo3.py --results-dir results

Required existing files:
    results/merged/{llama_1b,llama_3b,qwen_1p7b,qwen_4b}_{sciq,qasper}_combined.csv
    results/confirmation_bias/merged/{model}_sciq_bias_combined.csv
    results/demo3/csv/{model}_{sciq,sciq_bias_incorrect,qasper}.csv

Demo 2: first item in the five-answer JSON arrays (seed 42, when recorded).
Demo 3: one response per method. All metrics are computed on matched question IDs.
Exact-match and token F1 use the Demo 2 evaluation normalization convention.
Missing answers score zero; rows without gold answers do not enter accuracy metrics.
Missing scores are omitted only from score-based analyses. See README output.
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


def calculate_auroc(y, scores):
    """Mann-Whitney AUROC with average ranks for ties."""
    yy = np.asarray(y, dtype=float)
    ss = np.asarray(scores, dtype=float)
    valid = np.isfinite(yy) & np.isfinite(ss)
    yy, ss = yy[valid], ss[valid]
    npos = int((yy == 1).sum())
    nneg = int((yy == 0).sum())
    if not npos or not nneg:
        return np.nan
    ranks = pd.Series(ss).rank(method='average').to_numpy()
    return float((ranks[yy == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def calculate_aurc(y, scores):
    """Mean cumulative error risk under descending confidence-ranked coverage."""
    yy = np.asarray(y, dtype=float)
    ss = np.asarray(scores, dtype=float)
    valid = np.isfinite(yy) & np.isfinite(ss)
    yy, ss = yy[valid], ss[valid]
    if not len(yy):
        return np.nan
    # Stable order for equal scores. AURC is descriptive, not a causal effect.
    order = np.argsort(-ss, kind='stable')
    errors = (1 - yy[order]).astype(float)
    cumulative_risk = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    return float(cumulative_risk.mean())


def summarise(long):
    rows = []
    for (model, ds, method), g in long.groupby(['model', 'dataset', 'method'], sort=False):
        ev = g[g.evaluable]
        sc = ev[ev.score_valid]
        ss = g[g.score10.notna()]
        em = ev.exact_match.mean() if len(ev) else np.nan
        mean_score = ss.score10.mean() if len(ss) else np.nan
        rows.append(dict(
            model=model, model_label=MODEL_LABELS[model], dataset=ds,
            dataset_label=DATA_LABELS[ds], method=method,
            method_label=METHOD_LABELS[method], demo='Demo 2' if method in D2 else 'Demo 3',
            questions=len(g), evaluable_questions=len(ev),
            exact_match=em, token_f1=ev.token_f1.mean() if len(ev) else np.nan,
            mean_score_10=mean_score, mean_rank_score=sc.rank_score.mean(),
            scored_questions=len(sc), auroc=calculate_auroc(sc.exact_match, sc.rank_score),
            aurc=calculate_aurc(sc.exact_match, sc.rank_score),
            score_minus_accuracy=(mean_score / 10 - em) if
                np.isfinite(mean_score) and np.isfinite(em) else np.nan,
            cue_adoption=g.cue_adopted.mean() if ds == 'sciq_bias_incorrect' else np.nan,
        ))
    return pd.DataFrame(rows)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches='tight')
    plt.close(fig)


def groupbars(summary, ds, metric, out, ylabel, title, scale=1, ylim=None, annotate=False,
              include_ptrue=True):
    s = summary[summary.dataset == ds]
    if not include_ptrue:
        s = s[s.method != 'p_true']
    methods = [m for m in METHODS if include_ptrue or m != 'p_true']
    fig, ax = plt.subplots(figsize=(15.5, 6.5))
    x = np.arange(len(MODEL_KEYS))
    width = 0.82 / len(methods)
    for j, m in enumerate(methods):
        val = []
        for model in MODEL_KEYS:
            hit = s[(s.model == model) & (s.method == m)][metric]
            val.append(hit.iloc[0] * scale if len(hit) else np.nan)
        bars = ax.bar(x - .41 + (j + .5) * width, val, width=width,
                      label=METHOD_LABELS[m], color=METHOD_COLORS[m])
        if annotate:
            ax.bar_label(bars, fmt='%.2f', fontsize=6, padding=2, rotation=90)
    ax.set_xticks(x, [MODEL_LABELS[m] for m in MODEL_KEYS])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=5, frameon=False)
    ax.grid(axis='y', alpha=0.25)
    ax.set_axisbelow(True)
    _save(fig, out)


def method_group_box(data, out, value_col, title, ylabel, methods, ylim=None):
    vals, names, colors = [], [], []
    for m in methods:
        v = pd.to_numeric(data.loc[data.method == m, value_col], errors='coerce').dropna().to_numpy()
        if len(v):
            vals.append(v)
            names.append(METHOD_LABELS[m] + f'\n(n={len(v)})')
            colors.append(METHOD_COLORS[m])
    if not vals:
        return
    fig, ax = plt.subplots(figsize=(max(12, len(vals) * 1.5), 6.5))
    box = ax.boxplot(vals, tick_labels=names, patch_artist=True, showmeans=True,
                     showfliers=False)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(.4)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=.25)
    if ylim:
        ax.set_ylim(*ylim)
    _save(fig, out)


def correct_incorrect_box(data, out, title):
    data = data[(data.method != 'p_true') & data.evaluable & data.score10.notna()].copy()
    vals = []
    labels = []
    for status, label in [(1.0, 'Correct'), (0.0, 'Incorrect')]:
        v = data.loc[data.exact_match == status, 'score10'].to_numpy(dtype=float)
        if len(v):
            vals.append(v)
            labels.append(f'{label}\n(n={len(v):,})')
    if not vals:
        return
    fig, ax = plt.subplots(figsize=(8.5, 6))
    boxes = ax.boxplot(vals, tick_labels=labels, patch_artist=True,
                       showmeans=True, showfliers=False)
    for patch, color in zip(boxes['boxes'], ['#377b65', '#bb6868']):
        patch.set_facecolor(color)
        patch.set_alpha(.35)
    ax.set_title(title)
    ax.set_ylabel('Confidence 1–10 / bet $0–$10 (numeric comparison)')
    ax.set_ylim(-.3, 10.6)
    ax.grid(axis='y', alpha=.25)
    _save(fig, out)


def paired_data(long):
    s = long[long.dataset.isin(['sciq', 'sciq_bias_incorrect'])].copy()
    original = s[s.dataset == 'sciq'][
        ['model', 'global_idx', 'method', 'exact_match', 'token_f1', 'score10', 'cue_adopted']
    ].copy()
    incorrect = s[s.dataset == 'sciq_bias_incorrect'][
        ['model', 'global_idx', 'method', 'exact_match', 'token_f1', 'score10', 'cue_adopted']
    ].copy()
    pair = original.merge(incorrect, on=['model', 'global_idx', 'method'], how='inner',
                          suffixes=('_original', '_biased'), validate='one_to_one')
    target = len(MODEL_KEYS) * EXPECTED['sciq'] * len(METHODS)
    if len(pair) != target:
        print(f'WARNING: {len(pair):,}/{target:,} original/incorrect cue pairs matched')
    for metric in ['exact_match', 'token_f1', 'score10']:
        pair[f'{metric}_delta'] = pair[f'{metric}_biased'] - pair[f'{metric}_original']
    return pair


def bias_delta_plot(pair, metric, out, title, ylabel, scale=1, include_ptrue=True):
    stats = pair.groupby(['model', 'method'])[metric].mean().reset_index()
    methods = [m for m in METHODS if include_ptrue or m != 'p_true']
    fig, ax = plt.subplots(figsize=(15, 6.5))
    width = .82 / len(methods)
    x = np.arange(len(MODEL_KEYS))
    for j, m in enumerate(methods):
        v = []
        for model in MODEL_KEYS:
            hit = stats[(stats.model == model) & (stats.method == m)][metric]
            v.append(hit.iloc[0] * scale if len(hit) else np.nan)
        ax.bar(x - .41 + (j + .5) * width, v, width=width,
               color=METHOD_COLORS[m], label=METHOD_LABELS[m])
    ax.set_xticks(x, [MODEL_LABELS[m] for m in MODEL_KEYS])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.axhline(0, color='black', linewidth=.8)
    ax.grid(axis='y', alpha=.25)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.10), ncol=5, frameon=False)
    _save(fig, out)


def adoption_plot(summary, out):
    groupbars(summary, 'sciq_bias_incorrect', 'cue_adoption', out,
              'Answer exactly matches incorrect suggestion (%)',
              'Incorrect cue adoption by model and method (lexical proxy)',
              scale=100, ylim=(0, 100))


def make_plots(long, summary, root):
    outdir = root / 'demo3' / 'plots' / 'demo2_vs_demo3'
    outdir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(outdir / 'method_summary.csv', index=False)
    # Preserve record-level matched calculations for offline inspection.
    long.drop(columns=['refs', 'question', 'incorrect_cue'], errors='ignore').to_csv(
        outdir / 'matched_first_sample_metrics.csv', index=False)

    for ds in DATASETS:
        label = DATA_LABELS[ds]
        groupbars(summary, ds, 'exact_match', outdir / f'bar_exact_match_{ds}.png',
                  'Normalized exact-match accuracy (%)', f'{label} | Demo 2 vs Demo 3 accuracy',
                  scale=100, ylim=(0, 100))
        groupbars(summary, ds, 'token_f1', outdir / f'bar_token_f1_{ds}.png',
                  'Reference token F1 (%)', f'{label} | Demo 2 vs Demo 3 token F1',
                  scale=100, ylim=(0, 100))
        groupbars(summary, ds, 'mean_score_10', outdir / f'bar_confidence_{ds}.png',
                  'Self-reported confidence / bet amount (0–10)',
                  f'{label} | reported confidence and money betting',
                  ylim=(0, 10.5), include_ptrue=False)
        groupbars(summary, ds, 'auroc', outdir / f'bar_auroc_{ds}.png',
                  'AUROC for predicting exact-match correctness (higher is better)',
                  f'{label} | correctness discrimination', ylim=(0, 1.05))
        groupbars(summary, ds, 'aurc', outdir / f'bar_aurc_{ds}.png',
                  'Area under risk–coverage curve (lower is better)',
                  f'{label} | selective-prediction risk', ylim=(0, 1.05))

        subset = long[(long.dataset == ds) & long.score10.notna()]
        method_group_box(subset, outdir / f'box_confidence_by_method_{ds}.png',
                         'score10', f'{label} | confidence distributions by method',
                         'Confidence 1–10 / money bet $0–$10',
                         [m for m in METHODS if m != 'p_true'], ylim=(-.3, 10.6))
        correct_incorrect_box(subset, outdir / f'box_confidence_correct_incorrect_{ds}.png',
                              f'{label} | confidence on correct vs incorrect answers')

    # Confirmation-bias impact paired by the SAME SciQ item/model/method.
    pair = paired_data(long)
    pair.to_csv(outdir / 'paired_sciq_incorrect_cue_deltas.csv', index=False)
    bias_delta_plot(pair, 'exact_match_delta', outdir / 'bar_bias_delta_exact_match.png',
                    'Incorrect-cue effect vs original SciQ | accuracy',
                    'Percentage-point change: biased − original', scale=100)
    bias_delta_plot(pair, 'token_f1_delta', outdir / 'bar_bias_delta_token_f1.png',
                    'Incorrect-cue effect vs original SciQ | token F1',
                    'Token F1 percentage-point change: biased − original', scale=100)
    bias_delta_plot(pair, 'score10_delta', outdir / 'bar_bias_delta_confidence.png',
                    'Incorrect-cue effect vs original SciQ | reported score',
                    'Confidence/bet change (0–10 units)', include_ptrue=False)
    adoption_plot(summary, outdir / 'bar_bias_incorrect_cue_adoption.png')
    method_group_box(pair, outdir / 'box_bias_delta_confidence_by_method.png',
                     'score10_delta', 'Paired confidence/bet changes after incorrect cue',
                     'Biased − original score (−10 to +10)',
                     [m for m in METHODS if m != 'p_true'])
    # Bias gap: confidence/10 minus exact match. Descriptive, not formal calibration of bet amounts.
    s = summary[summary.method != 'p_true'].copy()
    fig, ax = plt.subplots(figsize=(15.5, 6))
    ds_keys = list(DATASETS)
    x = np.arange(len(ds_keys))
    width = .83 / len(METHODS[:-1])  # adjusted below for eight non-P(True) methods
    choices = [m for m in METHODS if m != 'p_true']
    width = .83 / len(choices)
    for j, method in enumerate(choices):
        val = [s[(s.dataset == ds) & (s.method == method)]['score_minus_accuracy'].mean() * 100
               for ds in ds_keys]
        ax.bar(x - .415 + (j + .5) * width, val, width=width,
               label=METHOD_LABELS[method], color=METHOD_COLORS[method])
    ax.set_xticks(x, [DATA_LABELS[ds] for ds in ds_keys])
    ax.set_title('Score minus exact-match accuracy across models (descriptive gap)')
    ax.set_ylabel('Mean score/10 minus exact-match accuracy (percentage points)')
    ax.axhline(0, color='black', linewidth=.8)
    ax.grid(axis='y', alpha=.25)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.12), ncol=4, frameon=False)
    _save(fig, outdir / 'bar_score_minus_accuracy_by_dataset.png')
    (outdir / 'README.txt').write_text(
        'Demo 2 vs Demo 3: matched-question figures\n\n'
        'Demo 2: only the first sample (seed 42) from the five-value JSON arrays.\n'
        'Demo 3: one response per method. Matching: model, dataset, global question index, gold references;\n'
        'exact original question text is also checked for SciQ/QASPER. Incorrect cue wording differs by design.\n'
        'P(True) is included for accuracy, F1, AUROC, AURC and cue copying; it is EXCLUDED\n'
        'from confidence figures because P(True) is a probability, not a 1–10 self-report.\n'
        'Money betting is compared numerically on 0–10 (bet $0–$10), but bets are NOT calibrated\n'
        'probabilities; the score-minus-accuracy gap is a descriptive scale comparison for bets.\n'
        'Accuracy is normalized exact match, a strict lower-bound proxy on QASPER; token F1\n'
        'provides a more lenient lexical comparison, not semantic/LLM-judged correctness.\n'
        'Unparsed answer = 0 accuracy/F1 (when a gold answer exists). Missing score = excluded\n'
        'from confidence/AUROC/AURC, so scored-question coverage is in method_summary.csv.\n'
        'AUROC: higher is better; AURC: lower is better; both use score as a ranking signal,\n'
        'including P(True) and dollar wagers. No GPU, no extra model calls or error bars.\n'
        'All box plots omit outlier marks only to avoid dense overlap; observations remain in quartiles.\n'
        'Paired bias comparisons control for the same item but include wording/reframing effects.\n'
        'Exact-string incorrect-cue copying is a proxy, not proof of confirmation bias.\n',
        encoding='utf-8')
    return outdir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=Path('results'),
                        help='Root 447_demo2/results, NOT results/demo3')
    parser.add_argument('--allow-partial', action='store_true',
                        help='Skip missing model/dataset files; intended for testing only')
    args = parser.parse_args()
    root = args.results_dir.resolve()
    long = load_comparisons(root, args.allow_partial)
    summary = summarise(long)
    outdir = make_plots(long, summary, root)
    print(f'\nSaved {len(list(outdir.glob("*.png")))} bar charts and box plots to: {outdir}')
    print(f'Also: {outdir / "method_summary.csv"}')
    print(f'      {outdir / "paired_sciq_incorrect_cue_deltas.csv"}')


if __name__ == '__main__':
    main()
