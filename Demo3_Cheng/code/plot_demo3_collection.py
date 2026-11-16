#!/usr/bin/env python3
"""Create bar charts and box plots for Demo 3 collection results.

Expected input: wide CSVs produced by demo3_export.py under results/demo3/csv/.
Outputs: PNG plots under results/demo3/plots/.

Only bar charts and box plots are generated.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

METHODS = [
    'speculative_editor',
    'opposing_counsel',
    'compliance_officer',
    'persona_panel',
    'money_betting',
]

METHOD_LABELS = {
    'speculative_editor': 'Speculative Editor',
    'opposing_counsel': 'Opposing Counsel',
    'compliance_officer': 'Compliance Officer',
    'persona_panel': 'Persona Panel',
    'money_betting': 'Money Betting',
}

MODEL_LABELS = {
    'llama_1b': 'Llama 1B',
    'llama_3b': 'Llama 3B',
    'qwen_1p7b': 'Qwen 1.7B',
    'qwen_4b': 'Qwen 4B',
}

DATASET_LABELS = {
    'sciq': 'SciQ',
    'sciq_bias_incorrect': 'SciQ Confirmation Bias (Incorrect Cue)',
    'qasper': 'QASPER',
}


def nice_model(x: str) -> str:
    return MODEL_LABELS.get(x, x)


def nice_dataset(x: str) -> str:
    return DATASET_LABELS.get(x, x)


def nice_method(x: str) -> str:
    return METHOD_LABELS.get(x, x)


def _to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({'true': True, 'false': False, '1': True, '0': False})


def load_long_results(results_dir: Path) -> pd.DataFrame:
    csv_dir = results_dir / 'csv'
    files = [p for p in csv_dir.glob('*.csv') if p.name != 'collection_coverage.csv']
    if not files:
        raise FileNotFoundError(f'No model CSVs found under {csv_dir}')

    pieces = []
    for path in sorted(files):
        wide = pd.read_csv(path)
        base_cols = [c for c in ['model', 'dataset', 'global_idx', 'paper_idx', 'question_idx', 'question'] if c in wide.columns]
        for method in METHODS:
            cols = {
                'answer': f'{method}_answer',
                'confidence': f'{method}_confidence',
                'bet_usd': f'{method}_bet_usd',
                'raw': f'{method}_raw',
                'input_tokens': f'{method}_input_tokens',
                'output_tokens': f'{method}_output_tokens',
                'elapsed_seconds': f'{method}_elapsed_seconds',
                'input_truncated': f'{method}_input_truncated',
                'panel_json_valid': f'{method}_panel_json_valid',
            }
            available = {k: v for k, v in cols.items() if v in wide.columns}
            part = wide[base_cols + list(available.values())].copy()
            rename_map = {v: k for k, v in available.items()}
            part = part.rename(columns=rename_map)
            part['method'] = method
            pieces.append(part)

    long = pd.concat(pieces, ignore_index=True)
    # Types and derived fields
    long['model_label'] = long['model'].map(nice_model)
    long['dataset_label'] = long['dataset'].map(nice_dataset)
    long['method_label'] = long['method'].map(nice_method)
    long['answer_present'] = long['answer'].notna() & (long['answer'].astype(str).str.strip() != '')
    if 'confidence' in long.columns:
        long['confidence'] = pd.to_numeric(long['confidence'], errors='coerce')
    else:
        long['confidence'] = np.nan
    if 'bet_usd' in long.columns:
        long['bet_usd'] = pd.to_numeric(long['bet_usd'], errors='coerce')
    else:
        long['bet_usd'] = np.nan
    for col in ['input_tokens', 'output_tokens', 'elapsed_seconds']:
        if col in long.columns:
            long[col] = pd.to_numeric(long[col], errors='coerce')
        else:
            long[col] = np.nan
    if 'input_truncated' in long.columns:
        long['input_truncated'] = _to_bool(long['input_truncated']).fillna(False)
    else:
        long['input_truncated'] = False
    if 'panel_json_valid' in long.columns:
        long['panel_json_valid'] = _to_bool(long['panel_json_valid'])
    else:
        long['panel_json_valid'] = np.nan

    long['score'] = np.where(long['method'] == 'money_betting', long['bet_usd'], long['confidence'])
    long['score_present'] = long['score'].notna()
    long['answer_chars'] = long['answer'].fillna('').astype(str).str.len()
    long['answer_words'] = long['answer'].fillna('').astype(str).str.split().str.len()
    long['model_dataset'] = long['model_label'] + '\n' + long['dataset_label']
    return long


def load_coverage(results_dir: Path) -> pd.DataFrame:
    candidates = [results_dir / 'csv' / 'collection_coverage.csv', Path('/mnt/data/collection_coverage.csv')]
    for path in candidates:
        if path.is_file():
            cov = pd.read_csv(path)
            cov['model_label'] = cov['model'].map(nice_model)
            cov['dataset_label'] = cov['dataset'].map(nice_dataset)
            cov['method_label'] = cov['method'].map(nice_method)
            cov['model_dataset'] = cov['model_label'] + '\n' + cov['dataset_label']
            cov['answer_parse_rate'] = 100 * cov['parsed_answers'] / cov['expected_questions']
            cov['score_parse_rate'] = 100 * cov['parsed_scores'] / cov['expected_questions']
            cov['completion_rate'] = 100 * cov['completed_responses'] / cov['expected_questions']
            return cov
    raise FileNotFoundError('collection_coverage.csv not found in results/demo3/csv or /mnt/data')


def grouped_bar(pivot: pd.DataFrame, title: str, ylabel: str, path: Path, ylim=None, annotate=True):
    if pivot.empty:
        return
    fig_w = max(10, 0.9 * len(pivot.index) + 4)
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    x = np.arange(len(pivot.index))
    n_series = len(pivot.columns)
    width = min(0.8 / max(n_series, 1), 0.22)
    offsets = (np.arange(n_series) - (n_series - 1) / 2.0) * width
    for i, col in enumerate(pivot.columns):
        vals = pivot[col].to_numpy(dtype=float)
        bars = ax.bar(x + offsets[i], vals, width=width, label=col)
        if annotate:
            ax.bar_label(bars, fmt='%.1f', padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=35, ha='right')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=0.3)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def simple_bar(series: pd.Series, title: str, ylabel: str, path: Path, ylim=None, annotate=True):
    if series.empty:
        return
    fig_w = max(8, 0.9 * len(series.index) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    x = np.arange(len(series.index))
    bars = ax.bar(x, series.to_numpy(dtype=float))
    if annotate:
        ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(series.index, rotation=35, ha='right')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=0.3)
    if ylim is not None:
        ax.set_ylim(*ylim)
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def boxplot_from_groups(groups: dict[str, list[float]], title: str, ylabel: str, path: Path, ylim=None):
    groups = {k: [v for v in vals if pd.notna(v)] for k, vals in groups.items()}
    groups = {k: vals for k, vals in groups.items() if len(vals) > 0}
    if not groups:
        return
    labels = list(groups)
    values = [groups[k] for k in labels]
    fig_w = max(9, 0.8 * len(labels) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    bp = ax.boxplot(values, tick_labels=labels, patch_artist=True, showmeans=True)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for patch, color in zip(bp['boxes'], [colors[i % len(colors)] for i in range(len(labels))]):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=0.3)
    if ylim is not None:
        ax.set_ylim(*ylim)
    plt.xticks(rotation=35, ha='right')
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results-dir', default='results/demo3', help='Demo 3 results directory')
    args = ap.parse_args()

    results_dir = Path(args.results_dir).resolve()
    plot_dir = results_dir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

    long = load_long_results(results_dir)
    cov = load_coverage(results_dir)

    # ---------- BAR CHARTS ----------
    # 1-2 Parse coverage by method and model/dataset combo
    answer_parse = cov.pivot(index='model_dataset', columns='method_label', values='answer_parse_rate')
    grouped_bar(answer_parse, 'Parsed answer rate by method and model/dataset', 'Parsed answers (%)',
                plot_dir / 'bar_answer_parse_rate_by_model_dataset.png', ylim=(0, 105))

    score_parse = cov.pivot(index='model_dataset', columns='method_label', values='score_parse_rate')
    grouped_bar(score_parse, 'Parsed score rate by method and model/dataset', 'Parsed scores (%)',
                plot_dir / 'bar_score_parse_rate_by_model_dataset.png', ylim=(0, 105))

    # 3 Mean confidence by method and dataset (exclude money)
    conf = long[long['method'] != 'money_betting']
    mean_conf_dataset = conf.groupby(['dataset_label', 'method_label'])['confidence'].mean().unstack()
    grouped_bar(mean_conf_dataset, 'Mean confidence by method and dataset', 'Mean confidence (1-10)',
                plot_dir / 'bar_mean_confidence_by_dataset.png', ylim=(0, 10.5))

    # 4 Mean confidence by method and model
    mean_conf_model = conf.groupby(['model_label', 'method_label'])['confidence'].mean().unstack()
    grouped_bar(mean_conf_model, 'Mean confidence by method and model', 'Mean confidence (1-10)',
                plot_dir / 'bar_mean_confidence_by_model.png', ylim=(0, 10.5))

    # 5 Mean bet by model and dataset
    bet = long[long['method'] == 'money_betting']
    mean_bet = bet.groupby(['model_dataset'])['bet_usd'].mean()
    simple_bar(mean_bet, 'Mean money bet by model/dataset', 'Mean bet ($0-$10)',
               plot_dir / 'bar_mean_bet_by_model_dataset.png', ylim=(0, 10.5))

    # 6 Mean output tokens by method and model
    mean_out_toks = long.groupby(['model_label', 'method_label'])['output_tokens'].mean().unstack()
    grouped_bar(mean_out_toks, 'Mean output tokens by method and model', 'Mean output tokens',
                plot_dir / 'bar_output_tokens_by_model.png')

    # 7 Mean latency by method and model
    mean_latency = long.groupby(['model_label', 'method_label'])['elapsed_seconds'].mean().unstack()
    grouped_bar(mean_latency, 'Mean latency by method and model', 'Mean elapsed seconds',
                plot_dir / 'bar_elapsed_seconds_by_model.png')

    # 8 Mean answer length (words) by method and dataset
    mean_answer_words = long.groupby(['dataset_label', 'method_label'])['answer_words'].mean().unstack()
    grouped_bar(mean_answer_words, 'Mean answer length by method and dataset', 'Mean answer length (words)',
                plot_dir / 'bar_answer_words_by_dataset.png')

    # 9 Persona panel JSON validity rate by model/dataset
    panel = long[long['method'] == 'persona_panel'].copy()
    panel['panel_json_valid'] = panel['panel_json_valid'].fillna(False)
    panel_rate = 100 * panel.groupby('model_dataset')['panel_json_valid'].mean()
    simple_bar(panel_rate, 'Persona panel valid JSON rate by model/dataset', 'Valid JSON rate (%)',
               plot_dir / 'bar_panel_json_valid_rate.png', ylim=(0, 105))

    # 10 Input truncation rate by method and dataset
    trunc = 100 * long.groupby(['dataset_label', 'method_label'])['input_truncated'].mean().unstack()
    grouped_bar(trunc, 'Input truncation rate by method and dataset', 'Truncated inputs (%)',
                plot_dir / 'bar_input_truncation_rate_by_dataset.png', ylim=(0, 105))

    # ---------- BOX PLOTS ----------
    # 11 Confidence distributions by method
    groups = {m: conf.loc[conf['method_label'] == m, 'confidence'].dropna().tolist()
              for m in [nice_method(m) for m in METHODS if m != 'money_betting']}
    boxplot_from_groups(groups, 'Confidence distribution by method', 'Confidence (1-10)',
                        plot_dir / 'box_confidence_by_method.png', ylim=(0.5, 10.5))

    # 12 Confidence distributions by model
    groups = {m: conf.loc[conf['model_label'] == m, 'confidence'].dropna().tolist()
              for m in conf['model_label'].dropna().unique()}
    boxplot_from_groups(groups, 'Confidence distribution by model', 'Confidence (1-10)',
                        plot_dir / 'box_confidence_by_model.png', ylim=(0.5, 10.5))

    # 13 Confidence distributions by dataset
    groups = {d: conf.loc[conf['dataset_label'] == d, 'confidence'].dropna().tolist()
              for d in conf['dataset_label'].dropna().unique()}
    boxplot_from_groups(groups, 'Confidence distribution by dataset', 'Confidence (1-10)',
                        plot_dir / 'box_confidence_by_dataset.png', ylim=(0.5, 10.5))

    # 14 Bet distributions by model
    groups = {m: bet.loc[bet['model_label'] == m, 'bet_usd'].dropna().tolist()
              for m in bet['model_label'].dropna().unique()}
    boxplot_from_groups(groups, 'Money betting distribution by model', 'Bet amount ($0-$10)',
                        plot_dir / 'box_bet_by_model.png', ylim=(-0.2, 10.5))

    # 15 Bet distributions by dataset
    groups = {d: bet.loc[bet['dataset_label'] == d, 'bet_usd'].dropna().tolist()
              for d in bet['dataset_label'].dropna().unique()}
    boxplot_from_groups(groups, 'Money betting distribution by dataset', 'Bet amount ($0-$10)',
                        plot_dir / 'box_bet_by_dataset.png', ylim=(-0.2, 10.5))

    # 16 Output token distributions by method
    groups = {m: long.loc[long['method_label'] == m, 'output_tokens'].dropna().tolist()
              for m in long['method_label'].dropna().unique()}
    boxplot_from_groups(groups, 'Output token distribution by method', 'Output tokens',
                        plot_dir / 'box_output_tokens_by_method.png')

    # 17 Latency distributions by method
    groups = {m: long.loc[long['method_label'] == m, 'elapsed_seconds'].dropna().tolist()
              for m in long['method_label'].dropna().unique()}
    boxplot_from_groups(groups, 'Latency distribution by method', 'Elapsed seconds',
                        plot_dir / 'box_elapsed_seconds_by_method.png')

    # 18 Answer-length distributions by method
    groups = {m: long.loc[long['method_label'] == m, 'answer_words'].dropna().tolist()
              for m in long['method_label'].dropna().unique()}
    boxplot_from_groups(groups, 'Answer length distribution by method', 'Answer length (words)',
                        plot_dir / 'box_answer_words_by_method.png')

    print(f'Saved plots to: {plot_dir}')
    for p in sorted(plot_dir.glob('*.png')):
        print(p.name)


if __name__ == '__main__':
    main()
