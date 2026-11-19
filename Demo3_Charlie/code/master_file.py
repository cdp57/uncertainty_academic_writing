#!/usr/bin/env python3
"""
Merge three result sources into one summary CSV.

  --regular-dir  : per_example_results*.csv   (unbiased aggregated results)
  --biased-dir   : per_example_results_biased*.csv (biased aggregated results, has `condition`)
  --panel-dir    : demo3_<model>_<dataset>[_bias_*].csv (per model/dataset panel answers:
                   speculative_editor, opposing_counsel, compliance_officer,
                   persona_panel, money_betting)

Outputs:
  summary.csv      one row per (dataset, model, condition)
  per_example.csv  merged example-level table (for debugging / further analysis)

Usage:
  python merge_summary.py --regular-dir regular/ --biased-dir biased/ \
      --panel-dir panels/ --out-dir out/ [--model-map llama_1b=meta-llama/Llama-3.2-1B-Instruct]
"""
import argparse
import glob
import json
import os
import re
import string

import numpy as np
import pandas as pd

PANELS = ["speculative_editor", "opposing_counsel", "compliance_officer",
          "persona_panel", "money_betting"]
BASE_METRICS = ["judge_correct", "rouge_l", "rouge_l_correct", "semantic_entropy",
                "discrete_semantic_entropy", "length_normalized_predictive_entropy",
                "predictive_entropy", "vanilla_self_report", "structured_self_report",
                "p_true_probability"]
UNCERTAINTY = [m for m in BASE_METRICS if m not in ("judge_correct", "rouge_l", "rouge_l_correct")]


# ---------------------------------------------------------------- helpers
def read_dir(path):
    files = sorted(glob.glob(os.path.join(path, "**", "*.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No CSVs under {path}")
    dfs = []
    for f in files:
        d = pd.read_csv(f)
        d["source_file"] = os.path.basename(f)
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def make_qkey(df):
    """Common example key: paper+question for QASPER-style data, else item index."""
    paper = df["paper_idx"] if "paper_idx" in df else pd.Series(np.nan, index=df.index)
    q = df["question_idx"] if "question_idx" in df else pd.Series(np.nan, index=df.index)
    item = df["item_idx"] if "item_idx" in df else pd.Series(np.nan, index=df.index)
    item = item.fillna(q)

    def fmt(p, qi, it):
        if pd.notna(p) and pd.notna(qi):
            return f"p{int(p)}_q{int(qi)}"
        return f"i{int(it)}" if pd.notna(it) else None

    return pd.Series([fmt(p, qi, it) for p, qi, it in zip(paper, q, item)], index=df.index)


def norm(s):
    s = str(s).lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def panel_correct(answer, refs_json):
    """Lenient match: normalized answer equals / contains / is contained in any reference."""
    if pd.isna(answer) or pd.isna(refs_json):
        return np.nan
    try:
        refs = json.loads(refs_json)
    except Exception:
        refs = [refs_json]
    if isinstance(refs, str):
        refs = [refs]
    a = norm(answer)
    if not a:
        return 0.0
    for r in refs:
        r = norm(r)
        if r and (a == r or r in a or (len(a) > 2 and a in r)):
            return 1.0
    return 0.0


def auroc(score, label):
    """AUROC of `score` for predicting label==1 (rank-based, ties averaged)."""
    d = pd.DataFrame({"s": score, "y": label}).dropna()
    n1, n0 = (d.y == 1).sum(), (d.y == 0).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = d.s.rank()
    return (r[d.y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


# ---------------------------------------------------------------- loading
def load_baseline(regular_dir, biased_dir, model_map):
    reg = read_dir(regular_dir)
    if "condition" not in reg:
        reg["condition"] = "unbiased"
    bia = read_dir(biased_dir)
    if "condition" not in bia:
        bia["condition"] = "biased"
    base = pd.concat([reg, bia], ignore_index=True)
    base["model"] = base["model"].replace(model_map)
    base["qkey"] = make_qkey(base)
    base = base.drop_duplicates(["dataset", "model", "condition", "qkey"])
    return base


def load_panels(panel_dir, model_map):
    df = read_dir(panel_dir)
    df["model"] = df["model"].replace(model_map)
    df["qkey"] = make_qkey(df)
    # condition: biased if a cue column is populated or the filename says so
    cue = df["incorrect_cue"].notna() if "incorrect_cue" in df else False
    fname_bias = df["source_file"].str.contains("bias", case=False)
    df["condition"] = np.where(cue | fname_bias, "biased", "unbiased")
    keep = ["dataset", "model", "condition", "qkey", "reference_answers_json"]
    for p in PANELS:
        keep += [f"{p}_answer", f"{p}_confidence", f"{p}_bet_usd"]
        df[f"{p}_correct"] = [panel_correct(a, r) for a, r in
                              zip(df[f"{p}_answer"], df["reference_answers_json"])]
        keep.append(f"{p}_correct")
    return df[keep].drop_duplicates(["dataset", "model", "condition", "qkey"])


# ---------------------------------------------------------------- summary
def summarize(merged):
    rows = []
    for (ds, model, cond), g in merged.groupby(["dataset", "model", "condition"]):
        row = {"dataset": ds, "model": model, "condition": cond,
               "n_baseline": int(g["judge_correct"].notna().sum()),
               "n_panel": int(g["persona_panel_answer"].notna().sum())
               if "persona_panel_answer" in g else 0}
        for m in BASE_METRICS:
            if m in g:
                row[f"mean_{m}"] = g[m].mean()
        # AUROC of confidence-type measures for predicting correctness
        # (entropies are negated so higher = more confident)
        for m in UNCERTAINTY:
            if m in g:
                sign = -1 if "entropy" in m else 1
                row[f"auroc_{m}"] = auroc(sign * g[m], g["judge_correct"])
        for p in PANELS:
            if f"{p}_answer" not in g:
                continue
            c = g[f"{p}_correct"]
            conf = pd.to_numeric(g[f"{p}_confidence"], errors="coerce")
            bet = pd.to_numeric(g[f"{p}_bet_usd"], errors="coerce")
            row[f"{p}_accuracy"] = c.mean()
            row[f"{p}_mean_confidence"] = conf.mean()
            row[f"{p}_mean_bet_usd"] = bet.mean()
            row[f"{p}_auroc_confidence"] = auroc(conf, c)
            row[f"{p}_auroc_bet"] = auroc(bet, c)
            # confidence rescaled to [0,1] if it looks like a percentage
            cs = conf / 100 if conf.max() and conf.max() > 1 else conf
            row[f"{p}_brier"] = ((cs - c) ** 2).mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "model", "condition"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regular-dir", required=False, default="Demo2_Charlie/figures/regular")
    ap.add_argument("--biased-dir", required=False, default="Demo2_Charlie/figures/biased")
    ap.add_argument("--panel-dir", required=False, default="Demo3_Cheng/results")
    ap.add_argument("--out-dir", default="Demo3_Charlie/results")
    ap.add_argument("--model-map", nargs="*", default=[],
                    help="alias=canonical pairs to reconcile model names across sources")
    a = ap.parse_args()
    model_map = dict(kv.split("=", 1) for kv in a.model_map)
    os.makedirs(a.out_dir, exist_ok=True)

    base = load_baseline(a.regular_dir, a.biased_dir, model_map)
    panels = load_panels(a.panel_dir, model_map)

    key = ["dataset", "model", "condition", "qkey"]
    merged = base.merge(panels, on=key, how="inner", indicator=True)
    
    print(merged["_merge"].value_counts().to_string())
    unmatched = merged[merged["_merge"] != "both"].groupby(["dataset", "model", "condition", "_merge"],
                                                          observed=True).size()
    if len(unmatched):
        print("\nUnmatched rows (check model-name / index alignment):\n", unmatched.to_string())
    merged = merged.drop(columns="_merge")

    summary = summarize(merged)
    print(merged['condition'])
    merged['p_true'] = 1 - merged['p_true']
    merged.to_csv(os.path.join(a.out_dir, "per_example.csv"), index=False)
    summary.to_csv(os.path.join(a.out_dir, "summary.csv"), index=False)
    print(f"\nWrote {len(summary)} summary rows to {a.out_dir}/summary.csv")


if __name__ == "__main__":
    main()