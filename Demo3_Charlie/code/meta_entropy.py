"""
Meta-entropy: predict answer correctness from uncertainty metrics with a logistic regression.

For every (dataset, condition, model) and each correctness target (LLM judge, ROUGE-L), runs 5-fold CV on
  * "all_entropy"     - the four entropy metrics together,
  * "all_confidence"  - self-report (vanilla, structured) and P(True) together,
  * "all_persona"     - the persona-elicited confidences together (speculative editor, opposing counsel,
                        compliance officer, persona panel),
  * "all_no_persona"  - entropy + confidence (everything except the persona confidences),
  * "all_metrics"     - everything together (entropy + confidence + persona),
  * one single-metric model per metric (baselines, to show what combining adds),
and writes AUROC / AURC plus the fitted model weights to one CSV. Comparing "all_metrics" with
"all_no_persona" (same rows, same folds) isolates what the persona confidences add.

The persona confidences come from separate persona-prompted generations, but here they are used as predictors
of the same target as every other metric: whether the greedy answer is correct.

Missing values: rows are never dropped for a missing feature. Gaps are filled with the median of the training
rows inside each CV fold (the imputer is part of the sklearn pipeline, so test rows never influence it). Only rows
with no correctness label are dropped. Features covered in fewer than --min-coverage of a group's rows are still
excluded, since an almost fully imputed feature is just a constant.

Folds: for QASPER, questions are grouped by paper (StratifiedGroupKFold on paper_idx), so no paper is split
across train and test (questions from one paper share context and are not independent). Datasets with
independent items (SciQ) use plain StratifiedKFold. Conditions (e.g. biased / unbiased) are analyzed
separately and never pooled: they have very different accuracy, and in SciQ the same questions appear in both.

Input: per_example.csv (columns dataset, model, condition, judge_correct and/or rouge_l_correct, the metric
columns, the *_confidence persona columns, and paper_idx for QASPER). Files without a condition column still work.
"""
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))

ENTROPY_FEATURES = [
    "semantic_entropy",
    "discrete_semantic_entropy",
    "predictive_entropy",
    "length_normalized_predictive_entropy",
]
CONFIDENCE_FEATURES = ["vanilla_self_report", "structured_self_report", "p_true_probability"]
# money_betting_confidence is left out on purpose: that persona reports a bet (money_betting_bet_usd), not a
# confidence, and the confidence column is empty in per_example.csv. Pass it explicitly to include it.
PERSONA_FEATURES = [
    "speculative_editor_confidence",
    "opposing_counsel_confidence",
    "compliance_officer_confidence",
    "persona_panel_confidence",
    "money_betting_bet_usd"
]
TARGETS = {"judge": "judge_correct", "rouge": "rouge_l_correct"}
MODEL_ORDER = ["llama3.2_1b", "qwen3_1.7b", "llama3.2_3b", "qwen3_4b"]


def parse_args():
    p = argparse.ArgumentParser(description="Logistic-regression meta-model over uncertainty metrics (5-fold CV)")
    p.add_argument("--input", type=str, default="Demo3_Charlie/results/per_example.csv",
                   help="Per-example CSV (dataset, model, condition, correctness, metric and persona columns).")
    p.add_argument("--output", type=str, default="Demo3_Charlie/results/meta_entropy_results.csv")
    p.add_argument("--entropy-features", nargs="+", default=ENTROPY_FEATURES)
    p.add_argument("--confidence-features", nargs="+", default=CONFIDENCE_FEATURES,
                   help="Self-report / P(True) columns. Skipped per group if missing or poorly covered.")
    p.add_argument("--persona-features", nargs="*", default=PERSONA_FEATURES,
                   help="Persona-elicited confidence columns. Skipped per group if missing or poorly covered. "
                        "Pass an empty list (--persona-features) to run without them.")
    p.add_argument("--condition-col", type=str, default="condition",
                   help="Column holding the experimental condition (e.g. biased / unbiased). Each value is "
                        "analyzed separately. Ignored if the column is absent.")
    p.add_argument("--conditions", nargs="+", default=None,
                   help="Only run these condition values (default: all found).")
    p.add_argument("--min-coverage", type=float, default=0.5,
                   help="Drop a feature for a (dataset, condition, model) if fewer than this fraction of rows have "
                        "it. Missing values of the kept features are median-imputed within each fold, but a feature "
                        "that is mostly imputed is close to a constant and only adds noise.")
    p.add_argument("--group-col", type=str, default="paper_idx",
                   help="Column whose values must not be split across train/test folds (default: paper_idx).")
    p.add_argument("--no-group", action="store_true",
                   help="Use plain StratifiedKFold everywhere (ignores --group-col).")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--C", type=float, default=1.0, help="Inverse L2 regularization strength.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ------------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------------
def compute_auroc(y, p):
    """AUROC for predicting a correct answer from predicted P(correct)."""
    if len(y) == 0 or len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def compute_aurc(y, p):
    """
    Area under the accuracy-coverage curve: answers sorted from most to least confident (highest P(correct)
    first). Higher is better; a random ranking scores about the overall accuracy. Same definition as the
    plotting script. The classic risk-based AURC is 1 minus this value.
    """
    n = len(y)
    if n == 0:
        return np.nan
    sorted_y = np.asarray(y, dtype=float)[np.argsort(-np.asarray(p), kind="stable")]
    coverages = np.linspace(1.0 / n, 1.0, n)
    accuracies = np.cumsum(sorted_y) / np.arange(1, n + 1)
    return float(trapz_fn(accuracies, coverages))


# ------------------------------------------------------------------------------
# Model + CV
# ------------------------------------------------------------------------------
def make_model(C):
    # Missing feature values are filled with the median of the *training* rows (the imputer is a pipeline step,
    # so it is re-fit inside every CV fold and never sees test rows). keep_empty_features keeps the column count
    # fixed if a feature is entirely missing in a training fold (it is then filled with 0 and carries no signal).
    # Standardizing makes the weights comparable across metrics on very different scales
    # (predictive entropy ~20, length-normalized PE ~0.5, confidence 1-10) and keeps the L2 penalty fair.
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         StandardScaler(),
                         LogisticRegression(C=C, max_iter=1000))


def extract_weights(pipe):
    """Weights on standardized features, the same weights in raw feature units, and both intercepts.
    "Raw" means the units of the imputed feature values (the imputer only fills gaps, it does not rescale)."""
    scaler, lr = pipe[1], pipe[2]
    w_std, b_std = lr.coef_[0], float(lr.intercept_[0])
    w_raw = w_std / scaler.scale_
    b_raw = b_std - float(np.sum(w_std * scaler.mean_ / scaler.scale_))
    return w_std, b_std, w_raw, b_raw


def make_splitter(folds, seed, grouped):
    if grouped:
        return StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)


def cross_validate(X, y, groups, folds, seed, C):
    splitter = make_splitter(folds, seed, groups is not None)
    oof = np.full(len(y), np.nan)
    fold_auroc, fold_aurc = [], []
    for train, test in splitter.split(X, y, groups):
        pipe = make_model(C).fit(X[train], y[train])
        p = pipe.predict_proba(X[test])[:, 1]
        oof[test] = p
        fold_auroc.append(compute_auroc(y[test], p))
        fold_aurc.append(compute_aurc(y[test], p))
    return oof, np.array(fold_auroc), np.array(fold_aurc)


def run_feature_set(sub, y, groups, feats, all_feats, args):
    X = sub[feats].to_numpy(dtype=float)
    oof, f_auroc, f_aurc = cross_validate(X, y, groups, args.folds, args.seed, args.C)
    aurc_pooled = compute_aurc(y, oof)

    row = {
        "features": "|".join(feats),
        "n_correct": int(y.sum()), "base_rate": float(y.mean()), "folds": args.folds,
        "auroc_mean": float(np.nanmean(f_auroc)), "auroc_std": float(np.nanstd(f_auroc, ddof=1)),
        "auroc_pooled": compute_auroc(y, oof),
        "aurc_mean": float(np.nanmean(f_aurc)), "aurc_std": float(np.nanstd(f_aurc, ddof=1)),
        "aurc_pooled": aurc_pooled, "aurc_risk_pooled": 1.0 - aurc_pooled,
    }

    # Final weights: fit on all rows of this (dataset, condition, model, target)
    w_std, b_std, w_raw, b_raw = extract_weights(make_model(args.C).fit(X, y))
    row["intercept_std"], row["intercept_raw"] = b_std, b_raw
    for f in all_feats:
        row[f"w_std_{f}"] = row[f"w_raw_{f}"] = np.nan
    for f, ws, wr in zip(feats, w_std, w_raw):
        row[f"w_std_{f}"], row[f"w_raw_{f}"] = float(ws), float(wr)
    return row


def build_feature_sets(entropy, confidence, persona):
    sets = []
    if len(entropy) > 1:
        sets.append(("all_entropy", entropy))
    if len(confidence) > 1:
        sets.append(("all_confidence", confidence))
    if len(persona) > 1:
        sets.append(("all_persona", persona))
    # Ablation pair: with vs without the persona confidences (identical rows and folds)
    if entropy and confidence and persona:
        sets.append(("all_no_persona", entropy + confidence))
    if sum(bool(g) for g in (entropy, confidence, persona)) > 1:
        sets.append(("all_metrics", entropy + confidence + persona))
    sets += [(f, [f]) for f in entropy + confidence + persona]
    return sets


def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    for col in ("dataset", "model"):
        if col not in df.columns:
            raise KeyError(f"'{col}' column missing from {args.input}")

    entropy_all = [f for f in args.entropy_features if f in df.columns]
    confidence_all = [f for f in args.confidence_features if f in df.columns]
    persona_all = [f for f in args.persona_features if f in df.columns]
    for name, req, have in [("entropy", args.entropy_features, entropy_all),
                            ("confidence", args.confidence_features, confidence_all),
                            ("persona", args.persona_features, persona_all)]:
        gone = [f for f in req if f not in have]
        if gone:
            print(f"Skipping {name} features not in the CSV: {gone}")
    if not entropy_all and not confidence_all and not persona_all:
        raise ValueError("None of the requested feature columns exist in the input CSV.")
    all_feats = entropy_all + confidence_all + persona_all

    # Conditions (e.g. biased / unbiased) are analyzed separately, never pooled
    group_keys = ["dataset", "model"]
    has_condition = args.condition_col in df.columns
    if has_condition:
        if args.conditions:
            df = df[df[args.condition_col].isin(args.conditions)]
            if df.empty:
                raise ValueError(f"No rows left after filtering {args.condition_col} to {args.conditions}.")
        group_keys.append(args.condition_col)
    elif args.conditions:
        print(f"--conditions ignored: no '{args.condition_col}' column in {args.input}")

    rows = []
    for key, g in df.groupby(group_keys, sort=False):
        ds, model = key[0], key[1]
        cond = key[2] if has_condition else "all"
        tag = f"{ds}/{cond}/{model}"
        # Features with too little coverage in this group would force most rows to be dropped
        keep, dropped = [], []
        for f in all_feats:
            cov = g[f].notna().mean()
            (keep if cov >= args.min_coverage else dropped).append((f, cov))
        entropy = [f for f, _ in keep if f in entropy_all]
        confidence = [f for f, _ in keep if f in confidence_all]
        persona = [f for f, _ in keep if f in persona_all]
        excluded = "|".join(f"{f}({c:.0%})" for f, c in dropped)
        if dropped:
            print(f"[{tag}] excluded for low coverage: {excluded}")
        if not entropy and not confidence and not persona:
            continue
        feats = entropy + confidence + persona
        feature_sets = build_feature_sets(entropy, confidence, persona)

        for target, tcol in TARGETS.items():
            if tcol not in g.columns:
                continue
            avail = int(g[tcol].notna().sum())
            # Only rows without a label are dropped. Missing feature values are median-imputed inside each fold,
            # and every feature set uses the same rows, so comparisons between sets are fair.
            sub = g.dropna(subset=[tcol]).reset_index(drop=True)
            n_missing_rows = int(sub[feats].isna().any(axis=1).sum())
            y = (sub[tcol].to_numpy(dtype=float) >= 0.5).astype(int)
            n_pos, n_neg = int(y.sum()), int((1 - y).sum())
            if min(n_pos, n_neg) < args.folds:
                print(f"[{tag}/{target}] skipped: {n_pos} correct / {n_neg} incorrect "
                      f"(need at least {args.folds} of each for {args.folds}-fold CV)")
                continue

            groups, scheme, n_groups = None, "StratifiedKFold", np.nan
            if (not args.no_group and args.group_col in sub.columns and sub[args.group_col].notna().all()
                    and sub[args.group_col].nunique() >= args.folds):
                groups = sub[args.group_col].to_numpy()
                scheme = f"StratifiedGroupKFold({args.group_col})"
                n_groups = int(sub[args.group_col].nunique())

            print(f"[{tag}/{target}] n={len(sub)} ({n_missing_rows} with >=1 imputed feature) | CV: {scheme}"
                  + (f", {n_groups} groups" if groups is not None else ""))
            for name, fs in feature_sets:
                rows.append({"dataset": ds, "condition": cond, "model": model, "target": target,
                             "feature_set": name,
                             "n": len(sub), "n_rows_with_label": avail, "n_rows_imputed": n_missing_rows,
                             "cv_scheme": scheme,
                             "n_groups": n_groups, "features_excluded_low_coverage": excluded,
                             **run_feature_set(sub, y, groups, fs, all_feats, args)})

    if not rows:
        print("No (dataset, condition, model, target) group had enough data; nothing written.")
        return

    out = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    out = out.sort_values(["dataset", "condition", "model", "target"],
                          key=lambda s: s.map(order).fillna(99) if s.name == "model" else s, kind="stable")
    meta_cols = ["dataset", "condition", "model", "target", "feature_set", "features", "n", "n_rows_with_label",
                 "n_rows_imputed", "n_correct",
                 "base_rate", "cv_scheme", "n_groups", "folds", "features_excluded_low_coverage",
                 "auroc_mean", "auroc_std", "auroc_pooled", "aurc_mean", "aurc_std", "aurc_pooled",
                 "aurc_risk_pooled", "intercept_std", "intercept_raw"]
    weight_cols = [f"w_std_{f}" for f in all_feats] + [f"w_raw_{f}" for f in all_feats]
    out = out[meta_cols + weight_cols]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\nSaved {len(out)} rows to: {args.output}\n")

    # Console summary: AUROC for every feature-set combination, one block per (dataset, condition, model, target)
    name_w = max(len(s) for s in out.feature_set)
    for (ds, cond, model, target), grp in out.groupby(["dataset", "condition", "model", "target"], sort=False):
        print(f"{ds}/{cond}/{model}/{target}  (n={int(grp.n.iloc[0])}, "
              f"{int(grp.n_rows_imputed.iloc[0])} rows with imputed values, acc {grp.base_rate.iloc[0]:.3f})")
        for _, r in grp.iterrows():
            print(f"    {r.feature_set:<{name_w}}  AUROC pooled {r.auroc_pooled:.3f} | "
                  f"fold mean {r.auroc_mean:.3f} +/- {r.auroc_std:.3f}")
        print()


if __name__ == "__main__":
    main()