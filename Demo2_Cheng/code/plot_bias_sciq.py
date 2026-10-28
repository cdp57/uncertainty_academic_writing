#!/usr/bin/env python3
"""Create eight presentation plots for the SciQ confirmation-bias experiment.

Reads ONLY existing saved experiment results. No model/GPU is needed.

Run after collection merging:
    python summarize_bias_sciq.py --phase collection --results-dir results
    python plot_bias_sciq.py --phase collection --results-dir results

Run after evaluation merging:
    python summarize_bias_sciq.py --phase evaluation --results-dir results
    python plot_bias_sciq.py --phase evaluation --results-dir results

Or, once both are complete:
    python plot_bias_sciq.py --phase all --results-dir results --pdf

Produces three collection and five evaluation plots under:
    results/confirmation_bias/plots/

Methodological notes:
- Each original SciQ question appears twice (correct cue / incorrect cue), and
  each baseline has five sampled answers. These are paired question variants.
- P(True) has ONE A/B probability per biased-question variant, not five.
- Judge TRUE rates only count successfully parsed judge responses, unlike the
  full-denominator exact-match rates. Parse coverage is reported and visualized.
- Lexical answer agreement and copying a wrong suggested word do not by
  themselves establish semantic correctness, bias endorsement, or calibration.
- The two additional evaluation figures compare biased SciQ with the original,
  unbiased SciQ evaluation for the SAME model and baseline. The three sets
  share SciQ references, but the question wording differs, so the change
  from original also includes rewriting effects.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Safe on SSH/login nodes without a graphical display.
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

MODEL_SPECS = [
    ("llama_1b_sciq_bias", "Llama 1B"),
    ("llama_3b_sciq_bias", "Llama 3B"),
    ("qwen_1p7b_sciq_bias", "Qwen 1.7B"),
    ("qwen_4b_sciq_bias", "Qwen 4B"),
]
MODELS = [x[0] for x in MODEL_SPECS]
MODEL_LABELS = dict(MODEL_SPECS)
METHODS = ["vanilla", "structured", "p_true", "caution"]
METHOD_LABELS = {"vanilla": "Standard", "structured": "Structured",
                 "p_true": "P(True)", "caution": "Caution"}
CONDITIONS = ["biased_correct", "biased_incorrect"]
CONDITION_LABELS = {"biased_correct": "Correct suggestion",
                    "biased_incorrect": "Incorrect suggestion"}
# Stable, legible grouping across plots.
CONDITION_COLORS = {"biased_correct": "#35866F", "biased_incorrect": "#C25A53"}
MODEL_COLORS = dict(zip(MODELS, plt.get_cmap("tab10").colors[:4]))
METHOD_MARKERS = {"vanilla": "o", "structured": "s", "p_true": "^", "caution": "D"}
# Exact original evaluation IDs and baseline names (different from biased summaries).
ORIGINAL_MODEL_IDS = {
    "llama_1b_sciq_bias": "meta-llama/Llama-3.2-1B-Instruct",
    "llama_3b_sciq_bias": "meta-llama/Llama-3.2-3B-Instruct",
    "qwen_1p7b_sciq_bias": "Qwen/Qwen3-1.7B",
    "qwen_4b_sciq_bias": "Qwen/Qwen3-4B",
}
ORIGINAL_METHOD_IDS = {
    "vanilla": "vanilla", "structured": "structured",
    "p_true": "p_true", "caution": "initial_test",
}
ANSWER_JSON_COLUMNS = {
    "vanilla": "vanilla_answers_json",
    "structured": "structured_answers_json",
    "p_true": "p_true_answers_json",
    "caution": "initial_answers_json",
}
CONFIDENCE_JSON_COLUMNS = {
    "vanilla": "vanilla_confidence_json",
    "structured": "structured_confidence_json",
    "caution": "initial_confidence_json",
}
READ_COLS = {"global_question_idx", "bias_condition", "p_true_probability_A"} | \
            set(ANSWER_JSON_COLUMNS.values()) | set(CONFIDENCE_JSON_COLUMNS.values())


def require_file(path: Path, action: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}\n{action}")
    return path


def as_list(cell, field, file):
    try:
        value = json.loads(cell) if isinstance(cell, str) else cell
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Bad JSON in {field} in {file}: {str(cell)[:80]}") from exc
    if not isinstance(value, list) or len(value) != 5:
        raise ValueError(f"Expected five items in {field} in {file}, got {value!r}")
    return value


def clean_number(v, lo=None, hi=None):
    try:
        x = float(v)
    except (ValueError, TypeError):
        return float("nan")
    if not math.isfinite(x):
        return float("nan")
    if lo is not None and x < lo or hi is not None and x > hi:
        return float("nan")
    return x


def normalize_answer(text):
    return " ".join(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def load_collection(root: Path):
    """Read only needed JSON/score columns, skipping large raw generations."""
    frames = []
    for tag, label in MODEL_SPECS:
        file = require_file(
            root / "merged" / f"{tag}_combined.csv",
            "Run: python summarize_bias_sciq.py --phase collection --results-dir results"
        )
        part = pd.read_csv(file, usecols=lambda c: c in READ_COLS,
                           keep_default_na=False, encoding="utf-8-sig")
        missing = READ_COLS - set(part.columns)
        if missing:
            raise ValueError(f"{file}: missing columns {sorted(missing)}")
        if len(part) != 2000 or part["global_question_idx"].nunique() != 1000:
            raise ValueError(f"{file}: expected 2000 rows from 1000 original questions; got {len(part)}")
        if set(part["bias_condition"]) != set(CONDITIONS):
            raise ValueError(f"{file}: expected both bias conditions")
        if part.duplicated(["global_question_idx", "bias_condition"]).any():
            raise ValueError(f"{file}: duplicate original-question/condition pairs")
        part["model"] = tag
        part["model_label"] = label
        part["_file"] = file.name
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def build_collection_stats(df):
    """Compute group summaries at the biased-question level."""
    confidence = []
    ptrue = []
    agreement = []
    for r in df.itertuples(index=False):
        model = r.model
        condition = r.bias_condition
        file = r._file if hasattr(r, "_file") else "merged collection"
        for method, col in CONFIDENCE_JSON_COLUMNS.items():
            items = as_list(getattr(r, col), col, file)
            for value in items:
                score = clean_number(value, 1, 10)
                if math.isfinite(score):
                    confidence.append((model, condition, method, score))
        p = clean_number(r.p_true_probability_A, 0, 1)
        if math.isfinite(p):
            ptrue.append((model, condition, p))
        for method, col in ANSWER_JSON_COLUMNS.items():
            items = as_list(getattr(r, col), col, file)
            text = [normalize_answer(x) for x in items]
            eligible = int(all(text))
            unanimous = int(len(set(text)) == 1) if eligible else None
            agreement.append((model, condition, method, eligible, unanimous))
    conf = pd.DataFrame(confidence, columns=["model", "condition", "baseline", "confidence"])
    ps = pd.DataFrame(ptrue, columns=["model", "condition", "ptrue"])
    agree = pd.DataFrame(agreement,
                         columns=["model", "condition", "baseline", "eligible", "unanimous"])
    if conf.empty or ps.empty or agree.empty:
        raise ValueError("Collection data does not contain valid confidence, P(True), and answer samples")
    return conf, ps, agree


def format_axes(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, alpha=0.20)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save_figure(fig, dest: Path, filename: str, pdf: bool):
    dest.mkdir(parents=True, exist_ok=True)
    png = dest / (filename + ".png")
    fig.savefig(png, dpi=175, bbox_inches="tight")
    if pdf:
        fig.savefig(dest / (filename + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png}")


def plot_confidence(conf, dest, pdf):
    means = conf.groupby(["model", "condition", "baseline"])["confidence"].mean()
    methods = ["vanilla", "structured", "caution"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for ax, tag in zip(axes.flat, MODELS):
        x = np.arange(len(methods))
        for cond, offset in zip(CONDITIONS, [-0.18, 0.18]):
            values = [means.get((tag, cond, m), np.nan) for m in methods]
            bars = ax.bar(x + offset, values, 0.35,
                          color=CONDITION_COLORS[cond], label=CONDITION_LABELS[cond])
            for bar, val in zip(bars, values):
                if math.isfinite(val):
                    ax.text(bar.get_x() + bar.get_width()/2, val + 0.10,
                            f"{val:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(MODEL_LABELS[tag], weight="bold")
        ax.set_xticks(x, [METHOD_LABELS[m] for m in methods])
        ax.set_ylim(0, 10.8)
        format_axes(ax)
    axes[0, 0].set_ylabel("Mean stated confidence (1–10)")
    axes[1, 0].set_ylabel("Mean stated confidence (1–10)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.95), frameon=False)
    fig.suptitle("Does a misleading suggestion change reported confidence?", y=0.995, fontsize=15)
    fig.text(.5, .015, "Each bar pools valid confidence scores across five answer samples per question.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .925))
    save_figure(fig, dest, "bias_collection_confidence", pdf)


def plot_ptrue(ps, dest, pdf):
    means = ps.groupby(["model", "condition"])["ptrue"].mean()
    counts = ps.groupby(["model", "condition"])["ptrue"].count()
    fig, ax = plt.subplots(figsize=(9, 5.8))
    x = np.arange(len(MODELS))
    for cond, offset in zip(CONDITIONS, [-0.18, 0.18]):
        vals = [means.get((tag, cond), np.nan) for tag in MODELS]
        bars = ax.bar(x + offset, vals, 0.35,
                      label=CONDITION_LABELS[cond], color=CONDITION_COLORS[cond])
        for b, value in zip(bars, vals):
            if math.isfinite(value):
                ax.text(b.get_x() + b.get_width()/2, value + .016,
                        f"{value:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x, [MODEL_LABELS[tag] for tag in MODELS])
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Mean P(True), option A probability")
    ax.set_title("P(True): probability assigned to the first brainstormed answer", pad=15)
    ax.legend(loc="upper center", ncol=2, frameon=False)
    format_axes(ax)
    fig.text(.5, .01, "One probability per question/condition (not five); higher means more self-belief, not proven accuracy.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .03, 1, 1))
    save_figure(fig, dest, "bias_collection_ptrue", pdf)
    for tag in MODELS:
        print("  P(True) coverage", MODEL_LABELS[tag],
              {k: int(counts.get((tag,k), 0)) for k in CONDITIONS})


def heatmap(ax, values, title, vmin, vmax, cmap, suffix="", fmt=".1f", labels=True):
    image = ax.imshow(values, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    ax.set_title(title, fontsize=11, weight="bold", pad=10)
    ax.set_xticks(np.arange(len(METHODS)), [METHOD_LABELS[m] for m in METHODS])
    ax.set_yticks(np.arange(len(MODELS)), [MODEL_LABELS[tag] for tag in MODELS])
    ax.tick_params(length=0)
    for i in range(len(MODELS)):
        for j in range(len(METHODS)):
            v = values[i,j]
            if math.isfinite(v):
                rgba = image.cmap(image.norm(v))
                brightness = .2126*rgba[0] + .7152*rgba[1] + .0722*rgba[2]
                ax.text(j,i, f"{v:{fmt}}{suffix}", ha="center", va="center", fontsize=9,
                        color="white" if brightness < .52 else "black")
            else:
                ax.text(j,i,"n/a",ha="center",va="center",fontsize=9)
    return image


def matrix_from(records, value_col, condition):
    lookup = {(r.model, r.baseline): getattr(r, value_col)
              for r in records.itertuples(index=False) if r.condition == condition}
    return np.array([[clean_number(lookup.get((model, method))) for method in METHODS]
                     for model in MODELS], dtype=float)


def plot_consistency(agree, dest, pdf):
    eligible = agree[agree["eligible"] == 1]
    rates = eligible.groupby(["model", "condition", "baseline"])["unanimous"].mean().reset_index()
    rates["rate"] = rates["unanimous"] * 100
    valid_counts = eligible.groupby(["model", "condition", "baseline"]).size()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
    images = []
    for ax, cond in zip(axes, CONDITIONS):
        mat = matrix_from(rates, "rate", cond)
        images.append(heatmap(ax, mat, CONDITION_LABELS[cond], 0, 100, "YlGnBu", "%"))
    fig.colorbar(images[-1], ax=axes, shrink=.74, label="Unanimous questions (%)", pad=.02)
    fig.suptitle("Answer consistency across five samples", fontsize=15, y=1.04)
    fig.text(.5, -.06, "Unanimous = all five normalized generated answers match (valid complete sets only); not accuracy.",
             ha="center", fontsize=10)
    save_figure(fig, dest, "bias_collection_answer_consistency", pdf)
    if len(eligible) != len(agree):
        print(f"  Note: {len(agree)-len(eligible)} of {len(agree)} five-answer sets have missing/unparseable answers")
    if valid_counts.min() < 900:
        print("  WARNING: Some agreement cells have fewer than 900/1000 eligible questions; interpret cautiously")


def load_evaluation(root):
    summ = require_file(root / "summary" / "evaluation_summary.csv",
        "Run: python summarize_bias_sciq.py --phase evaluation --results-dir results")
    pair = require_file(root / "summary" / "paired_bias_effect.csv",
        "Run: python summarize_bias_sciq.py --phase evaluation --results-dir results")
    e = pd.read_csv(summ)
    p = pd.read_csv(pair)
    summary_cols = {"model", "condition", "baseline", "n_answers", "exact_match_accuracy",
                    "token_f1", "self_judge_true_rate_valid_only", "judge_parse_rate",
                    "mentions_incorrect_suggestion_rate"}
    paired_cols = {"model", "baseline", "accuracy_drop_incorrect_minus_correct",
                   "f1_drop_incorrect_minus_correct"}
    for file, data, columns in [(summ, e, summary_cols), (pair, p, paired_cols)]:
        missing = columns - set(data.columns)
        if missing:
            raise ValueError(f"{file}: missing columns: {sorted(missing)}")
    for tag in MODELS:
        for cond in CONDITIONS:
            for method in METHODS:
                sub = e[(e.model==tag)&(e.condition==cond)&(e.baseline==method)]
                if len(sub)!=1:
                    raise ValueError(f"{summ}: expected one summary for {tag}/{cond}/{method}; got {len(sub)}")
        for method in METHODS:
            if len(p[(p.model==tag)&(p.baseline==method)])!=1:
                raise ValueError(f"{pair}: missing/duplicate paired summary for {tag}/{method}")
    counts = pd.to_numeric(e["n_answers"], errors="coerce")
    if counts.isna().any() or (counts <= 0).any():
        raise ValueError("Invalid n_answers in evaluation_summary.csv")
    return e, p


def load_original_sciq(results_dir: Path):
    """Read the unbiased SciQ evaluation used by summarize_all_results.py."""
    file = require_file(
        results_dir / "evaluation" / "summary" / "overall_summary.csv",
        "Run: python summarize_evaluation.py --results-dir results "
        "after completing the ORIGINAL (unbiased) SciQ evaluation"
    )
    original = pd.read_csv(file)
    required = {"model", "dataset", "baseline", "accuracy_exact_match", "token_f1"}
    missing = required - set(original.columns)
    if missing:
        raise ValueError(f"{file}: missing columns {sorted(missing)}")

    # Compare only SciQ, not QASPER. Use the actual original checkpoint model IDs.
    original = original[
        original["dataset"].astype(str).str.casefold().eq("sciq")
    ].copy()
    lookup = {}
    for tag in MODELS:
        for method in METHODS:
            match = original[
                original["model"].eq(ORIGINAL_MODEL_IDS[tag]) &
                original["baseline"].eq(ORIGINAL_METHOD_IDS[method])
            ]
            if len(match) != 1:
                raise ValueError(
                    f"{file}: expected exactly one original SciQ row for "
                    f"{MODEL_LABELS[tag]} / {METHOD_LABELS[method]}, got {len(match)}"
                )
            row = match.iloc[0]
            acc = clean_number(row["accuracy_exact_match"], 0, 1)
            f1 = clean_number(row["token_f1"], 0, 1)
            if not (math.isfinite(acc) and math.isfinite(f1)):
                raise ValueError(f"{file}: invalid original metric for {tag}/{method}")
            lookup[(tag, method)] = (acc, f1)
    return lookup


def plot_vs_original(e, original, dest, pdf, metric):
    """One plot per metric: original rate, and two cue-related changes in pp."""
    options = {
        "accuracy": (
            "exact_match_accuracy", 0,
            "Normalized exact-match accuracy", "bias_vs_original_accuracy"
        ),
        "token_f1": (
            "token_f1", 1,
            "Answer token F1", "bias_vs_original_token_f1"
        ),
    }
    col, metric_index, metric_label, filename = options[metric]
    biased_values = {
        (r.model, r.condition, r.baseline): clean_number(getattr(r, col), 0, 1)
        for r in e.itertuples(index=False)
    }
    original_mat = np.zeros((len(MODELS), len(METHODS)))
    good_delta = np.zeros_like(original_mat)
    bad_delta = np.zeros_like(original_mat)
    for mi, tag in enumerate(MODELS):
        for bi, method in enumerate(METHODS):
            reference = original[(tag, method)][metric_index]
            correct = biased_values.get((tag, "biased_correct", method), float("nan"))
            incorrect = biased_values.get((tag, "biased_incorrect", method), float("nan"))
            if not (math.isfinite(correct) and math.isfinite(incorrect)):
                raise ValueError(f"Missing valid biased metric: {tag} / {method} / {col}")
            original_mat[mi, bi] = reference * 100
            good_delta[mi, bi] = (correct - reference) * 100
            bad_delta[mi, bi] = (incorrect - reference) * 100

    max_abs = max(float(np.max(np.abs(good_delta))), float(np.max(np.abs(bad_delta))))
    limit = min(100, 5 * math.ceil(max(5, max_abs * 1.08) / 5))
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), layout="constrained")
    a0 = heatmap(axes[0], original_mat, "Original question: score", 0, 100,
                 "YlGnBu", "%", ".1f")
    a1 = heatmap(axes[1], good_delta, "Correct cue − original", -limit, limit,
                 "RdBu", " pp", "+.1f")
    a2 = heatmap(axes[2], bad_delta, "Incorrect cue − original", -limit, limit,
                 "RdBu", " pp", "+.1f")
    fig.colorbar(a0, ax=axes[0], orientation="horizontal", shrink=.84,
                 pad=.09, label="Unbiased SciQ baseline (%)")
    fig.colorbar(a2, ax=axes[1:], orientation="horizontal", shrink=.62,
                 pad=.09, label="Change from original (percentage points)")
    fig.suptitle(f"Confirmation-bias prompts vs original SciQ — {metric_label}",
                 fontsize=15, weight="bold")
    fig.text(.5, -.035,
             "Blue positive = higher than original; red negative = lower. "
             "Each cell is averaged over five answers/question. Changes may include wording effects.",
             ha="center", fontsize=10)
    save_figure(fig, dest, filename, pdf)
    print(f"  Largest incorrect-cue shift in {metric_label}: "
          f"{float(np.min(bad_delta)):+.1f} to {float(np.max(bad_delta)):+.1f} pp")


def plot_accuracy(e, dest, pdf):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.3), sharey=True)
    x = np.arange(len(METHODS))
    maximum = pd.to_numeric(e["exact_match_accuracy"], errors="coerce").max() * 100
    ceiling = min(100, max(15, math.ceil((maximum+10)/10)*10))
    for ax, tag in zip(axes.flat, MODELS):
        for cond, off in zip(CONDITIONS, [-.18, .18]):
            sub = e[(e.model==tag) & (e.condition==cond)].set_index("baseline")
            vals = [float(sub.loc[method, "exact_match_accuracy"])*100 for method in METHODS]
            bars = ax.bar(x+off, vals, .35,
                          color=CONDITION_COLORS[cond], label=CONDITION_LABELS[cond])
            for bar, value in zip(bars, vals):
                ax.annotate(f"{value:.1f}", (bar.get_x()+bar.get_width()/2, value),
                            xytext=(0,3), textcoords="offset points", ha="center", fontsize=7)
        ax.set_title(MODEL_LABELS[tag], weight="bold")
        ax.set_ylim(0, ceiling)
        ax.set_xticks(x, [METHOD_LABELS[m] for m in METHODS])
        format_axes(ax)
    axes[0,0].set_ylabel("Normalized exact-match accuracy (%)")
    axes[1,0].set_ylabel("Normalized exact-match accuracy (%)")
    fig.suptitle("Correct vs incorrect answer suggestions: reference accuracy", fontsize=15, y=.995)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",ncol=2,bbox_to_anchor=(.5,.954),frameon=False)
    fig.text(.5,.013,"Across all five samples per question; reference is the original SciQ correct answer.",
             ha="center",fontsize=10)
    fig.tight_layout(rect=(0,.04,1,.927))
    save_figure(fig,dest,"bias_evaluation_exact_accuracy",pdf)


def plot_paired_deltas(pair, e, dest, pdf):
    p = pair.copy()
    # Incorrect cue minus correct cue; negative means worse under incorrect suggestion.
    p["condition"] = "biased_incorrect"
    p["acc_change_pp"] = pd.to_numeric(p["accuracy_drop_incorrect_minus_correct"], errors="coerce")*100
    p["f1_change_pp"] = pd.to_numeric(p["f1_drop_incorrect_minus_correct"], errors="coerce")*100
    matrix_acc = matrix_from(p,"acc_change_pp","biased_incorrect")
    matrix_f1 = matrix_from(p,"f1_change_pp","biased_incorrect")
    e_wrong = e[e["condition"] == "biased_incorrect"].copy()
    e_wrong["wrong_mention_pct"] = pd.to_numeric(e_wrong["mentions_incorrect_suggestion_rate"],errors="coerce")*100
    matrix_wrong = matrix_from(e_wrong,"wrong_mention_pct","biased_incorrect")
    bound = max(10, np.nanmax(np.abs(np.concatenate([matrix_acc.ravel(),matrix_f1.ravel()]))))
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.5), layout="constrained")
    im0 = heatmap(axes[0],matrix_acc,"Accuracy change (pp)",-bound,bound,"RdBu"," pp","+.1f")
    im1 = heatmap(axes[1],matrix_f1,"Token F1 change (pp)",-bound,bound,"RdBu"," pp","+.1f")
    im2 = heatmap(axes[2],matrix_wrong,"Incorrect suggestion mentioned (%)",0,100,"Oranges","%")
    fig.colorbar(im0, ax=axes[:2], orientation="horizontal", shrink=.78, pad=.12,
                 label="Incorrect cue minus correct cue (percentage points)")
    fig.colorbar(im2, ax=axes[2], orientation="horizontal", shrink=.78, pad=.12,
                 label="Lexical mention rate (%)")
    fig.suptitle("Paired effects of an incorrect suggestion",y=1.045,fontsize=15)
    fig.text(.5,-.07,"Negative deltas: worse reference-match performance with an incorrect cue. Mentioning a term does NOT imply endorsing it.",
             ha="center",fontsize=10)
    save_figure(fig,dest,"bias_evaluation_paired_effects",pdf)


def plot_judge(e, dest, pdf):
    sub = e[e["condition"] == "biased_incorrect"].copy()
    fig, (ax, hx) = plt.subplots(1, 2, figsize=(13.7, 6.2), gridspec_kw={"width_ratios":[1.04,.96]})
    # Scatter compares different denominators: exact match all saved answers;
    # judge TRUE rate valid parsed judgments only. Circle size shows coverage.
    for model in MODELS:
        part = sub[sub["model"] == model]
        for r in part.itertuples(index=False):
            x = clean_number(r.exact_match_accuracy, 0, 1)*100
            y = clean_number(r.self_judge_true_rate_valid_only, 0, 1)*100
            coverage = clean_number(r.judge_parse_rate, 0, 1)
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(coverage)):
                continue
            ax.scatter(x,y,color=MODEL_COLORS[model],marker=METHOD_MARKERS[r.baseline],
                       s=30+170*coverage,alpha=.86,edgecolors="white",linewidths=.8)
            if r.baseline=="vanilla":
                ax.annotate(MODEL_LABELS[model],(x,y),xytext=(4,5),
                            textcoords="offset points",fontsize=8)
    ax.plot([0,100],[0,100],ls="--",lw=1.5,color="gray",alpha=.8)
    ax.set_xlim(-3,103); ax.set_ylim(-3,103)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Reference exact-match accuracy (%)")
    ax.set_ylabel("Self-judge says TRUE (valid judgments only, %)")
    ax.set_title("Incorrect cue: self-judgment vs reference")
    format_axes(ax)
    model_handles = [Line2D([0],[0],marker="o",markersize=8,linestyle="None",
                            markerfacecolor=MODEL_COLORS[m],markeredgecolor="white",
                            label=MODEL_LABELS[m]) for m in MODELS]
    method_handles = [Line2D([0],[0],marker=METHOD_MARKERS[m],markersize=8,
                             linestyle="None",markerfacecolor="#666666",color="#666666",
                             label=METHOD_LABELS[m]) for m in METHODS]
    ax.legend(handles=model_handles+method_handles, fontsize=8,
              loc="upper left",bbox_to_anchor=(0.02,1.03),framealpha=.9,ncol=2)

    sub["judge_parse_pct"] = pd.to_numeric(sub["judge_parse_rate"],errors="coerce")*100
    matrix_coverage = matrix_from(sub,"judge_parse_pct","biased_incorrect")
    im=heatmap(hx,matrix_coverage,"Judge verdict parsing coverage",0,100,"YlGnBu","%")
    fig.colorbar(im,ax=hx,shrink=.7,pad=.03,label="Valid judge outputs (%)")
    fig.suptitle("Self-judging under misleading suggestions",fontsize=15,y=.995)
    fig.text(.5,.016,"Dashed line: equal aggregate rates, not per-answer agreement. Point size reflects parse coverage; low coverage limits comparisons.",
             ha="center",fontsize=9)
    fig.subplots_adjust(left=.08,right=.94,top=.88,bottom=.14,wspace=.30)
    save_figure(fig,dest,"bias_evaluation_self_judgment",pdf)
    low=sub[pd.to_numeric(sub["judge_parse_rate"],errors="coerce") < .8]
    if not low.empty:
        print("WARNING: some judge TRUE rates use fewer than 80% valid verdicts:")
        for r in low.itertuples(index=False):
            print(f"  {MODEL_LABELS[r.model]} {METHOD_LABELS[r.baseline]}: {r.judge_parse_rate:.1%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir",type=Path,
                    default=Path(__file__).resolve().parent / "results")
    ap.add_argument("--phase",choices=["collection","evaluation","all"],default="all")
    ap.add_argument("--pdf",action="store_true",help="Save vector PDFs in addition to PNGs")
    args=ap.parse_args()
    root=args.results_dir / "confirmation_bias"
    out=root / "plots"
    if args.phase in ("collection","all"):
        collected=load_collection(root)
        confidence,ptrue,agreement=build_collection_stats(collected)
        plot_confidence(confidence,out,args.pdf)
        plot_ptrue(ptrue,out,args.pdf)
        plot_consistency(agreement,out,args.pdf)
    if args.phase in ("evaluation","all"):
        e,p=load_evaluation(root)
        plot_accuracy(e,out,args.pdf)
        plot_paired_deltas(p,e,out,args.pdf)
        plot_judge(e,out,args.pdf)
        # Compare exactly matched model/baseline scores to the original SciQ run.
        original = load_original_sciq(args.results_dir)
        plot_vs_original(e, original, out, args.pdf, "accuracy")
        plot_vs_original(e, original, out, args.pdf, "token_f1")
    print("COMPLETE. Plot folder:",out)

if __name__=="__main__":
    main()
