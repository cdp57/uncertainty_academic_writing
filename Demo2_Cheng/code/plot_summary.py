#!/usr/bin/env python3
"""Visualize Demo 2 *collection* results, before accuracy or judge evaluation.

Run from the 447_demo2 project root after `python merge_results.py`:

    python plot_summary.py --results-dir results
    python plot_summary.py --results-dir results --pdf

Reads only results/merged/ inputs. Writes figures to results/plots/.
Never loads any LLM, downloads datasets, saves support text, or uses gold labels.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = ["llama_1b", "llama_3b", "qwen_1p7b", "qwen_4b"]
DATASET_ORDER = ["qasper", "sciq"]
SAMPLE_SEEDS = [42, 123, 2026, 3407, 5678]
METHODS = {
    "vanilla": "Standard self-report",
    "structured": "Structured self-report",
    "caution": "Overconfidence warning",
}
ANSWER_COLUMNS = {
    "vanilla": "vanilla_answers_json",
    "structured": "structured_answers_json",
    "p_true": "p_true_answers_json",
    "caution": "initial_answers_json",
}
CONFIDENCE_COLUMNS = {
    "vanilla": "vanilla_confidence_json",
    "structured": "structured_confidence_json",
    "caution": "initial_confidence_json",
}
ALL_ANSWER_METHODS = {
    "vanilla": "Standard",
    "structured": "Structured",
    "p_true": "P(True) ideas",
    "caution": "Caution",
}
COLORS = dict(zip(ANSWER_COLUMNS, plt.get_cmap("tab10").colors[:4]))
FILE_PATTERN = re.compile(
    r"^(llama_(?:1b|3b)|qwen_(?:1p7b|4b))_(sciq|qasper)_combined\.csv$"
)


def nice_model(model: str) -> str:
    return {
        "llama_1b": "Llama 1B", "llama_3b": "Llama 3B",
        "qwen_1p7b": "Qwen 1.7B", "qwen_4b": "Qwen 4B",
    }.get(str(model), str(model))


def nice_dataset(ds: str) -> str:
    return "SciQ" if ds == "sciq" else "QASPER" if ds == "qasper" else str(ds)


def nice_label(model: str, ds: str) -> str:
    return f"{nice_model(model)} / {nice_dataset(ds)}"


def ordered_keys(keys):
    return sorted(set(keys), key=lambda k: (
        MODEL_ORDER.index(k[0]) if k[0] in MODEL_ORDER else 100,
        DATASET_ORDER.index(k[1]) if k[1] in DATASET_ORDER else 100,
        k,
    ))


def as_num(value, min_value=None, max_value=None):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(f):
        return np.nan
    if min_value is not None and f < min_value:
        return np.nan
    if max_value is not None and f > max_value:
        return np.nan
    return f


def json_array(value):
    if isinstance(value, list):
        return value
    try:
        arr = json.loads(value)
        return arr if isinstance(arr, list) else []
    except (TypeError, ValueError):
        return []


def norm_answer(s):
    """Lexical normalization only; different strings can mean the same thing."""
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    return " ".join(re.findall(r"\w+", s.casefold(), flags=re.UNICODE))


def safe_mean(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else np.nan


def safe_sd(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.std(ddof=1)) if arr.size >= 2 else np.nan


def annotate_heatmap(ax, array, format_spec=".2f", center=None):
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return
    lo, hi = finite.min(), finite.max()
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            v = array[i, j]
            if not np.isfinite(v):
                txt, c = "–", "black"
            else:
                txt = format(v, format_spec)
                midpoint = (lo + hi) / 2 if center is None else center
                c = "white" if v > midpoint and (center is None) else "black"
            ax.text(j, i, txt, ha="center", va="center", color=c, fontsize=8)


def plot_heatmap(array, row_labels, col_labels, title, name, save, vmin=None,
                 vmax=None, cmap="YlGnBu", fmt=".2f", center=None):
    if array.size == 0:
        return
    fig, ax = plt.subplots(figsize=(max(6, len(col_labels)*1.45+2),
                                     max(4, len(row_labels)*0.43+1.5)))
    im = ax.imshow(np.ma.masked_invalid(array), cmap=cmap, aspect="auto",
                   vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(col_labels)), col_labels)
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.set_title(title, pad=12)
    annotate_heatmap(ax, array, fmt, center=center)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.025)
    fig.tight_layout()
    save(fig, name)


def grouped_bars(df, cols, labels, title, ylabel, filename, save, ylim=None):
    """Grouped bars for one row per model/dataset, no error bars."""
    if df.empty:
        return
    x = np.arange(len(df))
    width = 0.8 / len(cols)
    fig, ax = plt.subplots(figsize=(max(11, 1.38*len(df)), 6))
    for i, col in enumerate(cols):
        arr = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        ax.bar(x + (i - (len(cols)-1)/2)*width, arr, width,
               label=labels[i], color=COLORS.get(col.split("_")[0]))
    ax.set_xticks(x, [nice_label(r.model, r.dataset) for r in df.itertuples()],
                  rotation=32, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.axhline(0, linewidth=0.7, color="gray", alpha=0.5)
    ax.legend(loc="best", fontsize=9)
    ax.grid(axis="y", alpha=0.17)
    fig.tight_layout()
    save(fig, filename)


def load_question_summaries(merged_dir):
    """Read five-sample JSON arrays only from the 8 merged collection CSVs."""
    question_records = []
    files = sorted(merged_dir.glob("*_combined.csv"))
    matched = 0
    required = (set(ANSWER_COLUMNS.values()) | set(CONFIDENCE_COLUMNS.values()) |
                {"p_true_probability_A"})
    for path in files:
        match = FILE_PATTERN.fullmatch(path.name)
        if not match:
            continue
        model, dataset = match.groups()
        head = pd.read_csv(path, nrows=0)
        missing = required - set(head.columns)
        if missing:
            print(f"Warning: skipping {path.name} because columns are missing: {sorted(missing)}")
            continue
        matched += 1
        # Never read or re-export the potentially long `support` column.
        frame = pd.read_csv(path, usecols=lambda c: c in required,
                            keep_default_na=False, dtype=str)
        for r in frame.to_dict("records"):
            q = {"model": model, "dataset": dataset,
                 "ptrue_prob": as_num(r.get("p_true_probability_A"), 0, 1)}
            for method, col in CONFIDENCE_COLUMNS.items():
                samples = json_array(r.get(col))
                numbers = np.array([as_num(x, 1, 10) for x in samples[:5]], dtype=float)
                if len(numbers) < 5:
                    numbers = np.pad(numbers, (0, 5-len(numbers)), constant_values=np.nan)
                valid = numbers[np.isfinite(numbers)]
                q[f"{method}_conf_mean"] = safe_mean(numbers)
                q[f"{method}_conf_sd"] = safe_sd(numbers)
                q[f"{method}_conf_parse"] = len(valid)/5
                q[f"{method}_high_conf"] = float(np.mean(valid >= 9)) if valid.size else np.nan
            for method, col in ANSWER_COLUMNS.items():
                answers = json_array(r.get(col))[:5]
                parsed = [norm_answer(a) for a in answers]
                valid = [a for a in parsed if a]
                q[f"{method}_answer_parse"] = len(valid)/5
                q[f"{method}_unique_count"] = float(len(set(valid))) if len(valid) >= 2 else np.nan
                q[f"{method}_consensus"] = (
                    max(Counter(valid).values())/len(valid) if len(valid) >= 2 else np.nan
                )
                # Only compare full five-sample sets for unanimity.
                q[f"{method}_unanimous"] = (
                    float(len(set(valid)) == 1) if len(valid) == 5 else np.nan
                )
            q["caution_minus_vanilla"] = (
                q["caution_conf_mean"] - q["vanilla_conf_mean"]
            )
            q["structured_minus_vanilla"] = (
                q["structured_conf_mean"] - q["vanilla_conf_mean"]
            )
            question_records.append(q)
        print(f"Loaded {len(frame)} pre-evaluation questions from {path.name}")
    if matched == 0:
        print("No compatible *_combined.csv files found; only the two original summary plots can be produced.")
    return pd.DataFrame(question_records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        default=Path(__file__).resolve().parent/"results")
    parser.add_argument("--pdf", action="store_true", help="Also save vector PDF figures")
    args = parser.parse_args()
    folder = args.results_dir / "merged"
    out = args.results_dir / "plots"
    out.mkdir(parents=True, exist_ok=True)
    if not folder.is_dir():
        parser.error(f"Missing merged results directory: {folder}. Run merge_results.py first.")

    saved = []
    def save(fig, stem):
        fig.savefig(out / f"{stem}.png", dpi=180, bbox_inches="tight")
        if args.pdf:
            fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
        saved.append(stem)
        plt.close(fig)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})

    # ------------------------------------------------------------------
    # Preserve the existing two outputs and their definitions.
    # ------------------------------------------------------------------
    a_path = folder / "across_sample_seed_mean_std.csv"
    p_path = folder / "ptrue_question_summary.csv"
    if not a_path.exists() or not p_path.exists():
        parser.error("Missing across_sample_seed_mean_std.csv or ptrue_question_summary.csv. "
                     "Run python merge_results.py --results-dir results first.")
    summary = pd.read_csv(a_path)
    ptrue = pd.read_csv(p_path)
    keys = ordered_keys(zip(summary.model, summary.dataset))
    summary = (summary.set_index(["model", "dataset"]).reindex(keys).reset_index())
    p_keys = ordered_keys(zip(ptrue.model, ptrue.dataset))
    ptrue = ptrue.set_index(["model", "dataset"]).reindex(p_keys).reset_index()
    labels = [nice_label(r.model, r.dataset) for r in summary.itertuples()]
    x = np.arange(len(summary))
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(11, 1.42*len(summary)), 6))
    for i, (method, title) in enumerate(METHODS.items()):
        col = f"{method}_mean_confidence_1to10"
        means = pd.to_numeric(summary[f"{col}_mean"], errors="coerce").to_numpy(float)
        stds = pd.to_numeric(summary[f"{col}_std"], errors="coerce").fillna(0).to_numpy(float)
        ax.bar(x + (i-1)*width, means, width, yerr=stds, label=title,
               capsize=3, color=COLORS[method])
    ax.set_ylabel("Mean confidence (1–10)")
    ax.set_title("Self-report confidence: mean and SD across five sample seeds")
    ax.set_xticks(x, labels, rotation=32, ha="right")
    ax.set_ylim(0, 10.5)
    ax.grid(axis="y", alpha=0.15)
    ax.legend()
    fig.tight_layout()
    save(fig, "self_report_confidence_five_seeds")

    labels = [nice_label(r.model, r.dataset) for r in ptrue.itertuples()]
    x = np.arange(len(ptrue))
    means = pd.to_numeric(ptrue.ptrue_mean_prob_A, errors="coerce")
    stds = pd.to_numeric(ptrue.ptrue_std_across_questions, errors="coerce").fillna(0)
    fig, ax = plt.subplots(figsize=(max(11, 1.38*len(ptrue)), 5.5))
    ax.bar(x, means, yerr=stds, capsize=3, color=COLORS["p_true"])
    ax.set_ylabel("P(A=True), normalized from A/B sequence likelihoods")
    ax.set_title("P(True): question-level mean and SD (one score per question)")
    ax.set_xticks(x, labels, rotation=32, ha="right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    save(fig, "ptrue_probability_question_sd")

    # ------------------------------------------------------------------
    # Seed means from merge_results.py. Each seed denotes one answer slot
    # per question, NOT an independent rerun of the full experiment.
    # ------------------------------------------------------------------
    seed_path = folder / "per_sample_seed_summary.csv"
    if seed_path.exists():
        seed_df = pd.read_csv(seed_path)
        seed_rows = [(m, d, meth) for m, d in keys for meth in METHODS]
        matrix = np.full((len(seed_rows), len(SAMPLE_SEEDS)), np.nan)
        for i, (model, ds, method) in enumerate(seed_rows):
            sub = seed_df[(seed_df.model == model) & (seed_df.dataset == ds)]
            for j, seed in enumerate(SAMPLE_SEEDS):
                rec = sub[pd.to_numeric(sub.sample_seed, errors="coerce") == seed]
                if not rec.empty:
                    matrix[i, j] = as_num(rec.iloc[0].get(f"{method}_mean_confidence_1to10"), 1, 10)
        plot_heatmap(matrix,
                     [f"{nice_label(m, d)} • {METHODS[meth]}" for m,d,meth in seed_rows],
                     [str(s) for s in SAMPLE_SEEDS],
                     "Mean confidence by sample seed (each seed used once per question)",
                     "confidence_by_sample_seed", save, vmin=1, vmax=10, fmt=".1f")
    else:
        print(f"Warning: missing {seed_path}; skipping seed-profile heatmap")

    confidence_heat = np.array([
        [as_num(summary.loc[i, f"{m}_mean_confidence_1to10_mean"], 1, 10)
         for m in METHODS] for i in range(len(summary))
    ])
    plot_heatmap(confidence_heat,
                 [nice_label(r.model, r.dataset) for r in summary.itertuples()],
                 ["Standard", "Structured", "Caution"],
                 "Mean self-reported confidence by baseline (1–10)",
                 "self_report_confidence_heatmap", save, vmin=1, vmax=10, fmt=".1f")

    p_matrix = np.full((len(MODEL_ORDER), len(DATASET_ORDER)), np.nan)
    for i, model in enumerate(MODEL_ORDER):
        for j, dataset in enumerate(DATASET_ORDER):
            sub = ptrue[(ptrue.model == model) & (ptrue.dataset == dataset)]
            if not sub.empty:
                p_matrix[i, j] = as_num(sub.iloc[0].ptrue_mean_prob_A, 0, 1)
    plot_heatmap(p_matrix, [nice_model(m) for m in MODEL_ORDER],
                 [nice_dataset(d) for d in DATASET_ORDER],
                 "P(True) mean probability by model and dataset",
                 "ptrue_probability_heatmap", save, vmin=0, vmax=1, fmt=".2f")

    questions = load_question_summaries(folder)
    if not questions.empty:
        keys = ordered_keys(zip(questions.model, questions.dataset))
        group = questions.groupby(["model", "dataset"], sort=False)
        aggregated = group.mean(numeric_only=True).reindex(keys).reset_index()
        plot_labels = [nice_label(m,d) for m,d in keys]

        # 1. Answer and confidence parsing coverage; missing values are visible.
        grouped_bars(aggregated,
                     [f"{m}_answer_parse" for m in ANSWER_COLUMNS],
                     list(ALL_ANSWER_METHODS.values()),
                     "Parsed answer coverage across five sampled responses",
                     "Fraction of answer slots parsed", "answer_parse_coverage", save, ylim=(0, 1.05))
        grouped_bars(aggregated,
                     [f"{m}_conf_parse" for m in METHODS],
                     list(METHODS.values()),
                     "Parsed confidence coverage across five sample seeds",
                     "Fraction of confidence slots parsed", "confidence_parse_coverage", save, ylim=(0, 1.05))

        # 2. Frequency of high numerical self-reports, conditional on parsing.
        grouped_bars(aggregated, [f"{m}_high_conf" for m in METHODS],
                     list(METHODS.values()),
                     "How often does the model report high confidence (9–10)?",
                     "Mean per-question fraction among parsed confidence scores", "high_confidence_fraction", save,
                     ylim=(0, 1.05))

        # 3. SD WITHIN each question's five seeded confidence answers.
        grouped_bars(aggregated, [f"{m}_conf_sd" for m in METHODS],
                     list(METHODS.values()),
                     "Within-question variability across five confidence samples",
                     "Mean question-level SD (1–10 scale)", "confidence_within_question_sd", save)

        # 4. Confidence shifts, paired within question (not answer correctness).
        delta_cols = ["structured_minus_vanilla", "caution_minus_vanilla"]
        delta_labels = ["Structured − standard", "Caution − standard"]
        delta_arr = aggregated[delta_cols].to_numpy(float)
        bound = max(1.0, np.nanmax(np.abs(delta_arr)) if np.isfinite(delta_arr).any() else 1)
        plot_heatmap(delta_arr, plot_labels, delta_labels,
                     "Change in mean confidence relative to standard self-report",
                     "prompt_confidence_shifts", save, vmin=-bound, vmax=bound,
                     cmap="RdBu_r", fmt="+.2f", center=0)

        # 5. Distinct answers and exact lexical unanimity across five seeds.
        # These are NOT semantic agreement or correctness measures.
        grouped_bars(aggregated,
                     [f"{m}_unique_count" for m in ANSWER_COLUMNS],
                     list(ALL_ANSWER_METHODS.values()),
                     "Mean number of distinct answer strings across five samples",
                     "Mean distinct normalized strings (1–5)",
                     "answer_lexical_diversity", save, ylim=(0, 5.3))
        grouped_bars(aggregated,
                     [f"{m}_unanimous" for m in ANSWER_COLUMNS],
                     list(ALL_ANSWER_METHODS.values()),
                     "Five-sample answer unanimity (exact lexical agreement)",
                     "Fraction of fully parsed questions with 5/5 identical strings",
                     "answer_lexical_unanimity", save, ylim=(0, 1.05))

        # 6. P(True) distributions: SD bars above hide distribution shape.
        datasets = [d for d in DATASET_ORDER if (questions.dataset == d).any()]
        fig, axes = plt.subplots(1, len(datasets), figsize=(6.1*len(datasets), 5.3),
                                 squeeze=False, sharey=True)
        for ax, ds in zip(axes[0], datasets):
            series, names = [], []
            for model in MODEL_ORDER:
                subset = questions[(questions.model == model) & (questions.dataset == ds)]
                vals = subset.ptrue_prob.dropna().to_numpy(float)
                if vals.size:
                    series.append(vals)
                    names.append(nice_model(model))
            if series:
                ax.boxplot(series, showfliers=False,
                           medianprops={"color": "black", "linewidth": 1.7})
                ax.set_xticks(range(1, len(names) + 1), names, rotation=25)
            ax.set_title(nice_dataset(ds))
            ax.set_ylim(-0.02, 1.02)
            ax.grid(axis="y", alpha=0.18)
            ax.set_ylabel("Question-level P(A=True)")
        fig.suptitle("P(True) distribution across questions (median and IQR)")
        fig.tight_layout()
        save(fig, "ptrue_probability_distribution")

        # 7. Relationship between perceived probability and self-report.
        # Both are model scores, NOT validated accuracy or calibration.
        for ds in datasets:
            fig, ax = plt.subplots(figsize=(8, 6))
            count = 0
            for model in MODEL_ORDER:
                sub = questions[(questions.model == model) & (questions.dataset == ds)]
                xvals = sub.vanilla_conf_mean.to_numpy(float) / 10
                yvals = sub.ptrue_prob.to_numpy(float)
                valid = np.isfinite(xvals) & np.isfinite(yvals)
                xvals, yvals = xvals[valid], yvals[valid]
                if len(xvals) < 2:
                    continue
                rho = float(np.corrcoef(xvals, yvals)[0, 1]) if (
                    np.std(xvals) > 0 and np.std(yvals) > 0) else np.nan
                corr_text = f"r={rho:.2f}" if np.isfinite(rho) else "r=n/a"
                ax.scatter(xvals, yvals, s=9, alpha=0.16,
                           label=f"{nice_model(model)} ({corr_text}, n={len(xvals)})")
                count += 1
            if not count:
                plt.close(fig)
                continue
            ax.set_xlim(-.02, 1.02)
            ax.set_ylim(-.02, 1.02)
            ax.set_xlabel("Question-level mean standard self-report confidence / 10")
            ax.set_ylabel("Question-level P(A=True)")
            ax.set_title(f"Relationship between two confidence estimates • {nice_dataset(ds)}")
            ax.grid(alpha=0.15)
            ax.legend(fontsize=9, markerscale=2.0)
            fig.tight_layout()
            save(fig, f"ptrue_vs_vanilla_confidence_{ds}")

        # 8. Does P(True) rise when five brainstormed answers have more lexical agreement?
        for ds in datasets:
            fig, ax = plt.subplots(figsize=(8, 5.6))
            found = False
            for model in MODEL_ORDER:
                sub = questions[(questions.model == model) & (questions.dataset == ds)]
                agg = (sub.groupby("p_true_consensus", dropna=True)
                         .ptrue_prob.agg(["mean", "count"]).reset_index())
                agg = agg[(agg["count"] >= 5) & agg["mean"].notna()]
                if not agg.empty:
                    ax.plot(agg.p_true_consensus, agg["mean"], "o-", linewidth=1.5,
                            label=nice_model(model))
                    found = True
            if not found:
                plt.close(fig)
                continue
            ax.set_xlim(0.18, 1.02)
            ax.set_ylim(0, 1.03)
            ax.set_xticks([.2, .4, .6, .8, 1.0])
            ax.set_xlabel("Largest share of identical normalized brainstormed answers")
            ax.set_ylabel("Mean P(A=True)")
            ax.set_title(f"P(True) vs lexical agreement among 5 brainstorms • {nice_dataset(ds)}")
            ax.grid(alpha=.17)
            ax.legend(fontsize=9)
            fig.tight_layout()
            save(fig, f"ptrue_vs_brainstorm_consensus_{ds}")

    print(f"Saved {len(saved)} pre-evaluation plot(s) to: {out.resolve()}")
    for stem in saved:
        print(" -", stem + ".png" + (" (+ PDF)" if args.pdf else ""))
    print("NOTE: These figures describe confidence, answer-string agreement, and parsing;")
    print("they do not use ground-truth accuracy, F1, or LLM-as-a-Judge verdicts.")
    print("NOTE: Error bars in the original confidence chart are SD across sample-seed means;")
    print("P(True) error bars are SD across QUESTIONS, not seeds.")


if __name__ == "__main__":
    main()
