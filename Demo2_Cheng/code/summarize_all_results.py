#!/usr/bin/env python3
"""Write a compact, presentation-ready CSDS 447 Demo 2 report.

Reads existing pre-evaluation and evaluation results; does not load any models.

Usage:
    python summarize_all_results.py --results-dir results
    python summarize_all_results.py --results-dir results --quiet

Outputs in the SAME DIRECTORY AS THIS SCRIPT:
    demo2_results_summary.txt
    demo2_results_summary.md

Required for evaluation sections:
    results/evaluation/summary/overall_summary.csv
Optional pre-evaluation inputs:
    results/merged/across_sample_seed_mean_std.csv
    results/merged/ptrue_question_summary.csv
    results/merged/*_combined.csv

This script intentionally does not report individual sample-seed comparisons,
seed-by-seed statistics, or standard deviations across sample positions.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

MODEL_IDS = ["llama_1b", "llama_3b", "qwen_1p7b", "qwen_4b"]
MODEL_LABELS = {
    "llama_1b": "Llama 1B", "llama_3b": "Llama 3B",
    "qwen_1p7b": "Qwen 1.7B", "qwen_4b": "Qwen 4B",
}
MODEL_ALIASES = {
    "meta-llama/Llama-3.2-1B-Instruct": "llama_1b",
    "meta-llama/Llama-3.2-3B-Instruct": "llama_3b",
    "Qwen/Qwen3-1.7B": "qwen_1p7b",
    "Qwen/Qwen3-4B": "qwen_4b",
}
DATASETS = ("qasper", "sciq")
BASELINES = ("vanilla", "structured", "p_true", "initial_test")
BASELINE_LABELS = {
    "vanilla": "Standard", "structured": "Structured",
    "p_true": "P(True)", "initial_test": "Caution",
}
COMBINED_FILE_RE = re.compile(
    r"^(llama_(?:1b|3b)|qwen_(?:1p7b|4b))_(sciq|qasper)_combined\.csv$"
)


def numeric(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else float("nan")
    except (ValueError, TypeError):
        return float("nan")


def is_finite(value):
    return math.isfinite(numeric(value))


def decimal(value, digits=3):
    return f"{numeric(value):.{digits}f}" if is_finite(value) else "n/a"


def pct(value, digits=1):
    return f"{numeric(value) * 100:.{digits}f}%" if is_finite(value) else "n/a"


def pp(value, digits=1):
    return f"{numeric(value) * 100:+.{digits}f} pp" if is_finite(value) else "n/a"


def score(value):
    return f"{numeric(value):.2f}" if is_finite(value) else "n/a"


def model_name(value):
    key = MODEL_ALIASES.get(str(value), str(value))
    return MODEL_LABELS.get(key, key)


def dataset_name(value):
    return "SciQ" if str(value).lower() == "sciq" else "QASPER"


def order(frame, include_baseline=False):
    if frame.empty:
        return frame
    out = frame.copy()
    out["_m"] = out["model"].map({m: i for i, m in enumerate(MODEL_IDS)}).fillna(99)
    out["_d"] = out["dataset"].map({d: i for i, d in enumerate(DATASETS)}).fillna(99)
    keys = ["_m", "_d"]
    if include_baseline:
        out["_b"] = out["baseline"].map({b: i for i, b in enumerate(BASELINES)}).fillna(99)
        keys.append("_b")
    return out.sort_values(keys).drop(columns=keys)


class Report:
    def __init__(self):
        self.text, self.markdown = [], []

    def heading(self, value, level=1):
        self.markdown.extend(["#" * level + " " + value, ""])
        self.text.extend([value, ("=" if level == 1 else "-") * len(value), ""])

    def paragraph(self, value):
        self.markdown.extend([value, ""])
        self.text.extend([value, ""])

    def bullet(self, value):
        self.markdown.append("- " + value)
        self.text.append("  - " + value)

    def end_bullets(self):
        self.markdown.append("")
        self.text.append("")

    def table(self, columns, rows):
        if not rows:
            self.paragraph("Data unavailable for this section.")
            return
        body = [[str(x).replace("\n", " ") for x in row] for row in rows]
        columns = [str(x) for x in columns]
        esc = lambda s: s.replace("|", "\\|")
        self.markdown.append("| " + " | ".join(map(esc, columns)) + " |")
        self.markdown.append("| " + " | ".join("---" for _ in columns) + " |")
        for row in body:
            self.markdown.append("| " + " | ".join(map(esc, row)) + " |")
        self.markdown.append("")
        widths = [max(len(row[i]) for row in [columns] + body) for i in range(len(columns))]
        self.text.append("  ".join(x.ljust(w) for x, w in zip(columns, widths)))
        self.text.append("  ".join("-" * w for w in widths))
        for row in body:
            self.text.append("  ".join(x.ljust(w) for x, w in zip(row, widths)))
        self.text.append("")

    def output(self):
        return ("\n".join(self.text).rstrip() + "\n",
                "\n".join(self.markdown).rstrip() + "\n")


def load_optional(path):
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "model" in df:
        df["model"] = df["model"].astype(str).replace(MODEL_ALIASES)
    if "dataset" in df:
        df["dataset"] = df["dataset"].astype(str).str.lower()
    return df


def load_data(root):
    ev = load_optional(root / "evaluation" / "summary" / "overall_summary.csv")
    if ev.empty:
        for alternative in (root / "evaluation" / "overall_summary.csv", root / "overall_summary.csv"):
            ev = load_optional(alternative)
            if not ev.empty:
                break
    conf = load_optional(root / "merged" / "across_sample_seed_mean_std.csv")
    ptrue = load_optional(root / "merged" / "ptrue_question_summary.csv")
    return ev, conf, ptrue


def validated_eval(ev):
    if ev.empty:
        return ev
    required = {"model", "dataset", "baseline", "accuracy_exact_match", "token_f1",
                "judge_true_rate", "judge_parse_rate", "mean_judge_confidence_1to10"}
    missing = required - set(ev.columns)
    if missing:
        raise ValueError(f"Evaluation summary missing columns: {sorted(missing)}")
    return order(ev, include_baseline=True)


def read_sample_list(text):
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return []
    return value[:5] if isinstance(value, list) else []


def canonical_answer(value):
    return " ".join(re.findall(r"\w+", str(value).casefold())) if value is not None else ""


def load_consistency(root):
    """Optional aggregate agreement across answer samples, without saving answers."""
    records = []
    fields = {
        "vanilla": "vanilla_answers_json",
        "structured": "structured_answers_json",
        "p_true": "p_true_answers_json",
        "initial_test": "initial_answers_json",
    }
    for file in sorted((root / "merged").glob("*_combined.csv")):
        match = COMBINED_FILE_RE.fullmatch(file.name)
        if not match:
            continue
        model, dataset = match.groups()
        cols = set(pd.read_csv(file, nrows=0).columns)
        present = {method: field for method, field in fields.items() if field in cols}
        if not present:
            continue
        # Read each merged file once, using only the four answer columns.
        data = pd.read_csv(file, usecols=list(present.values()), dtype=str, keep_default_na=False)
        for row in data.to_dict("records"):
            for method, field in present.items():
                answers = [canonical_answer(x) for x in read_sample_list(row[field])]
                good = [x for x in answers if x]
                records.append({
                    "model": model, "dataset": dataset, "baseline": method,
                    "n_distinct": len(set(good)) if len(good) >= 2 else float("nan"),
                    "unanimous": int(len(set(good)) == 1) if len(good) == 5 else float("nan"),
                    "parse_rate": len(good) / 5,
                })
    return pd.DataFrame(records)


def eval_lookup(ev):
    return {(str(row.model), str(row.dataset), str(row.baseline)): row
            for row in ev.itertuples(index=False)}


def model_rows(ev, baseline="vanilla"):
    return order(ev[ev["baseline"] == baseline]).to_dict("records") if not ev.empty else []


def section_overview(r, ev):
    r.heading("1. Vanilla baseline — reference metrics", 2)
    r.paragraph("This is the straightforward baseline. Values are proportions from 0 to 1, "
                "aggregated across the five generated answers per question.")
    rows = [(model_name(x["model"]), dataset_name(x["dataset"]),
             decimal(x["accuracy_exact_match"]), decimal(x["token_f1"]))
            for x in model_rows(ev)]
    r.table(["Model", "Dataset", "Accuracy (exact match)", "Token F1"], rows)


def section_baselines(r, ev):
    r.heading("2. How the four baselines compare", 2)
    r.paragraph("Higher is better for the two reference-based scores below. "
                "Standard = one-prompt answer + confidence; Structured = separate answer and confidence; "
                "P(True) = brainstormed answers with a separate truth-probability step; "
                "Caution = a prompt warning against overconfidence.")
    lookup = eval_lookup(ev)
    for metric, title in (("accuracy_exact_match", "Exact-match accuracy"),
                          ("token_f1", "Answer token F1")):
        r.heading(title, 3)
        rows = []
        for model in MODEL_IDS:
            for ds in DATASETS:
                if not any((model, ds, b) in lookup for b in BASELINES):
                    continue
                rows.append([model_name(model), dataset_name(ds)] + [
                    decimal(getattr(lookup[(model, ds, b)], metric))
                    if (model, ds, b) in lookup else "n/a" for b in BASELINES
                ])
        r.table(["Model", "Dataset", "Standard", "Structured", "P(True)", "Caution"], rows)

    r.heading("Best-performing baseline in each configuration", 3)
    winners = []
    for model in MODEL_IDS:
        for ds in DATASETS:
            group = [x for x in BASELINES if (model, ds, x) in lookup and
                     is_finite(lookup[(model, ds, x)].accuracy_exact_match)]
            if not group or (model, ds, "vanilla") not in lookup:
                continue
            best = max(group, key=lambda x: numeric(lookup[(model, ds, x)].accuracy_exact_match))
            best_val = numeric(lookup[(model, ds, best)].accuracy_exact_match)
            base = numeric(lookup[(model, ds, "vanilla")].accuracy_exact_match)
            winners.append((model_name(model), dataset_name(ds), BASELINE_LABELS[best],
                            decimal(best_val), pp(best_val-base)))
    r.table(["Model", "Dataset", "Best exact-match baseline", "Accuracy", "Gain vs standard"], winners)
    return lookup


def section_pre(r, conf, prob, consistency):
    r.heading("3. Pre-evaluation: what the models reported", 2)
    if not conf.empty:
        required = {"model", "dataset", "vanilla_mean_confidence_1to10_mean",
                    "structured_mean_confidence_1to10_mean", "caution_mean_confidence_1to10_mean"}
        if required <= set(conf.columns):
            r.heading("Mean reported confidence (1–10)", 3)
            r.paragraph("These numbers measure how confident the models said they were. "
                        "High confidence does not automatically mean high accuracy.")
            rows = [
                (model_name(x["model"]), dataset_name(x["dataset"]),
                 score(x["vanilla_mean_confidence_1to10_mean"]),
                 score(x["structured_mean_confidence_1to10_mean"]),
                 score(x["caution_mean_confidence_1to10_mean"]))
                for x in order(conf).to_dict("records")
            ]
            r.table(["Model", "Dataset", "Standard", "Structured", "Caution"], rows)
    if not prob.empty:
        required = {"model", "dataset", "ptrue_mean_prob_A"}
        if required <= set(prob.columns):
            r.heading("P(True): average assigned probability", 3)
            r.paragraph("P(True) is the model's estimated probability for option A (True) "
                        "when evaluating its first brainstormed answer. It is NOT measured correctness.")
            rows = []
            for x in order(prob).to_dict("records"):
                valid = numeric(x.get("ptrue_valid_questions"))
                total = numeric(x.get("question_count"))
                coverage = pct(valid / total) if total > 0 and is_finite(valid) else "n/a"
                rows.append((model_name(x["model"]), dataset_name(x["dataset"]),
                             decimal(x["ptrue_mean_prob_A"]), coverage))
            r.table(["Model", "Dataset", "Mean P(True)", "Available questions"], rows)
    if not consistency.empty:
        r.heading("Answer consistency across five responses", 3)
        r.paragraph("Shows the standard baseline only. 'Unanimous' means all five normalized "
                    "answers were identical, not necessarily correct.")
        grp = consistency[consistency.baseline == "vanilla"].groupby(["model", "dataset"], as_index=False).agg(
            distinct=("n_distinct", "mean"), unanimous=("unanimous", "mean"),
            parsed=("parse_rate", "mean"))
        rows = [(model_name(x["model"]), dataset_name(x["dataset"]),
                 score(x["distinct"]), pct(x["unanimous"]), pct(x["parsed"]))
                for x in order(grp).to_dict("records")]
        r.table(["Model", "Dataset", "Different answers / 5", "Unanimous", "Answer parse rate"], rows)
    if conf.empty and prob.empty and consistency.empty:
        r.paragraph("Pre-evaluation summary inputs were not found; this section is omitted.")


def section_judging(r, ev):
    r.heading("4. LLM-as-a-Judge: does the model approve its own answers?", 2)
    r.paragraph("Each model judges its own answers. The table uses the standard baseline "
                "so it is easy to compare with Section 1.")
    rows = []
    for x in model_rows(ev):
        difference = numeric(x["judge_true_rate"]) - numeric(x["accuracy_exact_match"])
        rows.append((model_name(x["model"]), dataset_name(x["dataset"]),
                     pct(x["accuracy_exact_match"]), pct(x["judge_true_rate"]),
                     pp(difference), pct(x["judge_parse_rate"]),
                     score(x["mean_judge_confidence_1to10"])))
    r.table(["Model", "Dataset", "Exact match", "Judge says TRUE", "Difference",
             "Valid judge outputs", "Judge confidence"], rows)
    r.paragraph("A positive difference means the model labeled more answers TRUE than exact "
                "matching counted as correct. It does not by itself show that the model judged "
                "incorrectly; correct paraphrases may fail exact matching. Judge TRUE rates "
                "exclude responses whose verdict could not be parsed.")


def section_observations(r, ev, conf, prob, consistency):
    r.heading("5. Main observations — talking points for the demo", 2)
    if ev.empty:
        r.bullet("Evaluation results are missing; run summarize_evaluation.py first.")
        r.end_bullets()
        return
    vanilla = ev[ev.baseline == "vanilla"].copy()
    if vanilla.empty:
        return
    for ds in ("sciq", "qasper"):
        sub = vanilla[vanilla.dataset == ds].copy()
        sub = sub[pd.to_numeric(sub.accuracy_exact_match, errors="coerce").notna()]
        if sub.empty:
            continue
        winner = sub.loc[sub.accuracy_exact_match.astype(float).idxmax()]
        r.bullet(f"{dataset_name(ds)}: {model_name(winner['model'])} had the highest standard-baseline "
                 f"exact-match accuracy ({pct(winner['accuracy_exact_match'])}) and token F1 "
                 f"of {pct(winner['token_f1'])}.")

    by_model = vanilla.set_index(["model", "dataset"])
    drops = []
    for model in MODEL_IDS:
        if (model, "sciq") in by_model.index and (model, "qasper") in by_model.index:
            sci = numeric(by_model.loc[(model, "sciq"), "accuracy_exact_match"])
            qas = numeric(by_model.loc[(model, "qasper"), "accuracy_exact_match"])
            if is_finite(sci) and is_finite(qas):
                drops.append((model, sci-qas))
    if drops and all(x > 0 for _, x in drops):
        avg = sum(x for _, x in drops) / len(drops)
        r.bullet(f"QASPER was harder under the exact-match measure: every model scored lower "
                 f"than on SciQ (average drop {pp(avg).removeprefix('+')}). "
                 "QASPER's longer, more varied answers also make exact matching stricter.")

    for family, small, large in (("Llama", "llama_1b", "llama_3b"),
                                  ("Qwen", "qwen_1p7b", "qwen_4b")):
        gains = []
        for ds in ("sciq", "qasper"):
            if (small, ds) in by_model.index and (large, ds) in by_model.index:
                a = numeric(by_model.loc[(small, ds), "accuracy_exact_match"])
                b = numeric(by_model.loc[(large, ds), "accuracy_exact_match"])
                if is_finite(a) and is_finite(b):
                    gains.append((ds, b-a))
        if gains:
            r.bullet(f"{family} model size: the larger model's standard-baseline accuracy "
                     + "; ".join(f"changed by {pp(diff)} on {dataset_name(ds)}" for ds, diff in gains) + ".")

    look = eval_lookup(ev)
    for method in ("structured", "p_true", "initial_test"):
        deltas = []
        for (model, ds, baseline), item in look.items():
            if baseline != method or (model, ds, "vanilla") not in look:
                continue
            d = numeric(item.accuracy_exact_match) - numeric(look[(model, ds, "vanilla")].accuracy_exact_match)
            if is_finite(d):
                deltas.append(d)
        if deltas:
            wins = sum(d > 1e-12 for d in deltas)
            ties = sum(abs(d) <= 1e-12 for d in deltas)
            r.bullet(f"{BASELINE_LABELS[method]} vs standard: higher exact-match accuracy "
                     f"in {wins}/{len(deltas)} model–dataset comparisons"
                     + (f" ({ties} ties)" if ties else "") + ". It is not a universal improvement.")

    strong = vanilla[pd.to_numeric(vanilla.judge_parse_rate, errors="coerce") >= 0.90].copy()
    if not strong.empty:
        strong["gap"] = pd.to_numeric(strong.judge_true_rate, errors="coerce") - pd.to_numeric(strong.accuracy_exact_match, errors="coerce")
        strong = strong[strong["gap"].notna()]
        if not strong.empty:
            top = strong.loc[strong.gap.idxmax()]
            r.bullet(f"Self-judging can be lenient: on {dataset_name(top['dataset'])}, "
                     f"{model_name(top['model'])} labeled {pct(top['judge_true_rate'])} of "
                     f"its valid standard-baseline judgments TRUE, versus {pct(top['accuracy_exact_match'])} "
                     "reference exact match. This gap needs semantic/human review before calling it judge error.")

    parsed = vanilla[pd.to_numeric(vanilla.judge_parse_rate, errors="coerce").notna()]
    if not parsed.empty:
        worst = parsed.loc[pd.to_numeric(parsed.judge_parse_rate).idxmin()]
        if numeric(worst.judge_parse_rate) < 0.9:
            r.bullet(f"Judge format reliability matters: {model_name(worst['model'])} on "
                     f"{dataset_name(worst['dataset'])} produced valid standard-baseline verdicts "
                     f"for only {pct(worst['judge_parse_rate'])} of the evaluated answers.")

    if not conf.empty and {"model", "dataset", "vanilla_mean_confidence_1to10_mean"} <= set(conf.columns):
        joined = vanilla.merge(conf[["model", "dataset", "vanilla_mean_confidence_1to10_mean"]],
                               on=["model", "dataset"], how="inner")
        joined = joined[joined["vanilla_mean_confidence_1to10_mean"].notna()]
        if not joined.empty:
            # A strong illustrative case: highest self-reported confidence on QASPER.
            qas = joined[joined.dataset == "qasper"]
            if not qas.empty:
                x = qas.loc[qas.vanilla_mean_confidence_1to10_mean.astype(float).idxmax()]
                r.bullet(f"Reported confidence is not accuracy: on QASPER, "
                         f"{model_name(x['model'])} reported {score(x['vanilla_mean_confidence_1to10_mean'])}/10 "
                         f"average standard-baseline confidence while exact-match accuracy was "
                         f"{pct(x['accuracy_exact_match'])}.")

    if not prob.empty and {"model", "dataset", "ptrue_mean_prob_A"} <= set(prob.columns):
        p = prob[pd.to_numeric(prob.ptrue_mean_prob_A, errors="coerce").notna()]
        if not p.empty:
            maximum = p.loc[p.ptrue_mean_prob_A.astype(float).idxmax()]
            r.bullet(f"P(True) scores also need calibration: the largest mean assigned probability "
                     f"was {decimal(maximum['ptrue_mean_prob_A'])} for "
                     f"{model_name(maximum['model'])} on {dataset_name(maximum['dataset'])}. "
                     "That score is a model estimate, not a measured accuracy rate.")

    if not consistency.empty:
        cs = consistency[consistency.baseline == "vanilla"].groupby(["model", "dataset"], as_index=False)["unanimous"].mean()
        cs = cs[pd.to_numeric(cs.unanimous, errors="coerce").notna()]
        if not cs.empty:
            high = cs.loc[cs.unanimous.idxmax()]
            r.bullet(f"Sampling consistency is visible even without checking correctness: "
                     f"{model_name(high['model'])} on {dataset_name(high['dataset'])} gave the same "
                     f"normalized standard-baseline answer in all five responses for "
                     f"{pct(high['unanimous'])} of eligible questions.")
    r.end_bullets()


def section_notes(r):
    r.heading("6. How to interpret the report", 2)
    r.bullet("Accuracy (exact match): the generated answer must match a reference after "
             "normalizing the text. A correct synonym or paraphrase may be counted wrong.")
    r.bullet("Token F1: how much the answer's words overlap with the reference. It is "
             "more flexible than exact match, but it cannot fully judge scientific meaning.")
    r.bullet("Self-judge TRUE rate: how often a model says its own answer is correct. "
             "This is NOT independent accuracy; valid judge-output coverage matters.")
    r.bullet("Confidence and P(True): what the model reports or assigns, not verified "
             "probabilities of being correct. Compare them with reference scores carefully.")
    r.bullet("Each baseline produced five answers per question. The tables aggregate those "
             "answers; they do not represent five separate experiments over the dataset.")
    r.bullet("QASPER token F1 measures answer overlap only (not official evidence F1). "
             "Long-form answers make wording-based comparisons especially limited.")
    r.bullet("All generation and judging used supporting context; the report and result "
             "tables do not include that support text.")
    r.end_bullets()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--quiet", action="store_true", help="Save without printing the full report")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    directory = args.results_dir.expanduser().resolve()
    evaluation, confidence, ptrue = load_data(directory)
    evaluation = validated_eval(evaluation)
    consistency = load_consistency(directory)
    if evaluation.empty and confidence.empty and ptrue.empty and consistency.empty:
        parser.error(f"No compatible result files found in {directory}")

    r = Report()
    r.heading("CSDS 447 Demo 2 — Results Summary")
    r.paragraph(f"Generated {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')} | Results: {directory}")
    r.paragraph("Models: Llama 3.2 1B/3B and Qwen3 1.7B/4B. Datasets: SciQ test "
                "and QASPER validation. All questions include supporting context.")
    if not evaluation.empty:
        section_overview(r, evaluation)
        section_baselines(r, evaluation)
    else:
        r.heading("Reference evaluation", 2)
        r.paragraph("Evaluation overall_summary.csv was not found; accuracy, F1 and self-judge tables are unavailable.")
    section_pre(r, confidence, ptrue, consistency)
    if not evaluation.empty:
        section_judging(r, evaluation)
        section_observations(r, evaluation, confidence, ptrue, consistency)
    section_notes(r)

    plain, md = r.output()
    (root / "demo2_results_summary.txt").write_text(plain, encoding="utf-8")
    (root / "demo2_results_summary.md").write_text(md, encoding="utf-8")
    if not args.quiet:
        print(plain)
    print(f"Saved {root / 'demo2_results_summary.txt'}")
    print(f"Saved {root / 'demo2_results_summary.md'}")


if __name__ == "__main__":
    main()
