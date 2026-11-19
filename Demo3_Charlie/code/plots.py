import os
import argparse
import glob
import json
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# NumPy 2.0+ compatibility for trapezoid integration
trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))

# Parameter-scaled X-axis mapping for Kuhn et al. style plots
MODEL_ORDER = [
    ("llama3.2_1b", "1B (Llama)", 1.0),
    ("qwen3_1.7b", "1.7B (Qwen)", 1.7),
    ("llama3.2_3b", "3B (Llama)", 3.0),
    ("qwen3_4b", "4B (Qwen)", 4.0),
]

# Aliases to match filenames/metadata to canonical model keys
MODEL_ALIASES = {
    "llama3.2_1b": ["llama3.2_1b", "llama32_1b", "llama_1b", "llama-1b", "llama1b"],
    "qwen3_1.7b": ["qwen3_1.7b", "qwen3_17b", "qwen_1.7b", "qwen_17b", "qwen17b"],
    "llama3.2_3b": ["llama3.2_3b", "llama32_3b", "llama_3b", "llama-3b", "llama3b"],
    "qwen3_4b": ["qwen3_4b", "qwen_4b", "qwen-4b", "qwen4b"],
}

# (column, display label, marker, color, higher-value-means-more-uncertain)
ENTROPY_MEASURES = [
    ("semantic_entropy", "Semantic Entropy", "s", "#1f77b4", True),
    ("discrete_semantic_entropy", "Discrete SE", "P", "#17becf", True),
    ("length_normalized_predictive_entropy", "LN-PE", "v", "#ff7f0e", True),
    ("predictive_entropy", "Predictive Entropy", "o", "#2ca02c", True),
]

CONFIDENCE_MEASURES = [
    ("vanilla_self_report", "Vanilla Self-Report", "^", "#9467bd", False),
    ("structured_self_report", "Structured Self-Report", "D", "#8c564b", False),
    ("p_true_probability", "P(True)", "x", "#d62728", False),
]

ALL_MEASURES = ENTROPY_MEASURES + CONFIDENCE_MEASURES

# Columns the LLM judge script adds to the entropy CSV
EVAL_SPECIFIC_COLS = ["correctness", "correct_answer_used", "greedy_response"]
JUDGE_COLUMNS = ["judge_verdict", "correctness"]
REFERENCE_COLUMNS = ["ground_truth", "correct_answer", "reference_answer"]
ANSWER_COLUMNS = ["greedy_response", "model_response"]  # greedy first: that is what the judge grades
ID_COLUMNS = ["paper_idx", "question_idx", "item_idx"]

# Self-report confidence lives in the 'combined' CSVs, with one value per sampled run (5 runs per question).
# For each measure: the per-run parsed list, the per-run raw model text (used to rescue values the original
# parser missed), and the legacy ' || '-joined string column.
CONFIDENCE_SOURCES = {
    "vanilla_self_report": {
        "label": "vanilla",
        "json": "vanilla_confidence_json",
        "raw": "vanilla_raw_json",
        "strings": ["vanilla_baseline_all_confidence", "vanilia_baseline_all_confidence"],
    },
    "structured_self_report": {
        "label": "structured",
        "json": "structured_confidence_json",
        "raw": "structured_raw_confidence_json",
        "strings": ["structured_self-report_all_confidence"],
    },
}
CONFIDENCE_RANGE = (1, 10)  # self-reported confidence is on a 1-10 scale
# "... my confidence in the answer as 7", "confidence level is 8 out of 10", "CONFIDENCE: 9", "10/10"
RAW_CONFIDENCE_PATTERN = re.compile(r"(?i)confiden\w*[^0-9\n%]{0,80}?(\d{1,2})(?![\d%]|\.\d)")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot AUROC, AURC, ROUGE-L, and Entropy metrics across datasets (Kuhn et al. style)")
    parser.add_argument("--entropy-dir", type=str, default="results_entropy",
                        help="Directory with CSVs from the semantic entropy script.")
    parser.add_argument("--confidence-dir", type=str, default="results_confidence",
                        help="Directory with CSVs containing self-report / P(True) confidence.")
    parser.add_argument("--evaluated-dir", type=str, default="results_evaluated",
                        help="Directory with CSVs written by the LLM judge script (evaluated_*.csv).")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Unified fallback directory if all CSVs share a single folder.")
    parser.add_argument("--output-dir", type=str, default="Demo3_Charlie/figures",
                        help="Directory to save plots and CSV summaries.")
    parser.add_argument("--summary-csv", type=str, default="Demo3_Charlie/results/meta_entropy_results.csv",
                        help="Meta-entropy summary CSV; drives the main / persona plots per dataset and condition.")
    parser.add_argument("--skip-legacy", action="store_true", default=True,
                        help="Only make the summary-CSV plots; skip the plots that rebuild metrics from the raw CSV dirs.")
    parser.add_argument("--rouge-threshold", type=float, default=0.25,
                        help="ROUGE-L F1 threshold for binary correctness (default: 0.25).")
    parser.add_argument("--no-raw-rescue", action="store_true",
                        help="Self-report confidence: use only values the original parser extracted. By default, "
                             "a run whose confidence was not parsed is re-parsed from its raw model text.")
    return parser.parse_args()


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def compute_rouge_l_f1(reference, candidate):
    """
    Word-token LCS ROUGE-L F1 between a reference answer and a model response.
    Returns (f1, reference_fully_covered).
    """
    if pd.isna(reference) or pd.isna(candidate):
        return 0.0, False

    ref_tokens = re.findall(r"\w+", str(reference).lower())
    cand_tokens = re.findall(r"\w+", str(candidate).lower())

    m, n = len(ref_tokens), len(cand_tokens)
    if m == 0 or n == 0:
        return 0.0, False

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == cand_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    precision = lcs_len / n
    recall = lcs_len / m

    if precision + recall == 0:
        return 0.0, False
    return (2 * precision * recall) / (precision + recall), (recall == 1.0)


def normalize_question(q):
    """Normalizes question strings for robust cross-file matching."""
    if pd.isna(q):
        return ""
    return re.sub(r'\s+', ' ', str(q).strip().lower())


def make_text_key(df):
    """<normalized question>#<n-th occurrence of that question>. Works for files without index columns."""
    nq = (df["question"].map(normalize_question) if "question" in df.columns
          else pd.Series([""] * len(df), index=df.index))
    return nq + "#" + nq.groupby(nq).cumcount().astype(str)


def make_key(df):
    """
    Per-row key used to join the entropy, judge and confidence CSVs.
    Questions repeat across QASPER papers ("What datasets do they use?"), so the question text alone is
    not unique: the key is <paper_idx>|<question_idx>|<question> (or <item_idx>|<question> for SciQ).
    Without index columns it falls back to the text key.
    """
    nq = (df["question"].map(normalize_question) if "question" in df.columns
          else pd.Series([""] * len(df), index=df.index))

    def as_int_str(col):
        return pd.to_numeric(df[col], errors="coerce").astype("Int64").astype(str)

    if "paper_idx" in df.columns and "question_idx" in df.columns:
        return as_int_str("paper_idx") + "|" + as_int_str("question_idx") + "|" + nq
    if "item_idx" in df.columns:
        return as_int_str("item_idx") + "|" + nq
    return make_text_key(df)


# ------------------------------------------------------------------------------
# Self-report confidence: first valid value across the full set of runs
# ------------------------------------------------------------------------------
def _valid_conf(v):
    """float in the 1-10 range, else None."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if np.isnan(v) or not (CONFIDENCE_RANGE[0] <= v <= CONFIDENCE_RANGE[1]):
        return None
    return v


def rescue_confidence_from_text(text):
    """Last 'confidence ... N' statement (1-10) in free text, else None. The last one wins because the
    verdict normally comes after the answer."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return None
    for v in reversed(RAW_CONFIDENCE_PATTERN.findall(str(text))):
        c = _valid_conf(v)
        if c is not None:
            return c
    return None


def _json_list(cell):
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return None
    try:
        v = json.loads(cell)
        return v if isinstance(v, list) else None
    except (TypeError, ValueError):
        return None


def per_run_inputs(row, spec):
    """(per-run parsed confidences, per-run raw texts) for one question."""
    parsed = _json_list(row.get(spec["json"]))
    if parsed is None:
        for col in spec["strings"]:
            cell = row.get(col)
            if cell is not None and not pd.isna(cell):
                parsed = [p.strip() for p in str(cell).split("||")]
                break
    return parsed or [], _json_list(row.get(spec["raw"])) or []


def first_valid_confidence(parsed, raw, rescue_raw=True):
    """
    Scans the runs in order and returns (confidence, run_index, source) for the first run that has a valid
    confidence: the originally parsed value, or (if rescue_raw) one re-parsed from that run's raw text.
    """
    for k in range(max(len(parsed), len(raw))):
        v = _valid_conf(parsed[k]) if k < len(parsed) else None
        if v is not None:
            return v, k, "parsed"
        if rescue_raw and k < len(raw):
            v = rescue_confidence_from_text(raw[k])
            if v is not None:
                return v, k, "raw_text"
    return np.nan, np.nan, ""


def valid_pairs(y_true, scores):
    """Rows where both the correctness label and the uncertainty score exist -> (binary labels, scores)."""
    y = pd.to_numeric(pd.Series(y_true).reset_index(drop=True), errors='coerce').to_numpy(dtype=float)
    s = pd.to_numeric(pd.Series(scores).reset_index(drop=True), errors='coerce').to_numpy(dtype=float)
    ok = ~np.isnan(y) & ~np.isnan(s)
    return (y[ok] >= 0.5).astype(int), s[ok]


def compute_auroc(y_binary, scores, is_entropy=True):
    """AUROC for predicting a correct answer. Entropy is negated so lower entropy = higher confidence."""
    if len(y_binary) == 0 or len(np.unique(y_binary)) < 2:
        return np.nan
    conf = -scores if is_entropy else scores
    return float(roc_auc_score(y_binary, conf))


def compute_aurc(y_binary, scores, is_entropy=True):
    """
    Area under the accuracy-coverage curve (answers sorted most- to least-confident).
    Note this is accuracy-based, so HIGHER is better (the classic risk-based AURC is lower-is-better).
    """
    n = len(y_binary)
    if n == 0:
        return np.nan
    conf = -scores if is_entropy else scores
    sorted_y = y_binary.astype(float)[np.argsort(-conf)]
    coverages = np.linspace(1.0 / n, 1.0, n)
    accuracies = np.cumsum(sorted_y) / np.arange(1, n + 1)
    return float(trapz_fn(accuracies, coverages))


def match_file_metadata(filename, df_head=None):
    """Determines dataset and canonical model key from filename or dataframe metadata."""
    filename_lower = filename.lower()
    fn_clean = re.sub(r'[^a-z0-9]', '', filename_lower)

    dataset = "sciq" if "sciq" in filename_lower else ("qasper" if "qasper" in filename_lower else None)
    if not dataset and df_head is not None and "dataset" in df_head.columns:
        ds_val = str(df_head["dataset"].iloc[0]).lower()
        dataset = "sciq" if "sciq" in ds_val else ("qasper" if "qasper" in ds_val else None)

    model_key = None
    for canonical_key, aliases in MODEL_ALIASES.items():
        if any(re.sub(r'[^a-z0-9]', '', a) in fn_clean for a in aliases):
            model_key = canonical_key
            break

    if not model_key and df_head is not None and "model" in df_head.columns:
        model_str_clean = re.sub(r'[^a-z0-9]', '', str(df_head["model"].iloc[0]).lower())
        for canonical_key, aliases in MODEL_ALIASES.items():
            if any(re.sub(r'[^a-z0-9]', '', a) in model_str_clean for a in aliases):
                model_key = canonical_key
                break

    return dataset, model_key


def load_entropy_df(path):
    df = pd.read_csv(path)
    df["key"] = make_key(df)
    df["tkey"] = make_text_key(df)
    return df.drop_duplicates(subset="key")


def load_eval_part(path):
    """
    Reads a judge-script output CSV and returns the columns needed downstream:
    key, correctness (judge verdict; NaN where the judge output couldn't be parsed),
    correct_answer_used (the reference the judge used), greedy_response (the answer that was judged).
    """
    df = pd.read_csv(path)
    df["key"] = make_key(df)
    df["tkey"] = make_text_key(df)

    corr_col = next((c for c in JUDGE_COLUMNS if c in df.columns and not df[c].isna().all()), None)
    ref_col = next((c for c in REFERENCE_COLUMNS if c in df.columns), None)
    ans_col = next((c for c in ANSWER_COLUMNS if c in df.columns), None)
    if corr_col is None:
        raise ValueError(f"{os.path.basename(path)}: no judge verdict column found ({JUDGE_COLUMNS})")
    if ans_col != "greedy_response":
        print(f"  WARNING: {os.path.basename(path)} has no 'greedy_response' column; using '{ans_col}' instead.")

    part = pd.DataFrame({"key": df["key"], "tkey": df["tkey"]})
    for c in ["question"] + ID_COLUMNS:
        if c in df.columns:
            part[c] = df[c]
    part["correctness"] = pd.to_numeric(df[corr_col], errors="coerce")
    part["correct_answer_used"] = df[ref_col] if ref_col else np.nan
    part["greedy_response"] = df[ans_col] if ans_col else np.nan
    return part.drop_duplicates(subset="key"), df


def parse_confidence_df(df, rescue_raw=True):
    """
    Parses self-report and P(True) confidence from a 'combined' CSV.

    Self-report: for each question, the FIRST valid confidence across the full set of sampled runs
    (not just run 0, which often failed to parse). Adds <measure>_run (0-based run used) and
    <measure>_source ('parsed' or 'raw_text') columns.
    Returns (frame keyed by key/tkey, per-measure parse-coverage stats).
    """
    out = pd.DataFrame({"key": make_key(df), "tkey": make_text_key(df)}, index=df.index)
    stats = []

    for measure, spec in CONFIDENCE_SOURCES.items():
        has_data = any(c in df.columns for c in [spec["json"], spec["raw"]] + spec["strings"])
        if not has_data:
            continue
        vals, runs, sources = [], [], []
        run0 = any_run = final = 0
        for _, row in df.iterrows():
            parsed, raw = per_run_inputs(row, spec)
            run0 += bool(parsed) and _valid_conf(parsed[0]) is not None
            any_run += not np.isnan(first_valid_confidence(parsed, raw, rescue_raw=False)[0])
            v, k, src = first_valid_confidence(parsed, raw, rescue_raw)
            final += not np.isnan(v)
            vals.append(v)
            runs.append(k)
            sources.append(src)
        out[measure] = vals
        out[measure + "_run"] = runs
        out[measure + "_source"] = sources
        stats.append({"baseline": spec["label"], "n_rows": len(df), "parsed_run0_only": run0,
                      "parsed_any_run": any_run, "with_raw_text_rescue": final,
                      "none_after": len(df) - final})

    # P(True): one value per question, already numeric (don't string-parse it: 1.2e-05 would be mangled)
    for col in ['p_true_probability_B', 'p_true_probability_A', 'p_true_probability']:
        if col in df.columns:
            out['p_true_probability'] = pd.to_numeric(df[col], errors='coerce')
            break

    return out.drop_duplicates(subset="key"), stats


def sem(values):
    v = pd.to_numeric(pd.Series(values), errors='coerce').dropna()
    return float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan


# ------------------------------------------------------------------------------
# Loading + metrics
# ------------------------------------------------------------------------------
def load_and_compute_metrics(entropy_dir, confidence_dir, evaluated_dir, fallback_dir=None, rouge_threshold=0.25,
                             rescue_raw=True):
    """
    Finds the entropy / judge / confidence CSV for every (dataset, model), joins them on a per-question key,
    and computes accuracy, AUROC and AURC for every uncertainty measure against both the LLM-judge and
    ROUGE-L correctness targets.

    Returns: metrics_by_dataset, entropy_bar_stats, per_example_df, summary_df, confidence_coverage_df
    """
    metrics_by_dataset = {"sciq": {}, "qasper": {}}
    entropy_bar_stats = {}
    per_example_frames = []
    summary_rows = []
    coverage_rows = []

    search_dirs = []
    if fallback_dir and os.path.exists(fallback_dir):
        search_dirs.append(fallback_dir)
    else:
        for d in [entropy_dir, confidence_dir, evaluated_dir]:
            if d and os.path.exists(d):
                search_dirs.append(d)

    all_files = []
    for d in search_dirs:
        all_files.extend(glob.glob(os.path.join(d, "*.csv")))

    entropy_map, confidence_map, evaluated_map = {}, {}, {}

    for f in sorted(set(all_files)):
        filename = os.path.basename(f).lower()
        try:
            df_temp = pd.read_csv(f, nrows=5)
        except Exception:
            continue

        ds, mk = match_file_metadata(filename, df_temp)
        if not ds or not mk:
            continue
        key = (ds, mk)

        cols = set(df_temp.columns)
        has_eval_cols = bool(cols & set(JUDGE_COLUMNS))
        has_entropy_cols = bool(cols & {"semantic_entropy", "predictive_entropy", "length_normalized_predictive_entropy"})
        has_conf_cols = bool(cols & {'vanilla_baseline_all_confidence', 'vanilia_baseline_all_confidence',
                                     'structured_self-report_all_confidence', 'p_true_probability_B',
                                     'p_true_probability'})

        # Judge-script outputs (evaluated_*.csv) also contain the entropy columns, so test for them first.
        if has_eval_cols or "judge" in filename or filename.startswith("evaluated"):
            evaluated_map[key] = f
        elif has_entropy_cols or "entropy" in filename:
            entropy_map[key] = f
        elif has_conf_cols or "confidence" in filename or "combined" in filename:
            confidence_map[key] = f

    all_keys = set(entropy_map) | set(confidence_map) | set(evaluated_map)

    for ds, mk in sorted(all_keys):
        tag = f"[{ds}/{mk}]"
        base = None

        eval_part, eval_raw = (None, None)
        if (ds, mk) in evaluated_map:
            eval_part, eval_raw = load_eval_part(evaluated_map[(ds, mk)])

        # 1. Entropy scores: prefer the standalone entropy CSV; the judge CSV carries the same columns.
        if (ds, mk) in entropy_map:
            base = load_entropy_df(entropy_map[(ds, mk)])
        elif eval_raw is not None and any(c for c, *_ in ENTROPY_MEASURES if c in eval_raw.columns):
            base = eval_raw.drop_duplicates(subset="key").copy()

        if base is not None:
            stats = {}
            for col_key, *_ in ENTROPY_MEASURES:
                if col_key in base.columns:
                    stats[col_key] = (pd.to_numeric(base[col_key], errors='coerce').mean(), sem(base[col_key]))
            entropy_bar_stats[(mk, ds)] = stats

        # 2. Judge results (correctness, reference used, greedy answer judged)
        merged = base
        if eval_part is not None:
            if merged is None:
                merged = eval_part
            else:
                merged = merged.drop(columns=[c for c in EVAL_SPECIFIC_COLS if c in merged.columns])
                extra = [c for c in eval_part.columns if c not in merged.columns or c == "key"]
                merged = merged.merge(eval_part[extra], on="key", how="inner")
            print(f"{tag} entropy rows: {0 if base is None else len(base)} | judged rows: {len(eval_part)} | "
                  f"joined: {len(merged)}")

        # 3. Self-report / P(True) confidence (left join so it never shrinks the entropy/judge sample)
        if (ds, mk) in confidence_map and merged is not None:
            conf_df, conf_stats = parse_confidence_df(pd.read_csv(confidence_map[(ds, mk)]), rescue_raw)
            for st in conf_stats:
                coverage_rows.append({"dataset": ds, "model": mk, **st})
                print(f"{tag} {st['baseline']} confidence parsed (of {st['n_rows']} questions): "
                      f"run 0 only {st['parsed_run0_only']} -> any run {st['parsed_any_run']} -> "
                      f"+raw-text rescue {st['with_raw_text_rescue']} (none: {st['none_after']})")
            conf_cols = [c for c in conf_df.columns if c not in ("key", "tkey")]

            # Join on paper/question index + text when they agree; otherwise on question text alone
            # (e.g. the SciQ combined file has no item index).
            join_key = "key"
            n_match = int(merged["key"].isin(conf_df["key"]).sum())
            if n_match < 0.9 * len(merged) and "tkey" in merged.columns:
                n_text = int(merged["tkey"].isin(conf_df["tkey"]).sum())
                if n_text > n_match:
                    join_key, n_match = "tkey", n_text
                    print(f"{tag} index keys don't line up with the confidence file; joining on question text "
                          f"(repeated questions are matched in file order).")
            conf_df = conf_df.drop_duplicates(subset=join_key)
            merged = merged.merge(conf_df[[join_key] + conf_cols], on=join_key, how="left")
            print(f"{tag} confidence rows matched: {n_match}/{len(merged)}")
            if n_match < 0.9 * len(merged):
                print(f"  WARNING: only {n_match}/{len(merged)} rows matched the confidence file; check that "
                      f"paper_idx / question_idx / question text agree between files.")

        if merged is None or merged.empty or "correctness" not in merged.columns:
            continue
        merged = merged.copy()

        # ROUGE-L of the greedy answer against the reference the judge used
        merged['rouge_l'], merged['exact'] = zip(*[
            compute_rouge_l_f1(r, c) for r, c in zip(merged["correct_answer_used"], merged["greedy_response"])
        ])
        merged['rouge_l_correctness'] = ((merged['rouge_l'] >= rouge_threshold) | merged['exact']).astype(float)

        judged = merged['correctness'].dropna()
        model_results = {
            "n_total": len(merged),
            "n_examples": {"judge": len(judged), "rouge": len(merged)},
            "judge_accuracy": judged.mean() if len(judged) else np.nan,
            "judge_sem": sem(judged),
            "recall_accuracy": merged['rouge_l_correctness'].mean(),
            "recall_sem": sem(merged['rouge_l_correctness']),
            "mean_rouge_score": merged['rouge_l'].mean(),
            "targets": {"judge": {}, "rouge": {}},
        }

        for target, tcol in [("judge", "correctness"), ("rouge", "rouge_l_correctness")]:
            for col_name, measure_label, _, _, is_entropy in ALL_MEASURES:
                if col_name not in merged.columns:
                    continue
                y, s = valid_pairs(merged[tcol], merged[col_name])
                auroc_v = compute_auroc(y, s, is_entropy)
                aurc_v = compute_aurc(y, s, is_entropy)
                model_results["targets"][target][measure_label] = {
                    "auroc": auroc_v, "aurc": aurc_v, "n": len(y)}
                summary_rows.append({
                    "dataset": ds, "model": mk, "target": target, "measure": measure_label,
                    "n_examples": len(y), "n_correct": int(y.sum()), "n_incorrect": int((1 - y).sum()),
                    "n_pair_total": len(merged),
                    "accuracy_on_these": float(y.mean()) if len(y) else np.nan,
                    "auroc": auroc_v, "aurc": aurc_v,
                })

        metrics_by_dataset[ds][mk] = model_results

        # Per-example record
        frame = pd.DataFrame({"dataset": ds, "model": mk}, index=merged.index)
        for c in ID_COLUMNS + ["question"]:
            if c in merged.columns:
                frame[c] = merged[c]
        frame["correct_answer_used"] = merged["correct_answer_used"]
        frame["greedy_response"] = merged["greedy_response"]
        frame["judge_correct"] = merged["correctness"]
        frame["rouge_l"] = merged["rouge_l"]
        frame["rouge_l_correct"] = merged["rouge_l_correctness"]
        for col_name, *_ in ALL_MEASURES:
            if col_name in merged.columns:
                frame[col_name] = merged[col_name]
            for suffix in ("_run", "_source"):  # which run's confidence was used, and how it was obtained
                if col_name + suffix in merged.columns:
                    frame[col_name + suffix] = merged[col_name + suffix]
        per_example_frames.append(frame)

    per_example_df = pd.concat(per_example_frames, ignore_index=True) if per_example_frames else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    coverage_df = pd.DataFrame(coverage_rows)
    return metrics_by_dataset, entropy_bar_stats, per_example_df, summary_df, coverage_df


def print_coverage(summary_df):
    """Prints how many examples each AUROC/AURC point is computed from."""
    if summary_df.empty:
        print("No AUROC/AURC results to report (no joined entropy + judge data found).")
        return
    order = {m: i for i, (m, _, _) in enumerate(MODEL_ORDER)}
    label_order = [lbl for _, lbl, *_ in ALL_MEASURES]
    for target in ["judge", "rouge"]:
        sub = summary_df[summary_df["target"] == target]
        if sub.empty:
            continue
        print("\n" + "=" * 78)
        print(f"Examples behind each AUROC/AURC point ({target.upper()} target)")
        print("=" * 78)
        for ds in ["sciq", "qasper"]:
            d = sub[sub["dataset"] == ds]
            if d.empty:
                continue
            pivot = d.pivot_table(index="model", columns="measure", values="n_examples", aggfunc="first")
            pivot = pivot.reindex(sorted(pivot.index, key=lambda m: order.get(m, 99)))
            pivot = pivot[[c for c in label_order if c in pivot.columns]].astype("Int64")
            print(f"\n{ds.upper()}:")
            print(pivot.to_string())
    print("\nA smaller n for the self-report measures means only rows with a parsed confidence are used.")


# ------------------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------------------
def _save(output_file):
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()


def plot_accuracy_comparison(metrics_by_dataset, output_file, recall_threshold=0.25):
    """Bar chart comparing LLM Judge accuracy vs. ROUGE-L accuracy of the greedy answer, with SEM error bars."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300, sharey=True)
    datasets = ["sciq", "qasper"]
    x = np.arange(len(MODEL_ORDER))
    bar_width = 0.35

    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        ds_data = metrics_by_dataset.get(ds, {})

        def get(key):
            return [ds_data.get(m, {}).get(key, np.nan) for m, _, _ in MODEL_ORDER]

        ax.bar(x - bar_width / 2, get('judge_accuracy'), width=bar_width, yerr=0,
               label="LLM Judge Accuracy", color="#1f77b4", capsize=3.5,
               error_kw={'ecolor': 'black', 'linewidth': 1.2, 'capthick': 1.2})
        ax.bar(x + bar_width / 2, get('recall_accuracy'), width=bar_width, yerr=0,
               label=f"Rouge-L Accuracy (≥{recall_threshold})", color="#ff7f0e", capsize=3.5,
               error_kw={'ecolor': 'black', 'linewidth': 1.2, 'capthick': 1.2})

        ticks = []
        for m, label, _ in MODEL_ORDER:
            n = ds_data.get(m, {}).get("n_examples", {}).get("judge")
            ticks.append(f"{label}\n(n={n})" if n else f"{label}\n(n=–)")
        ax.set_xticks(x)
        ax.set_xticklabels(ticks, rotation=0, fontsize=10)
        ax.set_title(f"({chr(97 + idx)}) {ds.upper()} Accuracy Comparison", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle='-', alpha=0.3, color='lightgrey')
        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Accuracy Rate", fontsize=11)
    axes[0].legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    _save(output_file)
    print(f"Saved Accuracy Comparison Bar Chart (with SEM error bars) to: {output_file}")


def plot_entropy_bars_with_sem(entropy_bar_stats, output_file):
    """Grouped bar charts for entropy measures with SEM error bars."""
    model_ds_order = [
        ("llama3.2_1b", "qasper", "Llama 1B / QASPER"), ("llama3.2_1b", "sciq", "Llama 1B / SciQ"),
        ("llama3.2_3b", "qasper", "Llama 3B / QASPER"), ("llama3.2_3b", "sciq", "Llama 3B / SciQ"),
        ("qwen3_1.7b", "qasper", "Qwen 1.7B / QASPER"), ("qwen3_1.7b", "sciq", "Qwen 1.7B / SciQ"),
        ("qwen3_4b", "qasper", "Qwen 4B / QASPER"), ("qwen3_4b", "sciq", "Qwen 4B / SciQ"),
    ]

    labels = []
    means_per_metric = {m[0]: [] for m in ENTROPY_MEASURES}
    sems_per_metric = {m[0]: [] for m in ENTROPY_MEASURES}

    for model_key, ds_key, display_label in model_ds_order:
        labels.append(display_label)
        entry = entropy_bar_stats.get((model_key, ds_key), {})
        for metric_key, *_ in ENTROPY_MEASURES:
            mean, sem_val = entry.get(metric_key, (np.nan, np.nan))
            means_per_metric[metric_key].append(mean)
            sems_per_metric[metric_key].append(0.0 if np.isnan(sem_val) else sem_val)

    x = np.arange(len(labels))
    num_metrics = len(ENTROPY_MEASURES)
    bar_width = 0.8 / num_metrics

    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    for i, (metric_key, metric_label, _, color, _) in enumerate(ENTROPY_MEASURES):
        offset = (i - (num_metrics - 1) / 2) * bar_width
        ax.bar(x + offset, means_per_metric[metric_key], width=bar_width, yerr=sems_per_metric[metric_key],
               label=metric_label, color=color, capsize=3.5,
               error_kw={'ecolor': 'black', 'linewidth': 1.2, 'capthick': 1.2})

    ax.set_ylabel("Uncertainty Score (Entropy)", fontsize=12, labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=10)
    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none', fontsize=11)
    ax.yaxis.grid(True, linestyle='-', alpha=0.3, color='lightgrey')
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    _save(output_file)
    print(f"Saved Entropy Grouped Bar Chart (with SEM error bars) to: {output_file}")


def plot_kuhn_style(metrics_by_dataset, metric_type="auroc", target="judge", output_file="figures/kuhn_auroc_plot.png"):
    """Side-by-side Kuhn et al. (2023) style plots for a target ('judge' or 'rouge'); x tick labels give n."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=300, sharey=True)
    datasets = ["sciq", "qasper"]
    dataset_titles = {"sciq": f"(a) SciQ ({target.upper()} Target)", "qasper": f"(b) QASPER ({target.upper()} Target)"}
    x_indices = np.arange(len(MODEL_ORDER))

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        dataset_data = metrics_by_dataset.get(dataset, {})

        for col_name, measure_label, marker, color, is_entropy in ALL_MEASURES:
            y_vals = [
                dataset_data.get(model_tag, {}).get("targets", {}).get(target, {})
                .get(measure_label, {}).get(metric_type, np.nan)
                for model_tag, _, _ in MODEL_ORDER
            ]
            if not np.all(np.isnan(y_vals)):
                ax.plot(x_indices, y_vals, label=measure_label, marker=marker, color=color,
                        linestyle='--', linewidth=1.8, markersize=7)

        ticks = []
        for model_tag, label, _ in MODEL_ORDER:
            n = dataset_data.get(model_tag, {}).get("n_examples", {}).get(target)
            ticks.append(f"{label}\n(n={n})" if n else f"{label}\n(n=–)")
        ax.set_xticks(x_indices)
        ax.set_xticklabels(ticks, rotation=0, fontsize=10)

        if idx == 0:
            ax.set_ylabel(metric_type.upper(), fontsize=12, labelpad=8)
        ax.set_xlabel(dataset_titles[dataset], fontsize=13, labelpad=10)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=11)

    # Pool legend entries from both panels (a measure may only appear in one dataset, e.g. self-report on QASPER)
    seen = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    order = [lbl for _, lbl, *_ in ALL_MEASURES if lbl in seen]
    handles, labels = [seen[l] for l in order], order
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.08),
                   ncol=4, frameon=False, fontsize=11)

    fig.text(0.5, -0.02,
             "n = examples with a judged greedy answer. Self-report measures only use rows with a parsed "
             "confidence (per-measure n in metric_summary.csv).",
             ha='center', va='top', fontsize=8.5, color='dimgray')

    plt.tight_layout()
    _save(output_file)
    print(f"Saved Kuhn et al. style {metric_type.upper()} plot ({target.upper()} Target) to: {output_file}")


# ------------------------------------------------------------------------------
# Meta-entropy summary plots (read from meta_entropy_results.csv)
# ------------------------------------------------------------------------------
# (feature_set in the summary csv, legend label, marker, color, linestyle, linewidth)
SUMMARY_MAIN = [
    ("semantic_entropy", "Semantic Entropy", "s", "#1f77b4", "--", 1.6),
    ("predictive_entropy", "Predictive Entropy", "o", "#2ca02c", "--", 1.6),
    ("length_normalized_predictive_entropy", "LN-PE", "v", "#ff7f0e", "--", 1.6),
    ("vanilla_self_report", "Vanilla Self-Report", "^", "#9467bd", "--", 1.6),
    ("p_true_probability", "P(True)", "x", "#d62728", "--", 1.6),
    ("all_metrics", "Meta-Uncertainty (all measures)", "*", "black", "-", 2.8),
]

SUMMARY_PERSONA = [
    ("speculative_editor_confidence", "Speculative Editor", "o", "#1f77b4", "--", 1.6),
    ("opposing_counsel_confidence", "Opposing Counsel", "s", "#2ca02c", "--", 1.6),
    ("compliance_officer_confidence", "Compliance Officer", "^", "#ff7f0e", "--", 1.6),
    ("persona_panel_confidence", "Persona Panel", "D", "#9467bd", "--", 1.6),
    ("money_betting_bet_usd", "Money Betting", "x", "#d62728", "--", 1.6),
    ("all_persona", "Meta-Uncertainty (personas only)", "*", "black", "-", 2.8),
]

# Condition panels available per dataset (QASPER only has an unbiased run in the summary csv)
CONDITION_TITLES = {"unbiased": "Unbiased", "biased": "Biased"}
DATASET_TITLES = {"sciq": "SciQ", "qasper": "QASPER"}
# Model families: one set of figures per (dataset, family)
MODEL_FAMILIES = {
    "llama": ("Llama", ["llama3.2_1b", "llama3.2_3b"]),
    "qwen": ("Qwen", ["qwen3_1.7b", "qwen3_4b"]),
}


def plot_summary_figure(summary, dataset, family, measures, metric, target, output_file, title_prefix):
    """
    One figure for one dataset and one model family (llama | qwen): a panel per condition (unbiased / biased) showing `metric` (auroc | aurc)
    vs. model size for each entry in `measures`. Points are the cross-validated mean, error bars the std over folds.
    """
    fam_label, fam_models = MODEL_FAMILIES[family]
    models = [(m, lbl, sz) for m, lbl, sz in MODEL_ORDER if m in fam_models]
    ds_df = summary[(summary["dataset"] == dataset) & (summary["target"] == target)
                    & (summary["model"].isin(fam_models))]
    conditions = [c for c in ["unbiased", "biased"] if c in ds_df["condition"].unique()]
    if not conditions:
        print(f"  (skipping {output_file}: no {dataset}/{target} rows in the summary csv)")
        return

    fig, axes = plt.subplots(1, len(conditions), figsize=(5.5 * len(conditions) + 0.5, 6), dpi=300,
                             sharey=True, squeeze=False)
    axes = axes[0]
    x_idx = np.arange(len(models))
    mean_col, std_col = f"{metric}_mean", f"{metric}_std"
    n_meas = len(measures)

    for ax, cond in zip(axes, conditions):
        cdf = ds_df[ds_df["condition"] == cond]
        for i, (feat, label, marker, color, ls, lw) in enumerate(measures):
            sub = cdf[cdf["feature_set"] == feat].set_index("model")
            if sub.empty:
                continue
            y = np.array([sub[mean_col].get(m, np.nan) for m, _, _ in models], dtype=float)
            e = np.array([sub[std_col].get(m, np.nan) for m, _, _ in models], dtype=float)
            if np.all(np.isnan(y)):
                continue
            # jitter = (i - (n_meas - 1) / 2) * 0.025  # tiny x offset so error bars don't overlap
            # np.nan_to_num(e)
            jitter = 0
            is_meta = feat.startswith("all_")
            ax.errorbar(x_idx + jitter, y, yerr=0, label=label, marker=marker, color=color,
                        linestyle=ls, linewidth=lw, markersize=11 if is_meta else 7, capsize=2.5,
                        elinewidth=0.9, alpha=1.0 if is_meta else 0.9, zorder=5 if is_meta else 3)

        ticks = []
        for m, lbl, _ in models:
            n = cdf.loc[cdf["model"] == m, "n"].max()
            ticks.append(f"{lbl}\n(n={int(n)})" if pd.notna(n) else f"{lbl}\n(n=–)")
        ax.set_xticks(x_idx)
        ax.set_xlim(-0.4, len(models) - 0.6)
        ax.set_xticklabels(ticks, fontsize=10)
        ax.set_xlabel(f"{DATASET_TITLES.get(dataset, dataset)} / {fam_label} — {CONDITION_TITLES[cond]}", fontsize=13, labelpad=10)
        if metric == "auroc":
            ax.axhline(0.5, color="grey", linestyle=":", linewidth=1, zorder=1)  # chance level
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle="-", alpha=0.25, color="lightgrey")
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", which="major", labelsize=11)

    axes[0].set_ylabel(metric.upper(), fontsize=12, labelpad=8)

    seen = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    order = [lbl for _, lbl, *_ in measures if lbl in seen]
    if order:
        fig.legend([seen[l] for l in order], order, loc="upper center", bbox_to_anchor=(0.5, 1.10),
                   ncol=3, frameon=False, fontsize=11)
    fig.suptitle(f"{title_prefix} ({fam_label}): {metric.upper()} ({target.upper()} target)", y=1.17, fontsize=13)
    fig.text(0.5, -0.02, "Mean ± std over CV folds. Dotted line = chance (AUROC 0.5)." if metric == "auroc"
             else "Mean ± std over CV folds. AURC here is accuracy-based: higher is better.",
             ha="center", va="top", fontsize=8.5, color="dimgray")
    plt.tight_layout()
    _save(output_file)
    print(f"Saved {title_prefix} {metric.upper()} plot ({dataset}, {target} target) to: {output_file}")


def make_summary_plots(summary_csv, output_dir):
    """Main (uncertainty measures + meta-entropy) and persona plots, per dataset and per model family,
    with biased + unbiased panels."""
    summary = pd.read_csv(summary_csv)
    for dataset in [d for d in ["sciq", "qasper"] if d in summary["dataset"].unique()]:
        for family in MODEL_FAMILIES:
            for target in ["judge", "rouge"]:
                for metric in ["auroc", "aurc"]:
                    plot_summary_figure(
                        summary, dataset, family, SUMMARY_MAIN, metric, target,
                        os.path.join(output_dir, f"main_{dataset}_{family}_{metric}_{target}.png"),
                        title_prefix=f"{DATASET_TITLES[dataset]} — uncertainty measures vs. Meta-Uncertainty")
                    plot_summary_figure(
                        summary, dataset, family, SUMMARY_PERSONA, metric, target,
                        os.path.join(output_dir, f"personas_{dataset}_{family}_{metric}_{target}.png"),
                        title_prefix=f"{DATASET_TITLES[dataset]} — personas")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # New: main + persona plots from the meta-entropy summary CSV (SciQ / QASPER, unbiased + biased)
    if os.path.exists(args.summary_csv):
        make_summary_plots(args.summary_csv, args.output_dir)
    else:
        print(f"WARNING: summary csv not found at {args.summary_csv}; skipping main/persona plots.")
    # if args.skip_legacy:
    #     return

    metrics_by_dataset, entropy_bar_stats, per_example_df, summary_df, coverage_df = load_and_compute_metrics(
        entropy_dir=args.entropy_dir,
        confidence_dir=args.confidence_dir,
        evaluated_dir=args.evaluated_dir,
        fallback_dir=args.input_dir,
        rouge_threshold=args.rouge_threshold,
        rescue_raw=not args.no_raw_rescue,
    )
    # if not coverage_df.empty:
    #     coverage_path = os.path.join(args.output_dir, "confidence_parse_coverage.csv")
    #     coverage_df.to_csv(coverage_path, index=False)
    #     print(f"Saved confidence parse coverage to: {coverage_path}")

    # CSVs: per-example uncertainty scores + reference + greedy answer, and per-measure metric summary
    # if not per_example_df.empty:
    #     order = {m: i for i, (m, _, _) in enumerate(MODEL_ORDER)}
    #     per_example_df = per_example_df.sort_values(
    #         by=["dataset", "model"], key=lambda s: s.map(order) if s.name == "model" else s, kind="stable")
    #     per_example_path = os.path.join(args.output_dir, "per_example_results.csv")
    #     per_example_df.to_csv(per_example_path, index=False)
    #     print(f"Saved per-example results ({len(per_example_df)} rows) to: {per_example_path}")
    # if not summary_df.empty:
    #     summary_path = os.path.join(args.output_dir, "metric_summary.csv")
    #     summary_df.to_csv(summary_path, index=False)
    #     print(f"Saved per-metric AUROC/AURC summary with example counts to: {summary_path}")

    # 1. Accuracy: LLM judge vs. ROUGE-L
    plot_accuracy_comparison(metrics_by_dataset, os.path.join(args.output_dir, "accuracy_judge_vs_rouge.png"),
                             recall_threshold=args.rouge_threshold)

    # 2. Entropy bars with SEM
    # plot_entropy_bars_with_sem(entropy_bar_stats, os.path.join(args.output_dir, "entropy_bars_sem.png"))

    # # 3-4. AUROC & AURC scaling curves for both targets
    # for target in ["judge", "rouge"]:
    #     for metric in ["auroc", "aurc"]:
    #         plot_kuhn_style(metrics_by_dataset, metric_type=metric, target=target,
    #                         output_file=os.path.join(args.output_dir, f"kuhn_{metric}_{target}_target.png"))

    # print_coverage(summary_df)


if __name__ == "__main__":
    main()