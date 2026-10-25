import os
import argparse
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Match reference figure aesthetic colors: Blue, Orange, Red
COLORS = ['#1f77b4', '#ff7f0e', '#d62728']

# Expected order of model/dataset combinations on X-axis
MODEL_DATASET_ORDER = [
    ("llama3.2_1b", "qasper", "Llama 1B / QASPER"),
    ("llama3.2_1b", "sciq", "Llama 1B / SciQ"),
    ("llama3.2_3b", "qasper", "Llama 3B / QASPER"),
    ("llama3.2_3b", "sciq", "Llama 3B / SciQ"),
    ("qwen3_1.7b", "qasper", "Qwen 1.7B / QASPER"),
    ("qwen3_1.7b", "sciq", "Qwen 1.7B / SciQ"),
    ("qwen3_4b", "qasper", "Qwen 4B / QASPER"),
    ("qwen3_4b", "sciq", "Qwen 4B / SciQ"),
]

METRICS_CONFIG = [
    ("semantic_entropy", "Semantic Entropy"),
    ("predictive_entropy", "Predictive Entropy"),
    ("length_normalized_predictive_entropy", "Length-Norm PE"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Grouped Bar Plot with Error Bars from Evaluation CSVs")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="results",
        help="Directory containing output CSV files from evaluate_semantic_entropy_2.py"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="figures/metrics_comparison_plot.png",
        help="File path to save the generated plot (.png or .pdf)"
    )
    parser.add_argument(
        "--error-bar",
        type=str,
        choices=["std", "sem"],
        default="sem",
        help="Error bar type: 'std' (Standard Deviation across items) or 'sem' (Standard Error of Mean)"
    )
    return parser.parse_args()


def load_and_aggregate_results(input_dir, error_type="sem"):
    """Scans input_dir for CSVs and computes mean and error per metric for each model/dataset combo."""
    aggregated_data = {}
    csv_files = glob.glob(os.path.join(input_dir, "results_*.csv"))

    if not csv_files:
        print(f"Warning: No matching CSV files found in directory '{input_dir}'")

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        df = pd.read_csv(file_path)

        # Extract model and dataset from filename or dataframe
        dataset = df["dataset"].iloc[0] if "dataset" in df.columns else ("sciq" if "sciq" in filename else "qasper")
        
        # Determine model tag from filename
        model_tag = filename.replace(f"results_{dataset}_", "").replace(".csv", "")

        key = (model_tag.lower(), dataset.lower())

        metrics_summary = {}
        for metric_key, _ in METRICS_CONFIG:
            if metric_key in df.columns:
                mean_val = df[metric_key].mean()
                if error_type == "sem":
                    err_val = df[metric_key].sem()
                else:
                    err_val = df[metric_key].std()
                metrics_summary[metric_key] = (mean_val, err_val)
            else:
                metrics_summary[metric_key] = (0.0, 0.0)

        aggregated_data[key] = metrics_summary

    return aggregated_data


def plot_grouped_bars(aggregated_data, error_type, output_file):
    """Generates the styled grouped bar plot with error caps."""
    labels = []
    means_per_metric = {m[0]: [] for m in METRICS_CONFIG}
    errs_per_metric = {m[0]: [] for m in METRICS_CONFIG}

    # Populate data aligned with MODEL_DATASET_ORDER
    for model_key, ds_key, display_label in MODEL_DATASET_ORDER:
        labels.append(display_label)
        
        # Try finding exact or partial match key in aggregated_data
        entry = aggregated_data.get((model_key, ds_key))
        if entry is None:
            # Fallback search for partial string matching
            for (m_k, d_k), val in aggregated_data.items():
                if ds_key in d_k and model_key.replace("_", "") in m_k.replace("_", "").replace("-", ""):
                    entry = val
                    break

        for metric_key, _ in METRICS_CONFIG:
            if entry and metric_key in entry:
                mean, err = entry[metric_key]
            else:
                mean, err = 0.0, 0.0
            means_per_metric[metric_key].append(mean)
            errs_per_metric[metric_key].append(err)

    x = np.arange(len(labels))
    num_metrics = len(METRICS_CONFIG)
    total_width = 0.75
    bar_width = total_width / num_metrics

    fig, ax = plt.subplots(figsize=(14, 6.5), dpi=300)

    # Plot bars for each metric group
    for i, (metric_key, metric_label) in enumerate(METRICS_CONFIG):
        offset = (i - (num_metrics - 1) / 2) * bar_width
        means = means_per_metric[metric_key]
        errs = errs_per_metric[metric_key]

        ax.bar(
            x + offset,
            means,
            width=bar_width,
            yerr=errs,
            label=metric_label,
            color=COLORS[i % len(COLORS)],
            edgecolor='none',
            capsize=3.5,
            error_kw={'ecolor': 'black', 'linewidth': 1.2, 'capthick': 1.2}
        )

    # Aesthetic styling matching reference plot
    ax.set_ylabel("Uncertainty Score (Entropy)", fontsize=12, labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=11)
    
    # Legend formatting
    ax.legend(
        loc='upper left',
        frameon=True,
        facecolor='white',
        edgecolor='none',
        fontsize=11
    )

    # Gridlines and Spines
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='-', alpha=0.3, color='lightgrey')
    ax.xaxis.grid(False)
    
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()

    # Save output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"Grouped bar chart plot saved successfully to: {output_file}")


def main():
    args = parse_args()
    aggregated_data = load_and_aggregate_results(args.input_dir, error_type=args.error_bar)
    plot_grouped_bars(aggregated_data, args.error_bar, args.output_file)


if __name__ == "__main__":
    main()