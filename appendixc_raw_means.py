from pathlib import Path
import pandas as pd
import numpy as np

# Change this to the parent directory containing:
# llama3_indirect_only, llama3_last_nonpad, llama3_mean_nonpad, llama3_target_tokens
BASE_DIR = Path("results_suppression_salience")

RUNS = {
    "indirect_only": BASE_DIR / "llama3_indirect_only",
    "last_nonpad": BASE_DIR / "llama3_last_nonpad",
    "mean_nonpad": BASE_DIR / "llama3_mean_nonpad",
    "target_tokens": BASE_DIR / "llama3_target_tokens",
}

OUT_DIR = BASE_DIR / "appendix_c_summaries"
OUT_DIR.mkdir(exist_ok=True)


def read_csv_if_exists(path):
    if path.exists():
        return pd.read_csv(path)
    print(f"Missing: {path}")
    return None


def first_existing(folder, names):
    for name in names:
        path = folder / name
        if path.exists():
            return path
    return None


all_layer_summaries = []
all_best_layers = []
all_probe_reports = []
all_condition_means = []

for run_name, folder in RUNS.items():
    print(f"\nProcessing {run_name}: {folder}")

    # Main Experiment 1 summary
    summary_path = first_existing(folder, [
        "salience_summary_by_layer.csv",
        "summary_by_layer.csv"
    ])
    summary = read_csv_if_exists(summary_path) if summary_path else None

    if summary is not None:
        summary["run"] = run_name
        all_layer_summaries.append(summary)

        # Find the primary suppression-vs-absent delta column
        delta_cols = [
            c for c in summary.columns
            if "mean_delta_suppressed_minus_absent" in c
        ]

        if delta_cols:
            delta_col = delta_cols[0]
            best = summary.loc[summary[delta_col].idxmax()].copy()
            best["run"] = run_name
            best["best_metric"] = delta_col
            all_best_layers.append(best)

        # Extract compact condition means if present
        keep_cols = [
            "run", "layer",
            "mean_absent", "sem_absent",
            "ci95_low_absent", "ci95_high_absent",
            "mean_suppressed", "sem_suppressed",
            "ci95_low_suppressed", "ci95_high_suppressed",
            "mean_mentioned", "sem_mentioned",
            "ci95_low_mentioned", "ci95_high_mentioned",
            "mean_delta_suppressed_minus_absent",
            "sem_delta_suppressed_minus_absent",
            "ci95_low_delta_suppressed_minus_absent",
            "ci95_high_delta_suppressed_minus_absent",
            "paired_p_suppressed_vs_absent",
            "wilcoxon_p_suppressed_vs_absent",
            "cohen_d_paired_suppressed_vs_absent",
            "n_pairs"
        ]
        existing = [c for c in keep_cols if c in summary.columns]
        all_condition_means.append(summary[existing])

    # Probe quality report
    probe_path = first_existing(folder, [
        "probe_report.csv",
        "probe_report_by_seed.csv"
    ])
    probe = read_csv_if_exists(probe_path) if probe_path else None

    if probe is not None:
        probe["run"] = run_name
        all_probe_reports.append(probe)


# Save full layer-wise appendix table
if all_layer_summaries:
    layer_summary = pd.concat(all_layer_summaries, ignore_index=True)
    layer_summary.to_csv(OUT_DIR / "appendix_c_full_layer_summary.csv", index=False)

if all_condition_means:
    condition_means = pd.concat(all_condition_means, ignore_index=True)
    condition_means.to_csv(OUT_DIR / "appendix_c_condition_means_by_layer.csv", index=False)

# Save best-layer summary
if all_best_layers:
    best_layers = pd.DataFrame(all_best_layers)

    desired = [
        "run",
        "layer",
        "mean_absent",
        "mean_suppressed",
        "mean_mentioned",
        "mean_delta_suppressed_minus_absent",
        "ci95_low_delta_suppressed_minus_absent",
        "ci95_high_delta_suppressed_minus_absent",
        "paired_p_suppressed_vs_absent",
        "wilcoxon_p_suppressed_vs_absent",
        "cohen_d_paired_suppressed_vs_absent",
        "n_pairs"
    ]

    existing = [c for c in desired if c in best_layers.columns]
    best_layers_compact = best_layers[existing]
    best_layers_compact.to_csv(OUT_DIR / "appendix_c_best_layer_summary.csv", index=False)

# Save probe-quality summaries
if all_probe_reports:
    probe_all = pd.concat(all_probe_reports, ignore_index=True)
    probe_all.to_csv(OUT_DIR / "appendix_c_probe_report_all.csv", index=False)

    metric_cols = [
        c for c in probe_all.columns
        if c in [
            "probe_accuracy",
            "probe_auc",
            "probe_f1",
            "probe_accuracy_mean",
            "probe_auc_mean",
            "probe_f1_mean"
        ]
    ]

    probe_summary = (
        probe_all
        .groupby("run")[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    probe_summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in probe_summary.columns
    ]

    probe_summary.to_csv(OUT_DIR / "appendix_c_probe_quality_summary.csv", index=False)


print("\nDone. Wrote appendix summaries to:")
print(OUT_DIR.resolve())

print("\nKey outputs:")
print("- appendix_c_best_layer_summary.csv")
print("- appendix_c_condition_means_by_layer.csv")
print("- appendix_c_full_layer_summary.csv")
print("- appendix_c_probe_quality_summary.csv")
print("- appendix_c_probe_report_all.csv")