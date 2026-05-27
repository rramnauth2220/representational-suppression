#!/usr/bin/env python3

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = ["Llama 3 8B Inst.", "Mistral 7B Inst.", "Gemma 7B IT"]


MAIN_OUTPUT_FILES = [
    "cross_model_summary.csv",
    "cross_model_layerwise.csv",
    "cross_model_probe_quality.csv",
    "cross_model_table.tex",
    "cross_model_delta_by_layer.png",
    "cross_model_peak_delta_bar.png",
    "cross_model_probe_auc_bar.png",
    "cross_model_report.md",
    "region_analysis/cross_model_layerwise_with_regions.csv",
    "region_analysis/region_summary.csv",
    "region_analysis/within_model_region_contrasts.csv",
    "region_analysis/between_model_region_contrasts.csv",
    "region_analysis/model_region_interaction_ols.txt",
    "region_analysis/region_summary_table.tex",
    "region_analysis/region_delta_by_model.png",
    "region_analysis/region_analysis_report.md",
    "ordering_analysis/ordering_model_summary.csv",
    "ordering_analysis/ordering_condition_summary.csv",
    "ordering_analysis/ordering_pairwise_tests.csv",
    "ordering_analysis/ordering_layerwise_summary.csv",
    "ordering_analysis/ordering_table.tex",
    "ordering_analysis/ordering_condition_means_by_model.png",
    "ordering_analysis/ordering_deltas_by_model.png",
    "ordering_analysis/ordering_analysis_report.md",
]


def format_report_value(value, precision: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "nan"
        return f"{value:.{precision}g}"
    return str(value)


def make_variable_table(rows: List[Tuple[str, Any]]) -> List[str]:
    table = ["| variable | value |", "|---|---|"]
    table.extend(f"| `{name}` | {format_report_value(value)} |" for name, value in rows)
    return table


def apply_model_order(df: pd.DataFrame) -> pd.DataFrame:
    order = {model: i for i, model in enumerate(MODEL_ORDER)}
    out = df.copy()
    if "model_short" not in out.columns:
        return out
    out["_order"] = out["model_short"].map(order).fillna(999)
    sort_cols = ["_order", "model_short"]
    if "layer" in out.columns:
        sort_cols.append("layer")
    out = out.sort_values(sort_cols).drop(columns=["_order"]).reset_index(drop=True)
    return out


def pretty_model_name(model: str) -> str:
    mapping = {
        "meta-llama/Meta-Llama-3-8B-Instruct": "Llama 3 8B Inst.",
        "mistralai/Mistral-7B-Instruct-v0.2": "Mistral 7B Inst.",
        "google/gemma-7b-it": "Gemma 7B IT",
        "google/gemma-1.1-7b-it": "Gemma 1.1 7B IT",
        "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5 7B Inst.",
        "Qwen/Qwen2-7B-Instruct": "Qwen2 7B Inst.",
    }
    return mapping.get(model, model.split("/")[-1].replace("-", " "))


def find_run_dirs(root: Path, required_files: Tuple[str, ...]) -> List[Path]:
    run_dirs = []
    for args_path in root.rglob("args.json"):
        run_dir = args_path.parent
        if all((run_dir / name).exists() for name in required_files):
            run_dirs.append(run_dir)
    return sorted(run_dirs)


def load_args(run_dir: Path) -> Dict:
    with open(run_dir / "args.json", "r", encoding="utf-8") as f:
        return json.load(f)


def choose_delta_column(summary: pd.DataFrame, preference: str = "indirect") -> Optional[str]:
    cols = list(summary.columns)
    if preference == "direct":
        candidates = ["mean_delta_suppressed_minus_absent"]
    elif preference == "mentioned":
        candidates = ["mean_delta_mentioned_minus_absent"]
    else:
        candidates = sorted(
            [
                c for c in cols
                if c.startswith("mean_delta_suppressed_indirect") and c.endswith("_minus_absent")
            ],
            key=lambda c: (len(c), c),
        )
        candidates += ["mean_delta_suppressed_minus_absent"]

    for col in candidates:
        if col in cols:
            return col

    any_delta = [c for c in cols if c.startswith("mean_delta_") and c.endswith("_minus_absent")]
    return sorted(any_delta)[0] if any_delta else None


def delta_column_to_condition(delta_col: str) -> str:
    return delta_col.replace("mean_delta_", "").replace("_minus_absent", "")


def associated_columns(delta_col: str) -> Dict[str, str]:
    condition = delta_column_to_condition(delta_col)
    return {
        "condition": condition,
        "mean_condition": f"mean_{condition}",
        "mean_absent": "mean_absent",
        "delta": delta_col,
        "ci_low": delta_col.replace("mean_", "ci95_low_"),
        "ci_high": delta_col.replace("mean_", "ci95_high_"),
        "p": f"paired_p_{condition}_vs_absent",
        "wilcoxon_p": f"wilcoxon_p_{condition}_vs_absent",
        "d": f"cohen_d_paired_{condition}_vs_absent",
    }


def latex_escape(s: str) -> str:
    return str(s).replace("_", "\\_")


def format_p(p: float) -> str:
    if pd.isna(p):
        return "--"
    if p < 0.001:
        return "$<.001$"
    return f"{p:.3f}"


def summarize_run(run_dir: Path, delta_preference: str):
    args = load_args(run_dir)
    summary = pd.read_csv(run_dir / "salience_summary_by_layer.csv")
    probe_seed = pd.read_csv(run_dir / "probe_report_by_seed.csv")

    delta_col = choose_delta_column(summary, delta_preference)
    if delta_col is None:
        raise ValueError(f"No usable delta column found in {run_dir}")

    cols = associated_columns(delta_col)
    condition = cols["condition"]
    best = summary.sort_values(delta_col, ascending=False).iloc[0]
    layer_vals = summary[delta_col].dropna().to_numpy(dtype=float)
    model = args.get("model", run_dir.name)

    record = {
        "model": model,
        "model_short": pretty_model_name(model),
        "run_dir": str(run_dir),
        "pooling": args.get("pooling", ""),
        "concepts_path": args.get("concepts_path", ""),
        "probe_seeds": args.get("probe_seeds", ""),
        "include_indirect": args.get("include_indirect", False),
        "only_indirect": args.get("only_indirect", False),
        "include_hard_negatives": args.get("include_hard_negatives", ""),
        "selected_condition": condition,
        "selected_delta_col": delta_col,
        "n_layers": int(len(summary)),
        "peak_layer": int(best["layer"]),
        "peak_mean_absent": float(best.get(cols["mean_absent"], np.nan)),
        f"peak_mean_{condition}": float(best.get(cols["mean_condition"], np.nan)),
        "peak_delta": float(best[delta_col]),
        "peak_ci95_low": float(best.get(cols["ci_low"], np.nan)),
        "peak_ci95_high": float(best.get(cols["ci_high"], np.nan)),
        "peak_paired_p": float(best.get(cols["p"], np.nan)),
        "peak_wilcoxon_p": float(best.get(cols["wilcoxon_p"], np.nan)),
        "peak_cohen_d": float(best.get(cols["d"], np.nan)),
        "mean_delta_across_layers": float(np.mean(layer_vals)) if len(layer_vals) else np.nan,
        "probe_accuracy": float(probe_seed["probe_accuracy"].mean()),
        "probe_auc": float(probe_seed["probe_auc"].mean()),
        "probe_f1": float(probe_seed["probe_f1"].mean()),
    }

    layerwise = pd.DataFrame({
        "model": model,
        "model_short": pretty_model_name(model),
        "run_dir": str(run_dir),
        "pooling": args.get("pooling", ""),
        "selected_condition": condition,
        "layer": summary["layer"],
        "mean_absent": summary.get("mean_absent", np.nan),
        "mean_condition": summary.get(cols["mean_condition"], np.nan),
        "mean_delta": summary.get(delta_col, np.nan),
        "ci95_low_delta": summary.get(cols["ci_low"], np.nan),
        "ci95_high_delta": summary.get(cols["ci_high"], np.nan),
        "paired_p": summary.get(cols["p"], np.nan),
        "cohen_d": summary.get(cols["d"], np.nan),
    })

    probe_quality = pd.DataFrame({
        "model": [model],
        "model_short": [pretty_model_name(model)],
        "run_dir": [str(run_dir)],
        "pooling": [args.get("pooling", "")],
        "probe_accuracy": [record["probe_accuracy"]],
        "probe_auc": [record["probe_auc"]],
        "probe_f1": [record["probe_f1"]],
    })
    return record, layerwise, probe_quality


def bootstrap_ci(values, n_boot=5000, ci=95, seed=42):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    stats = [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    alpha = (100 - ci) / 2
    return float(np.percentile(stats, alpha)), float(np.percentile(stats, 100 - alpha))


def bootstrap_diff_ci(a, b, n_boot=5000, ci=95, seed=42):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        aa = rng.choice(a, size=len(a), replace=True)
        bb = rng.choice(b, size=len(b), replace=True)
        diffs.append(np.mean(aa) - np.mean(bb))
    alpha = (100 - ci) / 2
    return float(np.mean(a) - np.mean(b)), float(np.percentile(diffs, alpha)), float(np.percentile(diffs, 100 - alpha))


def permutation_p_value(a, b, n_perm=10000, seed=42):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    observed = abs(np.mean(a) - np.mean(b))
    pooled = np.concatenate([a, b])
    n_a = len(a)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        diff = abs(np.mean(pooled[:n_a]) - np.mean(pooled[n_a:]))
        if diff >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)


def paired_bootstrap_diff(a, b, n_boot=5000, ci=95, seed=42):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~np.isnan(a) & ~np.isnan(b)
    diff = a[mask] - b[mask]
    if len(diff) == 0:
        return np.nan, np.nan, np.nan
    if len(diff) == 1:
        return float(diff[0]), float(diff[0]), float(diff[0])
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(np.arange(len(diff)), size=len(diff), replace=True)
        boots.append(np.mean(diff[idx]))
    alpha = (100 - ci) / 2
    return float(np.mean(diff)), float(np.percentile(boots, alpha)), float(np.percentile(boots, 100 - alpha))


def sign_flip_p_value(a, b, n_perm=10000, seed=42):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~np.isnan(a) & ~np.isnan(b)
    diff = a[mask] - b[mask]
    if len(diff) == 0:
        return np.nan
    observed = abs(np.mean(diff))
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=len(diff), replace=True)
        stat = abs(np.mean(diff * signs))
        if stat >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)


def cohen_d_paired(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~np.isnan(a) & ~np.isnan(b)
    diff = a[mask] - b[mask]
    if len(diff) < 2 or np.std(diff, ddof=1) == 0:
        return np.nan
    return float(np.mean(diff) / np.std(diff, ddof=1))


def assign_layer_regions(df):
    out = []
    for _, sub in df.groupby("model_short", sort=False):
        sub = sub.copy()
        layers = sorted(sub["layer"].unique())
        n = len(layers)
        layer_to_rank = {layer: i for i, layer in enumerate(layers)}
        regions = {}
        for layer in layers:
            frac = layer_to_rank[layer] / max(n - 1, 1)
            if frac < 1 / 3:
                regions[layer] = "early"
            elif frac < 2 / 3:
                regions[layer] = "middle"
            else:
                regions[layer] = "late"
        sub["region"] = sub["layer"].map(regions)
        sub["layer_rank_fraction"] = sub["layer"].map(lambda x: layer_to_rank[x] / max(n - 1, 1))
        out.append(sub)
    return pd.concat(out, ignore_index=True)


def make_region_summary(df, n_boot, seed):
    rows = []
    for (model, region), sub in df.groupby(["model_short", "region"], sort=False):
        vals = sub["mean_delta"].to_numpy(dtype=float)
        lo, hi = bootstrap_ci(vals, n_boot=n_boot, seed=seed)
        rows.append({
            "model_short": model,
            "region": region,
            "n_layers": len(vals),
            "mean_delta": float(np.mean(vals)),
            "sem_delta": float(pd.Series(vals).sem()),
            "ci95_low": lo,
            "ci95_high": hi,
            "min_delta": float(np.min(vals)),
            "max_delta": float(np.max(vals)),
        })
    order = {"early": 0, "middle": 1, "late": 2}
    return pd.DataFrame(rows).sort_values(["model_short", "region"], key=lambda s: s.map(order) if s.name == "region" else s)


def make_within_model_contrasts(df, n_boot, n_perm, seed):
    rows = []
    for model, sub in df.groupby("model_short", sort=False):
        for a, b in [("early", "middle"), ("early", "late"), ("middle", "late")]:
            va = sub[sub["region"] == a]["mean_delta"].to_numpy(dtype=float)
            vb = sub[sub["region"] == b]["mean_delta"].to_numpy(dtype=float)
            diff, lo, hi = bootstrap_diff_ci(va, vb, n_boot=n_boot, seed=seed)
            p = permutation_p_value(va, vb, n_perm=n_perm, seed=seed)
            rows.append({
                "model_short": model,
                "contrast": f"{a}_minus_{b}",
                "region_a": a,
                "region_b": b,
                "mean_a": float(np.mean(va)) if len(va) else np.nan,
                "mean_b": float(np.mean(vb)) if len(vb) else np.nan,
                "mean_diff": diff,
                "ci95_low": lo,
                "ci95_high": hi,
                "permutation_p": p,
            })
    return pd.DataFrame(rows)


def make_between_model_contrasts(df, n_boot, n_perm, seed):
    rows = []
    models = list(df["model_short"].drop_duplicates())
    for region in ["early", "middle", "late"]:
        sub_region = df[df["region"] == region]
        for m1, m2 in combinations(models, 2):
            v1 = sub_region[sub_region["model_short"] == m1]["mean_delta"].to_numpy(dtype=float)
            v2 = sub_region[sub_region["model_short"] == m2]["mean_delta"].to_numpy(dtype=float)
            diff, lo, hi = bootstrap_diff_ci(v1, v2, n_boot=n_boot, seed=seed)
            p = permutation_p_value(v1, v2, n_perm=n_perm, seed=seed)
            rows.append({
                "region": region,
                "model_a": m1,
                "model_b": m2,
                "contrast": f"{m1}_minus_{m2}",
                "mean_a": float(np.mean(v1)) if len(v1) else np.nan,
                "mean_b": float(np.mean(v2)) if len(v2) else np.nan,
                "mean_diff": diff,
                "ci95_low": lo,
                "ci95_high": hi,
                "permutation_p": p,
            })
    return pd.DataFrame(rows)


def try_fit_ols(df, outpath):
    try:
        import statsmodels.formula.api as smf
        text = smf.ols("mean_delta ~ C(model_short) * C(region)", data=df).fit().summary().as_text()
    except Exception as e:
        text = f"Could not fit statsmodels OLS interaction model.\n\nError: {repr(e)}"
    outpath.write_text(text, encoding="utf-8")


def resolve_condition(df: pd.DataFrame, preferred: str):
    cols = list(df.columns)
    if preferred in cols:
        return preferred
    candidates = [c for c in cols if c.startswith(preferred)]
    if candidates:
        return sorted(candidates, key=lambda c: (len(c), c))[0]
    indirect = [c for c in cols if c.startswith("suppressed_indirect")]
    if indirect:
        return sorted(indirect, key=lambda c: (len(c), c))[0]
    raise ValueError(f"Could not find condition {preferred!r}. Available: {cols}")


def analyze_ordering_run(run_dir: Path, condition_preferred: str, n_boot: int, n_perm: int, seed: int):
    args = load_args(run_dir)
    model = args.get("model", run_dir.name)
    model_short = pretty_model_name(model)
    df = pd.read_csv(run_dir / "salience_deltas.csv")
    condition = resolve_condition(df, condition_preferred)
    required = ["absent", "mentioned", condition]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{run_dir} missing required columns {missing}")

    condition_rows = []
    for cond in required:
        vals = df[cond].astype(float).to_numpy()
        lo, hi = bootstrap_ci(vals, n_boot=n_boot, seed=seed)
        condition_rows.append({
            "model": model,
            "model_short": model_short,
            "run_dir": str(run_dir),
            "condition": cond,
            "mean_score": float(np.nanmean(vals)),
            "ci95_low": lo,
            "ci95_high": hi,
            "n": int(np.sum(~np.isnan(vals))),
        })
    condition_summary = pd.DataFrame(condition_rows)

    comparisons = [
        ("mentioned", condition, "mentioned_minus_suppressed_indirect"),
        (condition, "absent", "suppressed_indirect_minus_absent"),
        ("mentioned", "absent", "mentioned_minus_absent"),
    ]
    test_rows = []
    for a, b, label in comparisons:
        mean_diff, lo, hi = paired_bootstrap_diff(df[a], df[b], n_boot=n_boot, seed=seed)
        p = sign_flip_p_value(df[a], df[b], n_perm=n_perm, seed=seed)
        d = cohen_d_paired(df[a], df[b])
        test_rows.append({
            "model": model,
            "model_short": model_short,
            "run_dir": str(run_dir),
            "comparison": label,
            "condition_a": a,
            "condition_b": b,
            "mean_a": float(np.nanmean(df[a].astype(float))),
            "mean_b": float(np.nanmean(df[b].astype(float))),
            "mean_diff": mean_diff,
            "ci95_low": lo,
            "ci95_high": hi,
            "cohen_d_paired": d,
            "permutation_p": p,
            "n_matched": int((~df[a].isna() & ~df[b].isna()).sum()),
        })
    pairwise = pd.DataFrame(test_rows)

    layer_rows = []
    for layer, sub in df.groupby("layer"):
        layer_rows.append({
            "model": model,
            "model_short": model_short,
            "run_dir": str(run_dir),
            "layer": layer,
            "condition": condition,
            "mean_absent": float(np.nanmean(sub["absent"])),
            "mean_suppressed_indirect": float(np.nanmean(sub[condition])),
            "mean_mentioned": float(np.nanmean(sub["mentioned"])),
            "delta_suppressed_indirect_minus_absent": float(np.nanmean(sub[condition] - sub["absent"])),
            "delta_mentioned_minus_suppressed_indirect": float(np.nanmean(sub["mentioned"] - sub[condition])),
            "delta_mentioned_minus_absent": float(np.nanmean(sub["mentioned"] - sub["absent"])),
            "n": len(sub),
        })
    layerwise = pd.DataFrame(layer_rows)

    def comp(label):
        return pairwise[pairwise["comparison"] == label].iloc[0]

    ms = comp("mentioned_minus_suppressed_indirect")
    sa = comp("suppressed_indirect_minus_absent")
    ma = comp("mentioned_minus_absent")
    model_summary = {
        "model": model,
        "model_short": model_short,
        "run_dir": str(run_dir),
        "suppression_condition": condition,
        "mean_absent": float(condition_summary[condition_summary["condition"] == "absent"]["mean_score"].iloc[0]),
        "mean_suppressed_indirect": float(condition_summary[condition_summary["condition"] == condition]["mean_score"].iloc[0]),
        "mean_mentioned": float(condition_summary[condition_summary["condition"] == "mentioned"]["mean_score"].iloc[0]),
        "mentioned_minus_suppressed_indirect": float(ms["mean_diff"]),
        "ms_ci95_low": float(ms["ci95_low"]),
        "ms_ci95_high": float(ms["ci95_high"]),
        "ms_p": float(ms["permutation_p"]),
        "ms_d": float(ms["cohen_d_paired"]),
        "suppressed_indirect_minus_absent": float(sa["mean_diff"]),
        "sa_ci95_low": float(sa["ci95_low"]),
        "sa_ci95_high": float(sa["ci95_high"]),
        "sa_p": float(sa["permutation_p"]),
        "sa_d": float(sa["cohen_d_paired"]),
        "mentioned_minus_absent": float(ma["mean_diff"]),
        "ma_ci95_low": float(ma["ci95_low"]),
        "ma_ci95_high": float(ma["ci95_high"]),
        "ma_p": float(ma["permutation_p"]),
        "ma_d": float(ma["cohen_d_paired"]),
        "ordering_supported": bool((ms["mean_diff"] > 0) and (ms["permutation_p"] < 0.05) and (sa["mean_diff"] > 0) and (sa["permutation_p"] < 0.05)),
        "n_matched": int(sa["n_matched"]),
    }
    return model_summary, condition_summary, pairwise, layerwise


def filtered_run_dirs(root: Path, require_only_indirect: bool, require_pooling: str = None):
    for run_dir in find_run_dirs(root, ("probe_report_by_seed.csv", "salience_summary_by_layer.csv")):
        args = load_args(run_dir)
        if require_only_indirect and not bool(args.get("only_indirect", False)):
            continue
        if require_pooling and args.get("pooling") != require_pooling:
            continue
        yield run_dir


def build_cross_model_tables(run_dirs, delta_preference: str):
    records = []
    layerwise_tables = []
    probe_tables = []

    for run_dir in run_dirs:
        record, layerwise, probe_quality = summarize_run(run_dir, delta_preference)
        records.append(record)
        layerwise_tables.append(layerwise)
        probe_tables.append(probe_quality)

    if not records:
        raise RuntimeError("No runs remained after filtering.")

    return (
        apply_model_order(pd.DataFrame(records)),
        apply_model_order(pd.concat(layerwise_tables, ignore_index=True)),
        apply_model_order(pd.concat(probe_tables, ignore_index=True)),
    )


def write_cross_model_outputs(summary_df: pd.DataFrame, layerwise: pd.DataFrame, probe_quality: pd.DataFrame, outdir: Path, args):
    summary_df.to_csv(outdir / "cross_model_summary.csv", index=False)
    layerwise.to_csv(outdir / "cross_model_layerwise.csv", index=False)
    probe_quality.to_csv(outdir / "cross_model_probe_quality.csv", index=False)

    make_cross_model_latex_table(summary_df, outdir / "cross_model_table.tex", args.caption, args.label)
    plot_delta_by_layer(layerwise, outdir / "cross_model_delta_by_layer.png", "Indirect suppression salience across model families")
    plot_peak_delta(summary_df, outdir / "cross_model_peak_delta_bar.png")
    plot_probe_auc(summary_df, outdir / "cross_model_probe_auc_bar.png")
    write_cross_model_report(summary_df, outdir / "cross_model_report.md")


def write_cross_model_report(summary_df: pd.DataFrame, outpath: Path):
    best = summary_df.sort_values("peak_delta", ascending=False).iloc[0]
    lines = [
        "# Experiment 4 Summary",
        "",
        "## cross_model_peak",
        "",
        *make_variable_table([
            ("model_short", best.get("model_short", "")),
            ("selected_condition", best.get("selected_condition", "")),
            ("pooling", best.get("pooling", "")),
            ("peak_layer", best.get("peak_layer", np.nan)),
            ("peak_delta", best.get("peak_delta", np.nan)),
            ("peak_ci95_low", best.get("peak_ci95_low", np.nan)),
            ("peak_ci95_high", best.get("peak_ci95_high", np.nan)),
            ("peak_paired_p", best.get("peak_paired_p", np.nan)),
            ("peak_cohen_d", best.get("peak_cohen_d", np.nan)),
            ("probe_auc", best.get("probe_auc", np.nan)),
        ]),
        "",
        "## outputs",
        "",
        *make_variable_table([(name, name) for name in MAIN_OUTPUT_FILES]),
    ]
    outpath.write_text("\n".join(lines), encoding="utf-8")


def make_cross_model_latex_table(summary_df: pd.DataFrame, outpath: Path, caption: str, label: str):
    rows = []
    for _, r in summary_df.iterrows():
        ci = f"[{r['peak_ci95_low']:.3f}, {r['peak_ci95_high']:.3f}]"
        rows.append(
            f"{latex_escape(r['model_short'])} & "
            f"{latex_escape(r['pooling'])} & "
            f"{r['probe_auc']:.3f} & "
            f"{r['probe_accuracy']:.3f} & "
            f"{int(r['peak_layer'])} & "
            f"{r['peak_delta']:.3f} & "
            f"{ci} & "
            f"{r['peak_cohen_d']:.2f} & "
            f"{format_p(r['peak_paired_p'])} \\\\"
        )
    table = """\\begin{table*}[t]
\\centering
\\small
\\begin{tabular}{l l c c c c c c c}
\\toprule
Model & Pooling & AUC & Acc. & Peak layer & Peak $\\Delta$ & 95\\% CI & $d$ & $p$ \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\caption{""" + caption + """}
\\label{""" + label + """}
\\end{table*}
"""
    outpath.write_text(table, encoding="utf-8")


def plot_delta_by_layer(layerwise: pd.DataFrame, outpath: Path, title: str):
    plt.figure(figsize=(7.2, 4.6))
    for model_short, sub in layerwise.groupby("model_short", sort=False):
        sub = sub.sort_values("layer")
        plt.plot(sub["layer"], sub["mean_delta"], label=model_short)
        if "ci95_low_delta" in sub and "ci95_high_delta" in sub:
            plt.fill_between(sub["layer"], sub["ci95_low_delta"], sub["ci95_high_delta"], alpha=0.15)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Layer")
    plt.ylabel("Suppression salience delta vs. absent")
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_peak_delta(summary_df: pd.DataFrame, outpath: Path):
    labels = summary_df["model_short"].tolist()
    x = np.arange(len(summary_df))
    y = summary_df["peak_delta"].to_numpy(dtype=float)
    lo = summary_df["peak_ci95_low"].to_numpy(dtype=float)
    hi = summary_df["peak_ci95_high"].to_numpy(dtype=float)
    plt.figure(figsize=(6.8, 4.3))
    plt.bar(x, y)
    plt.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), fmt="none", capsize=4, linewidth=1)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Peak delta vs. absent")
    plt.title("Peak indirect suppression salience by model")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_probe_auc(summary_df: pd.DataFrame, outpath: Path):
    labels = summary_df["model_short"].tolist()
    x = np.arange(len(summary_df))
    y = summary_df["probe_auc"].to_numpy(dtype=float)
    plt.figure(figsize=(6.8, 4.3))
    plt.bar(x, y)
    plt.axhline(0.5, linestyle="--", linewidth=1)
    plt.ylim(0, 1.0)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Mean probe AUC")
    plt.title("Probe quality across model families")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def write_region_outputs(layerwise: pd.DataFrame, outdir: Path, args):
    region_dir = outdir / "region_analysis"
    region_dir.mkdir(parents=True, exist_ok=True)

    df = assign_layer_regions(layerwise)
    region_summary = make_region_summary(df, args.n_boot, args.seed)
    within = make_within_model_contrasts(df, args.n_boot, args.n_perm, args.seed)
    between = make_between_model_contrasts(df, args.n_boot, args.n_perm, args.seed)

    df.to_csv(region_dir / "cross_model_layerwise_with_regions.csv", index=False)
    region_summary.to_csv(region_dir / "region_summary.csv", index=False)
    within.to_csv(region_dir / "within_model_region_contrasts.csv", index=False)
    between.to_csv(region_dir / "between_model_region_contrasts.csv", index=False)

    try_fit_ols(df, region_dir / "model_region_interaction_ols.txt")
    make_region_latex_table(region_summary, region_dir / "region_summary_table.tex")
    plot_region_summary(region_summary, region_dir / "region_delta_by_model.png")
    write_region_report(region_summary, within, between, region_dir / "region_analysis_report.md")
    return region_summary, within, between


def write_region_report(region_summary: pd.DataFrame, within: pd.DataFrame, between: pd.DataFrame, outpath: Path):
    best = region_summary.sort_values("mean_delta", ascending=False).iloc[0]
    lines = [
        "# Experiment 4 Region Analysis",
        "",
        "## top_region",
        "",
        *make_variable_table([
            ("model_short", best.get("model_short", "")),
            ("region", best.get("region", "")),
            ("n_layers", best.get("n_layers", np.nan)),
            ("mean_delta", best.get("mean_delta", np.nan)),
            ("ci95_low", best.get("ci95_low", np.nan)),
            ("ci95_high", best.get("ci95_high", np.nan)),
        ]),
        "",
        "## counts",
        "",
        *make_variable_table([
            ("n_region_rows", len(region_summary)),
            ("n_within_model_contrasts", len(within)),
            ("n_between_model_contrasts", len(between)),
        ]),
    ]
    outpath.write_text("\n".join(lines), encoding="utf-8")


def make_region_latex_table(region_summary: pd.DataFrame, outpath: Path):
    order = {"early": 0, "middle": 1, "late": 2}
    df = region_summary.copy()
    df["_region_order"] = df["region"].map(order)
    df = df.sort_values(["model_short", "_region_order"])
    lines = []
    for _, r in df.iterrows():
        ci = f"[{r['ci95_low']:.3f}, {r['ci95_high']:.3f}]"
        lines.append(
            f"{r['model_short']} & {r['region']} & {int(r['n_layers'])} & "
            f"{r['mean_delta']:.3f} & {ci} \\\\"
        )
    latex = r"""\begin{table}[t]
\centering
\small
\begin{tabular}{l l c c c}
\toprule
Model & Region & Layers & Mean $\Delta$ & 95\% CI \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\caption{Layer-region summary of indirect suppression salience across model families.}
\label{tab:layer-region-suppression}
\end{table}
"""
    outpath.write_text(latex, encoding="utf-8")


def plot_region_summary(region_summary: pd.DataFrame, outpath: Path):
    models = list(region_summary["model_short"].drop_duplicates())
    region_names = ["early", "middle", "late"]
    x = np.arange(len(models))
    width = 0.25
    plt.figure(figsize=(7.2, 4.5))
    for i, region in enumerate(region_names):
        vals, lows, highs = [], [], []
        for model in models:
            row = region_summary[(region_summary["model_short"] == model) & (region_summary["region"] == region)]
            if len(row) == 0:
                vals.append(np.nan)
                lows.append(np.nan)
                highs.append(np.nan)
            else:
                r = row.iloc[0]
                vals.append(r["mean_delta"])
                lows.append(r["ci95_low"])
                highs.append(r["ci95_high"])
        vals = np.asarray(vals, dtype=float)
        lows = np.asarray(lows, dtype=float)
        highs = np.asarray(highs, dtype=float)
        xpos = x + (i - 1) * width
        plt.bar(xpos, vals, width, label=region)
        plt.errorbar(xpos, vals, yerr=np.vstack([vals - lows, highs - vals]), fmt="none", capsize=3, linewidth=1)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(x, models, rotation=20, ha="right")
    plt.ylabel("Mean suppression salience delta")
    plt.title("Layer-region suppression salience by model")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def ordering_run_dirs(root: Path, require_only_indirect: bool, require_pooling: str = None):
    for run_dir in find_run_dirs(root, ("salience_deltas.csv",)):
        args = load_args(run_dir)
        if require_only_indirect and not bool(args.get("only_indirect", False)):
            continue
        if require_pooling and args.get("pooling") != require_pooling:
            continue
        yield run_dir


def write_ordering_outputs(root: Path, outdir: Path, args):
    ordering_dir = outdir / "ordering_analysis"
    ordering_dir.mkdir(parents=True, exist_ok=True)

    model_summaries = []
    condition_summaries = []
    pairwise_tests = []
    layerwise_summaries = []
    for run_dir in ordering_run_dirs(root, args.require_only_indirect, args.require_pooling):
        model_summary, condition_summary, pairwise, layerwise = analyze_ordering_run(
            run_dir, args.ordering_condition, args.n_boot, args.n_perm, args.seed
        )
        model_summaries.append(model_summary)
        condition_summaries.append(condition_summary)
        pairwise_tests.append(pairwise)
        layerwise_summaries.append(layerwise)

    if not model_summaries:
        raise RuntimeError("No ordering runs remained after filtering.")

    model_summary = apply_model_order(pd.DataFrame(model_summaries))
    condition_summary = apply_model_order(pd.concat(condition_summaries, ignore_index=True))
    pairwise = apply_model_order(pd.concat(pairwise_tests, ignore_index=True))
    layerwise = apply_model_order(pd.concat(layerwise_summaries, ignore_index=True))

    model_summary.to_csv(ordering_dir / "ordering_model_summary.csv", index=False)
    condition_summary.to_csv(ordering_dir / "ordering_condition_summary.csv", index=False)
    pairwise.to_csv(ordering_dir / "ordering_pairwise_tests.csv", index=False)
    layerwise.to_csv(ordering_dir / "ordering_layerwise_summary.csv", index=False)

    make_ordering_latex_table(model_summary, ordering_dir / "ordering_table.tex")
    plot_condition_means(model_summary, ordering_dir / "ordering_condition_means_by_model.png")
    plot_pairwise_deltas(model_summary, ordering_dir / "ordering_deltas_by_model.png")
    write_ordering_report(model_summary, pairwise, ordering_dir / "ordering_analysis_report.md")
    return model_summary, condition_summary, pairwise, layerwise


def write_ordering_report(model_summary: pd.DataFrame, pairwise: pd.DataFrame, outpath: Path):
    best = model_summary.sort_values("suppressed_indirect_minus_absent", ascending=False).iloc[0]
    lines = [
        "# Experiment 4 Ordering Analysis",
        "",
        "## top_ordering_effect",
        "",
        *make_variable_table([
            ("model_short", best.get("model_short", "")),
            ("suppression_condition", best.get("suppression_condition", "")),
            ("mean_absent", best.get("mean_absent", np.nan)),
            ("mean_suppressed_indirect", best.get("mean_suppressed_indirect", np.nan)),
            ("mean_mentioned", best.get("mean_mentioned", np.nan)),
            ("suppressed_indirect_minus_absent", best.get("suppressed_indirect_minus_absent", np.nan)),
            ("sa_p", best.get("sa_p", np.nan)),
            ("mentioned_minus_suppressed_indirect", best.get("mentioned_minus_suppressed_indirect", np.nan)),
            ("ms_p", best.get("ms_p", np.nan)),
            ("ordering_supported", best.get("ordering_supported", "")),
        ]),
        "",
        "## counts",
        "",
        *make_variable_table([
            ("n_model_rows", len(model_summary)),
            ("n_pairwise_rows", len(pairwise)),
        ]),
    ]
    outpath.write_text("\n".join(lines), encoding="utf-8")


def make_ordering_latex_table(model_summary: pd.DataFrame, outpath: Path):
    lines = []
    for _, r in model_summary.iterrows():
        lines.append(
            f"{r['model_short']} & "
            f"{r['mean_absent']:.3f} & "
            f"{r['mean_suppressed_indirect']:.3f} & "
            f"{r['mean_mentioned']:.3f} & "
            f"{r['suppressed_indirect_minus_absent']:.3f} & "
            f"[{r['sa_ci95_low']:.3f}, {r['sa_ci95_high']:.3f}] & "
            f"{format_p(r['sa_p'])} & "
            f"{r['mentioned_minus_suppressed_indirect']:.3f} & "
            f"{format_p(r['ms_p'])} \\\\"
        )
    latex = r"""\begin{table*}[t]
\centering
\small
\begin{tabular}{lcccccccc}
\toprule
Model & Absent & Supp. indirect & Mention & Supp.--Absent & 95\% CI & $p$ & Mention--Supp. & $p$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\caption{Condition-ordering analysis for Llama, Mistral, and Gemma indirect suppression experiments.}
\label{tab:condition-ordering}
\end{table*}
"""
    outpath.write_text(latex, encoding="utf-8")


def plot_condition_means(model_summary: pd.DataFrame, outpath: Path):
    models = model_summary["model_short"].tolist()
    conditions = [
        ("mean_absent", "absent"),
        ("mean_suppressed_indirect", "suppressed indirect"),
        ("mean_mentioned", "mentioned"),
    ]
    x = np.arange(len(models))
    width = 0.25
    plt.figure(figsize=(8, 4.8))
    for i, (col, label) in enumerate(conditions):
        plt.bar(x + (i - 1) * width, model_summary[col].to_numpy(dtype=float), width, label=label)
    plt.xticks(x, models, rotation=20, ha="right")
    plt.ylabel("Mean probe score")
    plt.title("Condition ordering across Llama, Mistral, and Gemma")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_pairwise_deltas(model_summary: pd.DataFrame, outpath: Path):
    models = model_summary["model_short"].tolist()
    comparisons = [
        ("suppressed_indirect_minus_absent", "supp. indirect - absent", "sa_ci95_low", "sa_ci95_high"),
        ("mentioned_minus_suppressed_indirect", "mentioned - supp. indirect", "ms_ci95_low", "ms_ci95_high"),
    ]
    x = np.arange(len(models))
    width = 0.32
    plt.figure(figsize=(8, 4.8))
    for i, (col, label, lo_col, hi_col) in enumerate(comparisons):
        vals = model_summary[col].to_numpy(dtype=float)
        lo = model_summary[lo_col].to_numpy(dtype=float)
        hi = model_summary[hi_col].to_numpy(dtype=float)
        xpos = x + (i - 0.5) * width
        plt.bar(xpos, vals, width, label=label)
        plt.errorbar(xpos, vals, yerr=np.vstack([vals - lo, hi - vals]), fmt="none", capsize=4, linewidth=1)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(x, models, rotation=20, ha="right")
    plt.ylabel("Mean paired difference")
    plt.title("Pairwise condition differences")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="results_cross_model")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--delta_preference", type=str, default="indirect", choices=["indirect", "direct", "mentioned"])
    parser.add_argument("--ordering_condition", type=str, default="suppressed_indirect")
    parser.add_argument("--require_only_indirect", action="store_true")
    parser.add_argument("--require_pooling", type=str, default=None)
    parser.add_argument("--n_boot", type=int, default=5000)
    parser.add_argument("--n_perm", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--caption",
        type=str,
        default=(
            "Cross-model replication of indirect suppression salience. "
            "Peak $\\Delta$ denotes the maximum layerwise difference between "
            "indirect suppression and concept-absent baselines."
        ),
    )
    parser.add_argument("--label", type=str, default="tab:cross-model-suppression")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    root = Path(args.root)
    outdir = Path(args.outdir) if args.outdir else root / "consolidated"
    outdir.mkdir(parents=True, exist_ok=True)

    run_dirs = list(filtered_run_dirs(root, args.require_only_indirect, args.require_pooling))
    if not run_dirs:
        raise RuntimeError("No cross-model runs remained after filtering.")

    summary_df, layerwise, probe_quality = build_cross_model_tables(run_dirs, args.delta_preference)
    write_cross_model_outputs(summary_df, layerwise, probe_quality, outdir, args)
    write_region_outputs(layerwise, outdir, args)
    write_ordering_outputs(root, outdir, args)

    print("\nDone.")
    print(f"Included {len(summary_df)} cross-model runs.")
    print(f"Outputs written to: {outdir}")
    print(summary_df[[
        "model_short",
        "pooling",
        "selected_condition",
        "probe_auc",
        "peak_layer",
        "peak_delta",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
