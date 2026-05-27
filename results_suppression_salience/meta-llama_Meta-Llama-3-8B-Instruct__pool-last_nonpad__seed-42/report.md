# Experiment 1 Summary

## run_config

| variable | value |
|---|---|
| `model` | meta-llama/Meta-Llama-3-8B-Instruct |
| `pooling` | last_nonpad |
| `concepts_path` | concepts.json |
| `probe_seeds` | 13,42,101 |
| `include_hard_negatives` | True |
| `indirect_mode` | first |

## probe_quality

| variable | value |
|---|---|
| `probe_accuracy_mean` | 0.778994 |
| `probe_auc_mean` | 0.860044 |
| `probe_f1_mean` | 0.753516 |

## peak_salience

| variable | value |
|---|---|
| `selected_delta_column` | mean_delta_suppressed_minus_absent |
| `peak_layer` | 3 |
| `mean_absent` | 0.389083 |
| `mean_suppressed` | 0.666773 |
| `mean_mentioned` | 0.365689 |
| `mean_delta_suppressed_minus_absent` | 0.27769 |
| `ci95_low_delta_suppressed_minus_absent` | 0.20856 |
| `ci95_high_delta_suppressed_minus_absent` | 0.33727 |
| `paired_p_suppressed_vs_absent` | 4.79629e-14 |
| `wilcoxon_p_suppressed_vs_absent` | 2.54294e-12 |
| `cohen_d_paired_suppressed_vs_absent` | 0.722255 |