# Experiment 1 Summary

## run_config

| variable | value |
|---|---|
| `model` | meta-llama/Meta-Llama-3-8B-Instruct |
| `pooling` | mean_nonpad |
| `concepts_path` | concepts.json |
| `probe_seeds` | 13,42,101 |
| `include_hard_negatives` | True |
| `include_indirect` | True |
| `only_indirect` | False |
| `indirect_mode` | first |

## probe_quality

| variable | value |
|---|---|
| `probe_accuracy_mean` | 0.826698 |
| `probe_auc_mean` | 0.918697 |
| `probe_f1_mean` | 0.82012 |

## peak_salience

| variable | value |
|---|---|
| `selected_delta_column` | mean_delta_suppressed_minus_absent |
| `peak_layer` | 1 |
| `mean_absent` | 0.160198 |
| `mean_suppressed` | 0.792374 |
| `mean_suppressed_indirect` | 0.680165 |
| `mean_mentioned` | 0.761065 |
| `mean_delta_suppressed_minus_absent` | 0.632177 |
| `ci95_low_delta_suppressed_minus_absent` | 0.581089 |
| `ci95_high_delta_suppressed_minus_absent` | 0.681067 |
| `paired_p_suppressed_vs_absent` | 8.57045e-52 |
| `wilcoxon_p_suppressed_vs_absent` | 4.59782e-24 |
| `cohen_d_paired_suppressed_vs_absent` | 2.11225 |