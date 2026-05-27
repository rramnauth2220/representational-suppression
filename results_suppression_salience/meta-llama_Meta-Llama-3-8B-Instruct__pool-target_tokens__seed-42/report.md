# Experiment 1 Summary

## run_config

| variable | value |
|---|---|
| `model` | meta-llama/Meta-Llama-3-8B-Instruct |
| `pooling` | target_tokens |
| `concepts_path` | concepts.json |
| `probe_seeds` | 13,42,101 |
| `include_hard_negatives` | True |
| `indirect_mode` | first |

## probe_quality

| variable | value |
|---|---|
| `probe_accuracy_mean` | 0.82766 |
| `probe_auc_mean` | 0.880075 |
| `probe_f1_mean` | 0.795504 |

## peak_salience

| variable | value |
|---|---|
| `selected_delta_column` | mean_delta_suppressed_minus_absent |
| `peak_layer` | 4 |
| `mean_absent` | 0.0727664 |
| `mean_suppressed` | 0.883009 |
| `mean_mentioned` | 0.872546 |
| `mean_delta_suppressed_minus_absent` | 0.810243 |
| `ci95_low_delta_suppressed_minus_absent` | 0.757788 |
| `ci95_high_delta_suppressed_minus_absent` | 0.86059 |
| `paired_p_suppressed_vs_absent` | 2.44276e-63 |
| `wilcoxon_p_suppressed_vs_absent` | 7.02092e-24 |
| `cohen_d_paired_suppressed_vs_absent` | 2.6627 |