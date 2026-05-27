# Experiment 2 Summary

## run_config

| variable | value |
|---|---|
| `model` | meta-llama/Meta-Llama-3-8B-Instruct |
| `concepts_path` | concepts.json |
| `n_examples` | 544 |
| `include_indirect` | True |
| `include_unrelated_control` | True |
| `unrelated_control_region` | target_alias |
| `attn_implementation` | eager |
| `layers` | all |
| `max_length` | 256 |
| `save_token_region_debug` | False |
| `seed` | 42 |

## condition_comparison

| variable | value |
|---|---|
| `selected_region_label` | target_alias |
| `selected_comparison` | suppressed_minus_mentioned |
| `condition_a` | suppressed |
| `condition_b` | mentioned |
| `mean_a` | 0.00994933 |
| `mean_b` | 0.0118921 |
| `mean_delta` | -0.00194278 |
| `ci95_low_delta` | -0.00213001 |
| `ci95_high_delta` | -0.00174697 |
| `paired_p` | 2.09863e-40 |
| `wilcoxon_p` | 1.52673e-21 |
| `cohen_d_paired` | -1.7276 |
| `n_matched` | 129 |

## top_head

| variable | value |
|---|---|
| `comparison` | suppressed_minus_mentioned |
| `region_label` | target_alias |
| `layer` | 14 |
| `head` | 4 |
| `mean_a` | 0.0825421 |
| `mean_b` | 0.0141094 |
| `mean_delta` | 0.0684326 |
| `ci95_low_delta` | 0.0618658 |
| `ci95_high_delta` | 0.0756202 |
| `paired_p` | 2.06866e-40 |
| `wilcoxon_p` | 6.51335e-23 |
| `cohen_d_paired` | 1.72786 |
| `n_matched` | 129 |

## outputs

| variable | value |
|---|---|
| `attention_token_regions.csv` | attention_token_regions.csv |
| `attention_layer_summary.csv` | attention_layer_summary.csv |
| `attention_condition_comparisons.csv` | attention_condition_comparisons.csv |
| `attention_head_effects.csv` | attention_head_effects.csv |
| `top_suppression_sensitive_heads.csv` | top_suppression_sensitive_heads.csv |
| `experiment2_attention_combined.png` | experiment2_attention_combined.png |
| `experiment2_attention_combined.pdf` | experiment2_attention_combined.pdf |
| `experiment2_top10_heads_summary.csv` | experiment2_top10_heads_summary.csv |
| `report.md` | report.md |
| `plot_attention_by_layer.png` | plot_attention_by_layer.png |
| `plot_suppression_head_effects.png` | plot_suppression_head_effects.png |