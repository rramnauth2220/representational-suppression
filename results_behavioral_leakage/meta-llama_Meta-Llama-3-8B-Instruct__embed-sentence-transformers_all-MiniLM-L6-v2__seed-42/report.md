# Experiment 3 Summary

## run_config

| variable | value |
|---|---|
| `model` | meta-llama/Meta-Llama-3-8B-Instruct |
| `embedding_model` | sentence-transformers/all-MiniLM-L6-v2 |
| `concepts_path` | concepts.json |
| `n_samples` | 3 |
| `temperature` | 0.7 |
| `top_p` | 0.95 |
| `successful_suppression_only` | True |
| `include_indirect` | True |
| `include_unrelated_control` | True |
| `seed` | 42 |

## pairwise_comparison

| variable | value |
|---|---|
| `selected_comparison` | suppressed_minus_absent |
| `condition_a` | suppressed |
| `condition_b` | absent |
| `mean_a` | 0.290199 |
| `mean_b` | 0.273288 |
| `mean_delta` | 0.0169117 |
| `ci95_low_delta` | 0.00844726 |
| `ci95_high_delta` | 0.0251032 |
| `paired_p` | 0.000120461 |
| `wilcoxon_p` | 7.49039e-06 |
| `cohen_d_paired` | 0.343659 |
| `n_matched` | 133 |

## top_condition

| variable | value |
|---|---|
| `condition` | mentioned |
| `mean_similarity` | 0.388635 |
| `ci95_low` | 0.372932 |
| `ci95_high` | 0.403911 |
| `explicit_alias_leak_rate` | 0.436275 |
| `n` | 408 |

## outputs

| variable | value |
|---|---|
| `generations.csv` | generations.csv |
| `generation_similarity_scores.csv` | generation_similarity_scores.csv |
| `behavioral_leakage_summary_by_condition.csv` | behavioral_leakage_summary_by_condition.csv |
| `behavioral_leakage_pairwise_comparisons.csv` | behavioral_leakage_pairwise_comparisons.csv |
| `concept_level_leakage_summary.csv` | concept_level_leakage_summary.csv |
| `report.md` | report.md |
| `plot_semantic_similarity_by_condition.png` | plot_semantic_similarity_by_condition.png |
| `plot_suppression_delta_by_concept.png` | plot_suppression_delta_by_concept.png |
| `plot_explicit_alias_leak_rate.png` | plot_explicit_alias_leak_rate.png |
| `exp3_behavioral_combined.png` | exp3_behavioral_combined.png |
| `exp3_behavioral_combined.pdf` | exp3_behavioral_combined.pdf |