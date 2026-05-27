# Cross-Model Suppression Salience Consolidation

## Summary

| model_short      | pooling     | selected_condition   |   probe_auc |   probe_accuracy |   peak_layer |   peak_delta |   peak_ci95_low |   peak_ci95_high |   peak_cohen_d |   peak_paired_p |
|:-----------------|:------------|:---------------------|------------:|-----------------:|-------------:|-------------:|----------------:|-----------------:|---------------:|----------------:|
| Llama 3 8B Inst. | mean_nonpad | suppressed_indirect  |    0.915918 |         0.824067 |            1 |     0.510766 |        0.443553 |         0.576813 |        1.25223 |     1.44462e-29 |
| Mistral 7B Inst. | mean_nonpad | suppressed_indirect  |    0.90009  |         0.8061   |            0 |     0.525931 |        0.455439 |         0.591019 |        1.33145 |     7.86724e-32 |
| Gemma 7B IT      | mean_nonpad | suppressed_indirect  |    0.909642 |         0.810908 |           28 |     0.481231 |        0.42647  |         0.535095 |        1.52857 |     2.73742e-37 |

## Suggested paper sentence

Across model families, indirect suppression increased probe-estimated concept recoverability relative to concept-absent baselines. The effect was evaluated under a fixed configuration using indirect-only suppression and mean pooling over non-padding tokens.

## Files generated

- `cross_model_summary.csv`
- `cross_model_layerwise.csv`
- `cross_model_probe_quality.csv`
- `cross_model_table.tex`
- `cross_model_delta_by_layer.png`
- `cross_model_peak_delta_bar.png`
- `cross_model_probe_auc_bar.png`