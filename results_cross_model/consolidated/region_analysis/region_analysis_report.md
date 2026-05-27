# Layer-Region Analysis

## Region Summary

| model_short      | region   |   n_layers |   mean_delta |   sem_delta |   ci95_low |   ci95_high |   min_delta |   max_delta |
|:-----------------|:---------|-----------:|-------------:|------------:|-----------:|------------:|------------:|------------:|
| Gemma 7B IT      | early    |         10 |     0.418091 |  0.00794576 |   0.403508 |    0.433195 |    0.380928 |    0.459286 |
| Gemma 7B IT      | middle   |          9 |     0.378776 |  0.00625867 |   0.367515 |    0.390716 |    0.353028 |    0.408786 |
| Gemma 7B IT      | late     |         10 |     0.418211 |  0.0123606  |   0.393372 |    0.439975 |    0.336047 |    0.481231 |
| Llama 3 8B Inst. | early    |         11 |     0.434844 |  0.02143    |   0.393189 |    0.472141 |    0.295244 |    0.510766 |
| Llama 3 8B Inst. | middle   |         11 |     0.33626  |  0.0193321  |   0.30076  |    0.371838 |    0.249534 |    0.445885 |
| Llama 3 8B Inst. | late     |         11 |     0.471783 |  0.00730723 |   0.457016 |    0.484051 |    0.411873 |    0.496089 |
| Mistral 7B Inst. | early    |         11 |     0.411694 |  0.0185679  |   0.379457 |    0.448917 |    0.343119 |    0.525931 |
| Mistral 7B Inst. | middle   |         11 |     0.327931 |  0.00837822 |   0.311405 |    0.343284 |    0.28026  |    0.367058 |
| Mistral 7B Inst. | late     |         11 |     0.375189 |  0.00845952 |   0.360574 |    0.39171  |    0.344258 |    0.430371 |

## Within-Model Region Contrasts

| model_short      | contrast           | region_a   | region_b   |   mean_a |   mean_b |    mean_diff |   ci95_low |   ci95_high |   permutation_p |
|:-----------------|:-------------------|:-----------|:-----------|---------:|---------:|-------------:|-----------:|------------:|----------------:|
| Llama 3 8B Inst. | early_minus_middle | early      | middle     | 0.434844 | 0.33626  |  0.0985837   |  0.0415445 |  0.152215   |      0.00379962 |
| Llama 3 8B Inst. | early_minus_late   | early      | late       | 0.434844 | 0.471783 | -0.0369393   | -0.081493  |  0.00252418 |      0.129687   |
| Llama 3 8B Inst. | middle_minus_late  | middle     | late       | 0.33626  | 0.471783 | -0.135523    | -0.17447   | -0.0960738  |      9.999e-05  |
| Mistral 7B Inst. | early_minus_middle | early      | middle     | 0.411694 | 0.327931 |  0.083763    |  0.0478167 |  0.123576   |      9.999e-05  |
| Mistral 7B Inst. | early_minus_late   | early      | late       | 0.411694 | 0.375189 |  0.0365047   |  9.677e-05 |  0.0774209  |      0.0883912  |
| Mistral 7B Inst. | middle_minus_late  | middle     | late       | 0.327931 | 0.375189 | -0.0472583   | -0.0700565 | -0.0257395  |      0.00049995 |
| Gemma 7B IT      | early_minus_middle | early      | middle     | 0.418091 | 0.378776 |  0.039315    |  0.0204343 |  0.0578112  |      0.00169983 |
| Gemma 7B IT      | early_minus_late   | early      | late       | 0.418091 | 0.418211 | -0.000119022 | -0.0257401 |  0.0283721  |      0.9952     |
| Gemma 7B IT      | middle_minus_late  | middle     | late       | 0.378776 | 0.418211 | -0.039434    | -0.0646216 | -0.0126981  |      0.0151985  |

## Between-Model Region Contrasts

| region   | model_a          | model_b          | contrast                                |   mean_a |   mean_b |   mean_diff |   ci95_low |   ci95_high |   permutation_p |
|:---------|:-----------------|:-----------------|:----------------------------------------|---------:|---------:|------------:|-----------:|------------:|----------------:|
| early    | Llama 3 8B Inst. | Mistral 7B Inst. | Llama 3 8B Inst._minus_Mistral 7B Inst. | 0.434844 | 0.411694 |  0.0231505  | -0.0326249 |  0.0724188  |      0.414859   |
| early    | Llama 3 8B Inst. | Gemma 7B IT      | Llama 3 8B Inst._minus_Gemma 7B IT      | 0.434844 | 0.418091 |  0.0167526  | -0.0266544 |  0.0566592  |      0.49565    |
| early    | Mistral 7B Inst. | Gemma 7B IT      | Mistral 7B Inst._minus_Gemma 7B IT      | 0.411694 | 0.418091 | -0.00639792 | -0.0410443 |  0.0330363  |      0.763224   |
| middle   | Llama 3 8B Inst. | Mistral 7B Inst. | Llama 3 8B Inst._minus_Mistral 7B Inst. | 0.33626  | 0.327931 |  0.00832981 | -0.0308488 |  0.0476286  |      0.69933    |
| middle   | Llama 3 8B Inst. | Gemma 7B IT      | Llama 3 8B Inst._minus_Gemma 7B IT      | 0.33626  | 0.378776 | -0.0425161  | -0.0801879 | -0.00562166 |      0.0712929  |
| middle   | Mistral 7B Inst. | Gemma 7B IT      | Mistral 7B Inst._minus_Gemma 7B IT      | 0.327931 | 0.378776 | -0.0508459  | -0.0708311 | -0.0320955  |      0.00059994 |
| late     | Llama 3 8B Inst. | Mistral 7B Inst. | Llama 3 8B Inst._minus_Mistral 7B Inst. | 0.471783 | 0.375189 |  0.0965945  |  0.0746097 |  0.116673   |      9.999e-05  |
| late     | Llama 3 8B Inst. | Gemma 7B IT      | Llama 3 8B Inst._minus_Gemma 7B IT      | 0.471783 | 0.418211 |  0.0535729  |  0.0282109 |  0.0813915  |      0.00139986 |
| late     | Mistral 7B Inst. | Gemma 7B IT      | Mistral 7B Inst._minus_Gemma 7B IT      | 0.375189 | 0.418211 | -0.0430216  | -0.0697999 | -0.0135826  |      0.00979902 |

## Suggested cautious wording

Layer-region analyses provide descriptive evidence for architectural variation in where indirect suppression salience is concentrated. Because regions contain relatively few layer-level observations and adjacent layers are not statistically independent, these tests should be interpreted as exploratory rather than confirmatory.