# Suppression Salience Experiments

This repository contains the code used in:

> **The Attentional White Bear Effect in Transformer Language Models**
>
> *tl;dr*: This study investigates whether suppression-oriented instructions reduce or preserve latent representations of prohibited concepts in transformer language models.

The repository evaluates suppression across four experiments:

1. **Latent Recoverability** (Experiment 1)
2. **Attention Allocation** (Experiment 2)
3. **Behavioral Semantic Leakage** (Experiment 3)
4. **Cross-Model Replication** (Experiment 4)

---

# Installation

The experiments require Python 3.9+.

Install dependencies:

```bash
pip install torch transformers pandas numpy scipy scikit-learn matplotlib tqdm
```

Experiment 3 additionally requires a sentence embedding model downloaded automatically through Hugging Face.

Some models (e.g., Llama and Gemma) require Hugging Face authentication and acceptance of model licenses.

---

# Concept Library

All experiments operate on a structured concept library:

```bash
--concepts_path concepts.json
```

Each concept entry contains:

- aliases
- indirect semantic descriptions
- contextual prompts
- positive examples
- matched negative examples
- optional hard negatives
- optional unrelated control concepts

The experiments construct matched prompting conditions from these entries.

---

# Prompt Conditions

Each concept–context pair is evaluated under five matched conditions:

| Condition | Description |
|------------|-------------|
| `absent` | Concept not mentioned |
| `mentioned` | Concept explicitly included |
| `suppressed` | Direct suppression ("Do not mention X") |
| `suppressed_indirect` | Indirect suppression via semantic description |
| `unrelated_suppression` | Suppression of an unrelated concept |

These conditions are used throughout the experiments.

---

# Experiment 1: Latent Recoverability

Experiment 1 trains layerwise probes to measure whether suppressed concepts remain recoverable from hidden representations.

## Example

```bash
python exp1_recoverability.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --concepts_path concepts.json \
  --outdir results/exp1_llama_mean \
  --pooling mean_nonpad \
  --include_indirect \
  --use_chat_template \
  --bf16 \
  --device_map_auto
```

## Probe Configuration

Default probe settings:

```text
Classifier: Logistic Regression
Solver: liblinear
Class Weights: balanced
C: 1.0
Max Iterations: 5000
Train/Test Split: 65/35
Probe Seeds: {13, 42, 101}
```

## Pooling Strategies

Supported pooling modes:

```text
last_nonpad
mean_nonpad
target_tokens
```

## Outputs

```text
probe_report_by_seed.csv
probe_report.csv
salience_probe_scores.csv
salience_deltas.csv
salience_summary_by_layer.csv
plot_salience_by_layer.png
plot_delta_by_layer.png
report.md
```

---

# Experiment 2: Attention Allocation

Experiment 2 evaluates whether suppression affects attention allocation toward concept-associated token regions.

## Example

```bash
python exp2_attention.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --concepts_path concepts.json \
  --outdir results/exp2_attention \
  --include_indirect \
  --include_unrelated_control \
  --use_chat_template \
  --bf16 \
  --device_map_auto \
  --attn_implementation eager
```

## Outputs

```text
attention_token_regions.csv
attention_layer_summary.csv
attention_condition_comparisons.csv
attention_head_effects.csv
top_suppression_sensitive_heads.csv
experiment2_attention_combined.png
experiment2_attention_combined.pdf
report.md
```

---

# Experiment 3: Behavioral Semantic Leakage

Experiment 3 evaluates whether suppression changes output behavior while preserving semantic proximity to the prohibited concept.

The procedure consists of:

1. Generation under matched prompting conditions.
2. Explicit alias-leak detection.
3. Semantic similarity scoring using sentence embeddings.

## Example

```bash
python exp3_leakage.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --concepts_path concepts.json \
  --outdir results/exp3_leakage \
  --include_indirect \
  --include_unrelated_control \
  --use_chat_template \
  --bf16 \
  --device_map_auto
```

## Generation Settings

Default decoding settings:

```text
Temperature: 0.7
Top-p: 0.95
Max New Tokens: 80
Samples per Prompt: 3
```

## Outputs

```text
generations.csv
generation_similarity_scores.csv
behavioral_leakage_summary_by_condition.csv
behavioral_leakage_pairwise_comparisons.csv
concept_level_leakage_summary.csv
plot_semantic_similarity_by_condition.png
plot_suppression_delta_by_concept.png
exp3_behavioral_combined.png
report.md
```

---

# Experiment 4: Cross-Model Replication

Experiment 4 aggregates Experiment 1 outputs across model families and evaluates cross-model consistency.

Supported model families include:

- Llama 3
- Mistral
- Gemma

## Example

After running Experiment 1 separately for each model:

```bash
python exp4_cross_model.py \
  --root results_cross_model \
  --outdir results_cross_model/consolidated \
  --delta_preference indirect \
  --ordering_condition suppressed_indirect \
  --require_only_indirect \
  --require_pooling mean_nonpad
```

## Outputs

```text
cross_model_summary.csv
cross_model_layerwise.csv
cross_model_probe_quality.csv
cross_model_table.tex
cross_model_delta_by_layer.png
cross_model_peak_delta_bar.png
cross_model_report.md
```

Additional analyses:

```text
region_analysis/
ordering_analysis/
```

---

# Statistical Analysis

Unless otherwise specified:

- Confidence intervals are computed using nonparametric bootstrap resampling.
- Paired comparisons preserve matched concept–context structure.
- Effect sizes are reported using paired Cohen's d.
- Significance testing uses paired t-tests, Wilcoxon signed-rank tests, or permutation tests as appropriate.

Randomness is controlled through:

```text
Global Seed: --seed
Probe Seeds: --probe_seeds
```

---

# Reproducibility

All experiments save the full command-line configuration as:

```text
args.json
```

within the output directory.

This file records model names, pooling strategies, probe settings, random seeds, and other experimental parameters necessary for reproducing a run.

---

# Citation

If you use this code, please cite:

```bibtex
[Citation to be added upon publication]
```