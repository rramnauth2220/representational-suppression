# Suppression Salience Experiment Report

Model: `meta-llama/Meta-Llama-3-8B-Instruct`
Pooling mode: `mean_nonpad`
Concept library: `concepts.json`
Probe split seeds: `13,42,101`
Hard negatives included: `True`
Indirect suppression mode: `first`

## Main Question

Does an explicit suppression instruction increase latent recoverability of a prohibited concept relative to a matched condition in which the concept is absent?

## Key Statistic

`delta = probe_score(suppressed) - probe_score(absent)`

A positive delta indicates that the prohibited concept is more recoverable under suppression than when absent.

## Probe Quality

Mean probe accuracy across seeds/layers/concepts: 0.824
Mean probe AUC across seeds/layers/concepts: 0.916
Mean probe F1 across seeds/layers/concepts: 0.816

## Interpretation Guardrail

This experiment supports claims about latent recoverability, not literal human-like ironic cognition. Strong claims require replication across models, prompt templates, pooling schemes, and larger concept libraries.