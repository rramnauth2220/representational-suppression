#!/usr/bin/env python3

import os
import re
import json
import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from scipy.stats import ttest_rel, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM


MAIN_OUTPUT_FILES = [
    "probe_report_by_seed.csv",
    "probe_report.csv",
    "salience_probe_scores.csv",
    "salience_deltas.csv",
    "salience_summary_by_layer.csv",
    "report.md",
    "plot_salience_by_layer.png",
    "plot_delta_by_layer.png",
]


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class EvalExample:
    concept: str
    context: str
    prompts: Dict[str, str]


# ----------------------------
# Utilities
# ----------------------------

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sanitize_filename(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def resolve_layers(layer_arg: str, num_hidden_layers: int) -> List[int]:
    """Resolve layer CLI input to hidden-state indices, including embeddings at 0."""
    if layer_arg == "all":
        return list(range(num_hidden_layers + 1))
    return parse_int_list(layer_arg)


def bootstrap_ci(values, n_boot: int = 2000, ci: float = 95.0, seed: int = 42):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))
    alpha = (100 - ci) / 2
    return float(np.percentile(means, alpha)), float(np.percentile(means, 100 - alpha))


def safe_ttest_rel(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~np.isnan(a) & ~np.isnan(b)
    if mask.sum() < 2:
        return np.nan, np.nan
    t, p = ttest_rel(a[mask], b[mask])
    return float(t), float(p)


def safe_wilcoxon(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~np.isnan(a) & ~np.isnan(b)
    if mask.sum() < 2:
        return np.nan, np.nan
    try:
        stat, p = wilcoxon(a[mask], b[mask], zero_method="wilcox")
        return float(stat), float(p)
    except ValueError:
        return np.nan, np.nan


def cohen_d_paired(a, b):
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    diff = diff[~np.isnan(diff)]
    if len(diff) < 2 or np.std(diff, ddof=1) == 0:
        return np.nan
    return float(np.mean(diff) / np.std(diff, ddof=1))


def load_concepts(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        concepts = json.load(f)
    required = ["contexts", "positive", "negative"]
    for concept, data in concepts.items():
        for key in required:
            if key not in data:
                raise ValueError(f"Concept {concept!r} is missing required key {key!r}")
        data.setdefault("aliases", [concept])
        data.setdefault("direct_aliases", data.get("aliases", [concept]))
        data.setdefault("indirect_descriptions", [])
        if not data["indirect_descriptions"] and data.get("indirect_description"):
            data["indirect_descriptions"] = [data["indirect_description"]]
        if not data["indirect_descriptions"] and data.get("description"):
            data["indirect_descriptions"] = [data["description"]]
        data.setdefault("negative_hard", [])
    return concepts


def select_indirect_descriptions(data: Dict[str, Any], args) -> List[str]:
    """Return the indirect concept descriptions used to build indirect suppression controls."""
    descriptions = list(data.get("indirect_descriptions", []))
    if not descriptions:
        return []
    if args.indirect_mode == "all":
        return descriptions[: args.max_indirect_per_concept]
    if args.indirect_mode == "random":
        rng = random.Random(args.seed)
        k = min(args.max_indirect_per_concept, len(descriptions))
        return rng.sample(descriptions, k=k)
    # default: first
    return descriptions[:1]


def get_probe_negatives(data: Dict[str, Any], include_hard_negatives: bool) -> List[str]:
    negatives = list(data["negative"])
    if include_hard_negatives:
        negatives.extend(data.get("negative_hard", []))
    return negatives


def get_matching_aliases(data: Dict[str, Any], concept: str) -> List[str]:
    """Use narrower aliases for explicit leakage/target-token matching when available."""
    return list(data.get("direct_aliases") or data.get("aliases") or [concept])


# ----------------------------
# Prompt construction
# ----------------------------

def maybe_chat_format(tokenizer, text: str, use_chat_template: bool, system_prompt: Optional[str] = None) -> str:
    if not use_chat_template:
        return text
    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        return text
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": text})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_eval_examples(
    concepts: Dict[str, Dict],
    tokenizer,
    use_chat_template: bool,
    system_prompt: Optional[str],
    include_indirect: bool,
    args,
) -> List[EvalExample]:
    examples = []
    for concept, data in concepts.items():
        indirect_descriptions = select_indirect_descriptions(data, args) if include_indirect else []
        for ctx in data["contexts"]:
            raw_prompts = {
                "absent": ctx,
                "mentioned": f"{ctx}\n\nInclude {concept} in the description.",
            }
            if not args.only_indirect:
                raw_prompts["suppressed"] = (
                    f"{ctx}\n\nDo not mention {concept}."
                )
            for j, description in enumerate(indirect_descriptions):
                suffix = "" if len(indirect_descriptions) == 1 else f"_{j+1}"
                raw_prompts[f"suppressed_indirect{suffix}"] = (
                    f"{ctx}\n\nDo not mention or allude to {description}."
                )
            prompts = {
                k: maybe_chat_format(tokenizer, v, use_chat_template, system_prompt)
                for k, v in raw_prompts.items()
            }
            examples.append(EvalExample(concept=concept, context=ctx, prompts=prompts))
    return examples


def build_probe_dataset(
    concept: str,
    data: Dict,
    tokenizer,
    use_chat_template: bool,
    system_prompt: Optional[str],
    args,
) -> Tuple[List[str], np.ndarray]:
    positives = [maybe_chat_format(tokenizer, p, use_chat_template, system_prompt) for p in data["positive"]]
    raw_negatives = get_probe_negatives(data, include_hard_negatives=args.include_hard_negatives)
    negatives = [maybe_chat_format(tokenizer, p, use_chat_template, system_prompt) for p in raw_negatives]
    prompts = positives + negatives
    labels = np.array([1] * len(positives) + [0] * len(negatives))
    return prompts, labels


# ----------------------------
# Model loading
# ----------------------------

def get_torch_dtype(args):
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return torch.float32


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = {
        "torch_dtype": get_torch_dtype(args),
        "trust_remote_code": args.trust_remote_code,
    }

    if args.device_map_auto:
        model_kwargs["device_map"] = "auto"

    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()

    if not args.device_map_auto:
        model = model.to(args.device)

    return tokenizer, model


def get_device(model, fallback="cuda"):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(fallback)


# ----------------------------
# Activation extraction
# ----------------------------

def forward_hidden_and_attn(model, tokenizer, prompt: str, max_length: int, output_attentions: bool = False):
    device = get_device(model)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(
            **inputs,
            output_hidden_states=True,
            output_attentions=output_attentions,
            use_cache=False,
        )
    return inputs, out


def alias_token_positions(tokenizer, input_ids: torch.Tensor, aliases: List[str]) -> List[int]:
    ids = input_ids.detach().cpu().tolist()
    decoded = [tokenizer.decode([tid]).strip().lower() for tid in ids]
    positions = []
    for i, piece in enumerate(decoded):
        for alias in aliases:
            # catches single-token aliases and many BPE fragments; conservative but useful
            if alias.lower() in piece or piece in alias.lower().split():
                positions.append(i)
    return sorted(set(positions))


def pool_hidden(
    hidden: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    mode: str,
    tokenizer=None,
    input_ids: Optional[torch.Tensor] = None,
    aliases: Optional[List[str]] = None,
) -> torch.Tensor:
    # hidden: [seq_len, hidden_dim]
    if attention_mask is None:
        valid = torch.ones(hidden.shape[0], dtype=torch.bool, device=hidden.device)
    else:
        valid = attention_mask.to(dtype=torch.bool, device=hidden.device)

    valid_idx = torch.where(valid)[0]
    if len(valid_idx) == 0:
        valid_idx = torch.arange(hidden.shape[0], device=hidden.device)

    if mode == "last":
        return hidden[-1]
    if mode == "last_nonpad":
        return hidden[valid_idx[-1]]
    if mode == "mean":
        return hidden.mean(dim=0)
    if mode == "mean_nonpad":
        return hidden[valid].mean(dim=0)
    if mode == "first":
        return hidden[valid_idx[0]]
    if mode == "target_tokens":
        if tokenizer is not None and input_ids is not None and aliases:
            positions = alias_token_positions(tokenizer, input_ids, aliases)
            if positions:
                return hidden[positions].mean(dim=0)
        # fallback keeps the run from crashing for absent controls
        return hidden[valid_idx[-1]]
    raise ValueError(f"Unknown pooling mode: {mode}")


def extract_layerwise_activations(
    model,
    tokenizer,
    prompts: List[str],
    layers: List[int],
    pooling: str,
    max_length: int,
    alias_lists: Optional[List[List[str]]] = None,
    desc: str = "Extracting activations",
) -> Dict[int, np.ndarray]:
    acts = {l: [] for l in layers}

    for i, prompt in enumerate(tqdm(prompts, desc=desc, leave=False)):
        inputs, out = forward_hidden_and_attn(model, tokenizer, prompt, max_length, output_attentions=False)
        aliases = alias_lists[i] if alias_lists is not None else None
        input_ids = inputs["input_ids"][0].detach().cpu()
        attn_mask = inputs.get("attention_mask", None)
        if attn_mask is not None:
            attn_mask = attn_mask[0].detach().cpu()

        for l in layers:
            h = out.hidden_states[l][0].detach().float().cpu()
            pooled = pool_hidden(h, attn_mask, pooling, tokenizer, input_ids, aliases)
            acts[l].append(pooled.numpy())

        del out, inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {l: np.stack(v) for l, v in acts.items()}


# ----------------------------
# Probe training
# ----------------------------

def train_probes(model, tokenizer, concepts, layers, pooling, max_length, outdir, args):
    probe_bank = {}
    rows = []
    split_seeds = parse_int_list(args.probe_seeds)

    for concept, data in tqdm(concepts.items(), desc="Training concept probes"):
        prompts, labels = build_probe_dataset(
            concept, data, tokenizer, args.use_chat_template, args.system_prompt, args
        )
        alias_lists = [get_matching_aliases(data, concept)] * len(prompts)
        layer_acts = extract_layerwise_activations(
            model, tokenizer, prompts, layers, pooling, max_length, alias_lists,
            desc=f"Probe activations: {concept}"
        )
        probe_bank[concept] = {}

        for l in layers:
            X = layer_acts[l]
            y = labels
            seed_probes = []

            for split_seed in split_seeds:
                can_stratify = len(y) >= 8 and len(np.unique(y)) == 2 and min(np.bincount(y)) >= 2
                if can_stratify:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=args.test_size, random_state=split_seed, stratify=y
                    )
                else:
                    X_train, X_test, y_train, y_test = X, X, y, y

                probe = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        max_iter=args.probe_max_iter,
                        class_weight="balanced",
                        C=args.probe_C,
                        solver="liblinear",
                        random_state=split_seed,
                    ),
                )
                probe.fit(X_train, y_train)
                pred = probe.predict(X_test)
                prob = probe.predict_proba(X_test)[:, 1]

                acc = accuracy_score(y_test, pred)
                f1 = f1_score(y_test, pred, zero_division=0)
                auc = roc_auc_score(y_test, prob) if len(set(y_test)) == 2 else np.nan

                rows.append({
                    "concept": concept,
                    "layer": l,
                    "split_seed": split_seed,
                    "probe_accuracy": acc,
                    "probe_auc": auc,
                    "probe_f1": f1,
                    "n_train": len(y_train),
                    "n_test": len(y_test),
                    "n_positive": int(y.sum()),
                    "n_negative": int((1 - y).sum()),
                })
                seed_probes.append(probe)

            probe_bank[concept][l] = seed_probes

    probe_report = pd.DataFrame(rows)
    probe_report.to_csv(os.path.join(outdir, "probe_report_by_seed.csv"), index=False)
    agg = (
        probe_report.groupby(["concept", "layer"])
        .agg(
            probe_accuracy_mean=("probe_accuracy", "mean"),
            probe_accuracy_std=("probe_accuracy", "std"),
            probe_auc_mean=("probe_auc", "mean"),
            probe_auc_std=("probe_auc", "std"),
            probe_f1_mean=("probe_f1", "mean"),
            probe_f1_std=("probe_f1", "std"),
            n_positive=("n_positive", "first"),
            n_negative=("n_negative", "first"),
        )
        .reset_index()
    )
    agg.to_csv(os.path.join(outdir, "probe_report.csv"), index=False)
    return probe_bank, probe_report, agg


# ----------------------------
# Experiment 1: salience
# ----------------------------

def evaluate_probe_scores(model, tokenizer, examples, concepts, probe_bank, layers, pooling, max_length, outdir):
    rows = []

    for ex in tqdm(examples, desc="Evaluating probe salience"):
        condition_names = list(ex.prompts.keys())
        prompts = list(ex.prompts.values())
        aliases = get_matching_aliases(concepts[ex.concept], ex.concept)
        alias_lists = [aliases] * len(prompts)

        layer_acts = extract_layerwise_activations(
            model, tokenizer, prompts, layers, pooling, max_length, alias_lists,
            desc=f"Eval activations: {ex.concept}"
        )

        for l in layers:
            X = layer_acts[l]
            probes = probe_bank[ex.concept][l]
            seed_scores = np.stack([probe.predict_proba(X)[:, 1] for probe in probes], axis=0)
            mean_scores = seed_scores.mean(axis=0)
            std_scores = seed_scores.std(axis=0) if seed_scores.shape[0] > 1 else np.zeros_like(mean_scores)

            for cond, score, score_std, prompt in zip(condition_names, mean_scores, std_scores, prompts):
                rows.append({
                    "concept": ex.concept,
                    "context": ex.context,
                    "layer": l,
                    "condition": cond,
                    "probe_score": float(score),
                    "probe_score_seed_std": float(score_std),
                    "prompt": prompt,
                })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "salience_probe_scores.csv"), index=False)

    wide = df.pivot_table(
        index=["concept", "context", "layer"],
        columns="condition",
        values="probe_score",
    ).reset_index()

    if "suppressed" in wide.columns and "absent" in wide.columns:
        wide["delta_suppressed_minus_absent"] = wide["suppressed"] - wide["absent"]
    if "mentioned" in wide.columns and "absent" in wide.columns:
        wide["delta_mentioned_minus_absent"] = wide["mentioned"] - wide["absent"]
    if "absent" in wide.columns:
        indirect_cols = [c for c in wide.columns if str(c).startswith("suppressed_indirect")]
        for c in indirect_cols:
            wide[f"delta_{c}_minus_absent"] = wide[c] - wide["absent"]
    if {"delta_suppressed_minus_absent", "delta_mentioned_minus_absent"}.issubset(wide.columns):
        wide["suppressed_fraction_of_mentioned"] = (
            wide["delta_suppressed_minus_absent"]
            / wide["delta_mentioned_minus_absent"].replace(0, np.nan)
        )

    wide.to_csv(os.path.join(outdir, "salience_deltas.csv"), index=False)

    summary_rows = []
    for l in layers:
        sub = wide[wide["layer"] == l]
        row = {"layer": l, "n_pairs": len(sub)}
        conditions = ["absent", "suppressed", "mentioned"] + [c for c in sub.columns if str(c).startswith("suppressed_indirect")]
        for cond in conditions:
            if cond in sub.columns:
                vals = sub[cond].astype(float)
                lo, hi = bootstrap_ci(vals, seed=42 + l)
                row[f"mean_{cond}"] = vals.mean()
                row[f"sem_{cond}"] = vals.sem()
                row[f"ci95_low_{cond}"] = lo
                row[f"ci95_high_{cond}"] = hi

        comparisons = [
            ("suppressed", "absent"),
            ("mentioned", "absent"),
            ("suppressed", "mentioned"),
        ]
        comparisons.extend((c, "absent") for c in conditions if str(c).startswith("suppressed_indirect"))
        comparisons.extend((c, "mentioned") for c in conditions if str(c).startswith("suppressed_indirect"))
        for a, b in comparisons:
            if a in sub.columns and b in sub.columns:
                delta = sub[a].astype(float) - sub[b].astype(float)
                lo, hi = bootstrap_ci(delta, seed=1000 + l)
                t, p = safe_ttest_rel(sub[a], sub[b])
                w, wp = safe_wilcoxon(sub[a], sub[b])
                row[f"mean_delta_{a}_minus_{b}"] = delta.mean()
                row[f"sem_delta_{a}_minus_{b}"] = delta.sem()
                row[f"ci95_low_delta_{a}_minus_{b}"] = lo
                row[f"ci95_high_delta_{a}_minus_{b}"] = hi
                row[f"paired_t_{a}_vs_{b}"] = t
                row[f"paired_p_{a}_vs_{b}"] = p
                row[f"wilcoxon_stat_{a}_vs_{b}"] = w
                row[f"wilcoxon_p_{a}_vs_{b}"] = wp
                row[f"cohen_d_paired_{a}_vs_{b}"] = cohen_d_paired(sub[a], sub[b])
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(outdir, "salience_summary_by_layer.csv"), index=False)
    return df, wide, summary


# ----------------------------
# Token dynamics
# ----------------------------

def evaluate_token_dynamics(model, tokenizer, examples, concepts, probe_bank, layers, max_length, outdir):
    rows = []
    for ex in tqdm(examples, desc="Evaluating token dynamics"):
        aliases = get_matching_aliases(concepts[ex.concept], ex.concept)
        for condition, prompt in ex.prompts.items():
            inputs, out = forward_hidden_and_attn(model, tokenizer, prompt, max_length, output_attentions=False)
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
            for l in layers:
                h = out.hidden_states[l][0].detach().float().cpu().numpy()
                probes = probe_bank[ex.concept][l]
                seed_scores = np.stack([probe.predict_proba(h)[:, 1] for probe in probes], axis=0)
                scores = seed_scores.mean(axis=0)
                for pos, score in enumerate(scores):
                    rows.append({
                        "concept": ex.concept,
                        "context": ex.context,
                        "condition": condition,
                        "layer": l,
                        "token_position": pos,
                        "token": tokens[pos],
                        "is_alias_token": pos in alias_token_positions(tokenizer, inputs["input_ids"][0].detach().cpu(), aliases),
                        "probe_score": float(score),
                    })
            del out, inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "token_dynamics.csv"), index=False)
    return df


# ----------------------------
# Attention diagnostics
# ----------------------------

def evaluate_attention_to_target(model, tokenizer, examples, concepts, layers, max_length, outdir):
    rows = []
    for ex in tqdm(examples, desc="Evaluating attention diagnostics"):
        aliases = get_matching_aliases(concepts[ex.concept], ex.concept)
        for condition, prompt in ex.prompts.items():
            if condition == "absent":
                continue
            inputs, out = forward_hidden_and_attn(model, tokenizer, prompt, max_length, output_attentions=True)
            if out.attentions is None:
                continue
            input_ids = inputs["input_ids"][0].detach().cpu()
            target_positions = alias_token_positions(tokenizer, input_ids, aliases)
            if not target_positions:
                continue
            seq_len = len(input_ids)
            for l in layers:
                attn = out.attentions[l][0].detach().float().cpu()  # [heads, query, key]
                for head in range(attn.shape[0]):
                    target_mass = attn[head, :, target_positions].sum(dim=-1)
                    rows.append({
                        "concept": ex.concept,
                        "context": ex.context,
                        "condition": condition,
                        "layer": l,
                        "head": head,
                        "target_positions": json.dumps(target_positions),
                        "mean_attention_to_target_tokens": float(target_mass.mean().item()),
                        "max_attention_to_target_tokens": float(target_mass.max().item()),
                        "seq_len": seq_len,
                    })
            del out, inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "attention_to_target.csv"), index=False)
    if len(df) > 0:
        summary = df.groupby(["condition", "layer"])["mean_attention_to_target_tokens"].agg(["mean", "sem"]).reset_index()
        summary.to_csv(os.path.join(outdir, "attention_summary_by_layer.csv"), index=False)
    return df


# ----------------------------
# Generation leakage
# ----------------------------

def generate_text(model, tokenizer, prompt, max_new_tokens, temperature, top_p):
    device = get_device(model)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    return full[len(prompt):].strip() if full.startswith(prompt) else full


def simple_semantic_leakage_score(text: str, aliases: List[str]) -> Tuple[float, List[str]]:
    text_l = text.lower()
    hits = []
    for a in aliases:
        if re.search(r"\b" + re.escape(a.lower()) + r"\b", text_l):
            hits.append(a)
    return float(len(hits) > 0), hits


def evaluate_generation_leakage(model, tokenizer, examples, concepts, n_samples, max_new_tokens, temperature, top_p, outdir):
    rows = []
    for ex in tqdm(examples, desc="Evaluating behavioral leakage"):
        aliases = get_matching_aliases(concepts[ex.concept], ex.concept)
        for condition, prompt in ex.prompts.items():
            for sample_idx in range(n_samples):
                gen = generate_text(model, tokenizer, prompt, max_new_tokens, temperature, top_p)
                leak_binary, hits = simple_semantic_leakage_score(gen, aliases)
                rows.append({
                    "concept": ex.concept,
                    "context": ex.context,
                    "condition": condition,
                    "sample_idx": sample_idx,
                    "prompt": prompt,
                    "generation": gen,
                    "explicit_or_alias_leak": leak_binary,
                    "matched_aliases": json.dumps(hits),
                })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "generation_leakage.csv"), index=False)
    summary = df.groupby(["condition", "concept"])["explicit_or_alias_leak"].agg(["mean", "sem", "count"]).reset_index()
    summary.to_csv(os.path.join(outdir, "generation_leakage_summary.csv"), index=False)
    return df, summary


# ----------------------------
# Plots and report
# ----------------------------

def plot_salience(summary: pd.DataFrame, outdir: str):
    plt.figure()
    plot_conditions = ["absent", "suppressed", "mentioned"]
    plot_conditions += sorted([c.replace("mean_", "") for c in summary.columns if c.startswith("mean_suppressed_indirect")])
    for cond in plot_conditions:
        col = f"mean_{cond}"
        if col in summary.columns:
            plt.plot(summary["layer"], summary[col], label=cond)
            lo = f"ci95_low_{cond}"
            hi = f"ci95_high_{cond}"
            if lo in summary.columns and hi in summary.columns:
                plt.fill_between(summary["layer"], summary[lo], summary[hi], alpha=0.15)
    plt.xlabel("Layer")
    plt.ylabel("Mean probe score")
    plt.title("Latent Concept Recoverability by Layer")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plot_salience_by_layer.png"), dpi=220)
    plt.close()

    plt.figure()
    delta_cols = [
        ("mean_delta_suppressed_minus_absent", "suppressed - absent"),
        ("mean_delta_mentioned_minus_absent", "mentioned - absent"),
    ]
    delta_cols += [(c, c.replace("mean_delta_", "").replace("_minus_", " - ")) for c in summary.columns if c.startswith("mean_delta_suppressed_indirect") and c.endswith("_minus_absent")]
    for col, label in delta_cols:
        if col in summary.columns:
            plt.plot(summary["layer"], summary[col], label=label)
            lo = col.replace("mean_", "ci95_low_")
            hi = col.replace("mean_", "ci95_high_")
            if lo in summary.columns and hi in summary.columns:
                plt.fill_between(summary["layer"], summary[lo], summary[hi], alpha=0.15)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Layer")
    plt.ylabel("Mean delta")
    plt.title("Suppression Salience Difference")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plot_delta_by_layer.png"), dpi=220)
    plt.close()


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


def select_report_delta_column(salience_summary: pd.DataFrame) -> Optional[str]:
    preferred = "mean_delta_suppressed_minus_absent"
    if preferred in salience_summary.columns:
        return preferred

    indirect_cols = [
        c for c in salience_summary.columns
        if c.startswith("mean_delta_suppressed_indirect") and c.endswith("_minus_absent")
    ]
    if indirect_cols:
        return sorted(indirect_cols, key=lambda c: (len(c), c))[0]

    delta_cols = [
        c for c in salience_summary.columns
        if c.startswith("mean_delta_") and c.endswith("_minus_absent")
    ]
    return sorted(delta_cols)[0] if delta_cols else None


def write_markdown_report(args, probe_report_seed, probe_report, salience_summary, leakage_summary, outdir):
    delta_col = select_report_delta_column(salience_summary)
    best = salience_summary.sort_values(delta_col, ascending=False).iloc[0] if delta_col else None

    report = [
        "# Experiment 1 Summary",
        "",
        "## run_config",
        "",
        *make_variable_table([
            ("model", args.model),
            ("pooling", args.pooling),
            ("concepts_path", args.concepts_path),
            ("probe_seeds", args.probe_seeds),
            ("include_hard_negatives", args.include_hard_negatives),
            ("include_indirect", args.include_indirect),
            ("only_indirect", args.only_indirect),
            ("indirect_mode", args.indirect_mode if args.include_indirect else "disabled"),
        ]),
        "",
        "## probe_quality",
        "",
        *make_variable_table([
            ("probe_accuracy_mean", probe_report_seed["probe_accuracy"].mean()),
            ("probe_auc_mean", probe_report_seed["probe_auc"].mean()),
            ("probe_f1_mean", probe_report_seed["probe_f1"].mean()),
        ]),
    ]

    if best is not None:
        report += [
            "",
            "## peak_salience",
            "",
            *make_variable_table([
                ("selected_delta_column", delta_col),
                ("peak_layer", int(best["layer"])),
                ("mean_absent", best.get("mean_absent", np.nan)),
                ("mean_suppressed", best.get("mean_suppressed", np.nan)),
                ("mean_suppressed_indirect", best.get("mean_suppressed_indirect", np.nan)),
                ("mean_mentioned", best.get("mean_mentioned", np.nan)),
                (delta_col, best.get(delta_col, np.nan)),
                (delta_col.replace("mean_", "ci95_low_"), best.get(delta_col.replace("mean_", "ci95_low_"), np.nan)),
                (delta_col.replace("mean_", "ci95_high_"), best.get(delta_col.replace("mean_", "ci95_high_"), np.nan)),
                (
                    f"paired_p_{delta_col.replace('mean_delta_', '').replace('_minus_absent', '')}_vs_absent",
                    best.get(f"paired_p_{delta_col.replace('mean_delta_', '').replace('_minus_absent', '')}_vs_absent", np.nan),
                ),
                (
                    f"wilcoxon_p_{delta_col.replace('mean_delta_', '').replace('_minus_absent', '')}_vs_absent",
                    best.get(f"wilcoxon_p_{delta_col.replace('mean_delta_', '').replace('_minus_absent', '')}_vs_absent", np.nan),
                ),
                (
                    f"cohen_d_paired_{delta_col.replace('mean_delta_', '').replace('_minus_absent', '')}_vs_absent",
                    best.get(f"cohen_d_paired_{delta_col.replace('mean_delta_', '').replace('_minus_absent', '')}_vs_absent", np.nan),
                ),
            ]),
        ]

    if leakage_summary is not None:
        report += ["", "## generation_leakage_summary", "", leakage_summary.to_markdown(index=False)]

    with open(os.path.join(outdir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))


# ----------------------------
# Main
# ----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--concepts_path", type=str, default="concepts.json")
    parser.add_argument("--outdir", type=str, default="results_suppression_salience")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--layers", type=str, default="all")
    parser.add_argument(
        "--pooling",
        type=str,
        default="last_nonpad",
        choices=["last", "last_nonpad", "mean", "mean_nonpad", "first", "target_tokens"],
    )
    parser.add_argument("--max_length", type=int, default=256)

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--load_4bit", action="store_true")
    parser.add_argument("--device_map_auto", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")

    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--system_prompt", type=str, default=None)
    parser.add_argument("--include_indirect", action="store_true")
    parser.add_argument("--only_indirect", action="store_true")
    parser.add_argument("--indirect_mode", type=str, default="first", choices=["first", "all", "random"])
    parser.add_argument("--max_indirect_per_concept", type=int, default=2)
    parser.add_argument("--include_hard_negatives", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--probe_seeds", type=str, default="13,42,101")
    parser.add_argument("--test_size", type=float, default=0.35)
    parser.add_argument("--probe_C", type=float, default=1.0)
    parser.add_argument("--probe_max_iter", type=int, default=5000)

    parser.add_argument("--run_token_dynamics", action="store_true")
    parser.add_argument("--attentions", action="store_true")
    parser.add_argument("--run_generation", action="store_true")
    parser.add_argument("--n_generation_samples", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)

    parser.add_argument("--seed", type=int, default=42)
    return parser


def make_output_dir(args) -> str:
    safe_model = sanitize_filename(args.model)
    run_name = f"{safe_model}__pool-{args.pooling}__seed-{args.seed}"
    return os.path.join(args.outdir, run_name)


def save_args(args, outdir: str):
    with open(os.path.join(outdir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)


def print_run_summary(args, concepts, examples, layers, outdir: str):
    print(f"Running model: {args.model}")
    print(f"Concepts: {len(concepts)}")
    print(f"Eval examples: {len(examples)}")
    print(f"Layers: {layers}")
    print(f"Pooling: {args.pooling}")
    print(f"Output directory: {outdir}")


def print_completion(outdir: str):
    print("\nDone.")
    print(f"Results written to: {outdir}")
    print("Main files:")
    for name in MAIN_OUTPUT_FILES:
        print(f"  {name}")


def run_experiment(args):
    seed_everything(args.seed)

    outdir = make_output_dir(args)
    os.makedirs(outdir, exist_ok=True)
    save_args(args, outdir)

    concepts = load_concepts(args.concepts_path)
    tokenizer, model = load_model(args)

    layers = resolve_layers(args.layers, model.config.num_hidden_layers)
    examples = build_eval_examples(
        concepts,
        tokenizer=tokenizer,
        use_chat_template=args.use_chat_template,
        system_prompt=args.system_prompt,
        include_indirect=args.include_indirect,
        args=args,
    )

    print_run_summary(args, concepts, examples, layers, outdir)

    probe_bank, probe_report_seed, probe_report = train_probes(
        model, tokenizer, concepts, layers, args.pooling, args.max_length, outdir, args
    )

    salience_df, salience_wide, salience_summary = evaluate_probe_scores(
        model, tokenizer, examples, concepts, probe_bank, layers, args.pooling, args.max_length, outdir
    )

    if args.run_token_dynamics:
        evaluate_token_dynamics(model, tokenizer, examples, concepts, probe_bank, layers, args.max_length, outdir)

    if args.attentions:
        evaluate_attention_to_target(model, tokenizer, examples, concepts, layers, args.max_length, outdir)

    leakage_summary = None
    if args.run_generation:
        _, leakage_summary = evaluate_generation_leakage(
            model, tokenizer, examples, concepts,
            args.n_generation_samples, args.max_new_tokens, args.temperature, args.top_p, outdir
        )

    plot_salience(salience_summary, outdir)
    write_markdown_report(args, probe_report_seed, probe_report, salience_summary, leakage_summary, outdir)

    print_completion(outdir)
    return outdir


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_experiment(args)


if __name__ == "__main__":
    main()
