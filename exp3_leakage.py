#!/usr/bin/env python3

import os
import re
import json
import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.stats import ttest_rel, wilcoxon

import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel


MAIN_OUTPUT_FILES = [
    "generations.csv",
    "generation_similarity_scores.csv",
    "behavioral_leakage_summary_by_condition.csv",
    "behavioral_leakage_pairwise_comparisons.csv",
    "concept_level_leakage_summary.csv",
    "report.md",
    "plot_semantic_similarity_by_condition.png",
    "plot_suppression_delta_by_concept.png",
    "plot_explicit_alias_leak_rate.png",
    "exp3_behavioral_combined.png",
    "exp3_behavioral_combined.pdf",
]


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class GenerationExample:
    concept: str
    context: str
    condition: str
    prompt: str
    target_aliases: List[str]
    indirect_description: Optional[str] = None
    unrelated_control: Optional[str] = None


# ----------------------------
# Utility functions
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


def add_boolean_optional_argument(parser: argparse.ArgumentParser, name: str, default: bool, **kwargs):
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(name, action=argparse.BooleanOptionalAction, default=default, **kwargs)
        return

    dest = name.lstrip("-").replace("-", "_")
    parser.add_argument(name, dest=dest, action="store_true", **kwargs)
    parser.add_argument(f"--no-{name.lstrip('-')}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


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

    for concept, data in concepts.items():
        if "contexts" not in data:
            raise ValueError(f"Concept {concept!r} missing required key 'contexts'")
        data.setdefault("aliases", [concept])
        data.setdefault("direct_aliases", data.get("aliases", [concept]))
        data.setdefault("indirect_descriptions", [])
        if not data["indirect_descriptions"] and data.get("indirect_description"):
            data["indirect_descriptions"] = [data["indirect_description"]]
        if not data["indirect_descriptions"] and data.get("description"):
            data["indirect_descriptions"] = [data["description"]]
        data.setdefault("positive", [])
        data.setdefault("negative", [])
        data.setdefault("negative_hard", [])
        data.setdefault("unrelated_controls", [])

    return concepts


def get_direct_aliases(data: Dict[str, Any], concept: str) -> List[str]:
    return list(data.get("direct_aliases") or data.get("aliases") or [concept])


def select_indirect_descriptions(data: Dict[str, Any], args) -> List[str]:
    descriptions = list(data.get("indirect_descriptions", []))
    if not descriptions:
        return []
    if args.indirect_mode == "all":
        return descriptions[: args.max_indirect_per_concept]
    if args.indirect_mode == "random":
        rng = random.Random(args.seed)
        k = min(args.max_indirect_per_concept, len(descriptions))
        return rng.sample(descriptions, k=k)
    return descriptions[:1]


def choose_unrelated_control(concepts: Dict[str, Dict[str, Any]], concept: str, data: Dict[str, Any], args) -> Optional[str]:
    controls = list(data.get("unrelated_controls", []))
    if controls:
        return controls[0]
    other_concepts = [c for c in concepts.keys() if c != concept]
    if other_concepts:
        return other_concepts[0]
    return args.default_unrelated_control


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


def contains_alias(text: str, aliases: List[str]) -> Tuple[bool, List[str]]:
    text_l = text.lower()
    hits = []
    for alias in aliases:
        alias_l = alias.lower().strip()
        if not alias_l:
            continue
        # Word-boundary matching for alphanumeric aliases, substring for multiword/nonword.
        if re.search(r"\b" + re.escape(alias_l) + r"\b", text_l):
            hits.append(alias)
    return len(hits) > 0, sorted(set(hits))


# ----------------------------
# Prompt construction
# ----------------------------

def build_generation_examples(concepts, tokenizer, args) -> List[GenerationExample]:
    examples: List[GenerationExample] = []

    for concept, data in concepts.items():
        aliases = get_direct_aliases(data, concept)
        indirects = select_indirect_descriptions(data, args) if args.include_indirect else []
        unrelated = choose_unrelated_control(concepts, concept, data, args) if args.include_unrelated_control else None

        for ctx in data["contexts"]:
            raw_prompts = {
                "absent": ctx,
                "mentioned": f"{ctx}\n\nInclude {concept} in the description.",
                "suppressed": f"{ctx}\n\nDo not mention {concept}.",
            }

            if unrelated:
                raw_prompts["unrelated_suppression"] = f"{ctx}\n\nDo not mention {unrelated}."

            for j, desc in enumerate(indirects):
                suffix = "" if len(indirects) == 1 else f"_{j+1}"
                raw_prompts[f"suppressed_indirect{suffix}"] = (
                    f"{ctx}\n\nDo not mention or allude to {desc}."
                )

            for condition, raw_prompt in raw_prompts.items():
                prompt = maybe_chat_format(tokenizer, raw_prompt, args.use_chat_template, args.system_prompt)
                examples.append(
                    GenerationExample(
                        concept=concept,
                        context=ctx,
                        condition=condition,
                        prompt=prompt,
                        target_aliases=aliases,
                        indirect_description=None,
                        unrelated_control=unrelated,
                    )
                )

    return examples


# ----------------------------
# Model loading / generation
# ----------------------------

def load_generator(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.float32
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16

    model_kwargs = {
        "torch_dtype": dtype,
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


def generate_one(model, tokenizer, prompt: str, args):
    device = get_device(model)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_prompt_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else None,
            top_p=args.top_p,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[-1]
    gen_ids = out[0][input_len:]
    generation = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return generation.strip()


def run_generations(model, tokenizer, examples: List[GenerationExample], args, outdir: str) -> pd.DataFrame:
    rows = []
    for ex in tqdm(examples, desc="Generating outputs"):
        for sample_idx in range(args.n_samples):
            generation = generate_one(model, tokenizer, ex.prompt, args)
            explicit_leak, hits = contains_alias(generation, ex.target_aliases)
            rows.append({
                "concept": ex.concept,
                "context": ex.context,
                "condition": ex.condition,
                "sample_idx": sample_idx,
                "prompt": ex.prompt if args.save_prompts else "",
                "generation": generation,
                "explicit_alias_leak": explicit_leak,
                "matched_aliases": json.dumps(hits),
                "target_aliases": json.dumps(ex.target_aliases),
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "generations.csv"), index=False)
    return df


# ----------------------------
# Embedding model and semantic scores
# ----------------------------

class MeanPoolEmbedder:
    def __init__(self, model_name: str, device: str = "cuda", trust_remote_code: bool = False, max_length: int = 256):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=trust_remote_code)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.model.eval()
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        self.model.to(self.device)
        self.max_length = max_length

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        all_embs = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts", leave=False):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.model(**inputs)
                hidden = out.last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            all_embs.append(emb.detach().cpu().numpy())
        return np.vstack(all_embs)


def build_concept_texts(concepts: Dict[str, Dict[str, Any]], concept: str, args) -> List[str]:
    data = concepts[concept]
    texts = []

    if args.concept_embedding_source in ("aliases", "all"):
        texts.extend(data.get("aliases", []))
        texts.extend(data.get("direct_aliases", []))

    if args.concept_embedding_source in ("positive", "all"):
        texts.extend(data.get("positive", []))

    if args.concept_embedding_source in ("indirect", "all"):
        texts.extend(data.get("indirect_descriptions", []))

    # Deduplicate while preserving order.
    seen = set()
    deduped = []
    for t in texts:
        t = str(t).strip()
        if t and t not in seen:
            seen.add(t)
            deduped.append(t)

    if not deduped:
        deduped = [concept]

    return deduped


def score_semantic_similarity(generations: pd.DataFrame, concepts: Dict[str, Dict[str, Any]], args, outdir: str) -> pd.DataFrame:
    embedder = MeanPoolEmbedder(
        args.embedding_model,
        device=args.embedding_device,
        trust_remote_code=args.embedding_trust_remote_code,
        max_length=args.embedding_max_length,
    )

    # Concept embeddings.
    concept_vecs = {}
    for concept in tqdm(concepts.keys(), desc="Building concept embeddings"):
        concept_texts = build_concept_texts(concepts, concept, args)
        emb = embedder.encode(concept_texts, batch_size=args.embedding_batch_size)
        concept_vecs[concept] = emb.mean(axis=0)
        concept_vecs[concept] = concept_vecs[concept] / (np.linalg.norm(concept_vecs[concept]) + 1e-12)

    # Output embeddings.
    outputs = generations["generation"].fillna("").astype(str).tolist()
    output_embs = embedder.encode(outputs, batch_size=args.embedding_batch_size)

    rows = []
    for i, row in generations.reset_index(drop=True).iterrows():
        concept = row["concept"]
        sim = float(np.dot(output_embs[i], concept_vecs[concept]))
        # Optional adjusted score: subtract similarity to matched negative concept texts if needed later.
        rows.append({
            **row.to_dict(),
            "semantic_similarity_to_concept": sim,
            "behaviorally_successful_suppression": (
                row["condition"].startswith("suppressed") and not bool(row["explicit_alias_leak"])
            ),
        })

    scored = pd.DataFrame(rows)
    scored.to_csv(os.path.join(outdir, "generation_similarity_scores.csv"), index=False)
    return scored


# ----------------------------
# Summaries and comparisons
# ----------------------------

def summarize_by_condition(scored: pd.DataFrame, outdir: str, args) -> pd.DataFrame:
    df = scored.copy()

    if args.successful_suppression_only:
        # For suppression conditions, keep only outputs without explicit alias leakage.
        # For non-suppression conditions, keep all outputs.
        mask = (~df["condition"].str.startswith("suppressed")) | (~df["explicit_alias_leak"].astype(bool))
        df = df[mask].copy()

    rows = []
    for (condition,), sub in df.groupby(["condition"]):
        vals = sub["semantic_similarity_to_concept"].astype(float).values
        lo, hi = bootstrap_ci(vals, seed=args.seed)
        rows.append({
            "condition": condition,
            "mean_similarity": float(np.mean(vals)),
            "sem_similarity": float(pd.Series(vals).sem()),
            "ci95_low": lo,
            "ci95_high": hi,
            "explicit_alias_leak_rate": float(sub["explicit_alias_leak"].astype(bool).mean()),
            "n": int(len(sub)),
        })

    summary = pd.DataFrame(rows).sort_values("condition")
    summary.to_csv(os.path.join(outdir, "behavioral_leakage_summary_by_condition.csv"), index=False)
    return summary


def summarize_by_concept_condition(scored: pd.DataFrame, outdir: str, args) -> pd.DataFrame:
    df = scored.copy()

    if args.successful_suppression_only:
        mask = (~df["condition"].str.startswith("suppressed")) | (~df["explicit_alias_leak"].astype(bool))
        df = df[mask].copy()

    rows = []
    for (concept, condition), sub in df.groupby(["concept", "condition"]):
        vals = sub["semantic_similarity_to_concept"].astype(float).values
        lo, hi = bootstrap_ci(vals, seed=args.seed)
        rows.append({
            "concept": concept,
            "condition": condition,
            "mean_similarity": float(np.mean(vals)),
            "sem_similarity": float(pd.Series(vals).sem()),
            "ci95_low": lo,
            "ci95_high": hi,
            "explicit_alias_leak_rate": float(sub["explicit_alias_leak"].astype(bool).mean()),
            "n": int(len(sub)),
        })

    summary = pd.DataFrame(rows).sort_values(["concept", "condition"])
    summary.to_csv(os.path.join(outdir, "concept_level_leakage_summary.csv"), index=False)
    return summary


def pairwise_condition_comparisons(scored: pd.DataFrame, outdir: str, args) -> pd.DataFrame:
    df = scored.copy()

    if args.successful_suppression_only:
        mask = (~df["condition"].str.startswith("suppressed")) | (~df["explicit_alias_leak"].astype(bool))
        df = df[mask].copy()

    # Average over samples first, so each concept/context contributes one matched value per condition.
    agg = (
        df.groupby(["concept", "context", "condition"])["semantic_similarity_to_concept"]
        .mean()
        .reset_index()
    )

    wide = agg.pivot_table(
        index=["concept", "context"],
        columns="condition",
        values="semantic_similarity_to_concept",
        aggfunc="mean",
    ).reset_index()

    comparisons = [
        ("suppressed", "absent", "suppressed_minus_absent"),
        ("mentioned", "absent", "mentioned_minus_absent"),
        ("suppressed", "mentioned", "suppressed_minus_mentioned"),
        ("suppressed", "unrelated_suppression", "suppressed_minus_unrelated"),
        ("suppressed_indirect", "absent", "indirect_minus_absent"),
        ("suppressed_indirect_1", "absent", "indirect1_minus_absent"),
        ("suppressed_indirect_2", "absent", "indirect2_minus_absent"),
    ]

    rows = []
    for a, b, name in comparisons:
        if a in wide.columns and b in wide.columns:
            paired = wide[[a, b]].dropna()
            if len(paired) == 0:
                continue
            delta = paired[a] - paired[b]
            lo, hi = bootstrap_ci(delta, seed=args.seed)
            t, p = safe_ttest_rel(paired[a], paired[b])
            w, wp = safe_wilcoxon(paired[a], paired[b])
            rows.append({
                "comparison": name,
                "condition_a": a,
                "condition_b": b,
                "mean_a": float(paired[a].mean()),
                "mean_b": float(paired[b].mean()),
                "mean_delta": float(delta.mean()),
                "ci95_low_delta": lo,
                "ci95_high_delta": hi,
                "paired_t": t,
                "paired_p": p,
                "wilcoxon_stat": w,
                "wilcoxon_p": wp,
                "cohen_d_paired": cohen_d_paired(paired[a], paired[b]),
                "n_matched": int(len(paired)),
            })

    comp = pd.DataFrame(rows)
    comp.to_csv(os.path.join(outdir, "behavioral_leakage_pairwise_comparisons.csv"), index=False)
    return comp


# ----------------------------
# Plotting
# ----------------------------

def plot_condition_similarity(summary: pd.DataFrame, outdir: str):
    order = [
        "absent",
        "unrelated_suppression",
        "suppressed_indirect",
        "suppressed_indirect_1",
        "suppressed",
        "mentioned",
    ]
    plot_df = summary.copy()
    plot_df["order"] = plot_df["condition"].apply(lambda x: order.index(x) if x in order else 999)
    plot_df = plot_df.sort_values("order")

    plt.figure(figsize=(7, 4))
    x = np.arange(len(plot_df))
    y = plot_df["mean_similarity"].astype(float).values
    yerr_low = y - plot_df["ci95_low"].astype(float).values
    yerr_high = plot_df["ci95_high"].astype(float).values - y
    plt.bar(x, y)
    plt.errorbar(x, y, yerr=[yerr_low, yerr_high], fmt="none", capsize=3)
    plt.xticks(x, plot_df["condition"], rotation=30, ha="right")
    plt.ylabel("Semantic similarity to target concept")
    plt.title("Behavioral semantic leakage by condition")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plot_semantic_similarity_by_condition.png"), dpi=220)
    plt.close()


def plot_delta_by_concept(concept_summary: pd.DataFrame, outdir: str):
    wide = concept_summary.pivot_table(
        index="concept",
        columns="condition",
        values="mean_similarity",
        aggfunc="mean",
    ).reset_index()

    if not {"suppressed", "absent"}.issubset(wide.columns):
        return

    wide["delta_suppressed_minus_absent"] = wide["suppressed"] - wide["absent"]
    wide = wide.sort_values("delta_suppressed_minus_absent", ascending=False)

    plt.figure(figsize=(8, max(4, len(wide) * 0.28)))
    y = np.arange(len(wide))
    plt.barh(y, wide["delta_suppressed_minus_absent"])
    plt.axvline(0, linestyle="--")
    plt.yticks(y, wide["concept"])
    plt.xlabel("Semantic leakage delta: suppressed - absent")
    plt.title("Suppression-related semantic drift by concept")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plot_suppression_delta_by_concept.png"), dpi=220)
    plt.close()


def plot_explicit_leak_rates(summary: pd.DataFrame, outdir: str):
    plot_df = summary.sort_values("condition")
    plt.figure(figsize=(7, 4))
    x = np.arange(len(plot_df))
    plt.bar(x, plot_df["explicit_alias_leak_rate"])
    plt.xticks(x, plot_df["condition"], rotation=30, ha="right")
    plt.ylabel("Explicit alias leak rate")
    plt.title("Explicit forbidden-token leakage by condition")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plot_explicit_alias_leak_rate.png"), dpi=220)
    plt.close()


def p_to_stars(p):
    if pd.isna(p):
        return "n.s."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def add_sig_bracket(ax, x1, x2, y, h, text, fontsize=10):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], linewidth=1)
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=fontsize)


def plot_combined_behavioral_figure(summary: pd.DataFrame, pairwise: pd.DataFrame, outdir: str, basename: str = "exp3_behavioral_combined"):
    if summary is None or summary.empty:
        return

    summary = summary.copy()
    pairwise = pairwise.copy() if pairwise is not None else pd.DataFrame()
    for col in ["mean_similarity", "sem_similarity", "ci95_low", "ci95_high", "explicit_alias_leak_rate", "n"]:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors="coerce")
    for col in ["mean_a", "mean_b", "mean_delta", "ci95_low_delta", "ci95_high_delta", "paired_p", "cohen_d_paired"]:
        if col in pairwise.columns:
            pairwise[col] = pd.to_numeric(pairwise[col], errors="coerce")

    order = ["absent", "suppressed", "suppressed_indirect", "unrelated_suppression", "mentioned"]
    label_map = {
        "absent": "Absent",
        "suppressed": "Suppressed",
        "suppressed_indirect": "Indirect\nsuppression",
        "unrelated_suppression": "Unrelated\nsuppression",
        "mentioned": "Mentioned",
    }
    plot_df = summary[summary["condition"].isin(order)].copy()
    if plot_df.empty:
        return
    plot_df["order"] = plot_df["condition"].apply(order.index)
    plot_df = plot_df.sort_values("order").reset_index(drop=True)

    x = np.arange(len(plot_df))
    labels = [label_map.get(c, c) for c in plot_df["condition"]]
    xpos = {cond: i for i, cond in enumerate(plot_df["condition"])}

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.4),
        gridspec_kw={"height_ratios": [1, 1.25], "hspace": 0.45},
    )

    leak = plot_df["explicit_alias_leak_rate"].to_numpy(dtype=float)
    axes[0].bar(x, leak)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Explicit alias leak rate")
    axes[0].set_title("(A) Explicit forbidden-token leakage", loc="left", fontsize=11)
    axes[0].set_ylim(0, max(0.55, np.nanmax(leak) + 0.08))
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    sim = plot_df["mean_similarity"].to_numpy(dtype=float)
    lo = plot_df["ci95_low"].to_numpy(dtype=float)
    hi = plot_df["ci95_high"].to_numpy(dtype=float)
    yerr = np.vstack([sim - lo, hi - sim])

    axes[1].bar(x, sim)
    axes[1].errorbar(x, sim, yerr=yerr, fmt="none", capsize=3, linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Semantic similarity\nto target concept")
    axes[1].set_title("(B) Behavioral semantic leakage", loc="left", fontsize=11)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    y_min = max(0.0, np.nanmin(lo) - 0.025)
    base_y = np.nanmax(hi) + 0.015
    bracket_gap = 0.022
    h = 0.005
    comparisons = [
        ("suppressed_minus_absent", "suppressed", "absent"),
        ("indirect_minus_absent", "suppressed_indirect", "absent"),
        ("suppressed_minus_unrelated", "suppressed", "unrelated_suppression"),
        ("suppressed_minus_mentioned", "suppressed", "mentioned"),
        ("mentioned_minus_absent", "mentioned", "absent"),
    ]
    plotted = 0
    for comp_name, cond_a, cond_b in comparisons:
        if cond_a not in xpos or cond_b not in xpos or pairwise.empty or "comparison" not in pairwise.columns:
            continue
        row = pairwise[pairwise["comparison"] == comp_name]
        if row.empty:
            continue
        row = row.iloc[0]
        delta = row.get("mean_delta", np.nan)
        stars = p_to_stars(row.get("paired_p", np.nan))
        sig_text = f"delta={delta:.3f} n.s." if stars == "n.s." else f"delta={delta:.3f}{stars}"
        y = base_y + plotted * bracket_gap
        add_sig_bracket(axes[1], xpos[cond_b], xpos[cond_a], y, h, sig_text, fontsize=8)
        plotted += 1

    axes[1].set_ylim(y_min, base_y + max(plotted, 1) * bracket_gap + 0.025)
    axes[1].text(
        0.99,
        0.02,
        "* p<.05, ** p<.01, *** p<.001",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
    )

    plt.savefig(os.path.join(outdir, f"{basename}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(outdir, f"{basename}.pdf"), bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Report
# ----------------------------

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


def select_pairwise_comparison(pairwise: pd.DataFrame) -> Optional[pd.Series]:
    if pairwise is None or len(pairwise) == 0:
        return None
    for name in ["suppressed_minus_absent", "indirect_minus_absent", "suppressed_minus_unrelated"]:
        match = pairwise[pairwise["comparison"] == name]
        if len(match) > 0:
            return match.iloc[0]
    return pairwise.iloc[0]


def write_report(args, condition_summary, pairwise, outdir: str):
    selected = select_pairwise_comparison(pairwise)
    lines = [
        "# Experiment 3 Summary",
        "",
        "## run_config",
        "",
        *make_variable_table([
            ("model", args.model),
            ("embedding_model", args.embedding_model),
            ("concepts_path", args.concepts_path),
            ("n_samples", args.n_samples),
            ("temperature", args.temperature),
            ("top_p", args.top_p),
            ("successful_suppression_only", args.successful_suppression_only),
            ("include_indirect", args.include_indirect),
            ("include_unrelated_control", args.include_unrelated_control),
            ("seed", args.seed),
        ]),
    ]

    if selected is not None:
        lines += [
            "",
            "## pairwise_comparison",
            "",
            *make_variable_table([
                ("selected_comparison", selected.get("comparison", "")),
                ("condition_a", selected.get("condition_a", "")),
                ("condition_b", selected.get("condition_b", "")),
                ("mean_a", selected.get("mean_a", np.nan)),
                ("mean_b", selected.get("mean_b", np.nan)),
                ("mean_delta", selected.get("mean_delta", np.nan)),
                ("ci95_low_delta", selected.get("ci95_low_delta", np.nan)),
                ("ci95_high_delta", selected.get("ci95_high_delta", np.nan)),
                ("paired_p", selected.get("paired_p", np.nan)),
                ("wilcoxon_p", selected.get("wilcoxon_p", np.nan)),
                ("cohen_d_paired", selected.get("cohen_d_paired", np.nan)),
                ("n_matched", selected.get("n_matched", np.nan)),
            ]),
        ]

    if condition_summary is not None and len(condition_summary) > 0:
        top_condition = condition_summary.sort_values("mean_similarity", ascending=False).iloc[0]
        lines += [
            "",
            "## top_condition",
            "",
            *make_variable_table([
                ("condition", top_condition.get("condition", "")),
                ("mean_similarity", top_condition.get("mean_similarity", np.nan)),
                ("ci95_low", top_condition.get("ci95_low", np.nan)),
                ("ci95_high", top_condition.get("ci95_high", np.nan)),
                ("explicit_alias_leak_rate", top_condition.get("explicit_alias_leak_rate", np.nan)),
                ("n", top_condition.get("n", np.nan)),
            ]),
        ]

    lines += [
        "",
        "## outputs",
        "",
        *make_variable_table([(name, name) for name in MAIN_OUTPUT_FILES]),
    ]

    with open(os.path.join(outdir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--concepts_path", type=str, default="concepts.json")
    parser.add_argument("--outdir", type=str, default="results_behavioral_leakage")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--load_4bit", action="store_true")
    parser.add_argument("--device_map_auto", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")

    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--system_prompt", type=str, default=None)

    parser.add_argument("--include_indirect", action="store_true")
    parser.add_argument("--indirect_mode", type=str, default="first", choices=["first", "all", "random"])
    parser.add_argument("--max_indirect_per_concept", type=int, default=2)

    add_boolean_optional_argument(parser, "--include_unrelated_control", default=True)
    parser.add_argument("--default_unrelated_control", type=str, default="flowers")

    parser.add_argument("--n_samples", type=int, default=3)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)

    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding_device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--embedding_batch_size", type=int, default=32)
    parser.add_argument("--embedding_max_length", type=int, default=256)
    parser.add_argument("--embedding_trust_remote_code", action="store_true")
    parser.add_argument(
        "--concept_embedding_source",
        type=str,
        default="all",
        choices=["aliases", "positive", "indirect", "all"],
        help="Texts used to build target concept embedding.",
    )

    add_boolean_optional_argument(parser, "--successful_suppression_only", default=True)
    parser.add_argument("--save_prompts", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    seed_everything(args.seed)

    safe_model = sanitize_filename(args.model)
    safe_embed = sanitize_filename(args.embedding_model)
    run_name = f"{safe_model}__embed-{safe_embed}__seed-{args.seed}"
    outdir = os.path.join(args.outdir, run_name)
    os.makedirs(outdir, exist_ok=True)

    with open(os.path.join(outdir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    concepts = load_concepts(args.concepts_path)

    print("Loading generator...")
    tokenizer, model = load_generator(args)

    examples = build_generation_examples(concepts, tokenizer, args)

    print(f"Running model: {args.model}")
    print(f"Concepts: {len(concepts)}")
    print(f"Generation examples: {len(examples)}")
    print(f"Samples per prompt: {args.n_samples}")
    print(f"Output directory: {outdir}")

    generations = run_generations(model, tokenizer, examples, args, outdir)

    # Free generator memory before loading embedding model if possible.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    scored = score_semantic_similarity(generations, concepts, args, outdir)

    condition_summary = summarize_by_condition(scored, outdir, args)
    concept_summary = summarize_by_concept_condition(scored, outdir, args)
    pairwise = pairwise_condition_comparisons(scored, outdir, args)

    plot_condition_similarity(condition_summary, outdir)
    plot_delta_by_concept(concept_summary, outdir)
    plot_explicit_leak_rates(condition_summary, outdir)
    plot_combined_behavioral_figure(condition_summary, pairwise, outdir)

    write_report(args, condition_summary, pairwise, outdir)

    print("\nDone.")
    print(f"Results written to: {outdir}")
    print("Main files:")
    for name in MAIN_OUTPUT_FILES:
        print(f"  {name}")


if __name__ == "__main__":
    main()
