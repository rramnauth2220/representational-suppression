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
from transformers import AutoTokenizer, AutoModelForCausalLM


MAIN_OUTPUT_FILES = [
    "attention_token_regions.csv",
    "attention_layer_summary.csv",
    "attention_condition_comparisons.csv",
    "attention_head_effects.csv",
    "top_suppression_sensitive_heads.csv",
    "experiment2_attention_combined.png",
    "experiment2_attention_combined.pdf",
    "experiment2_top10_heads_summary.csv",
    "report.md",
    "plot_attention_by_layer.png",
    "plot_suppression_head_effects.png",
]


@dataclass
class AttentionExample:
    concept: str
    context: str
    condition: str
    prompt: str
    region_label: str
    region_phrases: List[str]


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

    parser.add_argument(name, dest=name.lstrip("-").replace("-", "_"), action="store_true", **kwargs)
    parser.add_argument(f"--no-{name.lstrip('-')}", dest=name.lstrip("-").replace("-", "_"), action="store_false")
    parser.set_defaults(**{name.lstrip("-").replace("-", "_"): default})


def resolve_layers(layer_arg: str, num_hidden_layers: int) -> List[int]:
    if layer_arg == "all":
        return list(range(num_hidden_layers))
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

    for concept, data in concepts.items():
        if "contexts" not in data:
            raise ValueError(f"Concept {concept!r} is missing required key 'contexts'")
        data.setdefault("aliases", [concept])
        data.setdefault("direct_aliases", data.get("aliases", [concept]))
        data.setdefault("indirect_descriptions", [])
        if not data["indirect_descriptions"] and data.get("indirect_description"):
            data["indirect_descriptions"] = [data["indirect_description"]]
        if not data["indirect_descriptions"] and data.get("description"):
            data["indirect_descriptions"] = [data["description"]]
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


def unrelated_region_spec(unrelated: str, aliases: List[str], args) -> Tuple[str, List[str]]:
    mode = getattr(args, "unrelated_control_region", "target_alias")
    if mode == "target_alias":
        return "target_alias", aliases
    if mode == "unrelated_control":
        return "unrelated_control", [unrelated]
    raise ValueError(f"Unknown unrelated_control_region: {mode}")


def build_attention_examples(concepts, tokenizer, args) -> List[AttentionExample]:
    examples: List[AttentionExample] = []

    for concept, data in concepts.items():
        aliases = get_direct_aliases(data, concept)
        indirects = select_indirect_descriptions(data, args) if args.include_indirect else []
        unrelated = choose_unrelated_control(concepts, concept, data, args) if args.include_unrelated_control else None

        for ctx in data["contexts"]:
            raw = {
                "mentioned": (
                    f"{ctx}\n\nInclude {concept} in the description.",
                    "target_alias",
                    aliases,
                ),
                "suppressed": (
                    f"{ctx}\n\nDo not mention {concept}.",
                    "target_alias",
                    aliases,
                ),
            }

            if unrelated:
                region_label, phrases = unrelated_region_spec(unrelated, aliases, args)
                raw["unrelated_suppression"] = (
                    f"{ctx}\n\nDo not mention {unrelated}.",
                    region_label,
                    phrases,
                )

            for j, desc in enumerate(indirects):
                suffix = "" if len(indirects) == 1 else f"_{j+1}"
                raw[f"suppressed_indirect{suffix}"] = (
                    f"{ctx}\n\nDo not mention or allude to {desc}.",
                    "indirect_description",
                    [desc],
                )

            if args.include_context_region:
                raw["absent_context"] = (
                    ctx,
                    "context_region",
                    [ctx],
                )

            for condition, (text, region_label, phrases) in raw.items():
                prompt = maybe_chat_format(tokenizer, text, args.use_chat_template, args.system_prompt)
                examples.append(
                    AttentionExample(
                        concept=concept,
                        context=ctx,
                        condition=condition,
                        prompt=prompt,
                        region_label=region_label,
                        region_phrases=phrases,
                    )
                )

    return examples


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
    attn_implementation = getattr(args, "attn_implementation", None)
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
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


def normalize_piece(s: str) -> str:
    return s.replace("Ġ", "").replace("▁", "").strip().lower()


def find_phrase_positions_by_decoding(tokenizer, input_ids: torch.Tensor, phrases: List[str]) -> List[int]:
    ids_list = input_ids.detach().cpu().tolist()
    full_text = tokenizer.decode(ids_list, skip_special_tokens=False)
    positions = set()

    # Best-effort offset mapping over the decoded full text.
    try:
        enc = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc.get("offset_mapping", [])
        if len(offsets) == len(ids_list):
            text_l = full_text.lower()
            for phrase in phrases:
                phrase_l = phrase.lower().strip()
                if not phrase_l:
                    continue
                for match in re.finditer(re.escape(phrase_l), text_l):
                    start, end = match.span()
                    for i, (s, e) in enumerate(offsets):
                        if e > start and s < end:
                            positions.add(i)
    except Exception:
        pass

    # Fallback token-piece matching.
    if not positions:
        decoded_pieces = [normalize_piece(tokenizer.decode([tid])) for tid in ids_list]
        for phrase in phrases:
            phrase_l = phrase.lower().strip()
            if not phrase_l:
                continue
            phrase_words = set(re.findall(r"[a-zA-Z0-9']+", phrase_l))
            for i, piece in enumerate(decoded_pieces):
                if not piece:
                    continue
                if phrase_l in piece or piece in phrase_words:
                    positions.add(i)

    return sorted(positions)


def forward_attentions(model, tokenizer, prompt: str, max_length: int):
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
            output_attentions=True,
            output_hidden_states=False,
            use_cache=False,
        )
    return inputs, out


def attention_metric_row(ex: AttentionExample, example_id, layer, head, seq_len, mean_attention, max_attention, last_token_attention):
    return {
        "concept": ex.concept,
        "example_id": example_id,
        "condition": ex.condition,
        "region_label": ex.region_label,
        "layer": layer,
        "head": head,
        "mean_attention_to_region": mean_attention,
        "max_attention_to_region": max_attention,
        "attention_from_last_token_to_region": last_token_attention,
        "seq_len": seq_len,
    }


def add_attention_debug_columns(row: Dict[str, Any], ex: AttentionExample, region_positions: List[int]):
    row.update({
        "context": ex.context,
        "region_phrases": json.dumps(ex.region_phrases),
        "region_positions": json.dumps(region_positions),
        "prompt": ex.prompt,
    })
    return row


def matched_example_columns(df: pd.DataFrame) -> List[str]:
    if "example_id" in df.columns:
        return ["concept", "example_id"]
    if "context" in df.columns:
        return ["concept", "context"]
    return ["concept"]


def make_example_id_map(examples: List[AttentionExample]) -> Dict[Tuple[str, str], int]:
    example_ids = {}
    for ex in examples:
        key = (ex.concept, ex.context)
        if key not in example_ids:
            example_ids[key] = len(example_ids)
    return example_ids


def extract_attention_region_metrics(model, tokenizer, examples: List[AttentionExample], layers: List[int], args, outdir: str) -> pd.DataFrame:
    rows = []
    save_debug = getattr(args, "save_token_region_debug", False)
    example_ids = make_example_id_map(examples)

    for ex in tqdm(examples, desc="Extracting attention metrics"):
        example_id = example_ids[(ex.concept, ex.context)]
        inputs, out = forward_attentions(model, tokenizer, ex.prompt, args.max_length)
        if out.attentions is None:
            raise RuntimeError("Model did not return attentions. Some models require output_attentions=True support.")

        input_ids = inputs["input_ids"][0].detach().cpu()
        region_positions = find_phrase_positions_by_decoding(tokenizer, input_ids, ex.region_phrases)

        if not region_positions:
            if args.keep_missing_regions:
                row = attention_metric_row(
                    ex,
                    example_id=example_id,
                    layer=np.nan,
                    head=np.nan,
                    seq_len=int(input_ids.numel()),
                    mean_attention=np.nan,
                    max_attention=np.nan,
                    last_token_attention=np.nan,
                )
                if save_debug:
                    row = add_attention_debug_columns(row, ex, [])
                rows.append(row)
            del out, inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        seq_len = int(input_ids.numel())
        last_idx = seq_len - 1

        available_layers = len(out.attentions)
        valid_layers = [l for l in layers if 0 <= l < available_layers]

        for l in valid_layers:
            attn = out.attentions[l][0].detach().float().cpu()  # [heads, query, key]
            for head in range(attn.shape[0]):
                target_mass_by_query = attn[head, :, region_positions].sum(dim=-1)
                row = attention_metric_row(
                    ex,
                    example_id=example_id,
                    layer=int(l),
                    head=int(head),
                    seq_len=seq_len,
                    mean_attention=float(target_mass_by_query.mean().item()),
                    max_attention=float(target_mass_by_query.max().item()),
                    last_token_attention=float(target_mass_by_query[last_idx].item()),
                )
                if save_debug:
                    row = add_attention_debug_columns(row, ex, region_positions)
                rows.append(row)

        del out, inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "attention_token_regions.csv"), index=False)
    return df


def summarize_by_layer(df: pd.DataFrame, outdir: str) -> pd.DataFrame:
    valid = df.dropna(subset=["layer", "head"]).copy()
    summary_rows = []

    for (condition, region_label, layer), sub in valid.groupby(["condition", "region_label", "layer"]):
        vals = sub["mean_attention_to_region"].astype(float).values
        lo, hi = bootstrap_ci(vals, seed=100 + int(layer))
        summary_rows.append({
            "condition": condition,
            "region_label": region_label,
            "layer": int(layer),
            "mean_attention_to_region": float(np.mean(vals)),
            "sem_attention_to_region": float(pd.Series(vals).sem()),
            "ci95_low": lo,
            "ci95_high": hi,
            "n": int(len(vals)),
        })

    summary = pd.DataFrame(summary_rows).sort_values(["region_label", "condition", "layer"])
    summary.to_csv(os.path.join(outdir, "attention_layer_summary.csv"), index=False)
    return summary


def compute_head_effects(df: pd.DataFrame, outdir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    valid = df.dropna(subset=["layer", "head"]).copy()

    pivot_index = matched_example_columns(valid) + ["layer", "head", "region_label"]
    wide = valid.pivot_table(
        index=pivot_index,
        columns="condition",
        values="mean_attention_to_region",
        aggfunc="mean",
    ).reset_index()

    effect_rows = []
    comparisons = [
        ("suppressed", "mentioned", "suppressed_minus_mentioned"),
        ("suppressed", "unrelated_suppression", "suppressed_minus_unrelated"),
        ("suppressed_indirect", "mentioned", "indirect_minus_mentioned"),
        ("suppressed_indirect_1", "mentioned", "indirect1_minus_mentioned"),
        ("suppressed_indirect_2", "mentioned", "indirect2_minus_mentioned"),
    ]

    for (layer, head, region_label), sub in wide.groupby(["layer", "head", "region_label"]):
        row_base = {
            "layer": int(layer),
            "head": int(head),
            "region_label": region_label,
            "n_pairs": int(len(sub)),
        }
        for a, b, name in comparisons:
            if a in sub.columns and b in sub.columns:
                paired = sub[[a, b]].dropna()
                if len(paired) == 0:
                    continue
                delta = paired[a].values - paired[b].values
                lo, hi = bootstrap_ci(delta, seed=1000 + int(layer) * 100 + int(head))
                t, p = safe_ttest_rel(paired[a], paired[b])
                w, wp = safe_wilcoxon(paired[a], paired[b])
                d = cohen_d_paired(paired[a], paired[b])
                effect_rows.append({
                    **row_base,
                    "comparison": name,
                    "condition_a": a,
                    "condition_b": b,
                    "mean_a": float(paired[a].mean()),
                    "mean_b": float(paired[b].mean()),
                    "mean_delta": float(delta.mean()),
                    "sem_delta": float(pd.Series(delta).sem()),
                    "ci95_low_delta": lo,
                    "ci95_high_delta": hi,
                    "paired_t": t,
                    "paired_p": p,
                    "wilcoxon_stat": w,
                    "wilcoxon_p": wp,
                    "cohen_d_paired": d,
                    "n_matched": int(len(paired)),
                })

    effects = pd.DataFrame(effect_rows)
    if len(effects) > 0:
        effects = effects.sort_values(["comparison", "mean_delta"], ascending=[True, False])
    effects.to_csv(os.path.join(outdir, "attention_head_effects.csv"), index=False)

    top = effects[
        (effects["comparison"] == "suppressed_minus_mentioned")
        & (effects["region_label"] == "target_alias")
    ].copy() if len(effects) > 0 else pd.DataFrame()
    if len(top) > 0:
        top = top.sort_values("mean_delta", ascending=False)
    top.to_csv(os.path.join(outdir, "top_suppression_sensitive_heads.csv"), index=False)

    return effects, top


def summarize_condition_comparisons(df: pd.DataFrame, outdir: str) -> pd.DataFrame:
    valid = df.dropna(subset=["layer", "head"]).copy()
    agg = (
        valid.groupby(matched_example_columns(valid) + ["condition", "region_label"])
        ["mean_attention_to_region"]
        .mean()
        .reset_index()
    )

    wide = agg.pivot_table(
        index=matched_example_columns(agg) + ["region_label"],
        columns="condition",
        values="mean_attention_to_region",
        aggfunc="mean",
    ).reset_index()

    rows = []
    comparisons = [
        ("suppressed", "mentioned", "suppressed_minus_mentioned"),
        ("suppressed", "unrelated_suppression", "suppressed_minus_unrelated"),
        ("suppressed_indirect", "mentioned", "indirect_minus_mentioned"),
        ("suppressed_indirect_1", "mentioned", "indirect1_minus_mentioned"),
        ("suppressed_indirect_2", "mentioned", "indirect2_minus_mentioned"),
    ]

    for region_label, sub in wide.groupby("region_label"):
        for a, b, name in comparisons:
            if a in sub.columns and b in sub.columns:
                paired = sub[[a, b]].dropna()
                if len(paired) == 0:
                    continue
                delta = paired[a] - paired[b]
                lo, hi = bootstrap_ci(delta, seed=77)
                t, p = safe_ttest_rel(paired[a], paired[b])
                w, wp = safe_wilcoxon(paired[a], paired[b])
                rows.append({
                    "region_label": region_label,
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
    comp.to_csv(os.path.join(outdir, "attention_condition_comparisons.csv"), index=False)
    return comp


def plot_attention_by_layer(summary: pd.DataFrame, outdir: str):
    if len(summary) == 0:
        return

    for region_label in summary["region_label"].unique():
        sub = summary[summary["region_label"] == region_label]
        plt.figure()
        for condition in sorted(sub["condition"].unique()):
            cond = sub[sub["condition"] == condition].sort_values("layer")
            plt.plot(cond["layer"], cond["mean_attention_to_region"], label=condition)
            plt.fill_between(cond["layer"], cond["ci95_low"], cond["ci95_high"], alpha=0.15)

        plt.xlabel("Layer")
        plt.ylabel("Mean attention mass to region")
        plt.title(f"Attention to {region_label} region by layer")
        plt.legend()
        plt.tight_layout()
        safe = sanitize_filename(region_label)
        out = os.path.join(outdir, f"plot_attention_by_layer__{safe}.png")
        plt.savefig(out, dpi=220)
        plt.close()

    target_path = os.path.join(outdir, "plot_attention_by_layer__target_alias.png")
    if os.path.exists(target_path):
        import shutil
        shutil.copyfile(target_path, os.path.join(outdir, "plot_attention_by_layer.png"))


def plot_head_effects(effects: pd.DataFrame, outdir: str, comparison: str = "suppressed_minus_mentioned", top_k: int = 30):
    if len(effects) == 0:
        return
    sub = effects[
        (effects["comparison"] == comparison)
        & (effects["region_label"] == "target_alias")
    ].copy()
    if len(sub) == 0:
        return
    sub = sub.sort_values("mean_delta", ascending=False).head(top_k)
    labels = [f"L{int(r['layer'])}H{int(r['head'])}" for _, r in sub.iterrows()]

    plt.figure(figsize=(max(8, top_k * 0.25), 4))
    plt.bar(range(len(sub)), sub["mean_delta"])
    plt.axhline(0, linestyle="--")
    plt.xticks(range(len(sub)), labels, rotation=75, ha="right")
    plt.ylabel("Mean attention delta")
    plt.title("Top suppression-sensitive heads")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plot_suppression_head_effects.png"), dpi=220)
    plt.close()


def coerce_attention_plot_columns(layer_summary: pd.DataFrame, head_effects: pd.DataFrame):
    layer_summary = layer_summary.copy()
    head_effects = head_effects.copy()
    numeric_cols = {
        "layer",
        "head",
        "mean_delta",
        "ci95_low_delta",
        "ci95_high_delta",
        "mean_attention_to_region",
        "ci95_low",
        "ci95_high",
    }
    for df in [layer_summary, head_effects]:
        for col in numeric_cols.intersection(df.columns):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return layer_summary, head_effects


def plot_combined_attention_figure(layer_summary: pd.DataFrame, head_effects: pd.DataFrame, outdir: str, top_k: int = 10):
    layer_summary, head_effects = coerce_attention_plot_columns(layer_summary, head_effects)

    if layer_summary is None or layer_summary.empty or "region_label" not in layer_summary.columns:
        return

    layer_target = layer_summary[layer_summary["region_label"] == "target_alias"].copy()
    if (
        head_effects is not None
        and not head_effects.empty
        and {"region_label", "comparison"}.issubset(head_effects.columns)
    ):
        head_target = head_effects[
            (head_effects["region_label"] == "target_alias")
            & (head_effects["comparison"] == "suppressed_minus_mentioned")
        ].copy()
    else:
        head_target = pd.DataFrame(columns=["layer", "head", "mean_delta", "ci95_low_delta"])

    if layer_target.empty:
        return

    top_heads = head_target.sort_values("mean_delta", ascending=False).head(top_k)
    top_heads.to_csv(os.path.join(outdir, "experiment2_top10_heads_summary.csv"), index=False)

    layer_head_max = (
        head_target.groupby("layer", as_index=False)["mean_delta"]
        .max()
        .rename(columns={"mean_delta": "max_suppression_head_delta"})
    )
    positive_heads = head_target[head_target["ci95_low_delta"] > 0].copy()
    layer_head_count = (
        positive_heads.groupby("layer", as_index=False)["head"]
        .nunique()
        .rename(columns={"head": "n_positive_heads"})
    )

    layers = sorted(layer_target["layer"].dropna().astype(int).unique())
    if not layers:
        return
    layer_support = pd.DataFrame({"layer": layers})
    layer_head_max = layer_support.merge(layer_head_max, on="layer", how="left").fillna(0)
    layer_head_count = layer_support.merge(layer_head_count, on="layer", how="left").fillna(0)

    label_map = {
        "mentioned": "direct mention",
        "suppressed": "direct suppression",
        "suppressed_indirect": "indirect suppression",
        "suppressed_indirect_1": "indirect suppression",
        "unrelated_suppression": "unrelated suppression",
    }
    conditions_order = [
        "mentioned",
        "suppressed",
        "suppressed_indirect",
        "suppressed_indirect_1",
        "unrelated_suppression",
    ]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.4),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.1], "hspace": 0.12},
    )

    seen_labels = set()
    for cond in conditions_order:
        if cond not in set(layer_target["condition"]):
            continue
        sub = layer_target[layer_target["condition"] == cond].sort_values("layer").dropna(
            subset=["layer", "mean_attention_to_region", "ci95_low", "ci95_high"]
        )
        if sub.empty:
            continue
        label = label_map.get(cond, cond)
        if label in seen_labels:
            label = None
        else:
            seen_labels.add(label)

        x = sub["layer"].to_numpy(dtype=float)
        y = sub["mean_attention_to_region"].to_numpy(dtype=float)
        lo = sub["ci95_low"].to_numpy(dtype=float)
        hi = sub["ci95_high"].to_numpy(dtype=float)
        axes[0].plot(x, y, linewidth=2, label=label)
        axes[0].fill_between(x, lo, hi, alpha=0.14)

    axes[0].set_ylabel("Mean attention mass")
    axes[0].set_title("(A) Layerwise attention to concept-associated token regions", loc="left", fontsize=11)
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    x2 = layer_head_max["layer"].to_numpy(dtype=float)
    y2 = layer_head_max["max_suppression_head_delta"].to_numpy(dtype=float)
    axes[1].bar(x2, y2, width=0.8)
    axes[1].axhline(0, linestyle="--", linewidth=1)
    axes[1].set_ylabel("Max head delta")
    axes[1].set_xlabel("Layer")
    axes[1].set_title("(B) Largest suppression-sensitive head effect per layer", loc="left", fontsize=11)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    for _, row in layer_head_count.iterrows():
        n = int(row["n_positive_heads"])
        if n <= 0:
            continue
        layer = int(row["layer"])
        y = float(layer_head_max.loc[layer_head_max["layer"] == layer, "max_suppression_head_delta"].iloc[0])
        if y > 0:
            axes[1].text(layer, y + 0.001, str(n), ha="center", va="bottom", fontsize=7)

    axes[1].set_xticks(np.arange(min(layers), max(layers) + 1, 4))
    plt.savefig(os.path.join(outdir, "experiment2_attention_combined.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(outdir, "experiment2_attention_combined.pdf"), bbox_inches="tight")
    plt.close(fig)


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


def select_condition_comparison(condition_comparisons: pd.DataFrame) -> Optional[pd.Series]:
    if condition_comparisons is None or len(condition_comparisons) == 0:
        return None

    preferred = [
        ("target_alias", "suppressed_minus_mentioned"),
        ("target_alias", "suppressed_minus_unrelated"),
        ("target_alias", "indirect_minus_mentioned"),
    ]
    for region_label, comparison in preferred:
        match = condition_comparisons[
            (condition_comparisons["region_label"] == region_label)
            & (condition_comparisons["comparison"] == comparison)
        ]
        if len(match) > 0:
            return match.iloc[0]

    return condition_comparisons.iloc[0]


def select_top_head(top_heads: pd.DataFrame) -> Optional[pd.Series]:
    if top_heads is None or len(top_heads) == 0:
        return None
    return top_heads.sort_values("mean_delta", ascending=False).iloc[0]


def write_report(args, examples, condition_comparisons, top_heads, outdir: str):
    selected_comparison = select_condition_comparison(condition_comparisons)
    selected_head = select_top_head(top_heads)

    lines = [
        "# Experiment 2 Summary",
        "",
        "## run_config",
        "",
        *make_variable_table([
            ("model", args.model),
            ("concepts_path", args.concepts_path),
            ("n_examples", len(examples)),
            ("include_indirect", args.include_indirect),
            ("include_unrelated_control", args.include_unrelated_control),
            ("unrelated_control_region", args.unrelated_control_region),
            ("attn_implementation", args.attn_implementation),
            ("layers", args.layers),
            ("max_length", args.max_length),
            ("save_token_region_debug", args.save_token_region_debug),
            ("seed", args.seed),
        ]),
    ]

    if selected_comparison is not None:
        lines += [
            "",
            "## condition_comparison",
            "",
            *make_variable_table([
                ("selected_region_label", selected_comparison.get("region_label", "")),
                ("selected_comparison", selected_comparison.get("comparison", "")),
                ("condition_a", selected_comparison.get("condition_a", "")),
                ("condition_b", selected_comparison.get("condition_b", "")),
                ("mean_a", selected_comparison.get("mean_a", np.nan)),
                ("mean_b", selected_comparison.get("mean_b", np.nan)),
                ("mean_delta", selected_comparison.get("mean_delta", np.nan)),
                ("ci95_low_delta", selected_comparison.get("ci95_low_delta", np.nan)),
                ("ci95_high_delta", selected_comparison.get("ci95_high_delta", np.nan)),
                ("paired_p", selected_comparison.get("paired_p", np.nan)),
                ("wilcoxon_p", selected_comparison.get("wilcoxon_p", np.nan)),
                ("cohen_d_paired", selected_comparison.get("cohen_d_paired", np.nan)),
                ("n_matched", selected_comparison.get("n_matched", np.nan)),
            ]),
        ]

    if selected_head is not None:
        lines += [
            "",
            "## top_head",
            "",
            *make_variable_table([
                ("comparison", selected_head.get("comparison", "")),
                ("region_label", selected_head.get("region_label", "")),
                ("layer", selected_head.get("layer", np.nan)),
                ("head", selected_head.get("head", np.nan)),
                ("mean_a", selected_head.get("mean_a", np.nan)),
                ("mean_b", selected_head.get("mean_b", np.nan)),
                ("mean_delta", selected_head.get("mean_delta", np.nan)),
                ("ci95_low_delta", selected_head.get("ci95_low_delta", np.nan)),
                ("ci95_high_delta", selected_head.get("ci95_high_delta", np.nan)),
                ("paired_p", selected_head.get("paired_p", np.nan)),
                ("wilcoxon_p", selected_head.get("wilcoxon_p", np.nan)),
                ("cohen_d_paired", selected_head.get("cohen_d_paired", np.nan)),
                ("n_matched", selected_head.get("n_matched", np.nan)),
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--concepts_path", type=str, default="concepts.json")
    parser.add_argument("--outdir", type=str, default="results_attention_dynamics")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--layers", type=str, default="all")
    parser.add_argument("--max_length", type=int, default=256)

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--load_4bit", action="store_true")
    parser.add_argument("--device_map_auto", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Use eager attention when output_attentions=True is needed reliably.",
    )

    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--system_prompt", type=str, default=None)

    parser.add_argument("--include_indirect", action="store_true")
    parser.add_argument("--indirect_mode", type=str, default="first", choices=["first", "all", "random"])
    parser.add_argument("--max_indirect_per_concept", type=int, default=2)

    add_boolean_optional_argument(parser, "--include_unrelated_control", default=True)
    parser.add_argument("--default_unrelated_control", type=str, default="flowers")
    parser.add_argument(
        "--unrelated_control_region",
        type=str,
        default="target_alias",
        choices=["target_alias", "unrelated_control"],
        help=(
            "For unrelated_suppression prompts, measure attention to the original "
            "target alias or to the unrelated suppressed phrase."
        ),
    )

    parser.add_argument("--include_context_region", action="store_true")
    parser.add_argument("--keep_missing_regions", action="store_true")
    parser.add_argument(
        "--save_token_region_debug",
        action="store_true",
        help="Include prompt, context, region phrases, and token positions in attention_token_regions.csv.",
    )

    parser.add_argument("--top_k_heads", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    seed_everything(args.seed)

    safe_model = sanitize_filename(args.model)
    run_name = f"{safe_model}__attn_seed-{args.seed}"
    outdir = os.path.join(args.outdir, run_name)
    os.makedirs(outdir, exist_ok=True)

    with open(os.path.join(outdir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    tokenizer, model = load_model(args)
    concepts = load_concepts(args.concepts_path)

    layers = resolve_layers(args.layers, model.config.num_hidden_layers)

    examples = build_attention_examples(concepts, tokenizer, args)

    print(f"Running model: {args.model}")
    print(f"Concepts: {len(concepts)}")
    print(f"Attention examples: {len(examples)}")
    print(f"Layers: {layers}")
    print(f"Unrelated control region: {args.unrelated_control_region}")
    print(f"Output directory: {outdir}")

    attn_df = extract_attention_region_metrics(model, tokenizer, examples, layers, args, outdir)
    layer_summary = summarize_by_layer(attn_df, outdir)
    condition_comparisons = summarize_condition_comparisons(attn_df, outdir)
    head_effects, top_heads = compute_head_effects(attn_df, outdir)

    plot_attention_by_layer(layer_summary, outdir)
    plot_head_effects(head_effects, outdir, top_k=args.top_k_heads)
    plot_combined_attention_figure(layer_summary, head_effects, outdir)

    write_report(args, examples, condition_comparisons, top_heads, outdir)

    print("\nDone.")
    print(f"Results written to: {outdir}")
    print("Main files:")
    for name in MAIN_OUTPUT_FILES:
        print(f"  {name}")


if __name__ == "__main__":
    main()
