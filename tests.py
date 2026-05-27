import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import exp1_recoverability as exp1
import exp2_attention as exp2
import exp3_leakage as exp3
import exp4_cross_model as exp4


class DummyTokenizer:
    chat_template = "dummy"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        rendered = "|".join(f"{m['role']}:{m['content']}" for m in messages)
        if add_generation_prompt:
            rendered += "|assistant:"
        return rendered


class ConceptLoadingTests(unittest.TestCase):
    def test_exp1_load_concepts_adds_defaults_and_legacy_description(self):
        payload = {
            "white_bear": {
                "contexts": ["Describe a winter scene."],
                "positive": ["A polar bear crosses the ice."],
                "negative": ["A red car waits at a light."],
                "aliases": ["polar bear"],
                "description": "large white arctic mammal",
            }
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "concepts.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            concepts = exp1.load_concepts(str(path))

        data = concepts["white_bear"]
        self.assertEqual(data["direct_aliases"], ["polar bear"])
        self.assertEqual(data["indirect_descriptions"], ["large white arctic mammal"])
        self.assertEqual(data["negative_hard"], [])

    def test_exp1_load_concepts_rejects_missing_required_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "concepts.json"
            path.write_text(json.dumps({"x": {"contexts": []}}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "positive"):
                exp1.load_concepts(str(path))


class PromptConstructionTests(unittest.TestCase):
    def test_exp1_resolve_layers_includes_embedding_layer_for_all(self):
        self.assertEqual(exp1.resolve_layers("all", num_hidden_layers=3), [0, 1, 2, 3])
        self.assertEqual(exp1.resolve_layers("0,2", num_hidden_layers=12), [0, 2])

    def test_exp1_make_output_dir_sanitizes_model_name(self):
        args = SimpleNamespace(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            pooling="mean_nonpad",
            seed=42,
            outdir="results",
        )

        self.assertEqual(
            exp1.make_output_dir(args),
            "results\\meta-llama_Meta-Llama-3-8B-Instruct__pool-mean_nonpad__seed-42",
        )

    def test_exp1_build_eval_examples_respects_only_indirect(self):
        concepts = {
            "knife": {
                "contexts": ["Write a kitchen vignette."],
                "indirect_descriptions": ["sharp metal utensil", "cutting tool"],
            }
        }
        args = SimpleNamespace(
            indirect_mode="all",
            max_indirect_per_concept=2,
            only_indirect=True,
            seed=7,
        )

        examples = exp1.build_eval_examples(
            concepts,
            DummyTokenizer(),
            use_chat_template=False,
            system_prompt=None,
            include_indirect=True,
            args=args,
        )

        self.assertEqual(len(examples), 1)
        self.assertEqual(
            set(examples[0].prompts),
            {"absent", "mentioned", "suppressed_indirect_1", "suppressed_indirect_2"},
        )
        self.assertNotIn("suppressed", examples[0].prompts)

    def test_exp2_build_attention_examples_adds_region_labels(self):
        concepts = {
            "knife": {
                "contexts": ["Write a kitchen vignette."],
                "direct_aliases": ["chef knife"],
                "indirect_descriptions": ["sharp metal utensil"],
                "unrelated_controls": ["flower"],
            }
        }
        args = SimpleNamespace(
            include_indirect=True,
            include_unrelated_control=True,
            include_context_region=True,
            indirect_mode="first",
            max_indirect_per_concept=2,
            default_unrelated_control="cloud",
            unrelated_control_region="target_alias",
            use_chat_template=False,
            system_prompt=None,
            seed=7,
        )

        examples = exp2.build_attention_examples(concepts, DummyTokenizer(), args)
        by_condition = {ex.condition: ex for ex in examples}

        self.assertEqual(by_condition["mentioned"].region_label, "target_alias")
        self.assertEqual(by_condition["mentioned"].region_phrases, ["chef knife"])
        self.assertEqual(by_condition["unrelated_suppression"].region_label, "target_alias")
        self.assertEqual(by_condition["unrelated_suppression"].region_phrases, ["chef knife"])
        self.assertEqual(by_condition["suppressed_indirect"].region_label, "indirect_description")
        self.assertEqual(by_condition["absent_context"].region_phrases, ["Write a kitchen vignette."])

    def test_exp2_unrelated_control_region_can_measure_suppressed_phrase(self):
        concepts = {
            "knife": {
                "contexts": ["Write a kitchen vignette."],
                "direct_aliases": ["chef knife"],
                "unrelated_controls": ["flower"],
            }
        }
        args = SimpleNamespace(
            include_indirect=False,
            include_unrelated_control=True,
            include_context_region=False,
            indirect_mode="first",
            max_indirect_per_concept=2,
            default_unrelated_control="cloud",
            unrelated_control_region="unrelated_control",
            use_chat_template=False,
            system_prompt=None,
            seed=7,
        )

        examples = exp2.build_attention_examples(concepts, DummyTokenizer(), args)
        unrelated = {ex.condition: ex for ex in examples}["unrelated_suppression"]

        self.assertEqual(unrelated.region_label, "unrelated_control")
        self.assertEqual(unrelated.region_phrases, ["flower"])

    def test_chat_format_includes_system_and_user_messages(self):
        prompt = exp3.maybe_chat_format(
            DummyTokenizer(),
            "Do the task.",
            use_chat_template=True,
            system_prompt="Be concise.",
        )

        self.assertEqual(prompt, "system:Be concise.|user:Do the task.|assistant:")


class ExperimentTwoFigureTests(unittest.TestCase):
    def test_plot_combined_attention_figure_writes_expected_outputs(self):
        layer_summary = pd.DataFrame(
            {
                "condition": ["mentioned", "suppressed", "unrelated_suppression"] * 2,
                "region_label": ["target_alias"] * 6,
                "layer": [0, 0, 0, 1, 1, 1],
                "mean_attention_to_region": [0.2, 0.3, 0.25, 0.22, 0.35, 0.27],
                "ci95_low": [0.1, 0.2, 0.15, 0.12, 0.25, 0.17],
                "ci95_high": [0.3, 0.4, 0.35, 0.32, 0.45, 0.37],
            }
        )
        head_effects = pd.DataFrame(
            {
                "region_label": ["target_alias", "target_alias"],
                "comparison": ["suppressed_minus_mentioned", "suppressed_minus_mentioned"],
                "layer": [0, 1],
                "head": [2, 3],
                "mean_delta": [0.04, 0.08],
                "ci95_low_delta": [0.01, 0.02],
                "ci95_high_delta": [0.07, 0.12],
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            exp2.plot_combined_attention_figure(layer_summary, head_effects, tmp)
            outdir = Path(tmp)

            self.assertTrue((outdir / "experiment2_attention_combined.png").exists())
            self.assertTrue((outdir / "experiment2_attention_combined.pdf").exists())
            top_heads = pd.read_csv(outdir / "experiment2_top10_heads_summary.csv")

        self.assertEqual(list(top_heads["head"]), [3, 2])

    def test_plot_combined_attention_figure_handles_empty_head_effects(self):
        layer_summary = pd.DataFrame(
            {
                "condition": ["mentioned", "suppressed"],
                "region_label": ["target_alias", "target_alias"],
                "layer": [0, 0],
                "mean_attention_to_region": [0.2, 0.3],
                "ci95_low": [0.1, 0.2],
                "ci95_high": [0.3, 0.4],
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            exp2.plot_combined_attention_figure(layer_summary, pd.DataFrame(), tmp)
            outdir = Path(tmp)

            self.assertTrue((outdir / "experiment2_attention_combined.png").exists())
            self.assertTrue((outdir / "experiment2_attention_combined.pdf").exists())
            self.assertTrue((outdir / "experiment2_top10_heads_summary.csv").exists())


class ExperimentTwoOutputSizeTests(unittest.TestCase):
    def test_attention_metric_row_omits_debug_text_by_default(self):
        ex = exp2.AttentionExample(
            concept="knife",
            context="long context",
            condition="suppressed",
            prompt="long prompt",
            region_label="target_alias",
            region_phrases=["chef knife"],
        )

        row = exp2.attention_metric_row(
            ex,
            example_id=9,
            layer=1,
            head=2,
            seq_len=12,
            mean_attention=0.3,
            max_attention=0.6,
            last_token_attention=0.1,
        )

        self.assertEqual(
            set(row),
            {
                "concept",
                "example_id",
                "condition",
                "region_label",
                "layer",
                "head",
                "mean_attention_to_region",
                "max_attention_to_region",
                "attention_from_last_token_to_region",
                "seq_len",
            },
        )
        self.assertNotIn("prompt", row)
        self.assertNotIn("context", row)
        self.assertEqual(row["example_id"], 9)

    def test_attention_debug_columns_are_opt_in(self):
        ex = exp2.AttentionExample(
            concept="knife",
            context="long context",
            condition="suppressed",
            prompt="long prompt",
            region_label="target_alias",
            region_phrases=["chef knife"],
        )
        row = exp2.attention_metric_row(ex, 9, 1, 2, 12, 0.3, 0.6, 0.1)

        row = exp2.add_attention_debug_columns(row, ex, [4, 5])

        self.assertEqual(row["context"], "long context")
        self.assertEqual(row["prompt"], "long prompt")
        self.assertEqual(row["region_positions"], "[4, 5]")

    def test_make_example_id_map_groups_conditions_by_concept_context(self):
        examples = [
            exp2.AttentionExample("knife", "ctx a", "mentioned", "p1", "target_alias", ["knife"]),
            exp2.AttentionExample("knife", "ctx a", "suppressed", "p2", "target_alias", ["knife"]),
            exp2.AttentionExample("knife", "ctx b", "mentioned", "p3", "target_alias", ["knife"]),
        ]

        example_ids = exp2.make_example_id_map(examples)

        self.assertEqual(example_ids[("knife", "ctx a")], 0)
        self.assertEqual(example_ids[("knife", "ctx b")], 1)

    def test_attention_summaries_use_compact_example_id_for_matching(self):
        df = pd.DataFrame(
            {
                "concept": ["knife"] * 4,
                "example_id": [0, 0, 1, 1],
                "condition": ["mentioned", "suppressed", "mentioned", "suppressed"],
                "region_label": ["target_alias"] * 4,
                "layer": [0, 0, 0, 0],
                "head": [0, 0, 0, 0],
                "mean_attention_to_region": [0.2, 0.3, 0.4, 0.6],
                "max_attention_to_region": [0.3, 0.4, 0.5, 0.6],
                "attention_from_last_token_to_region": [0.1, 0.2, 0.3, 0.4],
                "seq_len": [10, 10, 11, 11],
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            comp = exp2.summarize_condition_comparisons(df, tmp)
            effects, top = exp2.compute_head_effects(df, tmp)

        self.assertEqual(comp["comparison"].iloc[0], "suppressed_minus_mentioned")
        self.assertAlmostEqual(float(comp["mean_delta"].iloc[0]), 0.15)
        self.assertEqual(effects["comparison"].iloc[0], "suppressed_minus_mentioned")
        self.assertAlmostEqual(float(effects["mean_delta"].iloc[0]), 0.15)
        self.assertEqual(len(top), 1)


class ExperimentTwoReportTests(unittest.TestCase):
    def test_write_report_uses_compact_variable_tables(self):
        args = SimpleNamespace(
            model="demo/model",
            concepts_path="concepts.json",
            include_indirect=True,
            include_unrelated_control=True,
            unrelated_control_region="target_alias",
            attn_implementation="eager",
            layers="all",
            max_length=256,
            save_token_region_debug=False,
            seed=42,
            top_k_heads=10,
        )
        examples = [
            exp2.AttentionExample(
                concept="knife",
                context="ctx",
                condition="suppressed",
                prompt="prompt",
                region_label="target_alias",
                region_phrases=["knife"],
            )
        ]
        condition_comparisons = pd.DataFrame(
            {
                "region_label": ["target_alias"],
                "comparison": ["suppressed_minus_mentioned"],
                "condition_a": ["suppressed"],
                "condition_b": ["mentioned"],
                "mean_a": [0.3],
                "mean_b": [0.2],
                "mean_delta": [0.1],
                "ci95_low_delta": [0.05],
                "ci95_high_delta": [0.15],
                "paired_p": [0.01],
                "wilcoxon_p": [0.02],
                "cohen_d_paired": [1.5],
                "n_matched": [17],
            }
        )
        top_heads = pd.DataFrame(
            {
                "region_label": ["target_alias"],
                "comparison": ["suppressed_minus_mentioned"],
                "layer": [3],
                "head": [7],
                "mean_a": [0.4],
                "mean_b": [0.1],
                "mean_delta": [0.3],
                "ci95_low_delta": [0.2],
                "ci95_high_delta": [0.4],
                "paired_p": [0.001],
                "wilcoxon_p": [0.002],
                "cohen_d_paired": [2.1],
                "n_matched": [17],
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            exp2.write_report(args, examples, condition_comparisons, top_heads, tmp)
            report = (Path(tmp) / "report.md").read_text(encoding="utf-8")

        self.assertIn("# Experiment 2 Summary", report)
        self.assertIn("## run_config", report)
        self.assertIn("## condition_comparison", report)
        self.assertIn("## top_head", report)
        self.assertIn("`selected_comparison` | suppressed_minus_mentioned", report)
        self.assertIn("`head` | 7", report)
        self.assertNotIn("Main Question", report)
        self.assertNotIn("Design Note", report)


class ExperimentThreeReportAndFigureTests(unittest.TestCase):
    def test_exp3_write_report_uses_compact_variable_tables(self):
        args = SimpleNamespace(
            model="demo/model",
            embedding_model="embed/model",
            concepts_path="concepts.json",
            n_samples=3,
            temperature=0.7,
            top_p=0.95,
            successful_suppression_only=True,
            include_indirect=True,
            include_unrelated_control=True,
            seed=42,
        )
        condition_summary = pd.DataFrame(
            {
                "condition": ["absent", "suppressed"],
                "mean_similarity": [0.1, 0.2],
                "ci95_low": [0.05, 0.15],
                "ci95_high": [0.15, 0.25],
                "explicit_alias_leak_rate": [0.0, 0.1],
                "n": [10, 10],
            }
        )
        pairwise = pd.DataFrame(
            {
                "comparison": ["suppressed_minus_absent"],
                "condition_a": ["suppressed"],
                "condition_b": ["absent"],
                "mean_a": [0.2],
                "mean_b": [0.1],
                "mean_delta": [0.1],
                "ci95_low_delta": [0.04],
                "ci95_high_delta": [0.16],
                "paired_p": [0.01],
                "wilcoxon_p": [0.02],
                "cohen_d_paired": [1.4],
                "n_matched": [10],
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            exp3.write_report(args, condition_summary, pairwise, tmp)
            report = (Path(tmp) / "report.md").read_text(encoding="utf-8")

        self.assertIn("# Experiment 3 Summary", report)
        self.assertIn("## run_config", report)
        self.assertIn("## pairwise_comparison", report)
        self.assertIn("`selected_comparison` | suppressed_minus_absent", report)
        self.assertNotIn("Main Question", report)
        self.assertNotIn("Interpretation Guardrail", report)

    def test_exp3_combined_behavioral_figure_writes_outputs(self):
        summary = pd.DataFrame(
            {
                "condition": ["absent", "suppressed", "mentioned"],
                "mean_similarity": [0.1, 0.2, 0.3],
                "ci95_low": [0.05, 0.15, 0.25],
                "ci95_high": [0.15, 0.25, 0.35],
                "explicit_alias_leak_rate": [0.0, 0.1, 0.2],
                "n": [10, 10, 10],
            }
        )
        pairwise = pd.DataFrame(
            {
                "comparison": ["suppressed_minus_absent"],
                "mean_delta": [0.1],
                "paired_p": [0.01],
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            exp3.plot_combined_behavioral_figure(summary, pairwise, tmp)
            outdir = Path(tmp)

            self.assertTrue((outdir / "exp3_behavioral_combined.png").exists())
            self.assertTrue((outdir / "exp3_behavioral_combined.pdf").exists())


class BehavioralLeakageTests(unittest.TestCase):
    def test_contains_alias_matches_case_insensitive_word_boundaries(self):
        leaked, hits = exp3.contains_alias("A polar bear appears.", ["Polar Bear", "bear"])

        self.assertTrue(leaked)
        self.assertEqual(hits, ["Polar Bear", "bear"])

    def test_contains_alias_does_not_match_inside_larger_word(self):
        leaked, hits = exp3.contains_alias("The scarecrow stood still.", ["crow"])

        self.assertFalse(leaked)
        self.assertEqual(hits, [])


class CrossModelSummaryTests(unittest.TestCase):
    def test_choose_delta_column_prefers_shortest_indirect_then_direct(self):
        summary = pd.DataFrame(
            columns=[
                "mean_delta_suppressed_minus_absent",
                "mean_delta_suppressed_indirect_2_minus_absent",
                "mean_delta_suppressed_indirect_minus_absent",
            ]
        )

        self.assertEqual(
            exp4.choose_delta_column(summary, "indirect"),
            "mean_delta_suppressed_indirect_minus_absent",
        )
        self.assertEqual(
            exp4.choose_delta_column(summary, "direct"),
            "mean_delta_suppressed_minus_absent",
        )

    def test_summarize_run_selects_peak_layer_and_probe_means(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "args.json").write_text(
                json.dumps(
                    {
                        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
                        "pooling": "mean_nonpad",
                        "include_indirect": True,
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                {
                    "layer": [0, 1],
                    "mean_absent": [0.10, 0.20],
                    "mean_suppressed_indirect": [0.30, 0.80],
                    "mean_delta_suppressed_indirect_minus_absent": [0.20, 0.60],
                    "ci95_low_delta_suppressed_indirect_minus_absent": [0.10, 0.40],
                    "ci95_high_delta_suppressed_indirect_minus_absent": [0.30, 0.80],
                    "paired_p_suppressed_indirect_vs_absent": [0.04, 0.01],
                    "wilcoxon_p_suppressed_indirect_vs_absent": [0.05, 0.02],
                    "cohen_d_paired_suppressed_indirect_vs_absent": [1.0, 2.0],
                }
            ).to_csv(run_dir / "salience_summary_by_layer.csv", index=False)
            pd.DataFrame(
                {
                    "probe_accuracy": [0.70, 0.90],
                    "probe_auc": [0.80, 1.00],
                    "probe_f1": [0.60, 0.80],
                }
            ).to_csv(run_dir / "probe_report_by_seed.csv", index=False)

            record, layerwise, probe_quality = exp4.summarize_run(run_dir, "indirect")

        self.assertEqual(record["model_short"], "Llama 3 8B Inst.")
        self.assertEqual(record["peak_layer"], 1)
        self.assertAlmostEqual(record["peak_delta"], 0.60)
        self.assertAlmostEqual(record["probe_auc"], 0.90)
        self.assertEqual(list(layerwise["mean_delta"]), [0.20, 0.60])
        self.assertAlmostEqual(float(probe_quality["probe_accuracy"].iloc[0]), 0.80)


class ExperimentFourTests(unittest.TestCase):
    def test_exp4_apply_model_order_sorts_preferred_models(self):
        df = pd.DataFrame(
            {
                "model_short": ["Gemma 7B IT", "Llama 3 8B Inst.", "Other"],
                "layer": [1, 1, 1],
                "mean_delta": [0.3, 0.1, 0.2],
            }
        )

        ordered = exp4.apply_model_order(df)

        self.assertEqual(list(ordered["model_short"]), ["Llama 3 8B Inst.", "Gemma 7B IT", "Other"])

    def test_exp4_cross_model_report_uses_compact_variable_tables(self):
        summary = pd.DataFrame(
            {
                "model_short": ["Llama 3 8B Inst.", "Gemma 7B IT"],
                "selected_condition": ["suppressed_indirect", "suppressed_indirect"],
                "pooling": ["mean_nonpad", "mean_nonpad"],
                "peak_layer": [4, 5],
                "peak_delta": [0.4, 0.2],
                "peak_ci95_low": [0.3, 0.1],
                "peak_ci95_high": [0.5, 0.3],
                "peak_paired_p": [0.01, 0.02],
                "peak_cohen_d": [1.2, 0.8],
                "probe_auc": [0.9, 0.85],
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            exp4.write_cross_model_report(summary, Path(tmp) / "cross_model_report.md")
            report = (Path(tmp) / "cross_model_report.md").read_text(encoding="utf-8")

        self.assertIn("# Experiment 4 Summary", report)
        self.assertIn("## cross_model_peak", report)
        self.assertIn("`model_short` | Llama 3 8B Inst.", report)
        self.assertIn("## outputs", report)


class ExperimentOneReportTests(unittest.TestCase):
    def test_write_markdown_report_uses_compact_variable_tables(self):
        args = SimpleNamespace(
            model="demo/model",
            pooling="mean_nonpad",
            concepts_path="concepts.json",
            probe_seeds="13,42",
            include_hard_negatives=True,
            include_indirect=False,
            only_indirect=False,
            indirect_mode="first",
        )
        probe_report_seed = pd.DataFrame(
            {
                "probe_accuracy": [0.8, 0.9],
                "probe_auc": [0.7, 0.9],
                "probe_f1": [0.6, 0.8],
            }
        )
        salience_summary = pd.DataFrame(
            {
                "layer": [0, 1],
                "mean_absent": [0.1, 0.2],
                "mean_suppressed": [0.3, 0.9],
                "mean_mentioned": [0.4, 0.95],
                "mean_delta_suppressed_minus_absent": [0.2, 0.7],
                "ci95_low_delta_suppressed_minus_absent": [0.1, 0.5],
                "ci95_high_delta_suppressed_minus_absent": [0.3, 0.9],
                "paired_p_suppressed_vs_absent": [0.05, 0.01],
                "wilcoxon_p_suppressed_vs_absent": [0.06, 0.02],
                "cohen_d_paired_suppressed_vs_absent": [1.2, 2.4],
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            exp1.write_markdown_report(
                args,
                probe_report_seed,
                probe_report=pd.DataFrame(),
                salience_summary=salience_summary,
                leakage_summary=None,
                outdir=tmp,
            )
            report = (Path(tmp) / "report.md").read_text(encoding="utf-8")

        self.assertIn("## probe_quality", report)
        self.assertIn("`probe_auc_mean`", report)
        self.assertIn("`selected_delta_column`", report)
        self.assertIn("`peak_layer` | 1", report)
        self.assertNotIn("Main Question", report)
        self.assertNotIn("Interpretation Guardrail", report)

    def test_write_markdown_report_falls_back_to_indirect_delta(self):
        args = SimpleNamespace(
            model="demo/model",
            pooling="mean_nonpad",
            concepts_path="concepts.json",
            probe_seeds="13,42",
            include_hard_negatives=True,
            include_indirect=True,
            only_indirect=True,
            indirect_mode="first",
        )
        probe_report_seed = pd.DataFrame(
            {
                "probe_accuracy": [0.8],
                "probe_auc": [0.9],
                "probe_f1": [0.7],
            }
        )
        salience_summary = pd.DataFrame(
            {
                "layer": [0, 1],
                "mean_absent": [0.1, 0.2],
                "mean_suppressed_indirect": [0.3, 0.8],
                "mean_mentioned": [0.4, 0.9],
                "mean_delta_suppressed_indirect_minus_absent": [0.2, 0.6],
                "ci95_low_delta_suppressed_indirect_minus_absent": [0.1, 0.5],
                "ci95_high_delta_suppressed_indirect_minus_absent": [0.3, 0.7],
                "paired_p_suppressed_indirect_vs_absent": [0.05, 0.01],
                "wilcoxon_p_suppressed_indirect_vs_absent": [0.06, 0.02],
                "cohen_d_paired_suppressed_indirect_vs_absent": [1.2, 2.4],
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            exp1.write_markdown_report(
                args,
                probe_report_seed,
                probe_report=pd.DataFrame(),
                salience_summary=salience_summary,
                leakage_summary=None,
                outdir=tmp,
            )
            report = (Path(tmp) / "report.md").read_text(encoding="utf-8")

        self.assertIn("`only_indirect` | True", report)
        self.assertIn("`selected_delta_column` | mean_delta_suppressed_indirect_minus_absent", report)
        self.assertIn("`peak_layer` | 1", report)


class LayerRegionTests(unittest.TestCase):
    def test_assign_layer_regions_uses_relative_layer_rank_per_model(self):
        df = pd.DataFrame(
            {
                "model_short": ["A"] * 6 + ["B"] * 3,
                "layer": [0, 1, 2, 3, 4, 5, 10, 20, 30],
                "mean_delta": np.arange(9, dtype=float),
            }
        )

        assigned = exp4.assign_layer_regions(df)

        self.assertEqual(
            list(assigned[assigned["model_short"] == "A"]["region"]),
            ["early", "early", "middle", "middle", "late", "late"],
        )
        self.assertEqual(
            list(assigned[assigned["model_short"] == "B"]["region"]),
            ["early", "middle", "late"],
        )


class ConditionOrderingTests(unittest.TestCase):
    def test_resolve_condition_falls_back_to_first_indirect_variant(self):
        df = pd.DataFrame(
            columns=["absent", "mentioned", "suppressed_indirect_2", "suppressed_indirect_1"]
        )

        self.assertEqual(exp4.resolve_condition(df, "suppressed_indirect"), "suppressed_indirect_1")

    def test_paired_bootstrap_diff_ignores_unmatched_nan_rows(self):
        mean_diff, lo, hi = exp4.paired_bootstrap_diff(
            [3.0, np.nan, 5.0],
            [1.0, 10.0, 2.0],
            n_boot=100,
            seed=1,
        )

        self.assertAlmostEqual(mean_diff, 2.5)
        self.assertLessEqual(lo, mean_diff)
        self.assertGreaterEqual(hi, mean_diff)


if __name__ == "__main__":
    unittest.main(verbosity=2)
