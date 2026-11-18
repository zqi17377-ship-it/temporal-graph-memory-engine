"""Long-Horizon Cross-Session Benchmark & Ebbinghaus Decay Pareto Sweep Runner.

Compares 4 Memory Baselines + 3-Point Consolidation Threshold Sweep (theta in {0.0, 0.25, 0.55})
across 4 longitudinal personas (20 sessions, 240 turns) and 60 structured QA probes.
Saves real empirical results to:
  - results/benchmark_results.json
  - results/evaluation_report.md
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.table import Table

from eval.dataset import PersonaHistory, QAProbe, build_benchmark_dataset
from graph_memory import (
    ContextGroundedReader,
    EmbeddingService,
    EpisodicVectorStore,
    HierarchicalGraphMemoryEngine,
    estimate_tokens,
)

CATEGORIES = [
    "single_hop_long_range",
    "fact_conflict_override",
    "temporal_historical_lineage",
    "multi_hop_graph_reasoning",
]


def evaluate_prediction(probe: QAProbe, prediction: str) -> tuple[bool, bool, str]:
    """Evaluate a grounded answer against expected keywords and forbidden stale values.

    Returns:
        (is_correct, has_stale_leakage, failure_reason)
    """
    pred_low = prediction.lower()
    if "insufficient_memory" in pred_low or not prediction.strip():
        return False, False, "missing_recall_evicted_or_unretrieved"

    stale_leaked = any(stale.lower() in pred_low for stale in probe.forbidden_stale)
    all_expected_present = all(exp.lower() in pred_low for exp in probe.expected_keywords)

    # For evolution sequence probes, also verify chronological order if >= 2 keywords
    if (
        all_expected_present
        and probe.category == "temporal_historical_lineage"
        and len(probe.expected_keywords) >= 2
    ):
        positions = [pred_low.find(exp.lower()) for exp in probe.expected_keywords]
        if positions != sorted(positions):
            return False, stale_leaked, "chronological_order_inversion"

    if stale_leaked:
        return False, True, "stale_conflict_hallucination"
    if not all_expected_present:
        if probe.category == "multi_hop_graph_reasoning":
            return False, False, "broken_multi_hop_chain"
        if probe.category == "temporal_historical_lineage":
            return False, False, "incomplete_historical_lineage"
        return False, False, "wrong_or_incomplete_entity"

    return True, False, "none"


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))
    return round(s[idx], 3)


def run_sliding_window_baseline(
    dataset: list[PersonaHistory],
    reader: ContextGroundedReader,
    window_size: int = 10,
) -> dict:
    """Baseline 1: Sliding Window keeping only the last `window_size` raw dialogue turns."""
    records: list[dict] = []
    latencies: list[float] = []
    tokens_list: list[int] = []

    for persona in dataset:
        recent_turns = persona.turns[-window_size:]
        for probe in persona.probes:
            t0 = time.perf_counter()
            ctx = "\n".join(
                f"[Turn {t.turn_id}] {persona.display_name}: {t.content}" for t in recent_turns
            )
            lat_ms = (time.perf_counter() - t0) * 1000.0
            ans = reader.answer_offline(probe.question, ctx, persona.display_name)
            ok, stale, reason = evaluate_prediction(probe, ans)

            latencies.append(lat_ms)
            tokens_list.append(estimate_tokens(ctx))
            records.append(
                {
                    "probe_id": probe.probe_id,
                    "category": probe.category,
                    "question": probe.question,
                    "prediction": ans,
                    "expected": probe.expected_keywords,
                    "correct": ok,
                    "stale_leakage": stale,
                    "failure_reason": reason,
                }
            )

    return _summarize_run("Sliding Window (Last 10 Turns)", records, tokens_list, latencies)


def run_full_context_baseline(
    dataset: list[PersonaHistory],
    reader: ContextGroundedReader,
) -> dict:
    """Baseline 2: Full History Context Stuffing (all 60 turns raw into prompt)."""
    records: list[dict] = []
    latencies: list[float] = []
    tokens_list: list[int] = []

    for persona in dataset:
        for probe in persona.probes:
            t0 = time.perf_counter()
            ctx = "\n".join(
                f"[Turn {t.turn_id}] {persona.display_name}: {t.content}" for t in persona.turns
            )
            lat_ms = (time.perf_counter() - t0) * 1000.0
            ans = reader.answer_offline(probe.question, ctx, persona.display_name)
            ok, stale, reason = evaluate_prediction(probe, ans)

            latencies.append(lat_ms)
            tokens_list.append(estimate_tokens(ctx))
            records.append(
                {
                    "probe_id": probe.probe_id,
                    "category": probe.category,
                    "question": probe.question,
                    "prediction": ans,
                    "expected": probe.expected_keywords,
                    "correct": ok,
                    "stale_leakage": stale,
                    "failure_reason": reason,
                }
            )

    return _summarize_run("Full History Context Stuffing", records, tokens_list, latencies)


def run_flat_vector_rag_baseline(
    dataset: list[PersonaHistory],
    reader: ContextGroundedReader,
    embedder: EmbeddingService,
    top_k: int = 5,
) -> dict:
    """Baseline 3: Flat Vector RAG Memory (Top-K cosine chunks, no graph, no conflict superseding)."""
    store = EpisodicVectorStore(":memory:", embedder=embedder)
    store.initialize_sync()

    for persona in dataset:
        for turn in persona.turns:
            store.add_turn_as_episode_sync(
                turn.model_copy(update={"salience_score": 0.5}),
                episode_id=f"flat_{persona.user_id}_{turn.turn_id:04d}",
            )

    records: list[dict] = []
    latencies: list[float] = []
    tokens_list: list[int] = []

    for persona in dataset:
        for probe in persona.probes:
            t0 = time.perf_counter()
            hits = store.search_sync(probe.question, user_id=persona.user_id, top_k=top_k)
            ctx = "\n".join(
                f"[Turn {ch.turn_id} | sim={sim:.2f}] {persona.display_name}: {ch.content}"
                for ch, sim in hits
            )
            lat_ms = (time.perf_counter() - t0) * 1000.0
            ans = reader.answer_offline(probe.question, ctx, persona.display_name)
            ok, stale, reason = evaluate_prediction(probe, ans)

            latencies.append(lat_ms)
            tokens_list.append(estimate_tokens(ctx))
            records.append(
                {
                    "probe_id": probe.probe_id,
                    "category": probe.category,
                    "question": probe.question,
                    "prediction": ans,
                    "expected": probe.expected_keywords,
                    "correct": ok,
                    "stale_leakage": stale,
                    "failure_reason": reason,
                }
            )

    store.close_sync()
    return _summarize_run(
        "Flat Vector RAG Memory (Top-5 Chunks)", records, tokens_list, latencies
    )


def run_ours_hierarchical_graph_engine(
    dataset: list[PersonaHistory],
    embedder: EmbeddingService,
    theta: float = 0.25,
    label: str | None = None,
) -> dict:
    """Method 4: Ours — Hierarchical Temporal Graph + Vector Memory with Conflict Resolution & Decay."""
    engine = HierarchicalGraphMemoryEngine(
        db_dir=":memory:",
        working_capacity=6,
        salience_threshold=0.35,
        decay_lambda=0.015,
        reinforcement_alpha=1.8,
        consolidation_theta=theta,
        embedder=embedder,
    )
    engine.initialize_sync()

    # Ingest longitudinal sessions, flush working memory, and run Ebbinghaus consolidation
    # at evaluation horizon (Turn 65) so all thresholds evaluate against an identical graph pool.
    for persona in dataset:
        engine.register_user(persona.user_id, persona.display_name)
        for turn in persona.turns:
            engine.ingest_turn_sync(turn, user_display_name=persona.display_name)
        engine.flush_working_memory_sync(persona.user_id)
        engine.consolidate_sync(current_turn=65, user_id=persona.user_id, theta=theta)

    # Measure cumulative graph edge compression across all users before QA probes
    all_edges = engine.temporal_graph.get_all_edges()
    active_pool = [e for e in all_edges if e.valid_until_turn is None]
    archived_edges = [e for e in active_pool if e.archived]
    retained_edges = [e for e in active_pool if not e.archived]
    edge_compression_pct = (
        round(100.0 * len(archived_edges) / len(active_pool), 2) if active_pool else 0.0
    )

    records: list[dict] = []
    latencies: list[float] = []
    tokens_list: list[int] = []

    for persona in dataset:
        for probe in persona.probes:
            ans, res = engine.answer_query_sync(
                user_id=persona.user_id,
                query=probe.question,
                current_turn=62,
                reinforce_on_read=False,  # Keep evaluation order-invariant across probes
            )
            ok, stale, reason = evaluate_prediction(probe, ans)
            latencies.append(res.latency_ms)
            tokens_list.append(res.prompt_tokens)
            records.append(
                {
                    "probe_id": probe.probe_id,
                    "category": probe.category,
                    "question": probe.question,
                    "prediction": ans,
                    "expected": probe.expected_keywords,
                    "correct": ok,
                    "stale_leakage": stale,
                    "failure_reason": reason,
                }
            )

    engine.close_sync()
    run_name = label or f"Ours: Hierarchical Temporal Graph + Vector (theta={theta:.2f})"
    summary = _summarize_run(run_name, records, tokens_list, latencies)
    summary["theta"] = theta
    summary["active_edges_retained"] = len(retained_edges)
    summary["active_edges_archived"] = len(archived_edges)
    summary["graph_compression_pct"] = edge_compression_pct
    return summary


def _summarize_run(
    name: str,
    records: list[dict],
    tokens_list: list[int],
    latencies: list[float],
) -> dict:
    total = len(records)
    correct = sum(1 for r in records if r["correct"])
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r)

    cat_acc: dict[str, float] = {}
    for cat in CATEGORIES:
        items = by_cat[cat]
        cat_acc[cat] = round(100.0 * sum(1 for x in items if x["correct"]) / max(1, len(items)), 2)

    # Conflict Resolution Precision across fact_conflict_override probes:
    # Precision = conflict_free_valid_answers / (conflict_free_valid_answers + stale_or_wrong_answers_attempted)
    conflict_items = by_cat["fact_conflict_override"]
    clean_conflict_hits = sum(1 for x in conflict_items if x["correct"] and not x["stale_leakage"])
    attempted_conflict = sum(
        1 for x in conflict_items if "insufficient_memory" not in x["prediction"].lower()
    )
    conflict_precision = (
        round(100.0 * clean_conflict_hits / attempted_conflict, 2)
        if attempted_conflict > 0
        else 0.0
    )
    stale_leak_rate = round(
        100.0 * sum(1 for x in conflict_items if x["stale_leakage"]) / max(1, len(conflict_items)),
        2,
    )

    failure_counts: dict[str, int] = defaultdict(int)
    failure_examples: list[dict] = []
    for r in records:
        if not r["correct"]:
            failure_counts[r["failure_reason"]] += 1
            if len(failure_examples) < 5:
                failure_examples.append(
                    {
                        "probe_id": r["probe_id"],
                        "category": r["category"],
                        "question": r["question"],
                        "prediction": r["prediction"],
                        "expected": r["expected"],
                        "reason": r["failure_reason"],
                    }
                )

    avg_tokens = round(sum(tokens_list) / max(1, len(tokens_list)), 1)
    return {
        "system": name,
        "total_probes": total,
        "correct_probes": correct,
        "overall_accuracy_pct": round(100.0 * correct / max(1, total), 2),
        "category_accuracy_pct": cat_acc,
        "conflict_resolution_precision_pct": conflict_precision,
        "stale_conflict_leakage_pct": stale_leak_rate,
        "avg_prompt_tokens": avg_tokens,
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "failure_attribution": dict(failure_counts),
        "sample_failures": failure_examples,
    }


def write_markdown_report(payload: dict, out_path: Path) -> None:
    baselines = payload["baselines"]
    sweep = payload["decay_pareto_sweep"]
    ds = payload["dataset_summary"]

    lines = [
        "# Long-Horizon Cross-Session Memory Benchmark Report",
        "",
        "## 1. Dataset & Experimental Protocol",
        f"- **Personas**: {ds['num_personas']} longitudinal CS/AI engineering users (`Alex Chen`, `Su Yan`, `Zhao Kai`, `Lin Wei`)",
        f"- **Sessions & Horizon**: {ds['total_sessions']} sessions, **{ds['total_turns']} dialogue turns** (~{ds['noise_turn_ratio_pct']}% casual chitchat/filler noise)",
        f"- **Explicit State Overrides & Retractions**: {ds['explicit_fact_overrides']} cross-session updates",
        f"- **Evaluation Probes**: **{ds['total_probes']} QA probes** (15 per category across 4 categories)",
        "",
        "## 2. Main Ablation: 4 Memory Architectures",
        "",
        "| Memory System | Overall Acc | Long-Range (15) | Conflict Override (15) | Historical Lineage (15) | Multi-Hop Graph (15) | Conflict Precision | Avg Tokens | Token Reduction vs Full | P95 Latency (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for b in baselines:
        ca = b["category_accuracy_pct"]
        lines.append(
            f"| **{b['system']}** | **{b['overall_accuracy_pct']:.1f}%** | "
            f"{ca['single_hop_long_range']:.1f}% | {ca['fact_conflict_override']:.1f}% | "
            f"{ca['temporal_historical_lineage']:.1f}% | {ca['multi_hop_graph_reasoning']:.1f}% | "
            f"{b['conflict_resolution_precision_pct']:.1f}% | {b['avg_prompt_tokens']:.1f} | "
            f"{b['token_reduction_vs_full_pct']:+.1f}% | {b['p95_latency_ms']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## 3. Ebbinghaus Forgetting Threshold Sweep (Compression vs Accuracy Pareto Frontier)",
            "",
            "| Consolidation Threshold $\\theta$ | Active Edges Retained | Edges Archived | Graph Compression Rate | Overall Acc | Multi-Hop Acc | Long-Range Acc | Conflict Override Acc | Avg Prompt Tokens |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for s in sweep:
        ca = s["category_accuracy_pct"]
        lines.append(
            f"| `$\\theta = {s['theta']:.2f}$` | {s['active_edges_retained']} | "
            f"{s['active_edges_archived']} | **{s['graph_compression_pct']:.1f}%** | "
            f"**{s['overall_accuracy_pct']:.1f}%** | {ca['multi_hop_graph_reasoning']:.1f}% | "
            f"{ca['single_hop_long_range']:.1f}% | {ca['fact_conflict_override']:.1f}% | "
            f"{s['avg_prompt_tokens']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## 4. Failure Case Attribution Across Architectures",
            "",
        ]
    )
    for b in baselines:
        fa = b["failure_attribution"]
        fa_str = ", ".join(f"`{k}`: {v}" for k, v in fa.items()) if fa else "Zero failures (0)"
        lines.append(f"- **{b['system']}**: {fa_str}")

    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    console = Console()
    dataset = build_benchmark_dataset()
    embedder = EmbeddingService(dim=128, use_openai=False)
    reader = ContextGroundedReader(use_openai=False)

    # Dataset statistics
    total_turns = sum(len(p.turns) for p in dataset)
    total_sessions = len({t.session_id for p in dataset for t in p.turns})
    total_probes = sum(len(p.probes) for p in dataset)
    filler_turns = sum(
        1
        for p in dataset
        for t in p.turns
        if t.content in set(__import__("eval.dataset", fromlist=["_FILLER_TEMPLATES"])._FILLER_TEMPLATES)
    )

    # Execute 4 core baselines
    b_window = run_sliding_window_baseline(dataset, reader, window_size=10)
    b_full = run_full_context_baseline(dataset, reader)
    b_flat = run_flat_vector_rag_baseline(dataset, reader, embedder, top_k=5)

    # Execute 3-point Ebbinghaus decay threshold sweep: theta in {0.0, 0.25, 0.55}
    s_000 = run_ours_hierarchical_graph_engine(
        dataset, embedder, theta=0.00, label="Ours (theta=0.00, No Decay Pruning)"
    )
    s_025 = run_ours_hierarchical_graph_engine(
        dataset,
        embedder,
        theta=0.25,
        label="Ours: Hierarchical Temporal Graph + Vector (theta=0.25)",
    )
    s_055 = run_ours_hierarchical_graph_engine(
        dataset, embedder, theta=0.55, label="Ours (theta=0.55, Aggressive Pruning)"
    )

    full_tokens = b_full["avg_prompt_tokens"]
    baselines = [b_window, b_full, b_flat, s_025]
    for item in baselines + [s_000, s_055]:
        item["token_reduction_vs_full_pct"] = round(
            100.0 * (1.0 - item["avg_prompt_tokens"] / max(1.0, full_tokens)), 2
        )

    sweep = [s_000, s_025, s_055]

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "benchmark_results.json"
    md_path = results_dir / "evaluation_report.md"

    payload = {
        "dataset_summary": {
            "num_personas": len(dataset),
            "total_sessions": total_sessions,
            "total_turns": total_turns,
            "noise_turns": filler_turns,
            "noise_turn_ratio_pct": round(100.0 * filler_turns / total_turns, 1),
            "explicit_fact_overrides": 21,
            "total_probes": total_probes,
            "probes_per_category": 15,
        },
        "baselines": baselines,
        "decay_pareto_sweep": sweep,
    }

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown_report(payload, md_path)

    # Pretty Rich CLI Summary Tables
    t1 = Table(title="Long-Horizon Cross-Session Memory Benchmark (60 Probes / 240 Turns)")
    t1.add_column("Memory Architecture", style="bold cyan")
    t1.add_column("Overall Acc", justify="right", style="bold green")
    t1.add_column("Long-Range", justify="right")
    t1.add_column("Conflict", justify="right")
    t1.add_column("Historical", justify="right")
    t1.add_column("Multi-Hop", justify="right")
    t1.add_column("Conflict Prec", justify="right")
    t1.add_column("Avg Tokens", justify="right")
    t1.add_column("Token Saving", justify="right")
    t1.add_column("P95 (ms)", justify="right")

    for b in baselines:
        ca = b["category_accuracy_pct"]
        t1.add_row(
            b["system"],
            f"{b['overall_accuracy_pct']:.1f}%",
            f"{ca['single_hop_long_range']:.1f}%",
            f"{ca['fact_conflict_override']:.1f}%",
            f"{ca['temporal_historical_lineage']:.1f}%",
            f"{ca['multi_hop_graph_reasoning']:.1f}%",
            f"{b['conflict_resolution_precision_pct']:.1f}%",
            f"{b['avg_prompt_tokens']:.1f}",
            f"{b['token_reduction_vs_full_pct']:+.1f}%",
            f"{b['p95_latency_ms']:.2f}",
        )
    console.print(t1)

    t2 = Table(title="Ebbinghaus Forgetting Sweep: Compression vs Accuracy Pareto Frontier")
    t2.add_column("Threshold theta", justify="center", style="bold magenta")
    t2.add_column("Retained Edges", justify="right")
    t2.add_column("Archived Edges", justify="right")
    t2.add_column("Compression %", justify="right", style="bold yellow")
    t2.add_column("Overall Acc", justify="right", style="bold green")
    t2.add_column("Multi-Hop Acc", justify="right")
    t2.add_column("Long-Range Acc", justify="right")
    t2.add_column("Avg Tokens", justify="right")

    for s in sweep:
        ca = s["category_accuracy_pct"]
        t2.add_row(
            f"{s['theta']:.2f}",
            str(s["active_edges_retained"]),
            str(s["active_edges_archived"]),
            f"{s['graph_compression_pct']:.1f}%",
            f"{s['overall_accuracy_pct']:.1f}%",
            f"{ca['multi_hop_graph_reasoning']:.1f}%",
            f"{ca['single_hop_long_range']:.1f}%",
            f"{s['avg_prompt_tokens']:.1f}",
        )
    console.print(t2)
    console.print(f"[bold green]Saved benchmark artifacts:[/bold green] {json_path} & {md_path}")


if __name__ == "__main__":
    main()
