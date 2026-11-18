"""Rich CLI Demo: 4-Session Cross-Session Memory, Conflict Overrides, Ebbinghaus Decay, & Multi-Hop Q&A."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from graph_memory import DialogueTurn, HierarchicalGraphMemoryEngine


def main() -> None:
    console = Console()
    console.print(
        Panel.fit(
            "[bold cyan]04-F2 Hierarchical & Temporal Graph Long-Term Memory Engine[/bold cyan]\n"
            "Tier 1 Working Buffer  |  Tier 2 SQLite Episodic Vectors  |  Tier 3 Bi-Temporal Property Graph",
            border_style="cyan",
        )
    )

    engine = HierarchicalGraphMemoryEngine(
        db_dir=":memory:",
        working_capacity=4,
        salience_threshold=0.35,
        decay_lambda=0.018,
        reinforcement_alpha=1.8,
        consolidation_theta=0.25,
    )
    engine.initialize_sync()
    engine.register_user("u_demo", "Alex Chen")

    sessions: list[tuple[str, list[tuple[int, str]]]] = [
        (
            "Session 1 (Onboarding & Initial Tech Stack)",
            [
                (1, "Currently I live in Beijing."),
                (2, "My primary database is MySQL."),
                (3, "I started leading Project Helios."),
                (4, "Had spicy ramen for lunch today before heading back to the lab."),
                (5, "I actively use PyTorch and I actively use LangChain."),
            ],
        ),
        (
            "Session 2 (City Relocation & Collaborator Onboarding)",
            [
                (12, "I relocated to Hangzhou."),
                (14, "For Project Helios, I teamed up with Dr. Mei Zhou."),
                (16, "Complaining about rainy campus weather all morning."),
            ],
        ),
        (
            "Session 3 (Database Migration & Library Deprecation)",
            [
                (24, "We switched our primary database from MySQL to PostgreSQL."),
                (27, "I stopped using LangChain."),
                (29, "Project Helios uses primary database PostgreSQL."),
            ],
        ),
        (
            "Session 4 (3-Hop Organization Link & Long-Horizon Consolidation)",
            [
                (36, "Dr. Mei Zhou is affiliated with Tsinghua AIR."),
                (38, "Dr. Mei Zhou targets conference NeurIPS."),
                (40, "Grabbed an iced oat latte from the corner cafe."),
            ],
        ),
    ]

    event_table = Table(title="Live Cross-Session Triple Extraction & Conflict Resolution Log")
    event_table.add_column("Session / Turn", style="cyan")
    event_table.add_column("Action", style="bold yellow")
    event_table.add_column("Triple (Subject, Predicate, Object)", style="white")
    event_table.add_column("Conflict / Superseding Lineage", style="magenta")

    for session_title, turn_items in sessions:
        for turn_id, text in turn_items:
            turn = DialogueTurn(
                turn_id=turn_id,
                session_id=session_title.split()[1],
                user_id="u_demo",
                content=text,
            )
            events = engine.ingest_turn_sync(turn, user_display_name="Alex Chen")
            for ev in events:
                s_disp = engine.temporal_graph.display_name(ev.subject)
                o_disp = engine.temporal_graph.display_name(ev.new_object)
                old_disp = (
                    engine.temporal_graph.display_name(ev.old_object) if ev.old_object else "-"
                )
                lineage_note = "-"
                if ev.action == "SUPERSEDED":
                    lineage_note = f"Closed '{old_disp}' (valid_until={turn_id}) -> Superseded by {ev.new_edge_id}"
                elif ev.action == "RETRACTED":
                    lineage_note = f"Retracted active '{o_disp}' at Turn {turn_id}"
                elif ev.action == "ADDED_COEXISTING":
                    lineage_note = f"Coexists alongside '{old_disp}'"

                event_table.add_row(
                    f"Turn {turn_id:02d}",
                    ev.action,
                    f"({s_disp}, {ev.predicate}, {o_disp})",
                    lineage_note,
                )
        engine.flush_working_memory_sync("u_demo")

    console.print(event_table)

    # Reinforce core Project Helios edge via an intermediate query, then inspect Ebbinghaus scores
    engine.retrieve_sync("u_demo", "Which project is Alex Chen leading?", current_turn=42)

    decay_table = Table(title="Ebbinghaus Retention Scores S(Delta t) Before Consolidation (Turn 45)")
    decay_table.add_column("Active Edge", style="cyan")
    decay_table.add_column("I_0", justify="right")
    decay_table.add_column("Access Count", justify="right")
    decay_table.add_column("Delta t", justify="right")
    decay_table.add_column("S(Delta t)", justify="right", style="bold green")
    decay_table.add_column("Decision (theta=0.25)", justify="center")

    for edge in engine.temporal_graph.get_active_edges(user_id="u_demo"):
        score = engine.forgetting_engine.retention_score(edge, current_turn=45)
        dt = max(0, 45 - edge.last_accessed_turn)
        decision = (
            "[bold red]ARCHIVE (Prune)[/bold red]"
            if score < 0.25
            else "[bold green]RETAIN[/bold green]"
        )
        decay_table.add_row(
            engine.temporal_graph.format_edge_readable(edge),
            f"{edge.base_importance:.2f}",
            str(edge.access_count),
            str(dt),
            f"{score:.3f}",
            decision,
        )
    console.print(decay_table)

    report = engine.consolidate_sync(current_turn=45, user_id="u_demo", theta=0.25)
    console.print(
        f"[bold yellow]Consolidation Pass @ Turn 45:[/bold yellow] "
        f"Archived {report.archived_count} stale chatter edges | "
        f"Retained {report.retained_count} core edges | "
        f"Compression Ratio = {report.compression_ratio * 100:.1f}%\n"
    )

    # Execute 4 showcase queries across Current, Historical, Retraction, and 3-Hop Reasoning
    queries = [
        ("Fact Override (City)", "Which city does Alex Chen currently live in?"),
        ("Fact Override (DB)", "What is Alex Chen's primary database right now?"),
        ("Explicit Retraction", "Which libraries does Alex Chen actively use now?"),
        ("Historical Lineage", "Which city did Alex Chen live in before moving to Hangzhou?"),
        (
            "3-Hop Graph Reasoning",
            "Which organization is the collaborator on the project led by Alex Chen affiliated with?",
        ),
    ]

    qa_table = Table(title="Hybrid Temporal-Aware Retrieval & Multi-Hop Q&A Results")
    qa_table.add_column("Probe Type", style="bold cyan")
    qa_table.add_column("Question", style="white")
    qa_table.add_column("Detected Intent", style="magenta")
    qa_table.add_column("Grounded Answer", style="bold green")
    qa_table.add_column("Tokens / Latency", justify="right")

    for q_type, question in queries:
        ans, res = engine.answer_query_sync("u_demo", question, current_turn=46)
        qa_table.add_row(
            q_type,
            question,
            res.intent.value,
            ans,
            f"{res.prompt_tokens} tok / {res.latency_ms:.2f} ms",
        )
    console.print(qa_table)
    engine.close_sync()


if __name__ == "__main__":
    main()
