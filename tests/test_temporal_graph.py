"""Tests for Temporal Knowledge Graph: canonicalization, point-in-time queries, 2-3 hop BFS/PPR, and SQLite recovery."""

from __future__ import annotations

from pathlib import Path

from graph_memory import (
    ConflictResolver,
    EntityType,
    ExtractedFact,
    ForgettingEngine,
    TemporalKnowledgeGraph,
)


def test_entity_alias_canonicalization() -> None:
    graph = TemporalKnowledgeGraph(":memory:")
    graph.initialize_sync()

    n1 = graph.upsert_node_sync("psql", EntityType.TECH_STACK, current_turn=1)
    n2 = graph.upsert_node_sync("PostgreSQL 16", EntityType.TECH_STACK, current_turn=8)
    n3 = graph.upsert_node_sync("PostgreSQL", EntityType.TECH_STACK, current_turn=12)

    assert n1.node_id == n2.node_id == n3.node_id
    assert n3.canonical_name == "PostgreSQL"
    assert "psql" in n3.aliases
    assert "PostgreSQL 16" in n3.aliases
    graph.close_sync()


def test_point_in_time_snapshot_vs_current_active_query() -> None:
    graph = TemporalKnowledgeGraph(":memory:")
    graph.initialize_sync()
    resolver = ConflictResolver()

    for turn_idx, db_name in [(3, "MySQL"), (18, "MongoDB"), (36, "PostgreSQL")]:
        resolver.apply_fact_sync(
            graph,
            ExtractedFact(
                subject="Project Atlas",
                subject_type=EntityType.PROJECT,
                predicate="primary_db",
                object=db_name,
                object_type=EntityType.TECH_STACK,
            ),
            user_id="u_atlas",
            current_turn=turn_idx,
        )

    snap_t10 = graph.get_edges_at_turn(10, user_id="u_atlas", predicate="primary_db")
    assert len(snap_t10) == 1
    assert graph.display_name(snap_t10[0].object) == "MySQL"

    snap_t25 = graph.get_edges_at_turn(25, user_id="u_atlas", predicate="primary_db")
    assert len(snap_t25) == 1
    assert graph.display_name(snap_t25[0].object) == "MongoDB"

    current = graph.get_active_edges(user_id="u_atlas", predicate="primary_db")
    assert len(current) == 1
    assert graph.display_name(current[0].object) == "PostgreSQL"
    graph.close_sync()


def test_two_and_three_hop_bfs_graph_traversal() -> None:
    graph = TemporalKnowledgeGraph(":memory:")
    graph.initialize_sync()
    resolver = ConflictResolver()

    triples = [
        (2, "Lin Wei", EntityType.PERSON, "works_on", "Project Nebula", EntityType.PROJECT),
        (14, "Project Nebula", EntityType.PROJECT, "collaborates_with", "Dr. Mei Zhou", EntityType.PERSON),
        (29, "Dr. Mei Zhou", EntityType.PERSON, "works_at", "Tsinghua AIR", EntityType.ORGANIZATION),
        (31, "Project Nebula", EntityType.PROJECT, "primary_db", "PostgreSQL", EntityType.TECH_STACK),
    ]
    for turn_idx, s, st, p, o, ot in triples:
        resolver.apply_fact_sync(
            graph,
            ExtractedFact(subject=s, subject_type=st, predicate=p, object=o, object_type=ot),
            user_id="u_lin",
            current_turn=turn_idx,
        )

    paths = graph.find_multi_hop_paths(["Lin Wei"], max_hops=3, user_id="u_lin", active_only=True)
    rendered_paths = [
        " -> ".join(graph.format_edge_readable(e) for e in path) for path in paths
    ]

    assert any(
        "(Lin Wei, works_on, Project Nebula)" in rp
        and "(Project Nebula, primary_db, PostgreSQL)" in rp
        for rp in rendered_paths
    )
    assert any(
        "(Lin Wei, works_on, Project Nebula)" in rp
        and "(Project Nebula, collaborates_with, Dr. Mei Zhou)" in rp
        and "(Dr. Mei Zhou, works_at, Tsinghua AIR)" in rp
        for rp in rendered_paths
    )
    graph.close_sync()


def test_personalized_pagerank_ranks_connected_ego_subgraph_above_disconnected_noise() -> None:
    graph = TemporalKnowledgeGraph(":memory:")
    graph.initialize_sync()
    resolver = ConflictResolver()
    fe = ForgettingEngine()

    resolver.apply_fact_sync(
        graph,
        ExtractedFact(
            subject="Su Yan",
            subject_type=EntityType.PERSON,
            predicate="works_on",
            object="Project Prism",
            object_type=EntityType.PROJECT,
            base_importance=0.92,
        ),
        user_id="u_su",
        current_turn=10,
    )
    resolver.apply_fact_sync(
        graph,
        ExtractedFact(
            subject="Project Prism",
            subject_type=EntityType.PROJECT,
            predicate="primary_db",
            object="ClickHouse",
            object_type=EntityType.TECH_STACK,
            base_importance=0.90,
        ),
        user_id="u_su",
        current_turn=12,
    )
    resolver.apply_fact_sync(
        graph,
        ExtractedFact(
            subject="Random Stranger",
            subject_type=EntityType.PERSON,
            predicate="mentioned_topic",
            object="Weather",
            object_type=EntityType.PREFERENCE,
            base_importance=0.25,
        ),
        user_id="u_su",
        current_turn=1,
    )

    ranked = graph.personalized_pagerank(
        seed_node_ids=["Su Yan"],
        current_turn=20,
        user_id="u_su",
        top_k_edges=2,
        forgetting_engine=fe,
    )
    assert len(ranked) == 2
    preds = {e.predicate for e in ranked}
    assert preds == {"works_on", "primary_db"}
    graph.close_sync()


def test_sqlite_persistence_and_cold_restart_recovery(tmp_path: Path) -> None:
    db_file = tmp_path / "persisted_graph.sqlite3"
    graph1 = TemporalKnowledgeGraph(db_file)
    graph1.initialize_sync()
    resolver = ConflictResolver()

    resolver.apply_fact_sync(
        graph1,
        ExtractedFact(
            subject="Zhao Kai",
            subject_type=EntityType.PERSON,
            predicate="lives_in",
            object="bj",
            object_type=EntityType.CITY,
        ),
        user_id="u_zhao",
        current_turn=4,
    )
    resolver.apply_fact_sync(
        graph1,
        ExtractedFact(
            subject="Zhao Kai",
            subject_type=EntityType.PERSON,
            predicate="lives_in",
            object="sz",
            object_type=EntityType.CITY,
        ),
        user_id="u_zhao",
        current_turn=22,
    )
    graph1.close_sync()

    # Re-open from SQLite file in a brand-new instance
    graph2 = TemporalKnowledgeGraph(db_file)
    graph2.initialize_sync()

    active = graph2.get_active_edges(user_id="u_zhao", predicate="lives_in")
    assert len(active) == 1
    assert graph2.display_name(active[0].object) == "Shenzhen"

    lineage = graph2.get_predicate_lineage("Zhao Kai", "lives_in", user_id="u_zhao")
    assert len(lineage) == 2
    assert graph2.display_name(lineage[0].object) == "Beijing"
    assert lineage[0].valid_until_turn == 22
    assert lineage[0].superseded_by == lineage[1].edge_id
    graph2.close_sync()
